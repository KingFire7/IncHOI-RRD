import os
import sys
import torch
import random
import warnings
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from pvic import build_detector
from utils_incremental import custom_collate, CustomisedDLE, DataFactory, get_base_dataset
from mir_utils import dynamic_replay_sort_by_confidence, interleave_replay_and_new_auto, load_mir, get_rare_nonrare_sets
from configs import base_detector_args, advanced_detector_args

import json

warnings.filterwarnings("ignore")

# === 新增函数: 实现论文 arXiv:2510.27020 的筛选分割逻辑 ===
def get_paper_split(args, num_phases=5):
    print(f"Generating {num_phases}-phase split following paper protocol...")
    print(f"Loading metadata from {args.hoi_path}...")

    with open(args.hoi_path, 'r') as f:
        meta_data = json.load(f)

    correspondence = meta_data['correspondence'] # list of [hoi_id, obj_id, verb_id]
    verbs_list = meta_data['verbs']

    # 1. 识别需要剔除的动作 ID
    # 论文剔除: no_interaction (通常 id=57)
    # 论文还剔除: 4 body motions (walk, run, jump, ?) 和 1 point_instr
    # 这里我们优先剔除 no_interaction，其他根据 HICO 标准 ID
    try:
        no_interaction_id = verbs_list.index('no_interaction')
    except ValueError:
        no_interaction_id = 57 # Fallback for standard HICO

    valid_hois = []

    # 2. 预处理筛选
    for item in correspondence:
        hoi_id, obj_id, verb_id = item[0], item[1], item[2]

        # 过滤 no_interaction
        if args.filter_no_interaction and verb_id == no_interaction_id:
            continue

        # 构造 item dict
        valid_hois.append({'id': hoi_id, 'object_index': obj_id, 'action_index': verb_id})

    print(f"Total valid HOI candidates after filtering: {len(valid_hois)}")

    # 随机打乱
    random.seed(args.seed)
    random.shuffle(valid_hois)

    # 3. 增量筛选分配 (New Object or New Relation -> Train, else -> Unseen)
    tasks = [[] for _ in range(num_phases)]
    unseen_hois = []

    seen_objects = set()
    seen_verbs = set()

    # 设置每阶段的类别配额 (参考论文 Table 4/5)
    # Phase 1-4: ~40, Phase 5: ~35. Total ~196 trained.
    if num_phases == 5:
        quotas = [40, 40, 40, 40, 40] # 最后一个阶段自适应剩余
    else:
        quotas = [17] * 10

    current_phase = 0

    for item in valid_hois:
        # 如果所有阶段都满了，剩下的全是 Unseen
        if current_phase >= num_phases:
            unseen_hois.append(item['id'])
            continue

        oid = item['object_index']
        vid = item['action_index']
        hid = item['id']

        is_new_obj = oid not in seen_objects
        is_new_verb = vid not in seen_verbs

        # 核心逻辑：引入新物体 或 新动作 -> 加入当前训练阶段
        if is_new_obj or is_new_verb:
            tasks[current_phase].append(hid)
            seen_objects.add(oid)
            seen_verbs.add(vid)

            # 检查配额
            if len(tasks[current_phase]) >= quotas[current_phase]:
                current_phase += 1
        else:
            # (Old Object + Old Verb) -> 归为 Unseen/Zero-Shot
            unseen_hois.append(hid)

    # 打印统计
    print("=== Paper Split Statistics ===")
    total_train = 0
    for i, t in enumerate(tasks):
        print(f"Task {i+1}: {len(t)} classes")
        total_train += len(t)
    print(f"Total Trained Classes: {total_train}")
    print(f"Unseen (Zero-Shot) Classes: {len(unseen_hois)}")

    # 重要：将 unseen 列表存入 args，供各进程使用
    args.unseen_classes = unseen_hois

    return tasks

    # 统计信息
    print("=== Paper Split Statistics ===")
    for i, t in enumerate(tasks):
        print(f"Task {i+1}: {len(t)} classes")
    print(f"Unseen (Zero-Shot) Classes: {len(unseen_hois)}")

    # 将 unseen_classes 存入 args 以便后续测试使用
    args.unseen_classes = unseen_hois

    return tasks

def get_task_splits(args, num_classes=600):
    if args.split_mode == 'paper_5phase':
        return get_paper_split(args, num_phases=5)
    elif args.split_mode == 'paper_10phase':
        return get_paper_split(args, num_phases=10)
    elif args.split_mode == 'random':
        classes = list(range(num_classes))
        random.seed(args.seed)
        random.shuffle(classes)
        # 简单均分
        num_tasks = 4
        task_size = len(classes) // num_tasks
        tasks = [classes[i * task_size : (i + 1) * task_size] for i in range(num_tasks)]
        return tasks
    # ... (保留 rare_first 等逻辑)
    return []

def get_samples_by_class(dataset, class_ids, max_per_class=None):
    samples = []
    class_to_samples = {cid: [] for cid in class_ids}
    for idx in range(len(dataset)):
        item = dataset[idx]
        # 修正：如果样本为None直接跳过
        if item is None:
            continue
        # 这里假定item的类别信息为item['hoi_id']，请根据你的真实数据结构修改
        hoi_id = item['hoi'] if isinstance(item, dict) else item[1]['hoi']
        # 修正：如果hoi_id是tensor，取int
        if isinstance(hoi_id, torch.Tensor):
            if hoi_id.numel() == 1:
                hoi_id = int(hoi_id.item())
            else:
                for hid in hoi_id:
                    hid_int = int(hid.item())
                    if hid_int in class_to_samples:
                        class_to_samples[hid_int].append(idx)
                continue
        if hoi_id in class_to_samples:
            class_to_samples[hoi_id].append(idx)
    for cid in class_ids:
        c_samples = class_to_samples[cid]
        if max_per_class is not None:
            c_samples = random.sample(c_samples, min(max_per_class, len(c_samples)))
        samples.extend(c_samples)
    return samples

def reset_subset(trainset, indices):
    """
    设置trainset.indices，并同步包裹Subset，修正由Replay索引合并带来的越界问题。
    """
    trainset.indices = list(indices)
    trainset.dataset = torch.utils.data.Subset(
        get_base_dataset(trainset.dataset),
        trainset.indices
    )

def main_incremental(rank, args, tasks, N_replay=50):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )
    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)

    # 用于累积已训练的类别
    trained_classes = []

    prev_ckpt = None
    engine = None
    per_sample_scores = None

    object_to_target = None
    if args.dataset == 'hicodet':
        args.num_verbs = 117
    elif args.dataset == 'vcoco':
        args.num_verbs = 24

    # 确保 Unseen Classes 在多进程中可用 (如果是从 Spawn 传入的 args，通常已有；否则需重新计算)
    if not hasattr(args, 'unseen_classes') and 'paper' in args.split_mode:
         # 子进程如果丢失 args.unseen_classes，需要重新运行一次 split 逻辑获取 (由于 seed 固定，结果一致)
         _ = get_task_splits(args, 600)

    for task_idx, task_classes in enumerate(tasks):

        if task_idx < args.start_task:
            print(f"=== Rank {rank}: Skipping Task {task_idx+1} ===")
        else:
            print(f"=== Rank {rank}: Training Task {task_idx+1} with {len(task_classes)} classes ===")
        trained_classes += task_classes

        # 构造训练集
        trainset = DataFactory(
            name=args.dataset, partition=args.partitions[0],
            data_root=args.data_root, filter_classes=task_classes
        )
        print(f"task{task_idx+1}使用的训练样本数量: {len(trainset)}")

        # ---- 统计 rare_set (for class-balanced KD) ----
        rare_set, _ = get_rare_nonrare_sets(trainset)
        # ---- MIR相关加载 ----
        mir_dict = {}
        mir_min, mir_max = 0.0, 1.0
        if task_idx > 0:
            mir_dict = load_mir(args.output_dir, task_idx)
            if len(mir_dict) > 0:
                mir_vals = [float(v) for v in mir_dict.values()]
                mir_min, mir_max = min(mir_vals), max(mir_vals)

        # 回放样本（从之前所有类别中每类抽N个样本）
        if args.use_replay and task_idx > 0:
            replay_classes = sum(tasks[:task_idx], [])
            prev_trainset = DataFactory(
                name=args.dataset, partition=args.partitions[0],
                data_root=args.data_root, filter_classes=replay_classes
            )
            N_replay = args.n_replay
            replay_indices = get_samples_by_class(prev_trainset, replay_classes, max_per_class=N_replay)
            print(f"task{task_idx+1}时增加回放样本数量: {len(replay_indices)}")

            if args.dynamic_replay:
                # === 1. 用上一task模型推理当前trainset所有样本 ===
                # prev_ckpt = os.path.join(args.output_dir, f"checkpoint_task{task_idx}.pth")
                assert os.path.exists(prev_ckpt), f"Replay排序需要上一个task模型: {prev_ckpt}"

                # _, per_sample_scores = engine.test_hico(return_per_sample_scores=True)
                assert per_sample_scores is not None, "需要先计算per_sample_scores"

                # === 2. 只筛选replay_indices的置信度 ===
                idx2score = {d['local_idx']: max([d['scores'][gt]
                                for gt in d['gt_classes'] if gt in d['scores']]) if d['scores'] else 0.0
                             for d in per_sample_scores}
                # 只保留replay_indices中的有效idx
                replay_conf_list = [(idx, idx2score.get(idx, 0.0)) for idx in replay_indices if idx in idx2score]
                # print(f"task{task_idx+1} Replay confidence list: {replay_conf_list}")

                # === 3. 排序与插入 ===
                replay_indices_sorted = dynamic_replay_sort_by_confidence(replay_conf_list, reverse=True)
                new_indices = [i for i in trainset.indices if i not in replay_indices_sorted]
                final_indices = interleave_replay_and_new_auto(
                    new_indices, replay_indices_sorted,
                    repeat=args.replay_repeat
                )
                print(f"task{task_idx+1}最终训练样本数量: {len(final_indices)}")
                trainset.indices = final_indices
                reset_subset(trainset, final_indices)
            else:
                all_indices = list(set(list(trainset.indices) + replay_indices))
                reset_subset(trainset, all_indices)
                # trainset.indices += replay_indices
                # trainset.indices = list(set(trainset.indices))
        else:
            replay_indices = []
            print(f"task{task_idx+1}时不使用回放样本")

        train_loader = DataLoader(
            dataset=trainset,
            collate_fn=custom_collate, batch_size=args.batch_size // args.world_size,
            num_workers=args.num_workers, pin_memory=True,
            sampler=DistributedSampler(
                trainset, num_replicas=args.world_size,
                rank=rank, drop_last=True)
        )

        # === [修改处] 测试集加载逻辑 ===
        # 根据 eval_mode 决定加载哪些类别
        if args.eval_mode == 'unseen':
            # 仅评估 Zero-Shot 类别
            eval_classes = getattr(args, 'unseen_classes', [])
            if len(eval_classes) == 0:
                if rank == 0: print("Warning: No unseen classes found in args. Loading from fallback logic.")
                # 再次尝试获取
                _ = get_task_splits(args, 600)
                eval_classes = getattr(args, 'unseen_classes', [])

            if rank == 0: print(f"Mode [Unseen]: Evaluating on {len(eval_classes)} zero-shot classes.")

        elif args.eval_mode == 'current':
            eval_classes = task_classes
        elif args.eval_mode == 'seen_valid':
            eval_classes = trained_classes
        elif args.eval_mode == 'all':
            eval_classes = list(range(600))
        else: # default
            eval_classes = trained_classes

        # 构造测试集（已训练所有类别）
        testset = DataFactory(
            name=args.dataset, partition=args.partitions[1],
            data_root=args.data_root, filter_classes=trained_classes
        )
        test_loader = DataLoader(
            dataset=testset,
            collate_fn=custom_collate, batch_size=args.batch_size // args.world_size,
            num_workers=args.num_workers, pin_memory=True,
            sampler=DistributedSampler(
                testset, num_replicas=args.world_size,
                rank=rank, drop_last=True)
        )

        if args.dataset == 'hicodet':
            def get_base_dataset(ds):
                while hasattr(ds, 'dataset'):
                    ds = ds.dataset
                return ds
            base_train_dataset = get_base_dataset(train_loader.dataset)
            object_to_target = base_train_dataset.object_to_verb
            args.num_verbs = 117
        elif args.dataset == 'vcoco':
            object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
            args.num_verbs = 24

        model = build_detector(args, object_to_target)

        # === [修改点 1]：定义当前任务的模型路径 ===
        current_task_ckpt = os.path.join(args.output_dir, f"checkpoint_task{task_idx+1}.pth")

        # === [修改点 2]：Eval Only 模式的特殊处理 ===
        if args.eval:
            if os.path.exists(current_task_ckpt):
                print(f"=== Rank {rank}: [Eval Only] Loading checkpoint {current_task_ckpt} ===")
                checkpoint = torch.load(current_task_ckpt, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])

                # 初始化 engine (复用之前的逻辑)
                engine = CustomisedDLE(
                    model, train_loader, test_loader, args,
                    filter_classes=eval_classes, # 注意这里使用的是你指定的 eval-mode 对应的类别
                    teacher_model=None, # 评估不需要 teacher
                    replay_indices=None, mir_dict=None, rare_set=None
                )

                # 直接测试
                ap, _ = engine.test_hico(return_per_sample_scores=True)
                if rank == 0:
                    print(f"[Eval Only] Task {task_idx+1} result on {args.eval_mode} set: mAP = {ap.mean():.4f}")

                # 必须更新 prev_ckpt 以便后续逻辑正常（虽然 Eval Only 不太依赖它）
                prev_ckpt = current_task_ckpt
                continue # 跳过后续训练步骤，直接进入下一个 Task
            else:
                print(f"Error: Checkpoint {current_task_ckpt} not found! Cannot evaluate.")
                continue

        # 加载上一次训练参数
        if prev_ckpt is not None and os.path.exists(prev_ckpt):
            print(f"=> Rank {rank}: Loading checkpoint {prev_ckpt}.")
            checkpoint = torch.load(prev_ckpt, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"=> Rank {rank}: PViC randomly initialised.")

        #加载教师模型（新增）
        teacher_model = None
        if args.use_distill and task_idx > 0 and prev_ckpt is not None and os.path.exists(prev_ckpt):
            if dist.get_rank() == 0:  # 只在0号卡加载
                teacher_model = build_detector(args, object_to_target)
                checkpoint = torch.load(prev_ckpt, map_location='cpu')
                teacher_model.load_state_dict(checkpoint['model_state_dict'])
                #teacher_model.eval()
                #teacher_model.to(rank)
                for p in teacher_model.parameters():
                    p.requires_grad = False
            else:
                teacher_model = None
                print(f"=> Rank {rank}: No use teacher model.")

        # print(f"=> Rank {rank}: Ready for DLE.")
        engine = CustomisedDLE(
            model, train_loader, test_loader, args,
            filter_classes=eval_classes,  # <--- 确保这里使用的是 eval_classes
            teacher_model=teacher_model,
            replay_indices=replay_indices if args.use_replay and task_idx > 0 else None,
            mir_dict=mir_dict, rare_set=rare_set, mir_min=mir_min, mir_max=mir_max
        )

        if task_idx < args.start_task:
            prev_ckpt = os.path.join(args.output_dir, f"checkpoint_task{task_idx+1}.pth")
        else:
            model.freeze_detector()
            param_dicts = [{"params": [p for p in model.parameters() if p.requires_grad]}]
            optim = torch.optim.AdamW(param_dicts, lr=args.lr_head, weight_decay=args.weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop, gamma=args.lr_drop_factor)
            engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

            engine(args.epochs)
            # 保存模型
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_task{task_idx+1}.pth")
            torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
            prev_ckpt = ckpt_path

        # 测试
        ap, per_sample_scores = engine.test_hico(return_per_sample_scores=True)
        if rank == 0:
            print(f"task{task_idx+1}测试集样本数: {len(testset)}，类别数: {len(trained_classes)}")
            print(f"[Task {task_idx+1}] mAP on {len(trained_classes)} classes: {ap.mean():.4f}")

if __name__ == '__main__':

    if "DETR" not in os.environ:
        raise KeyError(f"Specify the detector type with env. variable \"DETR\".")
    elif os.environ["DETR"] == "base":
        parser = argparse.ArgumentParser(parents=[base_detector_args(),])
        parser.add_argument('--detector', default='base', type=str)
        parser.add_argument('--raw-lambda', default=2.8, type=float)
    elif os.environ["DETR"] == "advanced":
        parser = argparse.ArgumentParser(parents=[advanced_detector_args(),])
        parser.add_argument('--detector', default='advanced', type=str)
        parser.add_argument('--raw-lambda', default=1.7, type=float)

    parser.add_argument('--kv-src', default='C5', type=str, choices=['C5', 'C4', 'C3'])
    parser.add_argument('--repr-dim', default=384, type=int)
    parser.add_argument('--triplet-enc-layers', default=1, type=int)
    parser.add_argument('--triplet-dec-layers', default=2, type=int)

    parser.add_argument('--alpha', default=.5, type=float)
    parser.add_argument('--gamma', default=.1, type=float)
    parser.add_argument('--box-score-thresh', default=.05, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--use-wandb', default=False, action='store_true')

    parser.add_argument('--port', default='1234', type=str)
    parser.add_argument('--seed', default=140, type=int)
    parser.add_argument('--world-size', default=8, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    #新增加的参数
    parser.add_argument('--n-replay', default=50, type=int, help='每类回放样本数量')
    parser.add_argument('--start-task', default=0, type=int)
    parser.add_argument('--use-replay', action='store_true', help='是否使用回放样本')
    parser.add_argument('--use-distill', action='store_true', help='是否启用知识蒸馏')
    parser.add_argument('--distill-loss-weight', default=1.0, type=float, help='蒸馏损失权重')
    # parser.add_argument('--use-cross-attn-distill', action='store_true', help='是否启用交叉注意力蒸馏')
    # parser.add_argument('--cross-attn-loss-weight', default=1.0, type=float, help='交叉注意力蒸馏损失权重')
    parser.add_argument('--dynamic-replay', action='store_true', help='是否启用动态回放')
    parser.add_argument('--replay-repeat', default=1, type=int, help='回放样本重复次数')

    parser.add_argument('--replay-distill', action='store_true', help='是否对回放样本增加蒸馏')
    parser.add_argument('--replay-distill-layer', default='feat', choices=['logits', 'feat'], help='蒸馏层，logits=输出层，feat=倒数第二层特征')
    parser.add_argument('--replay-distill-loss-weight', default=1.0, type=float, help='基础蒸馏loss权重')
    parser.add_argument('--replay-distill-rare-factor', default=2.0, type=float, help='稀有类蒸馏增强倍数')
    parser.add_argument('--replay-distill-mir-factor', default=2.0, type=float, help='混淆度蒸馏增强倍数')
    parser.add_argument('--use-attn-hint', action='store_true', help='是否启用教师Attention Hint')
    parser.add_argument('--attn-hint-alpha', default=0.05, type=float, help='Attention Hint权重')
    parser.add_argument('--attn-hint-epochs', default=0, type=int, help='仅前N个epoch使用Hint，为0则全程使用')

    parser.add_argument('--hoi-path', default='hoi_correspondence.json', type=str,
                        help='Path to hoi_correspondence.json')
    parser.add_argument('--rare-path', default='rare.json', type=str,
                        help='Path to rare.json')

    # === VAS-HOI 核心控制参数 ===
    parser.add_argument('--use-vas', action='store_true', default=False,
                        help='是否启用脆弱性感知前向抗干涉模块 (Module 2)')
    parser.add_argument('--vas-lambda', default=2.0, type=float,
                        help='抗干涉负向偏置的强度系数 (gamma)')
    # 可选：用于调试或消融，决定SIS的计算方式或来源
    parser.add_argument('--vas-sis-type', default='frequency', type=str,
                        choices=['frequency', 'kl_divergence', 'entropy'],
                        help='SIS(脆弱性标签)的计算来源')

    args = parser.parse_args()
    print(args)

    if not args.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    # === 修改: 调用新的任务分割函数 ===
    tasks = get_task_splits(args, num_classes=600)

    mp.spawn(main_incremental, nprocs=args.world_size, args=(args, tasks))
