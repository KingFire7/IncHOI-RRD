import os
import sys
import torch
import random
import warnings
import argparse
import time
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from pvic import build_detector
from utils_incremental import custom_collate, CustomisedDLE, DataFactory, get_base_dataset
from mir_utils import dynamic_replay_sort_by_confidence, interleave_replay_and_new_auto, load_mir, get_rare_nonrare_sets
from configs import base_detector_args, advanced_detector_args
from inchoi.rgip_utils import build_hico_object_verb_mappings

import json
import hashlib

warnings.filterwarnings("ignore")


def _json_default(obj):
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def _ensure_experiment_dirs(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if not getattr(args, 'checkpoint_dir', ''):
        args.checkpoint_dir = args.output_dir
    if not getattr(args, 'metrics_dir', ''):
        args.metrics_dir = os.path.join(args.output_dir, 'metrics')
    if not getattr(args, 'analysis_dir', ''):
        args.analysis_dir = os.path.join(args.output_dir, 'analysis')
    if not getattr(args, 'log_dir', ''):
        args.log_dir = os.path.join(args.output_dir, 'logs')
    if not getattr(args, 'config_dir', ''):
        args.config_dir = os.path.join(args.output_dir, 'configs')
    for path in [args.checkpoint_dir, args.metrics_dir, args.analysis_dir, args.log_dir, args.config_dir]:
        os.makedirs(path, exist_ok=True)


def _save_train_args(args):
    payload = vars(args).copy()
    for path in [
        os.path.join(args.output_dir, 'train_args.json'),
        os.path.join(args.config_dir, 'train_args.json'),
        os.path.join(args.checkpoint_dir, 'train_args.json'),
    ]:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def _append_jsonl(path, item):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False, default=_json_default) + '\n')


def _replay_cache_path(args, task_idx, replay_classes):
    cache_dir = getattr(args, 'replay_cache_dir', '') or os.environ.get(
        'RGIP_REPLAY_CACHE_DIR',
        os.path.join('outputs-rgip', 'cache', 'replay_indices')
    )
    os.makedirs(cache_dir, exist_ok=True)
    class_key = ','.join(map(str, replay_classes))
    class_hash = hashlib.md5(class_key.encode('utf-8')).hexdigest()[:12]
    split_seed = getattr(args, 'split_seed', None)
    if split_seed is None:
        split_seed = getattr(args, 'seed', 'none')
    filename = (
        f"{args.dataset}_{args.partitions[0]}_split{args.split_mode}"
        f"_splitseed{split_seed}_task{task_idx + 1}"
        f"_prev{len(replay_classes)}_n{args.n_replay}_{class_hash}.json"
    )
    return os.path.join(cache_dir, filename)


def _load_replay_indices_cache(path, replay_classes, n_replay):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if payload.get('n_replay') != n_replay:
            return None
        if payload.get('replay_classes') != list(map(int, replay_classes)):
            return None
        indices = payload.get('indices')
        if not isinstance(indices, list):
            return None
        return [int(i) for i in indices]
    except Exception as exc:
        print(f"[ReplayCache] Failed to load {path}: {exc}", flush=True)
        return None


def _save_replay_indices_cache(path, replay_classes, n_replay, indices):
    payload = {
        'replay_classes': list(map(int, replay_classes)),
        'n_replay': int(n_replay),
        'indices': [int(i) for i in indices],
        'timestamp': int(time.time()),
    }
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp_path, path)


def _save_task_eval(args, task_idx, trained_classes, ap, per_sample_scores, dataset):
    os.makedirs(args.metrics_dir, exist_ok=True)
    os.makedirs(args.analysis_dir, exist_ok=True)
    ap_list = ap.detach().cpu().tolist() if isinstance(ap, torch.Tensor) else []
    class_ap = {str(cls): float(ap_list[i]) for i, cls in enumerate(trained_classes[:len(ap_list)])}
    rare_all = set(getattr(dataset, 'rare', []))
    non_rare_all = set(getattr(dataset, 'non_rare', []))
    rare_vals = [class_ap[str(c)] for c in trained_classes if str(c) in class_ap and c in rare_all]
    nonrare_vals = [class_ap[str(c)] for c in trained_classes if str(c) in class_ap and c in non_rare_all]
    summary = {
        'task_idx': task_idx,
        'num_classes': len(trained_classes),
        'classes': trained_classes,
        'mAP': float(np.mean(ap_list)) if ap_list else 0.0,
        'rare_mAP': float(np.mean(rare_vals)) if rare_vals else 0.0,
        'non_rare_mAP': float(np.mean(nonrare_vals)) if nonrare_vals else 0.0,
        'class_ap': class_ap,
        'timestamp': int(time.time()),
    }
    with open(os.path.join(args.metrics_dir, f'task{task_idx}_eval.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)
    _append_jsonl(os.path.join(args.metrics_dir, 'task_eval_summary.jsonl'), summary)
    if per_sample_scores is not None:
        with open(os.path.join(args.analysis_dir, f'task{task_idx}_per_sample_scores.json'), 'w', encoding='utf-8') as f:
            json.dump(per_sample_scores, f, indent=2, ensure_ascii=False, default=_json_default)

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
    random.seed(getattr(args, 'split_seed', None) if getattr(args, 'split_seed', None) is not None else args.seed)
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
        random.seed(getattr(args, 'split_seed', None) if getattr(args, 'split_seed', None) is not None else args.seed)
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

    # Fast and safer path for our DataFactory(HICO-DET): use metadata only and
    # return original HICODet valid indices.  The previous implementation called
    # dataset[idx], which loaded images and returned subset-local indices; those
    # indices are not the same coordinate system used by reset_subset().
    if isinstance(dataset, DataFactory) and getattr(dataset, 'name', None) == 'hicodet':
        base_dataset = get_base_dataset(dataset.dataset)
        candidate_indices = dataset.indices if dataset.indices is not None else range(len(base_dataset))
        class_set = set(class_ids)
        for base_idx in candidate_indices:
            ann_idx = base_dataset._idx[int(base_idx)]
            ann = base_dataset.annotations[ann_idx]
            for hid in set(ann.get('hoi', [])) & class_set:
                class_to_samples[int(hid)].append(int(base_idx))

        for cid in class_ids:
            c_samples = class_to_samples[cid]
            if max_per_class is not None:
                c_samples = random.sample(c_samples, min(max_per_class, len(c_samples)))
            samples.extend(c_samples)
        return samples

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
    _ensure_experiment_dirs(args)
    if rank == 0:
        _save_train_args(args)
    dist.barrier()

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
        if task_idx < args.start_task:
            prev_ckpt = os.path.join(args.checkpoint_dir, f"checkpoint_task{task_idx+1}.pth")
            if rank == 0:
                print(f"=> Rank {rank}: Resume mode will use {prev_ckpt} as previous checkpoint.")
            continue

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
            N_replay = args.n_replay
            if rank == 0:
                replay_cache_path = _replay_cache_path(args, task_idx, replay_classes)
                replay_indices = None
                if not getattr(args, 'no_replay_cache', False):
                    replay_indices = _load_replay_indices_cache(
                        replay_cache_path, replay_classes, N_replay
                    )
                    if replay_indices is not None:
                        print(
                            f"[ReplayCache] Loaded {len(replay_indices)} replay indices from {replay_cache_path}",
                            flush=True
                        )
                if replay_indices is None:
                    print(
                        f"[ReplayCache] Building replay indices for task {task_idx+1}; "
                        f"classes={len(replay_classes)}, n_replay={N_replay}",
                        flush=True
                    )
                    prev_trainset = DataFactory(
                        name=args.dataset, partition=args.partitions[0],
                        data_root=args.data_root, filter_classes=replay_classes
                    )
                    replay_indices = get_samples_by_class(
                        prev_trainset, replay_classes, max_per_class=N_replay
                    )
                    if not getattr(args, 'no_replay_cache', False):
                        _save_replay_indices_cache(
                            replay_cache_path, replay_classes, N_replay, replay_indices
                        )
                        print(
                            f"[ReplayCache] Saved {len(replay_indices)} replay indices to {replay_cache_path}",
                            flush=True
                        )
            else:
                replay_indices = None
            replay_obj = [replay_indices]
            dist.broadcast_object_list(replay_obj, src=0)
            replay_indices = replay_obj[0]
            print(f"task{task_idx+1}时增加回放样本数量: {len(replay_indices)}")

            if args.dynamic_replay:
                if rank == 0:
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
                else:
                    final_indices = None
                final_obj = [final_indices]
                dist.broadcast_object_list(final_obj, src=0)
                final_indices = final_obj[0]
                print(f"task{task_idx+1}最终训练样本数量: {len(final_indices)}")
                reset_subset(trainset, final_indices)
            else:
                if rank == 0:
                    all_indices = sorted(set(list(trainset.indices) + replay_indices))
                else:
                    all_indices = None
                all_obj = [all_indices]
                dist.broadcast_object_list(all_obj, src=0)
                all_indices = all_obj[0]
                print(f"task{task_idx+1}最终训练样本数量: {len(all_indices)}")
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
            def _unwrap_dataset(ds):
                while hasattr(ds, 'dataset'):
                    ds = ds.dataset
                return ds
            base_train_dataset = _unwrap_dataset(train_loader.dataset)
            object_to_target = base_train_dataset.object_to_verb
            hoi_to_obj, hoi_to_verb, obj_verb_to_hoi, hoi_frequency = build_hico_object_verb_mappings(base_train_dataset)
            args.num_verbs = 117
        elif args.dataset == 'vcoco':
            object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
            hoi_to_obj, hoi_to_verb, obj_verb_to_hoi, hoi_frequency = {}, {}, {}, None
            args.num_verbs = 24

        model = build_detector(args, object_to_target)

        # === [修改点 1]：定义当前任务的模型路径 ===
        current_task_ckpt = os.path.join(args.checkpoint_dir, f"checkpoint_task{task_idx+1}.pth")

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
                    replay_indices=None, mir_dict=None, rare_set=None,
                    old_hoi_classes=sum(tasks[:task_idx], []),
                    current_hoi_classes=task_classes,
                    hoi_to_obj=hoi_to_obj,
                    hoi_to_verb=hoi_to_verb,
                    obj_verb_to_hoi=obj_verb_to_hoi,
                    object_to_verb=object_to_target,
                    hoi_frequency=hoi_frequency,
                    task_idx=task_idx + 1,
                    task_classes=task_classes,
                )

                # 直接测试
                ap, per_sample_scores_eval = engine.test_hico(return_per_sample_scores=True)
                if rank == 0:
                    print(f"[Eval Only] Task {task_idx+1} result on {args.eval_mode} set: mAP = {ap.mean():.4f}")
                    _save_task_eval(
                        args=args,
                        task_idx=task_idx + 1,
                        trained_classes=eval_classes.copy(),
                        ap=ap,
                        per_sample_scores=per_sample_scores_eval,
                        dataset=get_base_dataset(test_loader.dataset),
                    )

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
        if (args.use_distill or args.use_rgip) and task_idx > 0 and prev_ckpt is not None and os.path.exists(prev_ckpt):
            teacher_model = build_detector(args, object_to_target)
            checkpoint = torch.load(prev_ckpt, map_location='cpu')
            teacher_model.load_state_dict(checkpoint['model_state_dict'])
            teacher_device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
            teacher_model.to(teacher_device)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)
            print(f"=> Rank {rank}: Loaded frozen teacher model from {prev_ckpt}.")

        # print(f"=> Rank {rank}: Ready for DLE.")
        engine = CustomisedDLE(
            model, train_loader, test_loader, args,
            filter_classes=eval_classes,  # <--- 确保这里使用的是 eval_classes
            teacher_model=teacher_model,
            replay_indices=replay_indices if args.use_replay and task_idx > 0 else None,
            mir_dict=mir_dict, rare_set=rare_set, mir_min=mir_min, mir_max=mir_max,
            old_hoi_classes=sum(tasks[:task_idx], []),
            current_hoi_classes=task_classes,
            hoi_to_obj=hoi_to_obj,
            hoi_to_verb=hoi_to_verb,
            obj_verb_to_hoi=obj_verb_to_hoi,
            object_to_verb=object_to_target,
            hoi_frequency=hoi_frequency,
            task_idx=task_idx + 1,
            task_classes=task_classes,
        )

        resume_checkpoint = None
        completed_epochs = 0
        if args.resume and task_idx == args.start_task:
            if os.path.exists(args.resume):
                print(f"=> Rank {rank}: Resuming current task from {args.resume}.", flush=True)
                resume_checkpoint = torch.load(args.resume, map_location='cpu')
                target_model = engine._state.net.module if hasattr(engine._state.net, 'module') else model
                target_model.load_state_dict(resume_checkpoint['model_state_dict'])
                resumed_epoch = int(resume_checkpoint.get('epoch', 0) or 0)
                resumed_iteration = int(resume_checkpoint.get('iteration', 0) or 0)
                engine._state.epoch = resumed_epoch
                engine._state.iteration = resumed_iteration
                completed_epochs = resumed_epoch
                if rank == 0:
                    print(
                        f"=> Resume state loaded: completed_epoch={resumed_epoch}, "
                        f"iteration={resumed_iteration}, "
                        f"remaining_epochs={max(int(args.epochs) - resumed_epoch, 0)}.",
                        flush=True,
                    )
            else:
                raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")
        elif args.resume and rank == 0:
            print(
                f"=> Ignoring --resume for task {task_idx + 1}; "
                f"resume is only applied to start-task {args.start_task + 1}.",
                flush=True,
            )

        model.freeze_detector()
        param_dicts = [{"params": [p for p in model.parameters() if p.requires_grad]}]
        optim = torch.optim.AdamW(param_dicts, lr=args.lr_head, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop, gamma=args.lr_drop_factor)
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

        if resume_checkpoint is not None:
            if 'optim_state_dict' in resume_checkpoint:
                engine._state.optimizer.load_state_dict(resume_checkpoint['optim_state_dict'])
            if 'scheduler_state_dict' in resume_checkpoint and engine._state.lr_scheduler is not None:
                engine._state.lr_scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in resume_checkpoint and getattr(engine._state, 'scaler', None) is not None:
                engine._state.scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])

        if completed_epochs < int(args.epochs):
            engine(args.epochs)
        elif rank == 0:
            print(
                f"=> Current task already has {args.epochs} completed epochs in resume checkpoint; "
                "skipping training and saving task checkpoint.",
                flush=True,
            )
        # 保存模型
        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_task{task_idx+1}.pth")
        if rank == 0:
            torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
        dist.barrier()
        prev_ckpt = ckpt_path

        # 测试
        ap, per_sample_scores = engine.test_hico(return_per_sample_scores=True)
        if rank == 0:
            print(f"task{task_idx+1}测试集样本数: {len(testset)}，类别数: {len(trained_classes)}")
            print(f"[Task {task_idx+1}] mAP on {len(trained_classes)} classes: {ap.mean():.4f}")
            _save_task_eval(
                args=args,
                task_idx=task_idx + 1,
                trained_classes=trained_classes.copy(),
                ap=ap,
                per_sample_scores=per_sample_scores,
                dataset=get_base_dataset(test_loader.dataset),
            )

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
    parser.add_argument('--split-seed', default=None, type=int,
                        help='Seed for task/class split generation. Defaults to --seed.')
    parser.add_argument('--world-size', default=8, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--skip-initial-eval', action='store_true',
                        help='Skip the expensive evaluation that normally runs before each training task')
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
    parser.add_argument('--replay-cache-dir', default='', type=str,
                        help='Directory for cached replay sample indices. Defaults to outputs-rgip/cache/replay_indices.')
    parser.add_argument('--no-replay-cache', action='store_true',
                        help='Disable replay index cache and rebuild replay indices every run')

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
    parser.add_argument('--checkpoint-dir', default='', type=str,
                        help='Directory for checkpoint_task*.pth/latest.pth/best.pth. Defaults to output-dir.')
    parser.add_argument('--metrics-dir', default='', type=str,
                        help='Directory for epoch/task metric JSON files. Defaults to output-dir/metrics.')
    parser.add_argument('--analysis-dir', default='', type=str,
                        help='Directory for per-sample and analysis JSON files. Defaults to output-dir/analysis.')
    parser.add_argument('--log-dir', default='', type=str,
                        help='Directory for terminal logs produced by wrapper scripts. Defaults to output-dir/logs.')
    parser.add_argument('--config-dir', default='', type=str,
                        help='Directory for saved train_args.json. Defaults to output-dir/configs.')

    # === VAS-HOI 核心控制参数 ===
    parser.add_argument('--use-vas', action='store_true', default=False,
                        help='是否启用脆弱性感知前向抗干涉模块 (Module 2)')
    parser.add_argument('--vas-lambda', default=2.0, type=float,
                        help='抗干涉负向偏置的强度系数 (gamma)')
    # 可选：用于调试或消融，决定SIS的计算方式或来源
    parser.add_argument('--vas-sis-type', default='frequency', type=str,
                        choices=['frequency', 'kl_divergence', 'entropy'],
                        help='SIS(脆弱性标签)的计算来源')

    # === RGIP: Rarity-Guided Interaction Preservation ===
    parser.add_argument('--use-rgip', action='store_true',
                        help='Enable Rarity-Guided Interaction Preservation')
    parser.add_argument('--rgip-alpha', default=0.4, type=float,
                        help='SIS frequency-risk weight')
    parser.add_argument('--rgip-beta', default=0.3, type=float,
                        help='SIS margin-risk weight')
    parser.add_argument('--rgip-gamma', default=2.0, type=float,
                        help='SIS-to-training-weight strength')
    parser.add_argument('--rgip-topk', default=3, type=int,
                        help='Number of current-task confounders per replay pair')
    parser.add_argument('--rgip-eta-object', default=0.4, type=float,
                        help='Confounder score weight for shared object')
    parser.add_argument('--rgip-eta-query', default=0.4, type=float,
                        help='Confounder score weight for query/prototype similarity')
    parser.add_argument('--rgip-eta-sem', default=0.2, type=float,
                        help='Confounder score weight for predicate semantic similarity')
    parser.add_argument('--rgip-confounder-tau', default=0.2, type=float,
                        help='Temperature for confounder softmax weights')
    parser.add_argument('--rgip-lambda-attn', default=0.5, type=float,
                        help='Strength of content-attention modulation')
    parser.add_argument('--rgip-kappa', default=1.0, type=float,
                        help='Exponent for positional support in attention modulation')
    parser.add_argument('--rgip-attn-clamp', default=5.0, type=float,
                        help='Clamp magnitude for negative attention bias')
    parser.add_argument('--rgip-max-attn-pairs', default=16, type=int,
                        help='Max number of high-SIS HO pairs per image for RGIP attention modulation')
    parser.add_argument('--rgip-proto-momentum', default=0.9, type=float,
                        help='EMA momentum for current-task interaction prototypes')
    parser.add_argument('--rgip-int-loss-weight', default=1.0, type=float,
                        help='Overall weight for RGIP interaction distillation')
    parser.add_argument('--rgip-feat-weight', default=1.0, type=float,
                        help='Feature distillation weight')
    parser.add_argument('--rgip-logit-weight', default=1.0, type=float,
                        help='Old predicate distribution distillation weight')
    parser.add_argument('--rgip-hardneg-weight', default=0.5, type=float,
                        help='Hard new-predicate negative margin weight')
    parser.add_argument('--rgip-margin', default=0.2, type=float,
                        help='Margin for hard negative suppression')
    parser.add_argument('--rgip-temperature', default=2.0, type=float,
                        help='Temperature for predicate distribution KD')
    parser.add_argument('--rgip-use-semantic-confounder', action='store_true',
                        help='Use predicate semantic similarity when embeddings are available')
    parser.add_argument('--rgip-predicate-embedding-path', default='', type=str,
                        help='Optional path to predicate embeddings')
    parser.add_argument('--rgip-debug', action='store_true',
                        help='Log SIS, confounder and attention statistics')
    parser.add_argument('--rgip-timing', action='store_true',
                        help='Print per-stage timing for RGIP iterations on rank 0')
    parser.add_argument('--detect-anomaly', action='store_true',
                        help='Enable PyTorch autograd anomaly detection for debugging; keep disabled for normal training')

    args = parser.parse_args()
    print(args)

    if not args.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    # === 修改: 调用新的任务分割函数 ===
    tasks = get_task_splits(args, num_classes=600)

    mp.spawn(main_incremental, nprocs=args.world_size, args=(args, tasks))
