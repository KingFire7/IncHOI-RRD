#!/usr/bin/env python3
"""
evaluate_forgetting.py

完整修改版 — 确保包含训练时需要的所有超参数（优先使用 --train-args-file）。
目的：读取每个 task 的 checkpoint（checkpoint_task{t}.pth），在 HICO 测试集上评估每个 checkpoint 的 per-class AP（600），
并计算每个 task 的 mean AP、per-class forgetting、平均遗忘和 BWT 等指标，结果保存为 JSON / CSV。

使用示例（确保 DETR 环境变量与训练一致）：
export DETR=base
python evaluate_forgetting.py \
  --data-root ./hicodet \
  --output-dir ./outputs/... \
  --train-args-file /path/to/train_args.json \
  --num-tasks 4 --seed 140 --batch-size 2 --num-workers 4 --verbose

说明：
- 优先使用 --train-args-file（训练时保存的 args JSON）来完全一致地重建训练超参数；
  若未提供 train args 文件，脚本会从训练的 parent parser (base_detector_args / advanced_detector_args)
  读取参数，并对仍然缺失的关键字段用合理默认值填充（确保能构建模型并评估）。
- 若训练时有传入非默认的自定义参数，强烈建议使用 --train-args-file 或在命令行把那些参数也传给此脚本。
"""

import os
import sys
import argparse
import json
import time
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# 父解析器来自训练脚本的 configs（必须在 PYTHONPATH 中）
from configs import base_detector_args, advanced_detector_args

# 项目模块
from pvic import build_detector
from utils_incremental import DataFactory, custom_collate, get_base_dataset, CustomisedDLE

# ---------------------------
# Utilities
# ---------------------------
def split_tasks_identical(num_classes=600, num_tasks=4, seed=140):
    classes = list(range(num_classes))
    random.seed(seed)
    random.shuffle(classes)
    tasks = [classes[i * (num_classes // num_tasks):(i + 1) * (num_classes // num_tasks)] for i in range(num_tasks)]
    return tasks

def init_dist_for_single_process(master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method=f'tcp://{master_addr}:{master_port}', rank=0, world_size=1)

def build_data_loaders(args):
    trainset = DataFactory(name='hicodet', partition='train2015', data_root=args.data_root, filter_classes=None)
    testset = DataFactory(name='hicodet', partition='test2015', data_root=args.data_root, filter_classes=None)

    train_sampler = DistributedSampler(trainset, num_replicas=1, rank=0, drop_last=False)
    test_sampler = DistributedSampler(testset, num_replicas=1, rank=0, drop_last=False)

    train_loader = DataLoader(dataset=trainset, collate_fn=custom_collate,
                              batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, sampler=train_sampler)
    test_loader = DataLoader(dataset=testset, collate_fn=custom_collate,
                             batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=True, sampler=test_sampler)
    return train_loader, test_loader, trainset, testset

# ---------------------------
# Ensure defaults (fallback)
# ---------------------------
def ensure_defaults(args):
    """
    为构建模型和评估需要的关键超参数填充合理默认值。
    """
    # 基础参数 (所有模型共用或默认)
    defaults = {
        'repr_dim': 384,
        'hidden_dim': 256,
        'triplet_enc_layers': 1,
        'triplet_dec_layers': 2,
        'nheads': 8,
        'dim_feedforward': 2048,
        'enc_layers': 6,
        'dec_layers': 6,
        'position_embedding': 'sine',
        'position_embedding_dim': 128,
        'num_verbs': 117,
        'box_score_thresh': 0.05,
        'min_instances': 3,
        'max_instances': 15,
        'raw_lambda': 2.8,
        'set_cost_class': 1.0,
        'set_cost_bbox': 5.0,
        'set_cost_giou': 2.0,
        'bbox_loss_coef': 5.0,
        'giou_loss_coef': 2.0,
        'eos_coef': 0.1,
        'kv_src': getattr(args, 'kv_src', 'C5'),
        'device': getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'),
    }

    # 识别当前模型类型
    if not hasattr(args, 'detector'):
        args.detector = os.environ.get('DETR', 'base')

    # 针对 Swin-L (Advanced) 的特殊兜底参数
    if args.detector == 'advanced':
        advanced_defaults = {
            'backbone': 'swin_L_384_22k',
            'num_queries': 900,               # Advanced DETR 通常需要更多 queries
            'num_feature_levels': 4,          # 多尺度特征图
            'two_stage': True,                # 两阶段架构
            'mixed_selection': True,
            'look_forward_twice': True,
            'with_box_refine': True,
            'raw_lambda': 1.7,                # Advanced 默认的 lambda
            'kv_src': 'C5'                    # 确保 kv_src 默认正确
        }
        defaults.update(advanced_defaults)

    # 仅当 args 中没有该属性时，才用默认值填充
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

# ---------------------------
# Model loading (use same args as training)
# ---------------------------
def load_model_from_ckpt(args, obj_to_verb, ckpt_path, device):
    # dataset-specific num_verbs
    if getattr(args, 'dataset', 'hicodet') == 'hicodet':
        args.num_verbs = 117
    else:
        args.num_verbs = getattr(args, 'num_verbs', 24)

    if not hasattr(args, 'pretrained'):
        args.pretrained = ''

    # Build detector using full args (should match training)
    model = build_detector(args, obj_to_verb)

    # Load checkpoint (support both {'model_state_dict': ...} and raw state_dict)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("[WARN] strict load_state_dict failed:", e)
        print("[WARN] attempting load_state_dict(..., strict=False) to continue evaluation.")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(device)
    return model

# ---------------------------
# Metrics computation
# ---------------------------
def compute_task_specific_means(per_ckpt_ap, tasks):
    task_means = OrderedDict()
    for ck in sorted(per_ckpt_ap.keys()):
        ap_vec = per_ckpt_ap[ck]
        task_means[ck] = {}
        for k in range(1, ck + 1):
            classes = tasks[k-1]
            vals = ap_vec[classes]
            mean_ap = float(np.nanmean(vals))
            task_means[ck][k] = mean_ap
    return task_means

def compute_forgetting_and_summary(per_ckpt_ap, tasks):
    available_ck = sorted(per_ckpt_ap.keys())
    T = available_ck[-1]
    final_ap = per_ckpt_ap[T]
    num_classes = final_ap.shape[0]

    per_task = {}
    for k, cls_list in enumerate(tasks, start=1):
        cls_arr = np.array(cls_list, dtype=int)
        if k in per_ckpt_ap:
            initial_ap = per_ckpt_ap[k][cls_arr]
        else:
            initial_ap = np.full(cls_arr.shape, np.nan, dtype=float)
        final_on_task = final_ap[cls_arr]
        per_class_forgetting = (initial_ap - final_on_task)
        mean_forgetting = float(np.nanmean(per_class_forgetting))
        per_task[f"task_{k}"] = {
            "classes": cls_arr.tolist(),
            "initial_per_class_ap": np.nan_to_num(initial_ap, nan=float('nan')).tolist(),
            "final_per_class_ap": final_on_task.tolist(),
            "per_class_forgetting": np.nan_to_num(per_class_forgetting, nan=float('nan')).tolist(),
            "mean_forgetting": mean_forgetting,
            "initial_mean_ap": float(np.nanmean(initial_ap)),
            "final_mean_ap": float(np.nanmean(final_on_task))
        }

    per_class_initial = np.full(num_classes, np.nan, dtype=float)
    for k, cls_list in enumerate(tasks, start=1):
        cls_arr = np.array(cls_list, dtype=int)
        if k in per_ckpt_ap:
            per_class_initial[cls_arr] = per_ckpt_ap[k][cls_arr]
        else:
            per_class_initial[cls_arr] = np.nan
    per_class_forgetting = per_class_initial - final_ap
    mean_forgetting_all = float(np.nanmean(per_class_forgetting))

    bwt_list = []
    for k in range(1, len(tasks) + 1):
        if k in per_ckpt_ap:
            initial = per_ckpt_ap[k][np.array(tasks[k-1], dtype=int)]
            final_on_task = final_ap[np.array(tasks[k-1], dtype=int)]
            bwt_list.append(np.nanmean(final_on_task - initial))
    BWT = float(np.nanmean(bwt_list)) if len(bwt_list) > 0 else 0.0

    summary = {
        "final_map_all": float(np.nanmean(final_ap)),
        "per_class_forgetting": per_class_forgetting.tolist(),
        "per_class_initial_ap": np.nan_to_num(per_class_initial, nan=float('nan')).tolist(),
        "mean_forgetting_all": mean_forgetting_all,
        "BWT": BWT
    }
    return per_task, summary

# ---------------------------
# Argument parsing (reuse training parents, avoid duplicates)
# ---------------------------
def parse_args():
    if "DETR" not in os.environ:
        raise KeyError('Set env var "DETR" to "base" or "advanced" to match training config.')

    # 复用训练时的 parent parser
    if os.environ["DETR"] == "base":
        parent_parser = base_detector_args()
    else:
        parent_parser = advanced_detector_args()

    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=True)

    def add_if_not_exists(*opts, **kwargs):
        existing = set(parser._option_string_actions.keys())
        for o in opts:
            if o in existing:
                return
        parser.add_argument(*opts, **kwargs)

    # 核心评估参数
    add_if_not_exists('--data-root', required=True, help='root of hicodet dataset')
    add_if_not_exists('--output-dir', required=True, help='where checkpoints are stored')
    add_if_not_exists('--save-dir', default='eval_forgetting_results', help='where to save outputs')
    add_if_not_exists('--num-tasks', type=int, default=4)
    add_if_not_exists('--seed', type=int, default=140)
    add_if_not_exists('--batch-size', type=int, default=2)
    add_if_not_exists('--num-workers', type=int, default=4)
    add_if_not_exists('--train-args-file', default='', help='(optional) JSON file with training args')
    add_if_not_exists('--master-addr', default='127.0.0.1')
    add_if_not_exists('--master-port', default='29500')
    add_if_not_exists('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    add_if_not_exists('--verbose', action='store_true')
    # ======== 新增：Swin-L / Advanced DETR 常用超参数接口 ========
    add_if_not_exists('--backbone', default='', type=str, help="Name of the backbone (e.g., swin_L_384_22k)")
    add_if_not_exists('--num_queries', type=int, help="Number of queries (100 for base, 900 for advanced)")
    add_if_not_exists('--num_feature_levels', type=int, help="Number of feature levels (usually 4 for advanced)")
    add_if_not_exists('--two_stage', action='store_true', help="Use two stage DETR")
    add_if_not_exists('--with_box_refine', action='store_true', help="Use box refinement")
    add_if_not_exists('--mixed_selection', action='store_true', help="Use mixed selection")
    add_if_not_exists('--look_forward_twice', action='store_true', help="Use look forward twice")
    # ===============================================================

    args = parser.parse_args()

    # 优先使用保存的 JSON 训练参数覆盖
    taf = getattr(args, 'train_args_file', '')
    if taf:
        if not os.path.exists(taf):
            raise FileNotFoundError(f"train args file not found: {taf}")
        with open(taf, 'r') as f:
            saved = json.load(f)
        for k, v in saved.items():
            setattr(args, k, v)

    if not hasattr(args, 'dataset'):
        args.dataset = 'hicodet'

    return args

# ---------------------------
# Main evaluation flow
# ---------------------------
def main():
    args = parse_args()
    ensure_defaults(args)
    os.makedirs(args.save_dir, exist_ok=True)

    # 务必确保这里调用的是从 main_incremental 导入的正确划分函数
    from main_incremental import get_task_splits
    tasks = get_task_splits(args, num_classes=600)

    if getattr(args, 'verbose', False):
        print("Task splits:")
        for i, t in enumerate(tasks, start=1):
            print(f" Task {i}: count={len(t)}")

    init_dist_for_single_process(args.master_addr, args.master_port)

    # 构建一个 dummy train_loader（给 Engine 初始化用）
    trainset = DataFactory(name='hicodet', partition='train2015', data_root=args.data_root, filter_classes=None)
    train_sampler = DistributedSampler(trainset, num_replicas=1, rank=0, drop_last=False)
    train_loader = DataLoader(dataset=trainset, collate_fn=custom_collate, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    # 获取 obj_to_verb 映射
    base_train_dataset = get_base_dataset(train_loader.dataset)
    obj_to_verb = base_train_dataset.object_to_verb

    device = torch.device(args.device)

    per_ckpt_ap = {}

    # 遍历每个 Checkpoint
    for t_idx in range(1, args.num_tasks + 1):
        ckpt_path = os.path.join(args.output_dir, f'checkpoint_task{t_idx}.pth')
        if not os.path.exists(ckpt_path):
            print(f"[WARN] checkpoint {ckpt_path} not found -> skipping")
            continue

        # ================= 核心修改点 =================
        # 获取截至到当前 Task，模型学过的所有类别
        current_trained_classes = sum(tasks[:t_idx], [])
        print(f"\n--- Evaluating Checkpoint {t_idx} ---")
        print(f"Building Task-Aware test loader for {len(current_trained_classes)} classes...")

        # 动态构建当前 checkpoint 对应的裁剪版测试集（复现主程序的测试环境）
        testset = DataFactory(
            name='hicodet', partition='test2015',
            data_root=args.data_root,
            filter_classes=current_trained_classes
        )
        test_sampler = DistributedSampler(testset, num_replicas=1, rank=0, drop_last=False)
        test_loader = DataLoader(
            dataset=testset, collate_fn=custom_collate,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=True, sampler=test_sampler
        )
        # ==============================================

        if getattr(args, 'verbose', False):
            print(f"Loading checkpoint {ckpt_path} ...")
        model = load_model_from_ckpt(args, obj_to_verb, ckpt_path, device)

        # 这里传入 filter_classes=None 是为了让它返回 600 维的完整 AP 数组，不影响测试环境，但方便下面计算 CSV
        engine = CustomisedDLE(
            model, train_loader, test_loader, args,
            filter_classes=None, teacher_model=None,
            replay_indices=None, mir_dict=None, rare_set=None
        )

        ap_vec = engine.test_hico(return_per_sample_scores=False)
        if isinstance(ap_vec, torch.Tensor):
            ap_np = ap_vec.cpu().numpy()
        else:
            ap_np = np.array(ap_vec)

        per_ckpt_ap[t_idx] = ap_np

        # 严格复现主程序的 mAP 打印逻辑：只计算当前学过类别的平均分
        current_map = np.nanmean(ap_np[current_trained_classes])
        print(f"Checkpoint task{t_idx} evaluated (Task-Aware Mode): mAP = {current_map:.4f}")

        del engine
        del model
        torch.cuda.empty_cache()

    if len(per_ckpt_ap) == 0:
        print("No checkpoints evaluated. Exiting.")
        return

    # 生成详细结果和 CSV
    task_means = compute_task_specific_means(per_ckpt_ap, tasks)
    per_task_forgetting, overall_summary = compute_forgetting_and_summary(per_ckpt_ap, tasks)

    out = {
        "args": vars(args),
        "tasks": tasks,
        "available_checkpoints": sorted(list(per_ckpt_ap.keys())),
        "per_ckpt_ap_sampled": {str(k): v.tolist() for k, v in per_ckpt_ap.items()},
        "task_means_per_checkpoint": {str(k): {str(tk): v for tk, v in vdict.items()} for k, vdict in task_means.items()},
        "per_task_forgetting": per_task_forgetting,
        "overall_summary": overall_summary,
    }

    import time
    ts = int(time.time())
    out_json = os.path.join(args.save_dir, f'results_forgetting_taskaware_{ts}.json')
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved detailed results to {out_json}")

    import csv
    csv_path = os.path.join(args.save_dir, f'checkpoint_task_means_taskaware_{ts}.csv')
    max_tasks = args.num_tasks
    header = ["checkpoint"]
    for k in range(1, max_tasks + 1):
        header.append(f"task{k}_meanAP_if_present")
    rows = []
    for ck in sorted(per_ckpt_ap.keys()):
        row = [ck]
        for k in range(1, max_tasks + 1):
            if k <= ck:
                val = task_means[ck].get(k, float('nan'))
                row.append(val)
            else:
                row.append('')
        rows.append(row)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved CSV summary to {csv_path}")

    print("Overall final mAP (Task-Aware):", overall_summary.get('final_map_all'))

if __name__ == '__main__':
    main()
