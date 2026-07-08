#!/usr/bin/env python3
"""Generate figures/tables from one completed RGIP incremental run.

The script intentionally depends only on the Python standard library, numpy and
matplotlib, because the training environment does not necessarily provide
pandas/seaborn.
"""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict, OrderedDict
from statistics import mean, median

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pvic")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def markdown_table(rows, fieldnames=None):
    if not rows:
        return "(empty)\n"
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for row in rows:
        vals = []
        for key in fieldnames:
            val = row.get(key, "")
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def save_fig(fig, base_without_ext):
    fig.tight_layout()
    fig.savefig(base_without_ext + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(base_without_ext + ".pdf", bbox_inches="tight")
    plt.close(fig)


def safe_float(x):
    if x is None:
        return math.nan
    try:
        return float(x)
    except Exception:
        return math.nan


def summarise_values(values):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return {"count": 0, "mean": "", "median": "", "min": "", "max": ""}
    return {
        "count": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def dedupe_keep_last(rows, key_fn):
    ordered = OrderedDict()
    for row in rows:
        ordered[key_fn(row)] = row
    return list(ordered.values())


def group_mean(rows, group_keys, value_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_keys)].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        merged = {k: v for k, v in zip(group_keys, key)}
        for value_key in value_keys:
            vals = [
                safe_float(item.get(value_key))
                for item in items
                if value_key in item and not math.isnan(safe_float(item.get(value_key)))
            ]
            merged[value_key] = mean(vals) if vals else math.nan
        out.append(merged)
    return out


def parse_tag(args_obj, mode_hint):
    dataset = args_obj.get("dataset", "dataset")
    split_mode = args_obj.get("split_mode", "split")
    split_seed = args_obj.get("split_seed", "x")
    train_seed = args_obj.get("seed", "x")
    batch_size = args_obj.get("batch_size", "bs")
    epochs = args_obj.get("epochs", "ep")
    world_size = args_obj.get("world_size", "ws")
    replay = args_obj.get("n_replay", "nr")
    method = "rgipfull" if args_obj.get("use_rgip") else "baseline"
    return (
        f"{mode_hint}_{dataset}_{split_mode}_splitseed{split_seed}_"
        f"trainseed{train_seed}_{method}_ws{world_size}_bs{batch_size}_ep{epochs}_nr{replay}"
    )


def load_run(run_dir):
    metrics_dir = os.path.join(run_dir, "metrics")
    analysis_dir = os.path.join(run_dir, "analysis")
    config_path = os.path.join(run_dir, "configs", "train_args.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(run_dir, "train_args.json")
    args_obj = read_json(config_path)

    task_eval = []
    task_idx = 1
    while True:
        path = os.path.join(metrics_dir, f"task{task_idx}_eval.json")
        if not os.path.exists(path):
            break
        task_eval.append(read_json(path))
        task_idx += 1

    epoch_rows = read_jsonl(os.path.join(metrics_dir, "epoch_metrics.jsonl"))
    epoch_rows = dedupe_keep_last(
        epoch_rows,
        lambda r: (int(r.get("task_idx", -1)), int(r.get("epoch", -1))),
    )
    epoch_rows.sort(key=lambda r: (int(r.get("task_idx", 0)), int(r.get("epoch", 0))))

    rgip_rows = read_jsonl(os.path.join(metrics_dir, "rgip_iteration_stats.jsonl"))
    rgip_rows = dedupe_keep_last(
        rgip_rows,
        lambda r: (int(r.get("task_idx", -1)), int(r.get("iteration", -1))),
    )
    rgip_rows.sort(key=lambda r: (int(r.get("task_idx", 0)), int(r.get("iteration", 0))))

    per_sample = []
    for i in range(1, len(task_eval) + 1):
        path = os.path.join(analysis_dir, f"task{i}_per_sample_scores.json")
        if os.path.exists(path):
            per_sample.append((i, read_json(path)))

    return args_obj, task_eval, epoch_rows, rgip_rows, per_sample


def plot_stage_performance(out_root, tag, task_eval):
    out_dir = ensure_dir(os.path.join(out_root, "01_stage_performance"))
    desc = """说明：
本文件夹展示该次增量训练在每个阶段结束后的整体性能。
- mAP：当前已见 HOI 类别上的整体 AP 均值。
- Rare / Non-Rare：按 HICO-DET rare/non-rare 类别集合划分后的阶段性能。
- 这些结果来自 task*_eval.json，适合展示单次 run 的阶段趋势；不等价于 mean±std。
"""
    write_text(os.path.join(out_dir, "说明.txt"), desc)

    rows = []
    for r in task_eval:
        rows.append(
            {
                "task_idx": int(r["task_idx"]),
                "num_classes": int(r["num_classes"]),
                "mAP": float(r["mAP"]),
                "rare_mAP": float(r["rare_mAP"]),
                "non_rare_mAP": float(r["non_rare_mAP"]),
            }
        )
    write_csv(os.path.join(out_dir, f"stage_performance_{tag}.csv"), rows)
    write_text(
        os.path.join(out_dir, f"stage_performance_{tag}.md"),
        markdown_table(rows, ["task_idx", "num_classes", "mAP", "rare_mAP", "non_rare_mAP"]),
    )

    tasks = [r["task_idx"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(tasks, [r["mAP"] for r in rows], marker="o", label="mAP")
    ax.plot(tasks, [r["rare_mAP"] for r in rows], marker="s", label="Rare mAP")
    ax.plot(tasks, [r["non_rare_mAP"] for r in rows], marker="^", label="Non-Rare mAP")
    ax.set_xlabel("Incremental task")
    ax.set_ylabel("AP")
    ax.set_title("Stage-wise HICO-DET performance")
    ax.set_xticks(tasks)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, os.path.join(out_dir, f"stage_performance_curve_{tag}"))

    fig, ax = plt.subplots(figsize=(8.2, 2.6))
    ax.axis("off")
    table_data = [
        [
            r["task_idx"],
            r["num_classes"],
            f"{r['mAP']:.4f}",
            f"{r['rare_mAP']:.4f}",
            f"{r['non_rare_mAP']:.4f}",
        ]
        for r in rows
    ]
    tbl = ax.table(
        cellText=table_data,
        colLabels=["Task", "#Classes", "mAP", "Rare", "Non-Rare"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    ax.set_title("Final performance after each task", pad=12)
    save_fig(fig, os.path.join(out_dir, f"stage_performance_table_{tag}"))


def plot_epoch_curves(out_root, tag, epoch_rows):
    out_dir = ensure_dir(os.path.join(out_root, "02_epoch_training_curves"))
    desc = """说明：
本文件夹展示每个 task 内随 epoch 变化的验证性能曲线。
由于本次训练经历过断点重跑，脚本已按 (task_idx, epoch) 保留最后一条记录，避免失败前日志重复进入图表。
这些曲线适合观察收敛过程、选择 best epoch，以及判断 rare/non-rare 是否同步变化。
"""
    write_text(os.path.join(out_dir, "说明.txt"), desc)

    fields = ["task_idx", "epoch", "iteration", "mAP", "rare_mAP", "non_rare_mAP", "best_mAP_before_update"]
    write_csv(os.path.join(out_dir, f"epoch_metrics_dedup_{tag}.csv"), epoch_rows, fields)

    by_task = defaultdict(list)
    for r in epoch_rows:
        by_task[int(r["task_idx"])].append(r)

    best_rows = []
    for task, rows in sorted(by_task.items()):
        best = max(rows, key=lambda r: safe_float(r.get("mAP")))
        last = rows[-1]
        best_rows.append(
            {
                "task_idx": task,
                "best_epoch": int(best["epoch"]),
                "best_mAP": safe_float(best["mAP"]),
                "best_rare_mAP": safe_float(best["rare_mAP"]),
                "best_non_rare_mAP": safe_float(best["non_rare_mAP"]),
                "last_epoch": int(last["epoch"]),
                "last_mAP": safe_float(last["mAP"]),
            }
        )
    write_csv(os.path.join(out_dir, f"best_epoch_by_task_{tag}.csv"), best_rows)
    write_text(os.path.join(out_dir, f"best_epoch_by_task_{tag}.md"), markdown_table(best_rows))

    n_task = len(by_task)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), sharex=False, sharey=False)
    axes = axes.flatten()
    for ax, (task, rows) in zip(axes, sorted(by_task.items())):
        epochs = [int(r["epoch"]) for r in rows]
        ax.plot(epochs, [safe_float(r.get("mAP")) for r in rows], label="mAP")
        ax.plot(epochs, [safe_float(r.get("rare_mAP")) for r in rows], label="Rare")
        ax.plot(epochs, [safe_float(r.get("non_rare_mAP")) for r in rows], label="Non-Rare")
        ax.set_title(f"Task {task}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AP")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[n_task:]:
        ax.axis("off")
    fig.suptitle("Epoch-wise validation performance", y=1.02)
    save_fig(fig, os.path.join(out_dir, f"epoch_curves_map_rare_nonrare_{tag}"))


def introduced_groups(task_eval):
    groups = []
    seen = set()
    for r in task_eval:
        classes = [int(c) for c in r["classes"]]
        group = [c for c in classes if c not in seen]
        groups.append(group)
        seen.update(group)
    return groups


def plot_class_ap(out_root, tag, task_eval):
    out_dir = ensure_dir(os.path.join(out_root, "03_class_ap_heatmap"))
    desc = """说明：
本文件夹展示类别级 AP 的阶段变化。
- full600 heatmap：横轴为 HOI 类别，按引入阶段排序；纵轴为评估阶段；灰色表示该阶段尚未评估该类别。
- task_group_mean heatmap：把类别按引入任务分组，展示每个阶段对各组类别的平均 AP，更适合论文正文展示。
这些图可以支持“类别级性能变化/旧类保持情况”的分析，但不是多方法对比。
"""
    write_text(os.path.join(out_dir, "说明.txt"), desc)

    groups = introduced_groups(task_eval)
    ordered_classes = [c for group in groups for c in group]
    class_to_group = {c: i + 1 for i, group in enumerate(groups) for c in group}
    task_aps = [{int(k): float(v) for k, v in r["class_ap"].items()} for r in task_eval]

    matrix = np.full((len(task_eval), len(ordered_classes)), np.nan, dtype=float)
    rows = []
    for j, cls in enumerate(ordered_classes):
        row = {"class_id": cls, "introduced_task": class_to_group[cls]}
        for i, aps in enumerate(task_aps, 1):
            val = aps.get(cls, math.nan)
            matrix[i - 1, j] = val
            row[f"task{i}_ap"] = "" if math.isnan(val) else val
        rows.append(row)
    write_csv(os.path.join(out_dir, f"class_ap_matrix_{tag}.csv"), rows)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#d9d9d9")
    fig, ax = plt.subplots(figsize=(14, 3.8))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    boundaries = np.cumsum([len(g) for g in groups])
    for b in boundaries[:-1]:
        ax.axvline(b - 0.5, color="white", lw=1.2)
    centers = []
    start = 0
    for group in groups:
        centers.append(start + len(group) / 2 - 0.5)
        start += len(group)
    ax.set_xticks(centers)
    ax.set_xticklabels([f"Task {i}" for i in range(1, len(groups) + 1)])
    ax.set_yticks(range(len(task_eval)))
    ax.set_yticklabels([f"Eval T{i}" for i in range(1, len(task_eval) + 1)])
    ax.set_xlabel("HOI classes ordered by introduced task")
    ax.set_ylabel("Evaluation stage")
    ax.set_title("Class-wise AP heatmap")
    fig.colorbar(im, ax=ax, label="AP")
    save_fig(fig, os.path.join(out_dir, f"class_ap_heatmap_full600_{tag}"))

    group_matrix = np.full((len(task_eval), len(groups)), np.nan, dtype=float)
    group_rows = []
    for eval_idx, aps in enumerate(task_aps, 1):
        for group_idx, group in enumerate(groups, 1):
            vals = [aps[c] for c in group if c in aps]
            val = mean(vals) if vals else math.nan
            group_matrix[eval_idx - 1, group_idx - 1] = val
            group_rows.append(
                {
                    "eval_task": eval_idx,
                    "introduced_task_group": group_idx,
                    "num_classes": len(vals),
                    "mean_ap": "" if math.isnan(val) else val,
                }
            )
    write_csv(os.path.join(out_dir, f"task_group_mean_ap_{tag}.csv"), group_rows)
    write_text(os.path.join(out_dir, f"task_group_mean_ap_{tag}.md"), markdown_table(group_rows))

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(group_matrix, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"Intro T{i}" for i in range(1, len(groups) + 1)])
    ax.set_yticks(range(len(task_eval)))
    ax.set_yticklabels([f"Eval T{i}" for i in range(1, len(task_eval) + 1)])
    for i in range(group_matrix.shape[0]):
        for j in range(group_matrix.shape[1]):
            if not math.isnan(group_matrix[i, j]):
                ax.text(j, i, f"{group_matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    ax.set_title("Mean AP by introduced task group")
    fig.colorbar(im, ax=ax, label="AP")
    save_fig(fig, os.path.join(out_dir, f"task_group_mean_ap_heatmap_{tag}"))


def plot_forgetting(out_root, tag, task_eval):
    out_dir = ensure_dir(os.path.join(out_root, "04_forgetting_analysis"))
    desc = """说明：
本文件夹展示旧类别遗忘情况。
脚本以“类别被引入阶段的 AP”作为初始 AP，以 Task4 最终 AP 作为最终 AP，计算 AP drop = intro_AP - final_AP。
均值越大表示遗忘越明显；负值表示最终阶段反而更好。Task4 新引入类别没有未来阶段，因此不纳入遗忘柱状图。
"""
    write_text(os.path.join(out_dir, "说明.txt"), desc)

    groups = introduced_groups(task_eval)
    task_aps = [{int(k): float(v) for k, v in r["class_ap"].items()} for r in task_eval]
    final_aps = task_aps[-1]
    class_rows = []
    summary_rows = []
    box_values = []
    labels = []
    for intro_idx, group in enumerate(groups, 1):
        vals = []
        intro_vals = []
        final_vals = []
        for cls in group:
            intro_ap = task_aps[intro_idx - 1].get(cls, math.nan)
            final_ap = final_aps.get(cls, math.nan)
            if math.isnan(intro_ap) or math.isnan(final_ap):
                continue
            drop = intro_ap - final_ap
            class_rows.append(
                {
                    "class_id": cls,
                    "introduced_task": intro_idx,
                    "intro_ap": intro_ap,
                    "final_task4_ap": final_ap,
                    "ap_drop": drop,
                }
            )
            vals.append(drop)
            intro_vals.append(intro_ap)
            final_vals.append(final_ap)
        if vals:
            summary_rows.append(
                {
                    "introduced_task": intro_idx,
                    "num_classes": len(vals),
                    "intro_mean_ap": mean(intro_vals),
                    "final_mean_ap": mean(final_vals),
                    "mean_ap_drop": mean(vals),
                    "median_ap_drop": median(vals),
                    "positive_drop_ratio": sum(v > 0 for v in vals) / len(vals),
                }
            )
            if intro_idx < len(groups):
                box_values.append(vals)
                labels.append(f"Intro T{intro_idx}")
    write_csv(os.path.join(out_dir, f"forgetting_by_class_{tag}.csv"), class_rows)
    write_csv(os.path.join(out_dir, f"forgetting_summary_{tag}.csv"), summary_rows)
    write_text(os.path.join(out_dir, f"forgetting_summary_{tag}.md"), markdown_table(summary_rows))

    old_summary = [r for r in summary_rows if int(r["introduced_task"]) < len(groups)]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.bar(
        [f"Intro T{int(r['introduced_task'])}" for r in old_summary],
        [r["mean_ap_drop"] for r in old_summary],
        color="#4c78a8",
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("AP drop (introduced stage - final Task4)")
    ax.set_title("Average forgetting by introduced task group")
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, os.path.join(out_dir, f"forgetting_mean_drop_{tag}"))

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.boxplot(box_values, labels=labels, showmeans=True)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Class-level AP drop")
    ax.set_title("Distribution of class-level forgetting")
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, os.path.join(out_dir, f"forgetting_drop_boxplot_{tag}"))


def add_global_axis(rows, max_epoch=30):
    out = []
    for r in rows:
        rr = dict(r)
        task = int(rr.get("task_idx", 0))
        epoch = safe_float(rr.get("epoch"))
        rr["global_epoch"] = (task - 1) * max_epoch + epoch
        out.append(rr)
    return out


def plot_rgip_losses(out_root, tag, rgip_rows, max_epoch):
    out_dir = ensure_dir(os.path.join(out_root, "05_rgip_loss_components"))
    desc = """说明：
本文件夹展示 RGIP 训练内部损失。
- int_loss：interaction preservation 总损失。
- feat_kd / logit_kd / hardneg_loss：特征蒸馏、logit 蒸馏、hard negative 约束分量。
脚本已按 (task_idx, iteration) 保留最后一条记录，降低断点重跑日志重复的影响。
"""
    write_text(os.path.join(out_dir, "说明.txt"), desc)

    rgip_rows = add_global_axis(rgip_rows, max_epoch=max_epoch)
    fields = [
        "task_idx",
        "epoch",
        "global_epoch",
        "iteration",
        "loss",
        "rgip/int_loss",
        "rgip/feat_kd",
        "rgip/logit_kd",
        "rgip/hardneg_loss",
        "rgip/distilled_pairs",
    ]
    write_csv(os.path.join(out_dir, f"rgip_stats_dedup_{tag}.csv"), rgip_rows, fields)

    summary_rows = []
    for key in ["rgip/int_loss", "rgip/feat_kd", "rgip/logit_kd", "rgip/hardneg_loss"]:
        stat = summarise_values([r.get(key) for r in rgip_rows if key in r])
        stat["metric"] = key
        summary_rows.append(stat)
    summary_fields = ["metric", "count", "mean", "median", "min", "max"]
    write_csv(os.path.join(out_dir, f"rgip_loss_summary_{tag}.csv"), summary_rows, summary_fields)
    write_text(os.path.join(out_dir, f"rgip_loss_summary_{tag}.md"), markdown_table(summary_rows, summary_fields))

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    x = list(range(1, len(rgip_rows) + 1))
    axes[0].plot(x, [safe_float(r.get("rgip/int_loss")) for r in rgip_rows], lw=1.2, label="int_loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("RGIP interaction loss by logged iteration")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    for key, label in [
        ("rgip/feat_kd", "feat_kd"),
        ("rgip/logit_kd", "logit_kd"),
        ("rgip/hardneg_loss", "hardneg"),
    ]:
        axes[1].plot(x, [safe_float(r.get(key)) for r in rgip_rows], lw=1.0, label=label)
    axes[1].set_xlabel("Logged RGIP stat index")
    axes[1].set_ylabel("Loss component")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    save_fig(fig, os.path.join(out_dir, f"rgip_loss_components_by_iteration_{tag}"))

    by_epoch = group_mean(
        rgip_rows,
        ["task_idx", "epoch"],
        ["rgip/int_loss", "rgip/feat_kd", "rgip/logit_kd", "rgip/hardneg_loss"],
    )
    by_epoch = add_global_axis(by_epoch, max_epoch=max_epoch)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = [r["global_epoch"] for r in by_epoch]
    for key, label in [
        ("rgip/int_loss", "int_loss"),
        ("rgip/feat_kd", "feat_kd"),
        ("rgip/logit_kd", "logit_kd"),
        ("rgip/hardneg_loss", "hardneg"),
    ]:
        ax.plot(x, [safe_float(r.get(key)) for r in by_epoch], marker=".", lw=1.0, label=label)
    for t in sorted({int(r["task_idx"]) for r in by_epoch})[1:]:
        ax.axvline((t - 1) * max_epoch + 0.5, color="gray", lw=0.8, alpha=0.45)
    ax.set_xlabel("Global epoch index")
    ax.set_ylabel("Mean logged value per epoch")
    ax.set_title("RGIP loss components averaged by epoch")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_fig(fig, os.path.join(out_dir, f"rgip_loss_components_by_epoch_{tag}"))


def plot_sis_pair_weight(out_root, tag, rgip_rows, max_epoch):
    out_dir = ensure_dir(os.path.join(out_root, "06_sis_pair_weight"))
    desc = """说明：
本文件夹展示 RGIP 的稀缺性/干扰强度相关统计。
- sis_mean / sis_max：当前 batch 内 interaction interference score 的平均值和最大值。
- pair_weight_mean / pair_weight_max：根据 SIS 得到的样本权重统计。
这些图可用于说明稀缺性与干扰权重确实在训练中产生了非零分布。
"""
    write_text(os.path.join(out_dir, "说明.txt"), desc)

    rgip_rows = add_global_axis(rgip_rows, max_epoch=max_epoch)
    keys = ["rgip/sis_mean", "rgip/sis_max", "rgip/pair_weight_mean", "rgip/pair_weight_max"]
    summary_rows = []
    for key in keys:
        stat = summarise_values([r.get(key) for r in rgip_rows if key in r])
        stat["metric"] = key
        summary_rows.append(stat)
    fields = ["metric", "count", "mean", "median", "min", "max"]
    write_csv(os.path.join(out_dir, f"sis_pair_weight_summary_{tag}.csv"), summary_rows, fields)
    write_text(os.path.join(out_dir, f"sis_pair_weight_summary_{tag}.md"), markdown_table(summary_rows, fields))

    by_epoch = group_mean(rgip_rows, ["task_idx", "epoch"], keys)
    by_epoch = add_global_axis(by_epoch, max_epoch=max_epoch)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    x = [r["global_epoch"] for r in by_epoch]
    axes[0].plot(x, [safe_float(r.get("rgip/sis_mean")) for r in by_epoch], label="sis_mean")
    axes[0].plot(x, [safe_float(r.get("rgip/sis_max")) for r in by_epoch], label="sis_max")
    axes[0].set_ylabel("SIS")
    axes[0].set_title("SIS statistics by epoch")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(x, [safe_float(r.get("rgip/pair_weight_mean")) for r in by_epoch], label="pair_weight_mean")
    axes[1].plot(x, [safe_float(r.get("rgip/pair_weight_max")) for r in by_epoch], label="pair_weight_max")
    axes[1].set_xlabel("Global epoch index")
    axes[1].set_ylabel("Pair weight")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    save_fig(fig, os.path.join(out_dir, f"sis_pair_weight_curves_{tag}"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, key, title in zip(
        axes,
        ["rgip/sis_mean", "rgip/sis_max", "rgip/pair_weight_max"],
        ["SIS mean", "SIS max", "Pair weight max"],
    ):
        vals = [safe_float(r.get(key)) for r in rgip_rows if key in r and not math.isnan(safe_float(r.get(key)))]
        ax.hist(vals, bins=30, color="#4c78a8", alpha=0.85)
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.25)
    save_fig(fig, os.path.join(out_dir, f"sis_pair_weight_distribution_{tag}"))


def plot_replay_prototype(out_root, tag, rgip_rows, max_epoch):
    out_dir = ensure_dir(os.path.join(out_root, "07_replay_prototype_stats"))
    desc = """说明：
本文件夹展示 replay、蒸馏样本和 prototype bank 的训练统计。
- distilled_pairs：真正参与 interaction distillation 的 pair 数。
- num_replay_images / num_replay_pairs：batch 中 replay 样本及其 pair 数。
- prototype_count：当前已维护的 prototype 数。
这些图可用于说明 RGIP 额外训练信号的覆盖范围和训练成本。
"""
    write_text(os.path.join(out_dir, "说明.txt"), desc)

    rgip_rows = add_global_axis(rgip_rows, max_epoch=max_epoch)
    keys = [
        "rgip/distilled_pairs",
        "rgip/num_replay_images",
        "rgip/num_replay_pairs",
        "rgip/prototype_count",
        "rgip/same_object_ratio",
        "rgip/confounder_valid_ratio",
        "rgip/modulated_pair_count",
    ]
    summary_rows = []
    for key in keys:
        stat = summarise_values([r.get(key) for r in rgip_rows if key in r])
        stat["metric"] = key
        summary_rows.append(stat)
    fields = ["metric", "count", "mean", "median", "min", "max"]
    write_csv(os.path.join(out_dir, f"replay_prototype_summary_{tag}.csv"), summary_rows, fields)
    write_text(os.path.join(out_dir, f"replay_prototype_summary_{tag}.md"), markdown_table(summary_rows, fields))

    by_epoch = group_mean(
        rgip_rows,
        ["task_idx", "epoch"],
        ["rgip/distilled_pairs", "rgip/num_replay_images", "rgip/num_replay_pairs", "rgip/prototype_count"],
    )
    by_epoch = add_global_axis(by_epoch, max_epoch=max_epoch)
    x = [r["global_epoch"] for r in by_epoch]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    axes[0].plot(x, [safe_float(r.get("rgip/distilled_pairs")) for r in by_epoch], label="distilled_pairs")
    axes[0].plot(x, [safe_float(r.get("rgip/num_replay_images")) for r in by_epoch], label="num_replay_images")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Replay and distillation coverage")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(x, [safe_float(r.get("rgip/num_replay_pairs")) for r in by_epoch], label="num_replay_pairs")
    axes[1].set_xlabel("Global epoch index")
    axes[1].set_ylabel("Replay pairs")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    save_fig(fig, os.path.join(out_dir, f"replay_distillation_stats_{tag}"))

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.plot(x, [safe_float(r.get("rgip/prototype_count")) for r in by_epoch], marker=".", label="prototype_count")
    ax.set_xlabel("Global epoch index")
    ax.set_ylabel("Prototype count")
    ax.set_title("Prototype bank growth")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_fig(fig, os.path.join(out_dir, f"prototype_count_curve_{tag}"))


def parse_training_log(log_path):
    if not os.path.exists(log_path):
        return []
    pattern = re.compile(
        r"Epoch \[(\d+)/(\d+)\], Iter\. \[(\d+)/(\d+)\], Loss: ([0-9.]+), "
        r"Time\[Data/Iter\.\]: \[([0-9.]+)s/([0-9.]+)s\]"
    )
    rows = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            m = pattern.search(line)
            if not m:
                continue
            epoch, epochs, iter_idx, iter_total, loss, data_time, iter_time = m.groups()
            rows.append(
                {
                    "line_no": line_no,
                    "log_index": len(rows) + 1,
                    "epoch": int(epoch),
                    "epochs": int(epochs),
                    "iter": int(iter_idx),
                    "iter_total": int(iter_total),
                    "loss": float(loss),
                    "data_time_s": float(data_time),
                    "iter_window_time_s": float(iter_time),
                }
            )
    return rows


def plot_training_cost(out_root, tag, run_dir):
    out_dir = ensure_dir(os.path.join(out_root, "08_training_cost"))
    desc = """说明：
本文件夹从 train_fast4.log 解析终端输出中的 loss 与时间统计。
注意：日志经历过断点重跑，且 Time[Data/Iter.] 是训练引擎按打印窗口汇总/平均后的统计，不是严格单 iteration wall-clock。
因此该图适合展示训练成本的粗略趋势，不建议作为精确速度 benchmark。
"""
    write_text(os.path.join(out_dir, "说明.txt"), desc)

    log_path = os.path.join(run_dir, "logs", "train_fast4.log")
    rows = parse_training_log(log_path)
    write_csv(os.path.join(out_dir, f"training_log_samples_{tag}.csv"), rows)

    summary_rows = []
    for key in ["loss", "data_time_s", "iter_window_time_s"]:
        stat = summarise_values([r.get(key) for r in rows])
        stat["metric"] = key
        summary_rows.append(stat)
    fields = ["metric", "count", "mean", "median", "min", "max"]
    write_csv(os.path.join(out_dir, f"training_cost_summary_{tag}.csv"), summary_rows, fields)
    write_text(os.path.join(out_dir, f"training_cost_summary_{tag}.md"), markdown_table(summary_rows, fields))

    if not rows:
        return
    x = [r["log_index"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    axes[0].plot(x, [r["loss"] for r in rows], color="#4c78a8", lw=1.0)
    axes[0].set_ylabel("Printed loss")
    axes[0].set_title("Training loss from terminal log")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(x, [r["iter_window_time_s"] for r in rows], label="iter_window_time_s", lw=1.0)
    axes[1].plot(x, [r["data_time_s"] for r in rows], label="data_time_s", lw=1.0)
    axes[1].set_xlabel("Printed log index")
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Logged data/iteration time")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    save_fig(fig, os.path.join(out_dir, f"training_loss_time_from_log_{tag}"))


def plot_per_sample_scores(out_root, tag, per_sample):
    out_dir = ensure_dir(os.path.join(out_root, "09_per_sample_score_distribution"))
    desc = """说明：
本文件夹展示每个评估阶段保存的 per-sample GT 类置信度分布。
每个样本统计 scores 字典中的 mean/max/min 值；这些值可辅助分析样本层面的预测置信度，但不是 AP，也不能单独代表错误率。
"""
    write_text(os.path.join(out_dir, "说明.txt"), desc)

    flat_rows = []
    summary_rows = []
    data_for_box = []
    labels = []
    for task_idx, items in per_sample:
        vals = []
        for item in items:
            score_vals = [safe_float(v) for v in item.get("scores", {}).values()]
            score_vals = [v for v in score_vals if not math.isnan(v)]
            if not score_vals:
                continue
            mean_score = mean(score_vals)
            max_score = max(score_vals)
            min_score = min(score_vals)
            vals.append(mean_score)
            flat_rows.append(
                {
                    "task_idx": task_idx,
                    "local_idx": item.get("local_idx", ""),
                    "num_gt_classes": len(item.get("gt_classes", [])),
                    "num_scored_classes": len(score_vals),
                    "mean_gt_score": mean_score,
                    "max_gt_score": max_score,
                    "min_gt_score": min_score,
                }
            )
        if vals:
            stat = summarise_values(vals)
            stat["task_idx"] = task_idx
            summary_rows.append(stat)
            data_for_box.append(vals)
            labels.append(f"Task {task_idx}")
    write_csv(os.path.join(out_dir, f"per_sample_scores_flat_{tag}.csv"), flat_rows)
    fields = ["task_idx", "count", "mean", "median", "min", "max"]
    write_csv(os.path.join(out_dir, f"per_sample_score_summary_{tag}.csv"), summary_rows, fields)
    write_text(os.path.join(out_dir, f"per_sample_score_summary_{tag}.md"), markdown_table(summary_rows, fields))

    if not data_for_box:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.boxplot(data_for_box, labels=labels, showmeans=True)
    ax.set_ylabel("Mean GT-class score per sample")
    ax.set_title("Per-sample GT confidence distribution")
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, os.path.join(out_dir, f"per_sample_gt_score_distribution_{tag}"))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bins = np.linspace(0, 1, 31)
    for vals, label in zip(data_for_box, labels):
        ax.hist(vals, bins=bins, histtype="step", lw=1.6, label=label)
    ax.set_xlabel("Mean GT-class score per sample")
    ax.set_ylabel("Count")
    ax.set_title("Per-sample score histograms")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_fig(fig, os.path.join(out_dir, f"per_sample_gt_score_histogram_{tag}"))


def write_root_readme(out_root, tag, run_dir, args_obj, task_eval):
    final = task_eval[-1] if task_eval else {}
    text = f"""RGIP 实验图表汇总

来源 run:
{os.path.abspath(run_dir)}

训练参数标签:
{tag}

数据集/协议:
- dataset: {args_obj.get('dataset')}
- split_mode: {args_obj.get('split_mode')}
- split_seed: {args_obj.get('split_seed')}
- train_seed: {args_obj.get('seed')}
- world_size: {args_obj.get('world_size')}
- batch_size: {args_obj.get('batch_size')}
- epochs: {args_obj.get('epochs')}
- n_replay: {args_obj.get('n_replay')}
- use_replay: {args_obj.get('use_replay')}
- use_rgip: {args_obj.get('use_rgip')}

最终 Task{final.get('task_idx', '?')} 结果:
- mAP: {safe_float(final.get('mAP')):.6f}
- Rare mAP: {safe_float(final.get('rare_mAP')):.6f}
- Non-Rare mAP: {safe_float(final.get('non_rare_mAP')):.6f}

注意事项:
1. 本目录中的图表来自单个 split seed / train seed，不能直接作为 mean ± std。
2. epoch_metrics 与 rgip_iteration_stats 已按最后记录去重，以减少断点重跑日志重复的影响。
3. 本次 run 中 attention modulation 相关的 modulated_pair_count 为 0，因此没有生成 attention modulation 有效性图。
4. 注意力可视化图需要额外运行 inference_demo.py 或 attention-vis 脚本，本目录只包含当前训练输出能直接支持的图表。
"""
    write_text(os.path.join(out_root, "README.txt"), text)
    write_text(os.path.join(out_root, f"run_metadata_{tag}.json"), json.dumps(args_obj, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="outputs-rgip/fast4-splitrandom-splitseed140-trainseed140-20260629-233443",
    )
    parser.add_argument("--output-root", default="rgip_experiment_figures")
    parser.add_argument("--mode-hint", default="fast4")
    args = parser.parse_args()

    run_dir = args.run_dir
    out_root = ensure_dir(args.output_root)
    args_obj, task_eval, epoch_rows, rgip_rows, per_sample = load_run(run_dir)
    tag = parse_tag(args_obj, args.mode_hint)
    max_epoch = int(args_obj.get("epochs", 30))

    plot_stage_performance(out_root, tag, task_eval)
    plot_epoch_curves(out_root, tag, epoch_rows)
    plot_class_ap(out_root, tag, task_eval)
    plot_forgetting(out_root, tag, task_eval)
    plot_rgip_losses(out_root, tag, rgip_rows, max_epoch)
    plot_sis_pair_weight(out_root, tag, rgip_rows, max_epoch)
    plot_replay_prototype(out_root, tag, rgip_rows, max_epoch)
    plot_training_cost(out_root, tag, run_dir)
    plot_per_sample_scores(out_root, tag, per_sample)
    write_root_readme(out_root, tag, run_dir, args_obj, task_eval)

    print(os.path.abspath(out_root))


if __name__ == "__main__":
    main()
