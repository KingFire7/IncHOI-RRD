#!/usr/bin/env python3
# coding: utf-8
"""
evaluate_forgetting_viz.py

可视化 evaluate_forgetting.py 的输出，并支持按类别子集（full / rare / nonrare）输出 task learning and forgetting curves.

变化（本次）：
- 不在三张 learning 曲线图中显示图例（full/rare/nonrare 均不带 legend）
- 单独生成一个 legend 图文件 task_learning_legend.{pdf,png}，包含所有 Task 的图例，便于在论文中单独展示或放在图注下方
- 保持之前的字体放大、方形坐标轴、四边刻度等行为

用法与之前相同。
"""
import os
import json
import argparse
from typing import Optional, Dict, List, Tuple, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.0)


# -------------------------
# Utilities
# -------------------------
def load_results(results_json: str) -> Tuple[Dict[int, np.ndarray], Optional[List[List[int]]], Dict, Dict, Dict]:
    with open(results_json, 'r') as f:
        data = json.load(f)
    per_ckpt_ap = {int(k): np.array(v, dtype=float) for k, v in data['per_ckpt_ap_sampled'].items()}
    tasks = data.get('tasks', None)
    overall = data.get('overall_summary', {})
    per_task = data.get('per_task_forgetting', {})
    return per_ckpt_ap, tasks, per_task, overall, data.get('args', {})


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def sorted_ckpts_array(per_ckpt_ap: Dict[int, np.ndarray]):
    """Return (ckpt_indices_sorted, ap_matrix) where ap_matrix shape = (n_ckpts, n_classes)."""
    keys = sorted(per_ckpt_ap.keys())
    matrix = np.stack([per_ckpt_ap[k] for k in keys], axis=0)
    return keys, matrix


def load_rare_ids_from_file(path: str) -> Optional[List[int]]:
    if not path:
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] failed to load rare classes from {path}: {e}")
        return None

    if isinstance(data, dict):
        for key in ('rare', 'rare_ids', 'rare_classes'):
            if key in data and isinstance(data[key], list):
                try:
                    return [int(x) for x in data[key]]
                except Exception:
                    return None
        for v in data.values():
            if isinstance(v, list):
                try:
                    return [int(x) for x in v]
                except Exception:
                    continue
        return None

    if isinstance(data, list):
        try:
            return [int(x) for x in data]
        except Exception:
            return None

    return None


def _make_axis_square_and_ticks(ax: plt.Axes, fig: plt.Figure, figsize: Tuple[float, float]):
    try:
        w, h = figsize
        side = min(w, h)
    except Exception:
        side = float(figsize[0] if isinstance(figsize, (list, tuple)) else figsize)
    fig.set_size_inches(side, side)
    if hasattr(ax, "set_box_aspect"):
        try:
            ax.set_box_aspect(1.0)
        except Exception:
            pass
    else:
        try:
            ax.set_aspect('equal', adjustable='box')
        except Exception:
            pass

    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2,
                   top=True, right=True, bottom=True, left=True)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.0,
                   top=True, right=True, bottom=True, left=True)
    ax.minorticks_on()

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor('black')


# -------------------------
# Legend helper: create a standalone legend image for Tasks
# -------------------------
def plot_tasks_legend(num_tasks: int, outpath: str, text_size: int = 12, palette_name: str = "tab10"):
    """
    Create a standalone legend image with entries "Task 1", "Task 2", ...
    Saved to outpath.{pdf,png}.
    """
    if num_tasks <= 0:
        return
    palette = sns.color_palette(palette_name, num_tasks)
    # Determine ncol to make legend reasonably compact
    ncol = min(6, max(1, num_tasks))
    # figure width scaled by ncol
    fig_w = max(6.0, 1.2 * ncol)
    fig_h = 1.2 + (num_tasks // ncol) * 0.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    handles = []
    labels = []
    for i in range(num_tasks):
        h, = ax.plot([], [], color=palette[i % len(palette)], marker='o', linestyle='-', linewidth=2.0)
        handles.append(h)
        labels.append(f"Task {i+1}")
    leg = ax.legend(handles, labels, ncol=ncol, loc='center', frameon=False,
                    fontsize=max(8, text_size - 2), title='Tasks')
    try:
        leg.get_title().set_fontsize(max(8, text_size - 2))
    except Exception:
        pass
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outpath + ".pdf", bbox_inches='tight')
    fig.savefig(outpath + ".png", bbox_inches='tight', dpi=300)
    plt.close(fig)


# -------------------------
# Plotting functions (task curves accept class_filter and show_legend flag)
# -------------------------
def plot_task_learning_curves(per_ckpt_ap: Dict[int, np.ndarray],
                              tasks: List[List[int]],
                              outpath: str,
                              dpi: int = 300,
                              figsize=(6.5, 4.5),
                              title: Optional[str] = None,
                              y_limits: Optional[Tuple[float, float]] = None,
                              class_filter: Optional[Sequence[int]] = None,
                              show_legend: bool = False,
                              text_size: int = 12):
    """
    Plot mean AP for each task across checkpoints (learning curves).
    Legend is optional; for this workflow we set show_legend=False in each plot and create one legend file separately.
    """
    ckpts, ap_mat = sorted_ckpts_array(per_ckpt_ap)
    n_tasks = len(tasks)
    class_filter_set = set(class_filter) if class_filter is not None else None

    sns.set(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    palette = sns.color_palette("tab10", n_tasks)

    all_means = []
    plotted_any = False
    for t_idx, cls_list in enumerate(tasks, start=1):
        if class_filter_set is None:
            use_cls = cls_list
        else:
            use_cls = [c for c in cls_list if c in class_filter_set]
        if not use_cls:
            continue
        plotted_any = True
        means = []
        xs = []
        for j_ix, ck in enumerate(ckpts, start=1):
            if j_ix >= t_idx:
                vals = ap_mat[j_ix - 1, use_cls]
                m = np.nanmean(vals)
                means.append(m)
                xs.append(j_ix)
                if not np.isnan(m):
                    all_means.append(m)
        ax.plot(xs, means, label=f"Task {t_idx}", linewidth=2.0,
                marker='o', markersize=max(4, text_size/2.5),
                color=palette[(t_idx-1) % len(palette)])

    if not plotted_any:
        ax.text(0.5, 0.5, "No classes to plot for this filter", transform=ax.transAxes, ha='center', va='center', fontsize=text_size)
        ymin, ymax = 0.0, 1.0
    else:
        if y_limits is not None:
            ymin, ymax = float(y_limits[0]), float(y_limits[1])
        else:
            if len(all_means) == 0:
                ymin, ymax = 0.0, 1.0
            else:
                min_val = float(np.min(all_means))
                max_val = float(np.max(all_means))
                data_range = max_val - min_val
                if data_range < 0.25:
                    pad = max(0.02, data_range * 0.1)
                    ymin = max(0.0, min_val - pad)
                    ymax = min(1.0, max_val + pad)
                    if ymax - ymin < 1e-6:
                        ymin = max(0.0, min_val - 0.02)
                        ymax = min(1.0, max_val + 0.02)
                else:
                    ymin, ymax = 0.0, 1.0

    ax.set_xlabel("Task index", fontsize=text_size)
    ax.set_ylabel("Mean AP", fontsize=text_size)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(ckpts)
    ax.tick_params(axis='both', labelsize=max(8, text_size - 2))
    if title is not None:
        ax.set_title(title, fontsize=text_size + 2)

    _make_axis_square_and_ticks(ax, fig, figsize)

    # Do not place legend inside this plot (we will generate a separate legend image)
    fig.tight_layout()
    save_prefix = os.path.splitext(outpath)[0] if outpath.endswith(('.pdf', '.png')) else outpath
    fig.savefig(save_prefix + ".pdf", dpi=dpi, bbox_inches='tight')
    fig.savefig(save_prefix + ".png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_task_forgetting_curves(per_ckpt_ap: Dict[int, np.ndarray],
                                tasks: List[List[int]],
                                outpath: str,
                                dpi: int = 300,
                                figsize=(6.5, 4.5),
                                title: Optional[str] = None,
                                class_filter: Optional[Sequence[int]] = None,
                                text_size: int = 12):
    ckpts, ap_mat = sorted_ckpts_array(per_ckpt_ap)
    keys = ckpts
    n_tasks = len(tasks)
    class_filter_set = set(class_filter) if class_filter is not None else None

    sns.set(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    palette = sns.color_palette("tab10", n_tasks)

    plotted_any = False
    for t_idx, cls_list in enumerate(tasks, start=1):
        if class_filter_set is None:
            use_cls = cls_list
        else:
            use_cls = [c for c in cls_list if c in class_filter_set]
        if not use_cls:
            continue
        plotted_any = True

        if t_idx in keys:
            init_vec = ap_mat[keys.index(t_idx), use_cls]
        else:
            init_vec = np.nanmean(ap_mat[:, use_cls], axis=0)

        xs = []
        forget_means = []
        for j_ix, ck in enumerate(keys, start=1):
            if j_ix >= t_idx:
                cur_ap = ap_mat[j_ix - 1, use_cls]
                forgetting = np.nanmean(init_vec - cur_ap)
                xs.append(j_ix)
                forget_means.append(forgetting)
        ax.plot(xs, forget_means, label=f"Task {t_idx}", linewidth=2.0,
                marker='o', markersize=max(4, text_size/2.5),
                color=palette[(t_idx-1) % len(palette)])

    if not plotted_any:
        ax.text(0.5, 0.5, "No classes to plot for this filter", transform=ax.transAxes, ha='center', va='center', fontsize=text_size)

    ax.set_xlabel("Checkpoint (task index)", fontsize=text_size)
    ax.set_xticks(ckpts)
    ax.set_ylabel("Mean forgetting (initial - current)", fontsize=text_size)
    ax.tick_params(axis='both', labelsize=max(8, text_size - 2))
    ax.set_ylim(bottom=0.0)
    if title:
        ax.set_title(title, fontsize=text_size + 2)

    _make_axis_square_and_ticks(ax, fig, figsize)
    leg = ax.legend(title="Tasks", loc='best', fontsize=max(8, text_size - 2))
    try:
        leg.get_title().set_fontsize(max(8, text_size - 2))
    except Exception:
        pass
    fig.tight_layout()

    save_prefix = os.path.splitext(outpath)[0] if outpath.endswith(('.pdf', '.png')) else outpath
    fig.savefig(save_prefix + ".pdf", dpi=dpi, bbox_inches='tight')
    fig.savefig(save_prefix + ".png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# -------------------------
# Other plotting functions preserved (heatmap, topk, etc.)
# -------------------------
def plot_per_class_ap_heatmap(per_ckpt_ap: Dict[int, np.ndarray],
                              tasks: Optional[List[List[int]]],
                              outpath: str,
                              dpi: int = 300,
                              figsize=(8.0, 6.0),
                              cmap='viridis'):
    ckpts, ap_mat = sorted_ckpts_array(per_ckpt_ap)
    ap_mat_T = ap_mat.T
    n_classes = ap_mat_T.shape[0]
    if tasks is not None:
        order = [c for task in tasks for c in task]
    else:
        order = list(range(n_classes))
    ap_ordered = ap_mat_T[order, :]
    df = pd.DataFrame(ap_ordered, index=[f"cls_{c}" for c in order], columns=[f"ckpt_{k}" for k in ckpts])
    sns.set(style="white")
    plt.figure(figsize=figsize, dpi=dpi)
    ax = sns.heatmap(df, cmap=cmap, vmin=0.0, vmax=1.0, cbar_kws={'label': 'AP'})
    ax.set_ylabel("Classes (grouped by task)")
    ax.set_xlabel("Checkpoint")
    plt.tight_layout()
    save_prefix = os.path.splitext(outpath)[0] if outpath.endswith(('.pdf', '.png')) else outpath
    plt.savefig(save_prefix + ".pdf", bbox_inches='tight', dpi=dpi)
    plt.savefig(save_prefix + ".png", bbox_inches='tight', dpi=dpi)
    plt.close()


def plot_topk_forgetting(per_ckpt_ap: Dict[int, np.ndarray],
                         tasks: Optional[List[List[int]]],
                         outpath: str,
                         topk: int = 20,
                         dpi: int = 300,
                         figsize=(8.0, 6.0),
                         class_names: Optional[Dict[int, str]] = None):
    ckpts, ap_mat = sorted_ckpts_array(per_ckpt_ap)
    final_ap = ap_mat[-1]
    n_classes = ap_mat.shape[1]
    initial_ap = np.full(n_classes, np.nan, dtype=float)
    if tasks is not None:
        for k, cls_list in enumerate(tasks, start=1):
            ck = k
            cls_arr = np.array(cls_list, dtype=int)
            if ck in ckpts:
                initial_vec = ap_mat[ckpts.index(ck), cls_arr]
                initial_ap[cls_arr] = initial_vec
            else:
                stacked = np.stack([ap_mat[i, cls_arr] for i in range(ap_mat.shape[0])], axis=0)
                initial_ap[cls_arr] = np.nanmean(stacked, axis=0)
    else:
        initial_ap = ap_mat[0]
    forgetting = initial_ap - final_ap
    idx_sorted = np.argsort(np.nan_to_num(forgetting, nan=-1e6))[::-1]
    topk_idx = [i for i in idx_sorted if not np.isnan(forgetting[i])][:topk]
    labels = [class_names.get(i, f"cls_{i}") if class_names else f"cls_{i}" for i in topk_idx]
    init_vals = initial_ap[topk_idx]
    final_vals = final_ap[topk_idx]
    forget_vals = forgetting[topk_idx]
    x = np.arange(len(topk_idx))
    width = 0.32
    sns.set(style="whitegrid", font_scale=1.0)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.bar(x - width/2, init_vals, width=width, label='Initial AP', color='tab:blue')
    plt.bar(x + width/2, final_vals, width=width, label='Final AP', color='tab:orange')
    for i, fv in enumerate(forget_vals):
        plt.text(i, max(init_vals[i], final_vals[i]) + 0.02, f"{fv:.3f}", ha='center', va='bottom', fontsize=8)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel("AP")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.title(f"Top-{topk} Forgotten Classes (initial - final)")
    plt.tight_layout()
    save_prefix = os.path.splitext(outpath)[0] if outpath.endswith(('.pdf', '.png')) else outpath
    plt.savefig(save_prefix + ".pdf", bbox_inches='tight', dpi=dpi)
    plt.savefig(save_prefix + ".png", bbox_inches='tight', dpi=dpi)
    plt.close()

def plot_topk_forgetting_with_line(per_ckpt_ap: Dict[int, np.ndarray],
                                   tasks: Optional[List[List[int]]],
                                   outpath_prefix: str,
                                   topk: int = 20,
                                   class_names: Optional[Dict[str, str]] = None,
                                   dpi: int = 300,
                                   out_format: str = 'both'):
    """
    Plot top-K forgotten classes as grouped bars (initial vs final) and overlay a line
    showing decrease (percentage points). Saves to outpath_prefix.(pdf|png) according to out_format.
    """
    # Prepare data
    ckpts, ap_mat = sorted_ckpts_array(per_ckpt_ap)
    final_ap = ap_mat[-1]
    n_classes = ap_mat.shape[1]

    # compute initial_ap per-class using tasks mapping (same logic as other functions)
    initial_ap = np.full(n_classes, np.nan, dtype=float)
    if tasks is not None:
        for k, cls_list in enumerate(tasks, start=1):
            ck = k
            cls_arr = np.array(cls_list, dtype=int)
            if ck in ckpts:
                # ck is a checkpoint index; find its position in ckpts
                try:
                    pos = ckpts.index(ck)
                    initial_vec = ap_mat[pos, cls_arr]
                    initial_ap[cls_arr] = initial_vec
                except ValueError:
                    # fallback: mean across available ckpts
                    stacked = np.stack([ap_mat[i, cls_arr] for i in range(ap_mat.shape[0])], axis=0)
                    initial_ap[cls_arr] = np.nanmean(stacked, axis=0)
            else:
                stacked = np.stack([ap_mat[i, cls_arr] for i in range(ap_mat.shape[0])], axis=0)
                initial_ap[cls_arr] = np.nanmean(stacked, axis=0)
    else:
        # if no tasks mapping, use first checkpoint as initial
        initial_ap = ap_mat[0].copy()

    forgetting = initial_ap - final_ap  # fractions

    # rank by forgetting (largest first), exclude NaNs
    sort_idx = np.argsort(np.nan_to_num(forgetting, nan=-1e9))[::-1]
    topk_idx = [int(i) for i in sort_idx if not np.isnan(forgetting[i])]
    topk_idx = topk_idx[:topk]

    # labels and values
    labels = []
    for cid in topk_idx:
        if class_names:
            # support both string keys and int keys in mapping
            label = class_names.get(str(cid), class_names.get(cid, f"cls_{cid}"))
        else:
            label = f"cls_{cid}"
        labels.append(label)

    init_vals = initial_ap[topk_idx] * 100.0
    final_vals = final_ap[topk_idx] * 100.0
    forget_vals_pp = (initial_ap[topk_idx] - final_ap[topk_idx]) * 100.0  # percentage points

    # Plot
    plt.rcParams.update({'font.size': 10})
    fig, ax1 = plt.subplots(figsize=(max(6, topk * 0.35), 5), dpi=dpi)
    x = np.arange(len(topk_idx))
    width = 0.35

    ax1.bar(x - width/2, init_vals, width=width, label='Initial AP', color='#4C72B0', edgecolor='black')
    ax1.bar(x + width/2, final_vals, width=width, label='Final AP', color='#DD8452', edgecolor='black')
    ax1.set_ylabel('AP (%)', color='black')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    # Secondary axis for decrease (pp)
    ax2 = ax1.twinx()
    ax2.plot(x, forget_vals_pp, color='tab:red', marker='o', linestyle='-', linewidth=2.0, label='Decrease (pp)')
    ax2.set_ylabel('Decrease in AP (percentage points)', color='tab:red')
    # choose limits with small padding
    if len(forget_vals_pp) > 0:
        mn = float(np.nanmin(forget_vals_pp))
        mx = float(np.nanmax(forget_vals_pp))
        rng = max(1e-3, mx - mn)
        lower = min(0.0, mn - 0.1 * rng)
        upper = mx + 0.1 * rng
    else:
        lower, upper = 0.0, 10.0
    ax2.set_ylim(lower, max(upper, lower + 1e-3))
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # annotate bars and points
    for i in range(len(x)):
        ax1.text(x[i] - width/2, init_vals[i] + 1.0, f"{init_vals[i]:.1f}%", ha='center', va='bottom', fontsize=8)
        ax1.text(x[i] + width/2, final_vals[i] + 1.0, f"{final_vals[i]:.1f}%", ha='center', va='bottom', fontsize=8)
        ax2.text(x[i], forget_vals_pp[i] + (0.02 * (upper - lower)), f"{forget_vals_pp[i]:.1f}pp", color='tab:red', ha='center', va='bottom', fontsize=8)

    # combined legend (build from proxy artists)
    proxy_init = ax1.bar([], [], color='#4C72B0', edgecolor='black', label='Initial AP')
    proxy_final = ax1.bar([], [], color='#DD8452', edgecolor='black', label='Final AP')
    proxy_line = ax2.plot([], [], color='tab:red', marker='o', label='Decrease (pp)')[0]
    ax1.legend([proxy_init, proxy_final, proxy_line], ['Initial AP', 'Final AP', 'Decrease (pp)'],
               loc='upper right', frameon=True, fontsize=9)

    plt.title(f"Top-{len(topk_idx)} Forgotten Classes with AP Decrease (line)")
    fig.tight_layout()

    # Save according to out_format
    if out_format in ('both', 'pdf'):
        fig.savefig(outpath_prefix + ".pdf", bbox_inches='tight', dpi=dpi)
    if out_format in ('both', 'png'):
        fig.savefig(outpath_prefix + ".png", bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f"Saved top-K forgetting with line figure to {outpath_prefix}.*")

# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Visualize forgetting results")
    parser.add_argument('--results-json', required=True, help="JSON file generated by evaluate_forgetting.py")
    parser.add_argument('--out-dir', default='viz', help='output directory for figures')
    parser.add_argument('--topk', type=int, default=20, help='Top-k classes to show in bar plot')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--heatmap-cmap', default='viridis')
    parser.add_argument('--class-names-json', default='', help='Optional JSON mapping class_id -> name')
    parser.add_argument('--no-heatmap', action='store_true', help='Skip large per-class heatmap to save time')
    parser.add_argument('--fig-format', default='both', choices=['pdf', 'png', 'both'], help='save format')
    parser.add_argument('--topk-line', action='store_true', help='Draw top-K forgetting with line showing AP decrease on same figure')
    parser.add_argument('--topk-line-format', default='both', choices=['pdf', 'png', 'both'], help='format for topk-line figure')
    parser.add_argument('--learning-ylim', default='', help='Optional y-axis limits for learning curves as "ymin,ymax" e.g. "0.2,0.4". If omitted, auto-scale when appropriate.')
    parser.add_argument('--rare-classes-json', default='', help='Optional JSON file listing rare class ids (list of ints). If provided, outputs rare/nonrare separate learning/forgetting curves.')
    parser.add_argument('--rare-class-ids', default='', help='Optional comma-separated list of rare class ids, e.g. "0,5,10". Alternative to --rare-classes-json.')
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_outdir(args.out_dir)

    per_ckpt_ap, tasks, per_task, overall, run_args = load_results(args.results_json)

    class_names = None
    if args.class_names_json:
        try:
            with open(args.class_names_json, 'r') as f:
                class_names = json.load(f)
        except Exception:
            class_names = None

    # parse learning ylim if provided
    learning_ylim = None
    if args.learning_ylim:
        try:
            parts = args.learning_ylim.split(',')
            if len(parts) == 2:
                learning_ylim = (float(parts[0]), float(parts[1]))
        except Exception:
            learning_ylim = None

    # parse rare class ids (support various file formats)
    rare_ids = None
    if args.rare_classes_json:
        rare_ids = load_rare_ids_from_file(args.rare_classes_json)
        if rare_ids is None:
            print(f"[WARN] unable to parse rare classes from {args.rare_classes_json}; expected JSON list or object with key 'rare'/'rare_ids'.")
    if rare_ids is None and args.rare_class_ids:
        try:
            rare_ids = [int(x.strip()) for x in args.rare_class_ids.split(',') if x.strip() != ""]
        except Exception:
            rare_ids = None

    # Choose a larger text size for task learning curves (adjustable)
    learning_text_size = 16  # increase to enlarge fonts in learning plots

    # Task learning curves - FULL (no legend inside the plot)
    outprefix = os.path.join(args.out_dir, 'task_learning_curves_full')
    plot_task_learning_curves(per_ckpt_ap, tasks, outprefix, dpi=args.dpi,
                              figsize=(7.0, 4.2),
                              title=None,
                              y_limits=learning_ylim,
                              class_filter=None,
                              show_legend=False,
                              text_size=learning_text_size)

    # If rare ids provided, produce rare and nonrare learning plots (also no legend inside)
    if rare_ids is not None:
        any_ck = next(iter(per_ckpt_ap.keys()))
        n_classes = per_ckpt_ap[any_ck].shape[0]
        all_ids = list(range(n_classes))
        rare_set = set(rare_ids)
        nonrare_ids = [c for c in all_ids if c not in rare_set]

        outprefix_rare = os.path.join(args.out_dir, 'task_learning_curves_rare')
        plot_task_learning_curves(per_ckpt_ap, tasks, outprefix_rare, dpi=args.dpi,
                                  figsize=(7.0, 4.2),
                                  title=None,
                                  y_limits=learning_ylim,
                                  class_filter=rare_ids,
                                  show_legend=False,
                                  text_size=learning_text_size)

        outprefix_nonrare = os.path.join(args.out_dir, 'task_learning_curves_nonrare')
        plot_task_learning_curves(per_ckpt_ap, tasks, outprefix_nonrare, dpi=args.dpi,
                                  figsize=(7.0, 4.2),
                                  title=None,
                                  y_limits=learning_ylim,
                                  class_filter=nonrare_ids,
                                  show_legend=False,
                                  text_size=learning_text_size)

    # Generate standalone legend image for tasks (if tasks available)
    if tasks is not None and len(tasks) > 0:
        legend_out = os.path.join(args.out_dir, 'task_learning_legend')
        plot_tasks_legend(len(tasks), legend_out, text_size=learning_text_size)

    # Task forgetting curves - FULL
    outprefix = os.path.join(args.out_dir, 'task_forgetting_curves_full')
    plot_task_forgetting_curves(per_ckpt_ap, tasks, outprefix, dpi=args.dpi,
                                figsize=(7.0, 4.2),
                                title="Task-wise forgetting (initial - current) (full)",
                                class_filter=None,
                                text_size=12)

    # Rare/nonrare forgetting if requested
    if rare_ids is not None:
        outprefix_rare_f = os.path.join(args.out_dir, 'task_forgetting_curves_rare')
        plot_task_forgetting_curves(per_ckpt_ap, tasks, outprefix_rare_f, dpi=args.dpi,
                                    figsize=(7.0, 4.2),
                                    title="Task-wise forgetting (initial - current) (rare classes)",
                                    class_filter=rare_ids,
                                    text_size=12)

        outprefix_nonrare_f = os.path.join(args.out_dir, 'task_forgetting_curves_nonrare')
        plot_task_forgetting_curves(per_ckpt_ap, tasks, outprefix_nonrare_f, dpi=args.dpi,
                                    figsize=(7.0, 4.2),
                                    title="Task-wise forgetting (initial - current) (non-rare classes)",
                                    class_filter=nonrare_ids,
                                    text_size=12)

    # Per-class heatmap
    if not args.no_heatmap:
        outprefix = os.path.join(args.out_dir, 'per_class_ap_heatmap')
        plot_per_class_ap_heatmap(per_ckpt_ap, tasks, outprefix, dpi=args.dpi,
                                  figsize=(8.0, 10.0), cmap=args.heatmap_cmap)

    # Top-k bar (regular)
    outprefix = os.path.join(args.out_dir, f'top{args.topk}_forgetting_bar')
    plot_topk_forgetting(per_ckpt_ap, tasks, outprefix, topk=args.topk, dpi=args.dpi,
                         figsize=(10.0, 6.0), class_names=class_names)

    # Top-k with line if requested
    if args.topk_line:
        outprefix = os.path.join(args.out_dir, f'top{args.topk}_forgetting_with_line')
        plot_topk_forgetting_with_line(per_ckpt_ap, tasks, outprefix, topk=args.topk,
                                       class_names=class_names, dpi=args.dpi, out_format=args.topk_line_format)

    print(f"Saved figures to {args.out_dir}")


if __name__ == '__main__':
    main()