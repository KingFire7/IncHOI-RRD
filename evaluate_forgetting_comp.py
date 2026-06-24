#!/usr/bin/env python3
# coding: utf-8
"""
evaluate_forgetting_compare_topk_simple.py

比较多个 evaluate_forgetting.py 输出（JSON）的 top-K 忘记量（initial - final）并生成多种图表。
优化：
- 将图例移动至图表内部右上角，减少外部留白，并增加半透明白色背景。
- 移除了散点图中冗余的 Pearson 相关系数。
- 修复了柱状图/折线图 Y 轴底部留白的问题：当数据均大于0时，严格从0开始。
"""
import os
import argparse
import json
from typing import List, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------
# Global default font size
# ---------------------------
GLOBAL_FONT_SIZE = 24

# ---------------------------
# Styling helper
# ---------------------------
def apply_pub_style(use_latex: bool = False, base_fontsize: Optional[int] = None, font_family: str = "Times New Roman"):
    if base_fontsize is None:
        base_fontsize = GLOBAL_FONT_SIZE
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [font_family, "DejaVu Serif"],
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 2,
        "axes.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "lines.linewidth": 2.0,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "black",
    })
    if use_latex:
        plt.rcParams.update({"text.usetex": True})

# ---------------------------
# Data loading / processing
# ---------------------------
def load_results_json(results_json: str):
    with open(results_json, 'r') as f:
        data = json.load(f)
    per_ckpt_ap = {int(k): np.array(v, dtype=float) for k, v in data['per_ckpt_ap_sampled'].items()}
    tasks = data.get('tasks', None)
    return per_ckpt_ap, tasks, data.get('args', {})

def compute_initial_final(per_ckpt_ap: Dict[int, np.ndarray], tasks: Optional[List[List[int]]]):
    ckpts_sorted = sorted(per_ckpt_ap.keys())
    final_ap = per_ckpt_ap[ckpts_sorted[-1]].copy()
    n_classes = final_ap.shape[0]
    initial_ap = np.full(n_classes, np.nan, dtype=float)

    if tasks is None:
        initial_ap[:] = per_ckpt_ap[ckpts_sorted[0]][:]
        return initial_ap, final_ap

    for k, cls_list in enumerate(tasks, start=1):
        ck = k
        cls_arr = np.array(cls_list, dtype=int)
        if ck in per_ckpt_ap:
            initial_ap[cls_arr] = per_ckpt_ap[ck][cls_arr]
        else:
            stacked = np.stack([per_ckpt_ap[c][cls_arr] for c in ckpts_sorted], axis=0)
            initial_ap[cls_arr] = np.nanmean(stacked, axis=0)
    return initial_ap, final_ap

def topk_indices(forgetting: np.ndarray, k: int):
    idx = np.argsort(np.nan_to_num(forgetting, nan=-1e9))[::-1]
    topk = [int(i) for i in idx if not np.isnan(forgetting[i])]
    return topk[:k]

def save_fig(fig, outprefix: str, fmt: str = 'both', dpi: int = 300):
    if fmt in ('both', 'pdf'):
        fig.savefig(outprefix + ".pdf", dpi=dpi, bbox_inches='tight')
    if fmt in ('both', 'png'):
        fig.savefig(outprefix + ".png", dpi=dpi, bbox_inches='tight')

def compute_global_ylim_percent(per_results: List[Dict], topk: int, pad_ratio: float = 0.08):
    all_vals = []
    for r in per_results:
        idxs = r['topk_idx']
        vals = r['forgetting'][idxs] * 100.0
        if len(vals) < topk:
            vals = np.concatenate([vals, np.full(topk - len(vals), np.nan)])
        all_vals.append(vals)
    if len(all_vals) == 0:
        return (0.0, 10.0)
    stacked = np.stack(all_vals, axis=0)
    flattened = stacked.flatten()
    flattened = flattened[~np.isnan(flattened)]
    if flattened.size == 0:
        return (0.0, 10.0)

    vmin = float(np.min(flattened))
    vmax = float(np.max(flattened))

    rng = max(1e-3, vmax - min(vmin, 0.0))
    pad = rng * pad_ratio

    # 核心修复：如果数据最小值 >= 0，严格将 y 轴底部锁定在 0，不加负向 Padding
    if vmin >= 0.0:
        ymin = 0.0
    else:
        ymin = vmin - pad

    ymax = vmax + pad
    return (ymin, ymax)

# ---------------------------
# Plotting Functions
# ---------------------------
def plot_combined_bars(per_results: List[Dict], labels: List[str], topk: int, outpath: str, fmt: str,
                       width: float = 8.0, height: float = 4.5, dpi: int = 300):
    apply_pub_style()
    K = topk
    n_models = len(per_results)

    base_palette = sns.color_palette("tab10", n_colors=max(3, n_models))
    colors = list(base_palette)
    if n_models >= 1:
        colors[0] = '#CBDFDF'
        if n_models >= 2:
            if len(colors) < 2:
                colors = colors + ['#1CB3B0']
            colors[1] = '#1CB3B0'

    fig, ax = plt.subplots(figsize=(width, height))

    x = np.arange(1, K + 1)
    total_width = 0.8
    bar_width = total_width / n_models if n_models > 1 else total_width * 0.6
    offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, n_models)

    ymin, ymax = compute_global_ylim_percent(per_results, topk)
    ax.set_ylim(ymin, ymax)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='0.85', linewidth=0.8)

    bar_containers = []
    for m_idx, r in enumerate(per_results):
        idxs = r['topk_idx']
        vals = r['forgetting'][idxs]
        if len(vals) < K:
            vals = np.concatenate([vals, np.full(K - len(vals), np.nan)])
        y = vals * 100.0

        mask = ~np.isnan(y)
        xpos = x + offsets[m_idx]
        valid_x = xpos[mask]
        valid_y = y[mask]
        if valid_x.size > 0:
            color_for_model = colors[m_idx % len(colors)]
            # align='center' 默认行为，bar的底部默认从0开始
            bc = ax.bar(valid_x, valid_y, width=bar_width * 0.95,
                        color=color_for_model, edgecolor='black',
                        label=labels[m_idx] if m_idx < 20 else None, alpha=0.95)
            bar_containers.append(bc)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_xlim(0.5, K + 0.5)
    ax.set_xlabel("Rank")
    ax.set_ylabel("AP Decrease (%)")
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=6, prune='both'))
    ax.axhline(0.0, color='black', linewidth=1.2) # 加粗一点点0线基准

    if bar_containers:
        legend_handles = [bc.patches[0] for bc in bar_containers]
        legend_labels = labels[:len(bar_containers)]
        ax.legend(legend_handles, legend_labels, loc='upper right', frameon=True,
                  facecolor='white', framealpha=0.9, edgecolor='black')

    plt.tight_layout()
    save_fig(fig, outpath, fmt, dpi)
    plt.close(fig)

def plot_ranked_line(per_results: List[Dict], labels: List[str], topk: int, outpath: str, fmt: str,
                     width: float = 8.0, height: float = 4.5, dpi: int = 300):
    apply_pub_style()
    K = topk
    x_full = np.arange(1, K + 1)
    ymin, ymax = compute_global_ylim_percent(per_results, topk)
    fig, ax = plt.subplots(figsize=(width, height))

    palette = sns.color_palette("tab10", n_colors=max(3, len(per_results)))
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']
    for i, r in enumerate(per_results):
        idxs = r['topk_idx']
        vals = r['forgetting'][idxs]
        if len(vals) < K:
            vals = np.concatenate([vals, np.full(K - len(vals), np.nan)])
        y = vals * 100.0
        mask = ~np.isnan(y)
        ax.plot(x_full[mask], y[mask], label=labels[i], color=palette[i % len(palette)],
                linestyle='-', marker=markers[i % len(markers)], linewidth=2.0, markersize=max(6, GLOBAL_FONT_SIZE/2.5))

    ax.set_axisbelow(True)
    ax.grid(axis='y', color='0.85', linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x_full)
    ax.set_xticklabels([str(i) for i in x_full])
    ax.set_xlim(0.5, K + 0.5)
    ax.set_xlabel("Rank")
    ax.set_ylabel("AP Decrease (%)")
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=6, prune='both'))
    ax.axhline(0.0, color='black', linewidth=1.0)

    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='black')
    plt.tight_layout()
    save_fig(fig, outpath, fmt, dpi)
    plt.close(fig)

def plot_ranked_line_minimal(per_results: List[Dict], labels: List[str], topk: int, outpath: str, fmt: str,
                             width: float = 8.0, height: float = 4.0, dpi: int = 300):
    apply_pub_style()
    K = topk
    x_full = np.arange(1, K + 1)
    ymin, ymax = compute_global_ylim_percent(per_results, topk)
    fig, ax = plt.subplots(figsize=(width, height))

    palette = sns.color_palette("tab10", n_colors=max(3, len(per_results)))
    for i, r in enumerate(per_results):
        idxs = r['topk_idx']
        vals = r['forgetting'][idxs]
        if len(vals) < K:
            vals = np.concatenate([vals, np.full(K - len(vals), np.nan)])
        y = vals * 100.0
        mask = ~np.isnan(y)
        ax.plot(x_full[mask], y[mask], label=labels[i], color=palette[i % len(palette)],
                linestyle='-', marker=None, linewidth=2.0)

    ax.set_axisbelow(True)
    ax.grid(axis='y', color='0.85', linewidth=0.8)
    ax.xaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.xaxis.set_visible(False)
    ax.set_xlim(0.5, K + 0.5)
    ax.set_ylabel("AP Decrease (%)")
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=6, prune='both'))
    ax.axhline(0.0, color='black', linewidth=1.0)

    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='black')
    plt.tight_layout()
    save_fig(fig, outpath, fmt, dpi)
    plt.close(fig)

def plot_scatter_or_corr(per_results: List[Dict], labels: List[str], outpath: str, fmt: str,
                         dpi: int, width: float, height: float,
                         scatter_multiplier: int = 1, scatter_size: int = 20, scatter_cmap: str = 'turbo',
                         scatter_n: int = 0, scatter_random: bool = False, scatter_seed: Optional[int] = None):
    apply_pub_style()
    n = len(per_results)

    if len(per_results) == 0:
        print("[WARN] no per_results provided")
        return
    n_classes = per_results[0]['forgetting'].shape[0]

    if scatter_random and scatter_n > 0:
        rng = np.random.default_rng(scatter_seed)
        take_n = min(scatter_n, n_classes)
        sampled = rng.choice(np.arange(n_classes), size=take_n, replace=False)
        union_list = sorted(list(map(int, sampled)))
    else:
        union_set = set()
        for r in per_results:
            sorted_full = np.argsort(np.nan_to_num(r['forgetting'], nan=-1e9))[::-1]
            if scatter_n and scatter_n > 0:
                take_n = min(len(sorted_full), scatter_n)
            else:
                topk_len = len(r['topk_idx'])
                take_n = min(len(sorted_full), max(1, topk_len * scatter_multiplier))
            union_set.update(sorted_full[:take_n].tolist())
        union_list = sorted(list(union_set))

    if len(union_list) == 0:
        print("[WARN] union set empty, skipping scatter/corr plot.")
        return

    if n == 2:
        a = per_results[0]['forgetting'][union_list] * 100.0
        b = per_results[1]['forgetting'][union_list] * 100.0
        mean_decrease = np.nanmean(np.stack([a, b], axis=0), axis=0)

        fig, ax = plt.subplots(figsize=(width, height))

        sc = ax.scatter(a, b, s=scatter_size, c=mean_decrease, cmap=scatter_cmap,
                        edgecolors='black', linewidths=0.3, alpha=0.85, zorder=3)

        valid_vals = np.concatenate([a[~np.isnan(a)], b[~np.isnan(b)]])
        if valid_vals.size > 0:
            mn = np.nanpercentile(valid_vals, 1)
            mx = np.nanpercentile(valid_vals, 99)
            rng_val = mx - mn if mx > mn else 1.0

            plot_min = mn - 0.1 * rng_val
            plot_max = mx + 0.1 * rng_val

            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)
            ax.set_aspect('equal', adjustable='box')

            ax.plot([plot_min, plot_max], [plot_min, plot_max],
                    color='gray', linestyle='--', linewidth=1.5, zorder=1)

        ax.set_xlabel(f"{labels[0]} AP Decrease (%)")
        ax.set_ylabel(f"{labels[1]} AP Decrease (%)")

        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean AP Decrease (%)")

        ax.grid(True, alpha=0.4, color='0.7')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        save_fig(fig, outpath, fmt, dpi)
        plt.close(fig)
    else:
        mat = np.stack([r['forgetting'][union_list] * 100.0 for r in per_results], axis=0)
        n_models = mat.shape[0]
        corr = np.eye(n_models)
        for i in range(n_models):
            for j in range(i+1, n_models):
                a = mat[i]
                b = mat[j]
                valid = (~np.isnan(a)) & (~np.isnan(b))
                if valid.sum() > 1:
                    r_val, _ = pearsonr(a[valid], b[valid])
                else:
                    r_val = np.nan
                corr[i,j] = corr[j,i] = r_val
        fig, ax = plt.subplots(figsize=(width, height))
        sns.heatmap(corr, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="vlag", center=0, ax=ax, vmin=-1, vmax=1,
                    annot_kws={"size": GLOBAL_FONT_SIZE})
        ax.set_title("Pearson Correlation of AP Decreases Across Models")
        save_fig(fig, outpath, fmt, dpi)
        plt.close(fig)

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare top-K forgetting and produce publication-style plots")
    parser.add_argument('--results', nargs='+', required=True, help='Paths to evaluate_forgetting JSON files (order matters)')
    parser.add_argument('--labels', nargs='+', default=None, help='Labels for each result; default autogenerated')
    parser.add_argument('--topk', type=int, default=20, help='K for top-K per model (controls ranked/bar/line)')
    parser.add_argument('--out-dir', default='viz_compare', help='Output directory')
    parser.add_argument('--fmt', default='both', choices=['both', 'pdf', 'png'], help='Output format')
    parser.add_argument('--width', type=float, default=8.0, help='Figure width (inches)')
    parser.add_argument('--height', type=float, default=4.5, help='Figure height (inches)')
    parser.add_argument('--use-latex', action='store_true', help='Use LaTeX rendering (requires TeX)')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--scatter-multiplier', type=int, default=1,
                        help='Multiply per-model top-K length when building union for scatter (>=1).')
    parser.add_argument('--scatter-size', type=int, default=20, help='Scatter marker size (matplotlib s parameter).')
    parser.add_argument('--scatter-cmap', type=str, default='turbo', help='Colormap for scatter coloring.')
    parser.add_argument('--scatter-n', type=int, default=150,
                        help='If >0, use the top-N classes per model for scatter union (overrides scatter-multiplier).')
    parser.add_argument('--scatter-random', action='store_true',
                        help='If set, select scatter classes by RANDOM sampling of --scatter-n classes from the entire class set (overrides top-N selection).')
    parser.add_argument('--scatter-seed', type=int, default=None,
                        help='Random seed for scatter random sampling (used only if --scatter-random is set).')
    parser.add_argument('--font-size', type=int, default=16,
                        help='Base font size (points) to use for all generated figures (overrides default GLOBAL_FONT_SIZE).')
    args = parser.parse_args()

    global GLOBAL_FONT_SIZE
    GLOBAL_FONT_SIZE = int(args.font_size)

    os.makedirs(args.out_dir, exist_ok=True)

    per_results = []
    if args.labels is not None and len(args.labels) != len(args.results):
        raise ValueError("Number of labels must match number of results")
    for i, f in enumerate(args.results):
        per_ckpt_ap, tasks, run_args = load_results_json(f)
        initial_ap, final_ap = compute_initial_final(per_ckpt_ap, tasks)
        forgetting = initial_ap - final_ap
        idxs = topk_indices(forgetting, args.topk)
        per_results.append({
            'file': f,
            'initial_ap': initial_ap,
            'final_ap': final_ap,
            'forgetting': forgetting,
            'topk_idx': idxs
        })

    if args.labels is None:
        labels = [f"Run {i+1}" for i in range(len(per_results))]
    else:
        labels = args.labels

    for i, r in enumerate(per_results):
        mapping = {'rank_to_classid': r['topk_idx']}
        with open(os.path.join(args.out_dir, f"rank_to_classid_model{i+1}.json"), 'w') as fw:
            json.dump(mapping, fw, indent=2)

    outprefix = os.path.join(args.out_dir, f"combined_ranked_bar_top{args.topk}")
    plot_combined_bars(per_results, labels, args.topk, outprefix, args.fmt, width=args.width, height=args.height, dpi=args.dpi)

    outprefix2 = os.path.join(args.out_dir, f"ranked_line_top{args.topk}")
    plot_ranked_line(per_results, labels, args.topk, outprefix2, args.fmt, width=args.width, height=args.height, dpi=args.dpi)

    outprefix_min = os.path.join(args.out_dir, f"ranked_line_minimal_top{args.topk}")
    plot_ranked_line_minimal(per_results, labels, args.topk, outprefix_min, args.fmt, width=args.width, height=args.height - 0.5, dpi=args.dpi)

    outprefix3 = os.path.join(args.out_dir, f"scatter_or_corr_top{args.topk}")
    plot_scatter_or_corr(per_results, labels, outprefix3, args.fmt, dpi=args.dpi, width=args.width, height=args.height,
                         scatter_multiplier=args.scatter_multiplier, scatter_size=args.scatter_size, scatter_cmap=args.scatter_cmap,
                         scatter_n=args.scatter_n, scatter_random=args.scatter_random, scatter_seed=args.scatter_seed)

    rows = []
    for i, r in enumerate(per_results):
        for rank, cid in enumerate(r['topk_idx'], start=1):
            rows.append({
                'model': labels[i],
                'rank': rank,
                'class_id': int(cid),
                'initial_ap': float(r['initial_ap'][cid]) if not np.isnan(r['initial_ap'][cid]) else float('nan'),
                'final_ap': float(r['final_ap'][cid]) if not np.isnan(r['final_ap'][cid]) else float('nan'),
                'decrease_pp': float((r['initial_ap'][cid] - r['final_ap'][cid]) * 100.0) if (not np.isnan(r['initial_ap'][cid]) and not np.isnan(r['final_ap'][cid])) else float('nan')
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, f"top{args.topk}_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to {csv_path}")
    print(f"All figures saved to {args.out_dir}")

if __name__ == "__main__":
    main()