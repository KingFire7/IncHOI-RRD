import torch
import json
import os
from tqdm import tqdm
import random

def load_mir(output_dir, task_idx):
    mir_path = os.path.join(output_dir, f"mir_task{task_idx}.json")
    if os.path.exists(mir_path):
        with open(mir_path, 'r') as f:
            return json.load(f)
    else:
        return {}

# 获取rare和non-rare类别集合
def get_rare_nonrare_sets(trainset):
    base = trainset.dataset.dataset
    return set([int(x) for x in base.rare]), set([int(x) for x in base.non_rare])

def dynamic_replay_sort_by_confidence(conf_list, reverse=True):
    sorted_items = sorted(conf_list, key=lambda x: x[1], reverse=reverse)
    return [idx for idx, conf in sorted_items]

def repeat_list(lst, repeat):
    return (lst * repeat)[:len(lst) * repeat]

def interleave_replay_and_new_auto(new_indices, replay_indices_sorted, repeat=1):
    """
    自动均匀交错回放与新样本，无需手动指定间隔。
    """
    replay_indices_expanded = repeat_list(replay_indices_sorted, repeat)
    total = len(new_indices) + len(replay_indices_expanded)
    n_new, n_replay = len(new_indices), len(replay_indices_expanded)
    result = []
    i, j = 0, 0
    pos_new, pos_replay = 0, 0
    step_new = total / n_new if n_new > 0 else float('inf')
    step_replay = total / n_replay if n_replay > 0 else float('inf')

    ptr_new, ptr_replay = 0, 0
    cur_new, cur_replay = 0, 0

    for k in range(total):
        # 如果新或旧用完，直接补充剩下的
        if ptr_new >= n_new:
            result.extend(replay_indices_expanded[ptr_replay:])
            break
        if ptr_replay >= n_replay:
            result.extend(new_indices[ptr_new:])
            break
        # 轮盘式：谁“理应”在当前位置，谁就插入
        if (cur_new / step_new) <= (cur_replay / step_replay):
            result.append(new_indices[ptr_new])
            ptr_new += 1
            cur_new += 1
        else:
            result.append(replay_indices_expanded[ptr_replay])
            ptr_replay += 1
            cur_replay += 1
    return result