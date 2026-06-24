import os
import torch
import random
import json
import argparse
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 引入 pocket 库用于注意力热力图绘制
import pocket
import pocket.advis

from matplotlib.colors import ListedColormap
from pocket.advis.colours import build_continuous_cmap

# Defaults are repository-relative; override them through the CLI below.
MODEL_CKPT_BASELINE = 'outputs/baseline/latest.pth'
MODEL_CKPT_MFD = 'outputs/incremental/best.pth'
OUTPUT_DIR = 'infer_vis_results'
N_SAMPLE = 5
TEST_IMG_ROOT = 'hicodet/hico_20160224_det/images/test2015'
CORRESPONDENCE_JSON = 'hoi_correspondence.json'
DATA_ROOT = 'hicodet'
DATASET = 'hicodet'
DETECTOR_TYPE = 'base'
PARTITION = 'test2015'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ONLY_RARE = False

# ===== 指定图片名列表，为空则自动随机采样
IMG_SPECIFIC = []
# "HICO_test2015_00000019.jpg"
from utils_incremental import DataFactory, get_base_dataset
from pvic import build_detector
from configs import base_detector_args


def parse_demo_args():
    parser = argparse.ArgumentParser(description='Compare two HOI checkpoints and visualise attention.')
    parser.add_argument('--baseline-checkpoint', default=MODEL_CKPT_BASELINE)
    parser.add_argument('--incremental-checkpoint', default=MODEL_CKPT_MFD)
    parser.add_argument('--data-root', default=DATA_ROOT)
    parser.add_argument('--image-root', default=TEST_IMG_ROOT)
    parser.add_argument('--correspondence', default=CORRESPONDENCE_JSON)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--num-samples', type=int, default=N_SAMPLE)
    parser.add_argument('--images', nargs='*', default=IMG_SPECIFIC,
                        help='Optional image file names; otherwise sample from image-root.')
    parser.add_argument('--only-rare', action='store_true', default=ONLY_RARE)
    return parser.parse_args()

def load_names_and_corres(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    coco_names = data['COCO']
    verb_names = data['verbs']
    hico_object_names = data['objects']
    correspondence = data['correspondence']
    return coco_names, verb_names, hico_object_names, correspondence

def get_transform():
    df = DataFactory(DATASET, PARTITION, DATA_ROOT)
    return df.transforms

def build_args_and_obj2verb():
    parser = argparse.ArgumentParser(parents=[base_detector_args(),])
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
    # 强制单机单卡模式，防止模型阻塞
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--distributed', default=False, action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    args = parser.parse_args([])
    args.detector = DETECTOR_TYPE
    args.num_verbs = 117
    args.raw_lambda = 2.8 if DETECTOR_TYPE == 'base' else 1.7
    df = DataFactory(DATASET, PARTITION, DATA_ROOT)
    base_dataset = get_base_dataset(df.dataset)
    obj_to_verb = base_dataset.object_to_verb
    return args, obj_to_verb

def load_model(args, obj_to_verb, ckpt_path):
    model = build_detector(args, obj_to_verb)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(DEVICE)
    return model

def preprocess_image(img_path, transforms):
    img = Image.open(img_path).convert('RGB')
    img_tensor, _ = transforms(img, {})
    if torch.is_tensor(img_tensor):
        np_img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        np_img = np.clip((np_img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)
        np_img = (np_img * 255).astype(np.uint8)
    else:
        np_img = np.array(img_tensor)
    return np_img, img_tensor

@torch.no_grad()
def inference(model, img_tensor):
    img_tensor = img_tensor.to(DEVICE)

    # 注册钩子拦截注意力权重
    attn_weights = []
    hook = model.decoder.layers[-1].qk_attn.register_forward_hook(
        lambda self, input, output: attn_weights.append(output[1])
    )

    results = model([img_tensor])

    # 移除钩子防止内存泄漏
    hook.remove()

    res = results[0] if isinstance(results, list) else results
    # 将截获的注意力附加在输出字典中
    if attn_weights:
        res['attn_weights'] = attn_weights[0]

    return res

def build_coco_to_hico_object_map(coco_names, hico_object_names):
    coco2hico = {}
    for coco_id, coco_name in enumerate(coco_names):
        if coco_name in hico_object_names:
            hico_obj_id = hico_object_names.index(coco_name)
            coco2hico[coco_id] = hico_obj_id
    return coco2hico

def build_hoi2category(correspondence, rare_set, nonrare_set):
    hoi_cat_map = {}
    for idx, (hoi_id, obj_id, verb_id) in enumerate(correspondence):
        if hoi_id in rare_set:
            hoi_cat_map[(obj_id, verb_id)] = 'rare'
        elif hoi_id in nonrare_set:
            hoi_cat_map[(obj_id, verb_id)] = 'nonrare'
        else:
            hoi_cat_map[(obj_id, verb_id)] = 'unknown'
    return hoi_cat_map

def visualize(image_np, results, coco_names, verb_names, img_name, model_tag, outdir,
              hoi_cat_map=None, coco2hico=None):
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(image_np)
    boxes = results['boxes'].cpu().numpy()
    img_h, img_w = int(results['size'][0]), int(results['size'][1])
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)
    pairing = results['pairing'].cpu().numpy()
    scores = results['scores'].cpu().numpy()
    labels = results['labels'].cpu().numpy()
    objects = results['objects'].cpu().numpy()

    best_for_pair = {}
    for i, pair in enumerate(pairing):
        if scores[i] < 0.15:
            continue
        key = tuple(pair.tolist())  # (human_idx, obj_idx)
        if key not in best_for_pair or scores[i] > best_for_pair[key][0]:
            best_for_pair[key] = (scores[i], labels[i], objects[i], i)

    has_rare = False
    for (human_idx, obj_idx), (score, verb_id, coco_obj_id, i) in best_for_pair.items():
        h_box = boxes[human_idx]
        o_box = boxes[obj_idx]
        h_rect = plt.Rectangle((h_box[0], h_box[1]), h_box[2] - h_box[0], h_box[3] - h_box[1],
                               fill=False, edgecolor='blue', linewidth=2)
        o_rect = plt.Rectangle((o_box[0], o_box[1]), o_box[2] - o_box[0], o_box[3] - o_box[1],
                               fill=False, edgecolor='orange', linewidth=2)
        ax.add_patch(h_rect)
        ax.add_patch(o_rect)
        objn = coco_names[coco_obj_id] if coco_obj_id < len(coco_names) else str(coco_obj_id)
        verbn = verb_names[verb_id] if verb_id < len(verb_names) else str(verb_id)
        hoi_cat = ""
        hoi_cat_label = "unknown"
        if coco2hico is not None:
            hico_obj_id = coco2hico.get(coco_obj_id, None)
            if hico_obj_id is None:
                hoi_cat_label = "unknown-object"
            else:
                if hoi_cat_map is not None:
                    hoi_cat_label = hoi_cat_map.get((hico_obj_id, verb_id), "unknown")
            hoi_cat = f" ({hoi_cat_label})"
        if hoi_cat_label == "rare":
            has_rare = True
        text_str = f'H{human_idx}: {objn}/{verbn}{hoi_cat}/{score:.2f}'
        ax.text(o_box[0], o_box[1]-15-30*human_idx, text_str, color='orange', fontsize=16, weight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='orange'))
        ax.text(h_box[0], h_box[1]-10, f'Human {human_idx}', color='blue', fontsize=16,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue'))
    ax.axis('off')
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    save_name = f"{os.path.splitext(img_name)[0]}-{model_tag}.png"
    plt.savefig(os.path.join(outdir, save_name), dpi=120)
    plt.close(fig)

    # 额外返回 best_for_pair 字典，供后续注意力提取使用
    return has_rare, best_for_pair

def save_attention_maps(image_np, results, img_name, model_tag, outdir, best_for_pair, coco_names, verb_names):
    """提取并保存纯净版注意力可视化图像"""
    if 'attn_weights' not in results or 'x' not in results:
        return

    img_h, img_w = int(results['size'][0]), int(results['size'][1])
    attn = results['attn_weights']
    x_indices = results['x']

    # 构建层级文件夹结构: OUTPUT_DIR / 图片名 / 模型名 /
    base_img_name = os.path.splitext(img_name)[0]
    attn_dir = os.path.join(outdir, base_img_name, model_tag)
    os.makedirs(attn_dir, exist_ok=True)

    for (human_idx, obj_idx), (score, verb_id, coco_obj_id, idx) in best_for_pair.items():
        ho_pair_idx = x_indices[idx]

        # 将一维序列 reshape 回二维空间网格 (假定有 8 个头，空间下采样率为 32)
        try:
            # 加入 .cpu() 将张量从 GPU 移至 CPU
            attn_map = attn[0, :, ho_pair_idx].cpu().reshape(8, math.ceil(img_h / 32), math.ceil(img_w / 32))
        except Exception as e:
            print(f"注意力特征图 reshape 失败，请检查模型架构: {e}")
            continue

        objn = coco_names[coco_obj_id] if coco_obj_id < len(coco_names) else str(coco_obj_id)
        verbn = verb_names[verb_id] if verb_id < len(verb_names) else str(verb_id)

        custom_red = build_continuous_cmap(
            rgb_x=[1.0],                  # 终点位置
            rgb_v=[(1.0, 0.8, 0.0)],      # 终点颜色为纯红色 (R, G, B)
            alpha_v=[0.0, 0.95]            # 透明度从 0.0(完全透明) 渐变到 0.8(不透明)
        )
        # 遍历输出 8 个注意力头
        for head_idx in range(8):
            # 获取绝对无框、无字的原图副本，并转换为 PIL Image 格式
            clean_image_np = image_np.copy()
            clean_image_pil = Image.fromarray(clean_image_np)

            # 文件名：包含对齐信息的注意力图
            save_path = os.path.join(attn_dir, f"pair{idx}_H{human_idx}_{objn}_{verbn}_head{head_idx+1}.png")

            # 使用 pocket 中的接口，渲染热力图并保存
            pocket.advis.heatmap(clean_image_pil, attn_map[head_idx: head_idx+1], save_path=save_path, c_maps=custom_red)
            plt.close('all')

if __name__ == "__main__":
    demo_args = parse_demo_args()
    MODEL_CKPT_BASELINE = demo_args.baseline_checkpoint
    MODEL_CKPT_MFD = demo_args.incremental_checkpoint
    DATA_ROOT = demo_args.data_root
    TEST_IMG_ROOT = demo_args.image_root
    CORRESPONDENCE_JSON = demo_args.correspondence
    OUTPUT_DIR = demo_args.output_dir
    N_SAMPLE = demo_args.num_samples
    IMG_SPECIFIC = demo_args.images
    ONLY_RARE = demo_args.only_rare

    coco_names, verb_names, hico_object_names, correspondence = load_names_and_corres(CORRESPONDENCE_JSON)
    args, obj_to_verb = build_args_and_obj2verb()
    transforms = get_transform()
    model_baseline = load_model(args, obj_to_verb, MODEL_CKPT_BASELINE)
    model_mfd = load_model(args, obj_to_verb, MODEL_CKPT_MFD)
    models = [("baseline", model_baseline), ("MFD", model_mfd)]

    df = DataFactory(DATASET, PARTITION, DATA_ROOT)
    base_dataset = get_base_dataset(df.dataset)
    rare_set = set(base_dataset.rare)
    nonrare_set = set(base_dataset.non_rare)
    hoi_cat_map = build_hoi2category(correspondence, rare_set, nonrare_set)
    coco2hico = build_coco_to_hico_object_map(coco_names, hico_object_names)

    all_imgs = sorted([f for f in os.listdir(TEST_IMG_ROOT) if f.endswith(".jpg")])
    if IMG_SPECIFIC:
        candidate_imgs = [img for img in IMG_SPECIFIC if img in all_imgs]
    else:
        candidate_imgs = all_imgs.copy()

    # Step 1: 预采样
    if ONLY_RARE:
        rare_img_names = []
        print("预采样并筛查含rare组合的图片...")
        random.shuffle(candidate_imgs)
        for img_name in tqdm(candidate_imgs, desc="Rare-filtering"):
            img_path = os.path.join(TEST_IMG_ROOT, img_name)
            image_np, img_tensor = preprocess_image(img_path, transforms)
            results = inference(model_baseline, img_tensor)
            boxes = results['boxes'].cpu().numpy()
            pairing = results['pairing'].cpu().numpy()
            scores = results['scores'].cpu().numpy()
            labels = results['labels'].cpu().numpy()
            objects = results['objects'].cpu().numpy()
            best_for_pair = {}
            for i, pair in enumerate(pairing):
                if scores[i] < 0.15:
                    continue
                key = tuple(pair.tolist())
                if key not in best_for_pair or scores[i] > best_for_pair[key][0]:
                    best_for_pair[key] = (scores[i], labels[i], objects[i], i)
            has_rare = False
            for (human_idx, obj_idx), (score, verb_id, coco_obj_id, i) in best_for_pair.items():
                hico_obj_id = coco2hico.get(coco_obj_id, None)
                if hico_obj_id is None:
                    continue
                hoi_cat_label = hoi_cat_map.get((hico_obj_id, verb_id), "unknown")
                if hoi_cat_label == "rare":
                    has_rare = True
                    break
            if has_rare:
                rare_img_names.append(img_name)
            if len(rare_img_names) >= N_SAMPLE:
                break
        if len(rare_img_names) < N_SAMPLE:
            print(f"警告：只采集到 {len(rare_img_names)} 张含rare类的图片。")
        img_names = rare_img_names[:N_SAMPLE]
    else:
        if IMG_SPECIFIC:
            img_names = [img for img in IMG_SPECIFIC if img in all_imgs][:N_SAMPLE]
        else:
            img_names = random.sample(all_imgs, min(N_SAMPLE, len(all_imgs)))

    # Step 2: 正式输出N_SAMPLE张可视化图片（包含原图可视化与注意力机制可视化）
    for img_name in tqdm(img_names, desc="Processing selected images"):
        img_path = os.path.join(TEST_IMG_ROOT, img_name)
        image_np, img_tensor = preprocess_image(img_path, transforms)

        for model_tag, model in models:
            # 1. 运行推理获取基础输出及注意力特征
            results = inference(model, img_tensor)

            # 2. 绘制基础带有框线和标签的可视化（接收过滤后的有效预测项）
            has_rare, best_for_pair = visualize(
                image_np, results,
                coco_names, verb_names, img_name,
                model_tag, OUTPUT_DIR,
                hoi_cat_map=hoi_cat_map,
                coco2hico=coco2hico
            )

            # 3. 生成无框纯净版的注意力图像组合并独立保存
            save_attention_maps(
                image_np, results, img_name,
                model_tag, OUTPUT_DIR,
                best_for_pair, coco_names, verb_names
            )
