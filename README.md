# Towards Class-Incremental Human-Object Interaction Detection via Rarity-Aware Relational Distillation

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

本文是 Rarity-Aware Relational Distillation（RRD）的 PyTorch 实现，基于 [PViC](https://github.com/fredzzhang/pvic) 解决 HOI 类增量学习问题。项目在 HICO-DET 上按任务逐步引入 HOI 类，并组合动态回放、稀有类自适应蒸馏、注意力提示与脆弱性感知抑制，以减轻灾难性遗忘。

论文 *Towards Class-Incremental Human-Object Interaction Detection via Rarity-Aware Relational Distillation* 当前正在审稿。

> 当前仓库是研究代码整理版。为避免改变已有实验结果，核心训练逻辑保持原状；模型权重、数据集、日志和可视化结果不纳入 Git。

## 代码结构

```text
.
├── main_incremental.py       # 类增量训练/逐任务评估入口
├── utils_incremental.py      # 增量数据包装与训练引擎
├── mir_utils.py              # 动态回放辅助函数
├── pvic.py                   # PViC 与增量模块主体
├── transformers.py           # HOI Transformer 模块
├── configs.py                # DETR/H-DETR 公共参数
├── evaluate_forgetting.py    # 逐 checkpoint 遗忘评估
├── evaluate_forgetting_viz.py# 单组结果可视化
├── evaluate_forgetting_comp.py # 多组结果比较
├── inference_demo.py         # 双模型推理与注意力可视化
├── hoi_correspondence.json   # HICO 类、物体、动作对应关系
├── rare.json                 # rare HOI 类列表
├── detr/ h_detr/ pocket/     # 上游 Git 子模块
├── hicodet/ vcoco/           # 数据接口 Git 子模块
└── docs/
    ├── STRUCTURE.md          # 模块关系与数据流
    └── USAGE.md              # 安装、训练、评估命令
```

原始 PViC 的训练与测试说明保留在 [docs.md](docs.md)。本项目新增部分见 [结构说明](docs/STRUCTURE.md) 和 [使用说明](docs/USAGE.md)。

## 快速开始

```bash
git clone --recurse-submodules git@github.com:KingFire7/IncHOI-RRD.git
cd IncHOI-RRD
pip install -e pocket

# 单机 8 卡、4 个随机增量任务
DETR=base python main_incremental.py \
  --data-root ./hicodet \
  --pretrained checkpoints/detr-r50-hicodet.pth \
  --output-dir outputs/inc-random \
  --world-size 8 \
  --use-replay --n-replay 20 \
  --use-distill
```

`DETR=base` 使用 DETR-R50；`DETR=advanced` 使用 H-DETR/Deformable DETR。详细参数与评估流程见 [docs/USAGE.md](docs/USAGE.md)。

## 方法概览

RRD 使用统一的教师—学生框架保存历史交互知识。动态回放依据样本重要性优先保留易受干扰的历史实例；关系拓扑对齐约束注意力所表达的全局空间结构；稀有类自适应的成对蒸馏则加强长尾类别的语义稳定性。仓库同时支持标准多任务增量划分与新概念发现划分。

## 输出约定

每个任务保存 `checkpoint_taskN.pth`，训练参数保存到输出目录。`outputs/`、`checkpoints/`、日志、图像和数据集均已加入 `.gitignore`，请使用独立存储发布权重和实验结果。

## 致谢与许可

本项目建立在 Frederic Z. Zhang 等人的 PViC 代码之上，并沿用其 DETR、H-DETR、Pocket、HICO-DET 与 V-COCO 依赖。原论文与引用信息见 [PViC 文档](docs.md)。代码沿用 [BSD 3-Clause License](LICENSE)。

## 联系与引用

问题与建议可提交 Issue，或联系 `hujiaming1214@stu.xjtu.edu.cn`。

```bibtex
@article{hu2026inchoi,
  title   = {Towards Class-Incremental Human-Object Interaction Detection via Rarity-Aware Relational Distillation},
  author  = {Hu, Jiaming and others},
  journal = {Computer Vision and Image Understanding},
  note    = {Under Review},
  year    = {2026}
}
```
