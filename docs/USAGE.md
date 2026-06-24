# 使用说明

## 1. 环境与数据

建议沿用原 PViC 环境（Python 3.8、PyTorch 1.8、torchvision 0.9），其他依赖与训练参数见 [原始 PViC 说明](../docs.md)。

```bash
git submodule update --init --recursive
pip install -e pocket
cd h_detr/models/ops
python setup.py build install
cd ../../..
```

将 HICO-DET 图像放在 `hicodet/hico_20160224_det/`，将预训练检测器放在 `checkpoints/`。这两个目录不会被 Git 跟踪。

## 2. 增量训练

基础 DETR 示例：

```bash
DETR=base python main_incremental.py \
  --data-root ./hicodet \
  --hoi-path ./hoi_correspondence.json \
  --rare-path ./rare.json \
  --pretrained checkpoints/detr-r50-hicodet.pth \
  --output-dir outputs/inc-random \
  --world-size 8 --batch-size 16 \
  --split-mode random \
  --use-replay --n-replay 20 \
  --use-distill --distill-loss-weight 1.0
```

高级检测器使用 `DETR=advanced`，并按需增加 backbone、query 数和 checkpoint 参数。`--start-task N` 可从已有任务继续；程序会在输出目录查找 `checkpoint_taskN.pth`。

任务划分可选 `random`、`paper_5phase`、`paper_10phase`。`rare_first` 与 `non_rare_first` 参数目前保留为实验接口，但当前实现尚未给出划分逻辑，不建议直接使用。

## 3. 遗忘评估

```bash
DETR=base python evaluate_forgetting.py \
  --data-root ./hicodet \
  --output-dir outputs/inc-random \
  --save-dir eval_forgetting_results/inc-random \
  --train-args-file outputs/inc-random/train_args.json
```

绘图：

```bash
python evaluate_forgetting_viz.py \
  --results-json eval_forgetting_results/inc-random/results_forgetting_<timestamp>.json \
  --out-dir viz/inc-random \
  --rare-classes-json rare.json
```

比较多组结果时运行 `python evaluate_forgetting_comp.py --help` 查看输入格式。

## 4. 推理与注意力可视化

```bash
python inference_demo.py \
  --baseline-checkpoint outputs/baseline/latest.pth \
  --incremental-checkpoint outputs/incremental/best.pth \
  --data-root ./hicodet \
  --output-dir infer_vis_results/demo \
  --num-samples 5
```

## 5. 常见问题

- 训练入口依赖 NCCL 和 CUDA，`--world-size` 应等于参与训练的 GPU 数。
- `DETR` 环境变量必须是 `base` 或 `advanced`。
- checkpoint 只保存 `model_state_dict`，恢复训练时优化器状态不会自动恢复。
- 大文件不要强行提交到 Git；权重建议通过 Release、对象存储或模型托管平台发布。
