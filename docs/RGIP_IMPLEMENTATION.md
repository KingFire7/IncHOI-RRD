# RGIP implementation notes

This project keeps the original PViC entrypoints runnable while adding
Rarity-Guided Interaction Preservation (RGIP) behind `--use-rgip`.

## Main files

- `main_incremental.py`
  - Adds RGIP arguments.
  - Loads a frozen teacher on every rank when `--use-rgip` or `--use-distill` is active after task 1.
  - Passes old/current HOI classes and object-verb mappings into the training engine.

- `utils_incremental.py`
  - Keeps the old training iteration when `--use-rgip` is off.
  - Uses a separate RGIP iteration when `--use-rgip` is on:
    teacher replay forward -> SIS/context -> student forward -> interaction distillation -> prototype update.

- `pvic.py`
  - Adds pair-level classification weights.
  - Returns pair metadata needed by RGIP:
    `pair_labels`, `pair_objects`, `pair_prior`, `pair_global_indices`, `pair_idx_in_image`.

- `transformers.py`
  - Splits decoder cross-attention into content logits and position logits.
  - Applies RGIP negative bias only to content logits.

- `inchoi/rgip_utils.py`
  - Central RGIP state and utilities:
    SIS, replay pair context, prototype bank, confounder selection, interaction distillation.

## Validation script

Use:

```bash
bash scripts/verify_rgip_experiments.sh syntax
```

Available experiment modes:

```bash
bash scripts/verify_rgip_experiments.sh baseline
bash scripts/verify_rgip_experiments.sh weighted
bash scripts/verify_rgip_experiments.sh kd
bash scripts/verify_rgip_experiments.sh full
bash scripts/verify_rgip_experiments.sh no-hardneg
bash scripts/verify_rgip_experiments.sh freq-only
bash scripts/verify_rgip_experiments.sh same-object
```

Override smoke-test defaults through environment variables, for example:

```bash
WORLD_SIZE=4 BATCH_SIZE=16 EPOCHS=30 N_REPLAY=50 \
OUTPUT_DIR=outputs/rgip_full \
bash scripts/verify_rgip_experiments.sh full
```

## Real experiment launcher

Use `scripts/run_rgip_experiments.sh` for real training and analysis. It creates
an `outputs-rgip` run directory by default:

```text
outputs-rgip/<run_name>/
├── checkpoints/   # checkpoint_task*.pth, latest.pth, best.pth, train_args.json
├── logs/          # terminal output captured by tee
├── metrics/       # epoch_metrics.jsonl, task*_eval.json, task_eval_summary.jsonl
├── analysis/      # per-sample scores, forgetting results, visualisation outputs
└── configs/       # train_args.json
```

Main commands:

```bash
bash scripts/run_rgip_experiments.sh baseline
bash scripts/run_rgip_experiments.sh replay-distill
bash scripts/run_rgip_experiments.sh weighted
bash scripts/run_rgip_experiments.sh kd
bash scripts/run_rgip_experiments.sh full
bash scripts/run_rgip_experiments.sh paper5-full
bash scripts/run_rgip_experiments.sh paper10-full
```

Typical full run:

```bash
CUDA_DEVICES=0,1,2,3 WORLD_SIZE=4 PER_GPU_BATCH=16 EPOCHS=30 N_REPLAY=50 \
PRETRAINED=checkpoints/detr-r50-hicodet.pth \
SPLIT_SEED=140 SEED=140 \
RUN_NAME=hico4-full-seed140-train140 \
bash scripts/run_rgip_experiments.sh full
```

For mean/std experiments, sweep `SPLIT_SEED` for class/task partitions and
`SEED` for training randomness independently.

`BATCH_SIZE` is the global batch size. The training code uses
`BATCH_SIZE / WORLD_SIZE` as the per-GPU batch size.

- Fast 4-GPU setting: `CUDA_DEVICES=0,1,2,3 WORLD_SIZE=4 PER_GPU_BATCH=16`
  gives global batch 64 and uses memory roughly similar to single-GPU batch 16
  on each 3090.
  If you want linear learning-rate scaling, add `LR_HEAD=4e-4`; otherwise the
  default `1e-4` is more conservative.
- More comparable setting to old single-GPU global batch 16:
  `CUDA_DEVICES=0,1,2,3 WORLD_SIZE=4 BATCH_SIZE=16`.
  This uses per-GPU batch 4 and changes the optimisation less, but the speedup
  is smaller.

After training:

```bash
RUN_DIR=outputs-rgip/hico4-full-seed140-train140 \
CHECKPOINT_DIR=$RUN_DIR/checkpoints \
CONFIG_DIR=$RUN_DIR/configs \
ANALYSIS_DIR=$RUN_DIR/analysis \
bash scripts/run_rgip_experiments.sh forgetting
```

## Current engineering choice

The prototype bank uses local per-rank EMA updates. This keeps the first RGIP
implementation simple and stable. If exact cross-rank prototype synchronisation
is required, add an all-reduce step over prototype sums/counts before the EMA
update.
