#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/verify_rgip_experiments.sh syntax
#   bash scripts/verify_rgip_experiments.sh baseline
#   bash scripts/verify_rgip_experiments.sh weighted
#   bash scripts/verify_rgip_experiments.sh kd
#   bash scripts/verify_rgip_experiments.sh full
#
# The default mode is "syntax", which is intentionally light-weight.  Training
# modes use small defaults for smoke testing; override them with environment
# variables when running full experiments.

MODE="${1:-syntax}"

CONDA_ENV="${CONDA_ENV:-pvic}"
DETR_TYPE="${DETR_TYPE:-base}"
WORLD_SIZE="${WORLD_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-1}"
N_REPLAY="${N_REPLAY:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rgip_verify}"
DATA_ROOT="${DATA_ROOT:-hicodet}"
PORT="${PORT:-12355}"
SPLIT_MODE="${SPLIT_MODE:-random}"
START_TASK="${START_TASK:-0}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export DETR="${DETR_TYPE}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

COMMON_ARGS=(
  --world-size "${WORLD_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --n-replay "${N_REPLAY}"
  --output-dir "${OUTPUT_DIR}"
  --data-root "${DATA_ROOT}"
  --port "${PORT}"
  --split-mode "${SPLIT_MODE}"
  --start-task "${START_TASK}"
  --num-workers 2
)

run_train() {
  python main_incremental.py "${COMMON_ARGS[@]}" "$@"
}

case "${MODE}" in
  syntax)
    python -m py_compile \
      pvic.py \
      transformers.py \
      utils_incremental.py \
      main_incremental.py \
      inchoi/rgip_utils.py \
      inference_demo.py
    python - <<'PY'
from inchoi.rgip_utils import RarityGuidedState, build_hico_object_verb_mappings
print("RGIP imports OK:", RarityGuidedState.__name__, build_hico_object_verb_mappings.__name__)
PY
    ;;

  baseline)
    # Replay baseline: old replay path, no RGIP.
    run_train --use-replay
    ;;

  weighted)
    # Weighted replay only: SIS weights classification loss; no attention or interaction KD.
    run_train \
      --use-replay \
      --use-rgip \
      --rgip-int-loss-weight 0 \
      --rgip-lambda-attn 0 \
      --rgip-debug
    ;;

  kd)
    # Rarity-guided replay + feature/logit interaction distillation; no attention modulation.
    run_train \
      --use-replay \
      --use-rgip \
      --rgip-lambda-attn 0 \
      --rgip-int-loss-weight 1 \
      --rgip-hardneg-weight 0.5 \
      --rgip-debug
    ;;

  no-hardneg)
    # Full SIS and attention, without hard negative margin.
    run_train \
      --use-replay \
      --use-rgip \
      --rgip-hardneg-weight 0 \
      --rgip-debug
    ;;

  full)
    # Full RGIP: SIS-weighted replay, interaction KD, prototype confounders and attention modulation.
    run_train \
      --use-replay \
      --use-rgip \
      --rgip-alpha 0.4 \
      --rgip-beta 0.3 \
      --rgip-gamma 2.0 \
      --rgip-topk 3 \
      --rgip-eta-object 0.4 \
      --rgip-eta-query 0.4 \
      --rgip-eta-sem 0.0 \
      --rgip-lambda-attn 0.5 \
      --rgip-int-loss-weight 1.0 \
      --rgip-feat-weight 1.0 \
      --rgip-logit-weight 1.0 \
      --rgip-hardneg-weight 0.5 \
      --rgip-debug
    ;;

  freq-only)
    # Ablation: SIS only uses rarity frequency.
    run_train \
      --use-replay \
      --use-rgip \
      --rgip-alpha 1.0 \
      --rgip-beta 0.0 \
      --rgip-lambda-attn 0 \
      --rgip-debug
    ;;

  same-object)
    # Ablation: confounder selection relies only on shared object.
    run_train \
      --use-replay \
      --use-rgip \
      --rgip-eta-object 1.0 \
      --rgip-eta-query 0.0 \
      --rgip-eta-sem 0.0 \
      --rgip-debug
    ;;

  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Available modes: syntax, baseline, weighted, kd, no-hardneg, full, freq-only, same-object" >&2
    exit 2
    ;;
esac

