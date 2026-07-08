#!/usr/bin/env bash
set -euo pipefail

# Real RGIP experiment launcher.
#
# Main training examples:
#   bash scripts/run_rgip_experiments.sh baseline
#   bash scripts/run_rgip_experiments.sh weighted
#   bash scripts/run_rgip_experiments.sh kd
#   bash scripts/run_rgip_experiments.sh full
#   bash scripts/run_rgip_experiments.sh full-rd-tuned
#   bash scripts/run_rgip_experiments.sh fast4
#   bash scripts/run_rgip_experiments.sh paper5-full
#   bash scripts/run_rgip_experiments.sh paper10-full
#
# bash scripts/run_rgip_experiments.sh baseline        # 普通 replay baseline
# bash scripts/run_rgip_experiments.sh replay-distill  # 旧回放蒸馏 baseline
# bash scripts/run_rgip_experiments.sh weighted        # 只用 SIS 加权
# bash scripts/run_rgip_experiments.sh kd              # SIS + interaction KD，无 attention modulation
# bash scripts/run_rgip_experiments.sh full            # 完整 RGIP
# bash scripts/run_rgip_experiments.sh full-rd-tuned   # replay-distill + 轻量 RGIP，快速强基线版本
# bash scripts/run_rgip_experiments.sh no-hardneg      # 去掉 hard negative
# bash scripts/run_rgip_experiments.sh freq-only       # frequency-only SIS
# bash scripts/run_rgip_experiments.sh same-object     # same-object confounder only
# bash scripts/run_rgip_experiments.sh paper5-full     # IRD 5-phase New Concept
# bash scripts/run_rgip_experiments.sh paper10-full    # IRD 10-phase New Concept
#
# Analysis examples after training:
#   RUN_DIR=outputs-rgip/full-seed140-train140 bash scripts/run_rgip_experiments.sh forgetting
#   RESULTS_JSON=outputs-rgip/.../analysis/forgetting/results_xxx.json bash scripts/run_rgip_experiments.sh viz-forgetting
#   RUN_DIR=outputs-rgip/... BASELINE_CKPT=... INCREMENTAL_CKPT=... bash scripts/run_rgip_experiments.sh attention-vis

MODE="${1:-full}"

CONDA_ENV="${CONDA_ENV:-pvic}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2}"
DETR_TYPE="${DETR_TYPE:-base}"
DATASET="${DATASET:-hicodet}"
DATA_ROOT="${DATA_ROOT:-hicodet}"
PRETRAINED="${PRETRAINED:-checkpoints/detr-r50-hicodet.pth}"
HOI_PATH="${HOI_PATH:-hoi_correspondence.json}"
RARE_PATH="${RARE_PATH:-rare.json}"

GPU_COUNT="$(awk -F',' '{print NF}' <<< "${CUDA_DEVICES}")"
WORLD_SIZE="${WORLD_SIZE:-${GPU_COUNT}}"
PER_GPU_BATCH="${PER_GPU_BATCH:-16}"
BATCH_SIZE="${BATCH_SIZE:-$((PER_GPU_BATCH * WORLD_SIZE))}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EPOCHS="${EPOCHS:-30}"
N_REPLAY="${N_REPLAY:-50}"
REPLAY_REPEAT="${REPLAY_REPEAT:-1}"
PORT="${PORT:-12355}"
SEED="${SEED:-140}"
SPLIT_SEED="${SPLIT_SEED:-${SEED}}"
START_TASK="${START_TASK:-0}"
EVAL_MODE="${EVAL_MODE:-seen_valid}"
SPLIT_MODE="${SPLIT_MODE:-random}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs-rgip}"

case "${MODE}" in
  paper5-full)
    SPLIT_MODE="paper_5phase"
    ;;
  paper10-full)
    SPLIT_MODE="paper_10phase"
    ;;
esac

RUN_NAME="${RUN_NAME:-${MODE}-split${SPLIT_MODE}-splitseed${SPLIT_SEED}-trainseed${SEED}-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
METRICS_DIR="${METRICS_DIR:-${RUN_DIR}/metrics}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${RUN_DIR}/analysis}"
CONFIG_DIR="${CONFIG_DIR:-${RUN_DIR}/configs}"

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}" "${LOG_DIR}" "${METRICS_DIR}" "${ANALYSIS_DIR}" "${CONFIG_DIR}"

if (( BATCH_SIZE % WORLD_SIZE != 0 )); then
  echo "BATCH_SIZE=${BATCH_SIZE} must be divisible by WORLD_SIZE=${WORLD_SIZE} because main_incremental.py uses batch_size/world_size per GPU." >&2
  exit 2
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export DETR="${DETR_TYPE}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

COMMON_ARGS=(
  --dataset "${DATASET}"
  --data-root "${DATA_ROOT}"
  --pretrained "${PRETRAINED}"
  --output-dir "${RUN_DIR}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --log-dir "${LOG_DIR}"
  --metrics-dir "${METRICS_DIR}"
  --analysis-dir "${ANALYSIS_DIR}"
  --config-dir "${CONFIG_DIR}"
  --hoi-path "${HOI_PATH}"
  --rare-path "${RARE_PATH}"
  --world-size "${WORLD_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --epochs "${EPOCHS}"
  --n-replay "${N_REPLAY}"
  --replay-repeat "${REPLAY_REPEAT}"
  --replay-cache-dir "${REPLAY_CACHE_DIR:-${OUTPUT_ROOT}/cache/replay_indices}"
  --port "${PORT}"
  --seed "${SEED}"
  --split-seed "${SPLIT_SEED}"
  --start-task "${START_TASK}"
  --split-mode "${SPLIT_MODE}"
  --eval-mode "${EVAL_MODE}"
)

if [[ -n "${LR_HEAD:-}" ]]; then
  COMMON_ARGS+=(--lr-head "${LR_HEAD}")
fi
if [[ -n "${LR_DROP:-}" ]]; then
  COMMON_ARGS+=(--lr-drop "${LR_DROP}")
fi
if [[ -n "${WEIGHT_DECAY:-}" ]]; then
  COMMON_ARGS+=(--weight-decay "${WEIGHT_DECAY}")
fi
if [[ -n "${CLIP_MAX_NORM:-}" ]]; then
  COMMON_ARGS+=(--clip-max-norm "${CLIP_MAX_NORM}")
fi
if [[ -n "${PRINT_INTERVAL:-}" ]]; then
  COMMON_ARGS+=(--print-interval "${PRINT_INTERVAL}")
fi
if [[ -n "${RESUME:-}" ]]; then
  COMMON_ARGS+=(--resume "${RESUME}")
fi
if [[ "${RGIP_TIMING:-0}" == "1" ]]; then
  COMMON_ARGS+=(--rgip-timing)
fi
if [[ "${SKIP_INITIAL_EVAL:-0}" == "1" ]]; then
  COMMON_ARGS+=(--skip-initial-eval)
fi

if [[ "${DETR_TYPE}" == "advanced" ]]; then
  COMMON_ARGS+=(
    --backbone "${BACKBONE:-swin_large}"
    --use-checkpoint
    --drop-path-rate "${DROP_PATH_RATE:-0.5}"
    --num-queries-one2one "${NUM_QUERIES_ONE2ONE:-900}"
    --num-queries-one2many "${NUM_QUERIES_ONE2MANY:-1500}"
  )
fi

run_and_log() {
  local log_name="$1"
  shift
  local log_file="${LOG_DIR}/${log_name}.log"
  {
    echo
    echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
    echo "Run directory: ${RUN_DIR}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICES}"
    echo "WORLD_SIZE: ${WORLD_SIZE}"
    echo "Global batch size: ${BATCH_SIZE}"
    echo "Per-GPU batch size: $((BATCH_SIZE / WORLD_SIZE))"
    echo "Checkpoint directory: ${CHECKPOINT_DIR}"
    echo "Metrics directory: ${METRICS_DIR}"
    echo "Analysis directory: ${ANALYSIS_DIR}"
    echo "Log file: ${log_file}"
    echo "Command:"
    printf ' %q' "$@"
    echo
  } | tee "${RUN_DIR}/command.txt" | tee -a "${RUN_DIR}/command_history.txt"
  "$@" 2>&1 | tee -a "${log_file}"
}

run_train() {
  run_and_log "train_${MODE}" python main_incremental.py "${COMMON_ARGS[@]}" "$@"
}

case "${MODE}" in
  baseline)
    # Replay baseline: normal replay, no RGIP.
    run_train --use-replay
    ;;

  replay-distill)
    # Old replay distillation baseline.
    run_train \
      --use-replay \
      --use-distill \
      --distill-loss-weight "${DISTILL_LOSS_WEIGHT:-1.0}" \
      --replay-distill
    ;;

  weighted)
    # RGIP weighted replay only: SIS weights classification loss.
    run_train \
      --use-replay \
      --use-rgip \
      --rgip-int-loss-weight 0 \
      --rgip-lambda-attn 0 \
      --rgip-debug
    ;;

  kd)
    # RGIP weighted replay + interaction distillation, no attention modulation.
    run_train \
      --use-replay \
      --use-rgip \
      --rgip-lambda-attn 0 \
      --rgip-int-loss-weight "${RGIP_INT_LOSS_WEIGHT:-1.0}" \
      --rgip-hardneg-weight "${RGIP_HARDNEG_WEIGHT:-0.5}" \
      --rgip-debug
    ;;

  full|fast4|full4)
    # Full RGIP for the default four-task HICO split unless SPLIT_MODE is overridden.
    run_train \
      --use-replay \
      --use-rgip \
      --rgip-alpha "${RGIP_ALPHA:-0.4}" \
      --rgip-beta "${RGIP_BETA:-0.3}" \
      --rgip-gamma "${RGIP_GAMMA:-2.0}" \
      --rgip-topk "${RGIP_TOPK:-3}" \
      --rgip-eta-object "${RGIP_ETA_OBJECT:-0.4}" \
      --rgip-eta-query "${RGIP_ETA_QUERY:-0.4}" \
      --rgip-eta-sem "${RGIP_ETA_SEM:-0.0}" \
      --rgip-lambda-attn "${RGIP_LAMBDA_ATTN:-0.5}" \
      --rgip-max-attn-pairs "${RGIP_MAX_ATTN_PAIRS:-16}" \
      --rgip-int-loss-weight "${RGIP_INT_LOSS_WEIGHT:-1.0}" \
      --rgip-feat-weight "${RGIP_FEAT_WEIGHT:-1.0}" \
      --rgip-logit-weight "${RGIP_LOGIT_WEIGHT:-1.0}" \
      --rgip-hardneg-weight "${RGIP_HARDNEG_WEIGHT:-0.5}" \
      --rgip-debug
    ;;

  full-rd-tuned|rgip-rd-tuned)
    # Strong replay-distill baseline plus lightweight RGIP regularisation.
    # This is intended as a fast stabilised variant when vanilla RGIP full is
    # weaker than replay-distill.
    run_train \
      --use-replay \
      --use-distill \
      --distill-loss-weight "${DISTILL_LOSS_WEIGHT:-1.0}" \
      --replay-distill \
      --use-rgip \
      --rgip-alpha "${RGIP_ALPHA:-0.4}" \
      --rgip-beta "${RGIP_BETA:-0.2}" \
      --rgip-gamma "${RGIP_GAMMA:-0.5}" \
      --rgip-topk "${RGIP_TOPK:-3}" \
      --rgip-eta-object "${RGIP_ETA_OBJECT:-0.4}" \
      --rgip-eta-query "${RGIP_ETA_QUERY:-0.4}" \
      --rgip-eta-sem "${RGIP_ETA_SEM:-0.0}" \
      --rgip-lambda-attn "${RGIP_LAMBDA_ATTN:-0.05}" \
      --rgip-max-attn-pairs "${RGIP_MAX_ATTN_PAIRS:-16}" \
      --rgip-int-loss-weight "${RGIP_INT_LOSS_WEIGHT:-0.2}" \
      --rgip-feat-weight "${RGIP_FEAT_WEIGHT:-0.5}" \
      --rgip-logit-weight "${RGIP_LOGIT_WEIGHT:-0.5}" \
      --rgip-hardneg-weight "${RGIP_HARDNEG_WEIGHT:-0.05}" \
      --rgip-margin "${RGIP_MARGIN:-0.1}" \
      --rgip-debug
    ;;

  paper5-full)
    COMMON_ARGS+=(--filter-no-interaction)
    run_train \
      --use-replay \
      --use-rgip \
      --rgip-alpha "${RGIP_ALPHA:-0.4}" \
      --rgip-beta "${RGIP_BETA:-0.3}" \
      --rgip-gamma "${RGIP_GAMMA:-2.0}" \
      --rgip-topk "${RGIP_TOPK:-3}" \
      --rgip-eta-object "${RGIP_ETA_OBJECT:-0.4}" \
      --rgip-eta-query "${RGIP_ETA_QUERY:-0.4}" \
      --rgip-eta-sem "${RGIP_ETA_SEM:-0.0}" \
      --rgip-lambda-attn "${RGIP_LAMBDA_ATTN:-0.5}" \
      --rgip-max-attn-pairs "${RGIP_MAX_ATTN_PAIRS:-16}" \
      --rgip-int-loss-weight "${RGIP_INT_LOSS_WEIGHT:-1.0}" \
      --rgip-debug
    ;;

  paper10-full)
    COMMON_ARGS+=(--filter-no-interaction)
    run_train \
      --use-replay \
      --use-rgip \
      --rgip-alpha "${RGIP_ALPHA:-0.4}" \
      --rgip-beta "${RGIP_BETA:-0.3}" \
      --rgip-gamma "${RGIP_GAMMA:-2.0}" \
      --rgip-topk "${RGIP_TOPK:-3}" \
      --rgip-eta-object "${RGIP_ETA_OBJECT:-0.4}" \
      --rgip-eta-query "${RGIP_ETA_QUERY:-0.4}" \
      --rgip-eta-sem "${RGIP_ETA_SEM:-0.0}" \
      --rgip-lambda-attn "${RGIP_LAMBDA_ATTN:-0.5}" \
      --rgip-max-attn-pairs "${RGIP_MAX_ATTN_PAIRS:-16}" \
      --rgip-int-loss-weight "${RGIP_INT_LOSS_WEIGHT:-1.0}" \
      --rgip-debug
    ;;

  no-hardneg)
    run_train --use-replay --use-rgip --rgip-hardneg-weight 0 --rgip-debug
    ;;

  freq-only)
    run_train --use-replay --use-rgip --rgip-alpha 1.0 --rgip-beta 0.0 --rgip-lambda-attn 0 --rgip-debug
    ;;

  same-object)
    run_train --use-replay --use-rgip --rgip-eta-object 1.0 --rgip-eta-query 0.0 --rgip-eta-sem 0.0 --rgip-debug
    ;;

  forgetting)
    mkdir -p "${ANALYSIS_DIR}/forgetting"
    run_and_log "forgetting_${MODE}" python evaluate_forgetting.py \
      --data-root "${DATA_ROOT}" \
      --output-dir "${CHECKPOINT_DIR}" \
      --num-tasks "${NUM_TASKS:-4}" \
      --seed "${SEED}" \
      --batch-size "${EVAL_BATCH_SIZE:-2}" \
      --master-port "${FORGETTING_PORT:-12365}" \
      --save-dir "${ANALYSIS_DIR}/forgetting" \
      --train-args-file "${CONFIG_DIR}/train_args.json"
    ;;

  viz-forgetting)
    if [[ -z "${RESULTS_JSON:-}" ]]; then
      echo "Please set RESULTS_JSON=/path/to/results_forgetting_xxx.json" >&2
      exit 2
    fi
    mkdir -p "${ANALYSIS_DIR}/forgetting_viz"
    run_and_log "viz_forgetting" python evaluate_forgetting_viz.py \
      --results-json "${RESULTS_JSON}" \
      --out-dir "${ANALYSIS_DIR}/forgetting_viz" \
      --topk "${TOPK:-20}" \
      --dpi "${DPI:-300}" \
      --topk-line \
      --rare-classes-json "${RARE_PATH}"
    ;;

  attention-vis)
    BASELINE_CKPT="${BASELINE_CKPT:-${CHECKPOINT_DIR}/checkpoint_task1.pth}"
    INCREMENTAL_CKPT="${INCREMENTAL_CKPT:-${CHECKPOINT_DIR}/checkpoint_task4.pth}"
    mkdir -p "${ANALYSIS_DIR}/attention_vis"
    run_and_log "attention_vis" python inference_demo.py \
      --baseline-checkpoint "${BASELINE_CKPT}" \
      --incremental-checkpoint "${INCREMENTAL_CKPT}" \
      --data-root "${DATA_ROOT}" \
      --image-root "${IMAGE_ROOT:-${DATA_ROOT}/hico_20160224_det/images/test2015}" \
      --correspondence "${HOI_PATH}" \
      --output-dir "${ANALYSIS_DIR}/attention_vis" \
      --num-samples "${NUM_VIS_SAMPLES:-5}" \
      --attn-cmap "${ATTN_CMAP:-jet}" \
      --attn-alpha "${ATTN_ALPHA:-0.62}" \
      --attn-dim "${ATTN_DIM:-0.28}"
    ;;

  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Available modes: baseline, replay-distill, weighted, kd, full, full-rd-tuned, fast4, full4, paper5-full, paper10-full, no-hardneg, freq-only, same-object, forgetting, viz-forgetting, attention-vis" >&2
    exit 2
    ;;
esac
