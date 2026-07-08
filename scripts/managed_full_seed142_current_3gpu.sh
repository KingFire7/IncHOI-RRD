#!/usr/bin/env bash
set -euo pipefail

cd /data/hujm/pvic

ROOT="outputs-rgip/full-seeds-20260708"
RUN_NAME="full-rd-tuned-resnet50-splitrandom-splitseed142-trainseed142"
RUN_DIR="${ROOT}/${RUN_NAME}"
LOG="${ROOT}/managed_${RUN_NAME}_$(date +%Y%m%d-%H%M%S).log"
REPORT="${ROOT}/PROGRESS.md"

mkdir -p "${ROOT}" "${RUN_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"
}

append_report() {
  {
    echo
    echo "$*"
  } >> "${REPORT}"
}

on_exit() {
  local status=$?
  if [[ "${status}" -eq 0 ]]; then
    log "Current-server ResNet full seed142 finished successfully."
  else
    log "Current-server ResNet full seed142 failed/interrupted with status ${status}."
  fi
  append_report "- End: $(date '+%Y-%m-%d %H:%M:%S'), status=${status}, run=\`${RUN_DIR}\`"
  return "${status}"
}
trap on_exit EXIT

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pvic

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

log "Starting current-server ResNet full seed142."
log "Run dir: ${RUN_DIR}"
log "GPUs: 0,1,2; world_size=3; per_gpu_batch=21"
append_report "### Current server ResNet full seed142"
append_report "- Start: $(date '+%Y-%m-%d %H:%M:%S')"
append_report "- Run dir: \`${RUN_DIR}\`"
append_report "- GPUs: 0,1,2; WORLD_SIZE=3; PER_GPU_BATCH=21"

CUDA_DEVICES=0,1,2 \
WORLD_SIZE=3 \
PER_GPU_BATCH=21 \
NUM_WORKERS=2 \
PORT=12442 \
SEED=142 \
SPLIT_SEED=142 \
SPLIT_MODE=random \
START_TASK=0 \
OUTPUT_ROOT="${ROOT}" \
RUN_NAME="${RUN_NAME}" \
RUN_DIR="${RUN_DIR}" \
PRETRAINED=checkpoints/detr-r50-hicodet.pth \
RGIP_LAMBDA_ATTN=0.15 \
RGIP_MAX_ATTN_PAIRS=16 \
SKIP_INITIAL_EVAL=1 \
bash scripts/run_rgip_experiments.sh full-rd-tuned
