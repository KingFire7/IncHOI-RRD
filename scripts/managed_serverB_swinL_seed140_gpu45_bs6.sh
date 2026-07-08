#!/usr/bin/env bash
set -euo pipefail

cd /data02/James/pvic-rgip-current

ROOT="/data04/James/outputs-rgip-serverB"
RUN_NAME="full-rd-tuned-swinL-serverB-splitrandom-splitseed140-trainseed140-ws2-pgb3-bs6"
RUN_DIR="${ROOT}/${RUN_NAME}"
LOG="${ROOT}/managed_${RUN_NAME}_$(date +%Y%m%d-%H%M%S).log"
REPORT="${ROOT}/PROGRESS.md"

mkdir -p "${ROOT}" "${RUN_DIR}" "${RUN_DIR}/logs" "${RUN_DIR}/metrics" "${RUN_DIR}/checkpoints" "${RUN_DIR}/analysis" "${RUN_DIR}/configs"
mkdir -p checkpoints hicodet
ln -sfn /data02/James/pvic/checkpoints/h-defm-detr-swinL-dp0-mqs-lft-iter-2stg-hicodet.pth checkpoints/h-defm-detr-swinL-dp0-mqs-lft-iter-2stg-hicodet.pth
rm -f hicodet/hico_20160224_det hicodet/detections
ln -s /data02/James/SCG/spatially-conditioned-graphs/hicodet/hico_20160224_det hicodet/hico_20160224_det
ln -s /data02/James/SCG/spatially-conditioned-graphs/hicodet/detections hicodet/detections

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
    log "ServerB Swin-L full seed140 bs6 finished successfully."
  else
    log "ServerB Swin-L full seed140 bs6 failed/interrupted with status ${status}."
  fi
  append_report "- End: $(date '+%Y-%m-%d %H:%M:%S'), status=${status}, run=\`${RUN_DIR}\`"
  return "${status}"
}
trap on_exit EXIT

source /data01/James/miniconda3/etc/profile.d/conda.sh
conda activate /home/James/.conda/envs/pvic

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

log "Starting ServerB Swin-L full seed140 bs6."
log "Run dir: ${RUN_DIR}"
log "GPUs: 4,5; world_size=2; per_gpu_batch=3; global_batch=6"
log "Pretrained: checkpoints/h-defm-detr-swinL-dp0-mqs-lft-iter-2stg-hicodet.pth"
append_report "### ServerB Swin-L full seed140 bs6"
append_report "- Start: $(date '+%Y-%m-%d %H:%M:%S')"
append_report "- Run dir: \`${RUN_DIR}\`"
append_report "- GPUs: 4,5; WORLD_SIZE=2; PER_GPU_BATCH=3; BATCH_SIZE=6"
append_report "- Detector: advanced, backbone=swin_large"

DETR_TYPE=advanced \
BACKBONE=swin_large \
DROP_PATH_RATE=0.5 \
NUM_QUERIES_ONE2ONE=900 \
NUM_QUERIES_ONE2MANY=1500 \
CUDA_DEVICES=4,5 \
WORLD_SIZE=2 \
PER_GPU_BATCH=3 \
NUM_WORKERS=1 \
PORT=23478 \
SEED=140 \
SPLIT_SEED=140 \
SPLIT_MODE=random \
START_TASK=0 \
OUTPUT_ROOT="${ROOT}" \
RUN_NAME="${RUN_NAME}" \
RUN_DIR="${RUN_DIR}" \
DATA_ROOT=/data02/James/pvic-rgip-current/hicodet \
PRETRAINED=checkpoints/h-defm-detr-swinL-dp0-mqs-lft-iter-2stg-hicodet.pth \
RGIP_LAMBDA_ATTN=0.15 \
RGIP_MAX_ATTN_PAIRS=16 \
SKIP_INITIAL_EVAL=1 \
bash scripts/run_rgip_experiments.sh full-rd-tuned
