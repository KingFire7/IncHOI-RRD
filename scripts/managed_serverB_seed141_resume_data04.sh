#!/usr/bin/env bash
set -euo pipefail

cd /data02/James/pvic-rgip-current

ROOT="/data04/James/outputs-rgip-serverB"
RUN_NAME="full-rd-tuned-serverB-splitrandom-splitseed141-trainseed141"
RUN_DIR="${ROOT}/${RUN_NAME}"
RESUME_CKPT="${RUN_DIR}/checkpoints/latest.pth"
LOG="${ROOT}/managed_${RUN_NAME}_resume_data04_$(date +%Y%m%d-%H%M%S).log"
REPORT="${ROOT}/PROGRESS.md"

mkdir -p "${ROOT}" "${RUN_DIR}" "${RUN_DIR}/logs" "${RUN_DIR}/metrics" "${RUN_DIR}/checkpoints" "${RUN_DIR}/analysis" "${RUN_DIR}/configs"
mkdir -p hicodet
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
    log "ServerB seed141 data04 resume finished successfully."
  else
    log "ServerB seed141 data04 resume failed/interrupted with status ${status}."
  fi
  append_report "- End: $(date '+%Y-%m-%d %H:%M:%S'), status=${status}, run=\`${RUN_DIR}\`"
  return "${status}"
}
trap on_exit EXIT

if [[ ! -f "${RESUME_CKPT}" ]]; then
  log "Missing resume checkpoint: ${RESUME_CKPT}"
  exit 2
fi

source /data01/James/miniconda3/etc/profile.d/conda.sh
conda activate /home/James/.conda/envs/pvic

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

log "Starting ServerB seed141 data04 resume."
log "Run dir: ${RUN_DIR}"
log "Resume checkpoint: ${RESUME_CKPT}"
log "GPUs: 1,2,3; world_size=3; per_gpu_batch=21"
log "Data root: /data02/James/pvic-rgip-current/hicodet"
append_report "### ServerB seed141 data04 resume"
append_report "- Start: $(date '+%Y-%m-%d %H:%M:%S')"
append_report "- Run dir: \`${RUN_DIR}\`"
append_report "- Resume checkpoint: \`${RESUME_CKPT}\`"
append_report "- GPUs: 1,2,3; WORLD_SIZE=3; PER_GPU_BATCH=21"

CUDA_DEVICES=1,2,3 \
WORLD_SIZE=3 \
PER_GPU_BATCH=21 \
NUM_WORKERS=2 \
PORT=23474 \
SEED=141 \
SPLIT_SEED=141 \
SPLIT_MODE=random \
START_TASK=3 \
OUTPUT_ROOT="${ROOT}" \
RUN_NAME="${RUN_NAME}" \
RUN_DIR="${RUN_DIR}" \
DATA_ROOT=/data02/James/pvic-rgip-current/hicodet \
PRETRAINED=checkpoints/detr-r50-hicodet.pth \
RESUME="${RESUME_CKPT}" \
RGIP_LAMBDA_ATTN=0.15 \
RGIP_MAX_ATTN_PAIRS=16 \
SKIP_INITIAL_EVAL=1 \
bash scripts/run_rgip_experiments.sh full-rd-tuned
