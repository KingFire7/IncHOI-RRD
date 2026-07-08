#!/usr/bin/env bash
set -euo pipefail

cd /data/hujm/pvic

ROOT="outputs-rgip/attnfix-fullrun-20260706-030113"
RUN_DIR="${ROOT}/full-rd-tuned-attnfix-ddpfix-resume-task2-reboot-20260706-2137"
RESUME_CKPT="${RUN_DIR}/checkpoints/latest.pth"
LOG="${ROOT}/managed_latest_resume_2gpu-$(date +%Y%m%d-%H%M%S).log"
REPORT="${ROOT}/PROGRESS.md"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"
}

append_report() {
  {
    echo
    echo "$*"
  } >> "${REPORT}"
}

task_triplet() {
  local run_dir="$1"
  local task_id="$2"
  python - "$run_dir" "$task_id" <<'PY'
import json, os, sys
run_dir, task_id = sys.argv[1], sys.argv[2]
p = os.path.join(run_dir, 'metrics', f'task{task_id}_eval.json')
if not os.path.exists(p):
    print('nan nan nan')
else:
    r = json.load(open(p))
    print(r.get('mAP', 'nan'), r.get('rare_mAP', 'nan'), r.get('non_rare_mAP', 'nan'))
PY
}

on_exit() {
  local status=$?
  if [[ "${status}" -eq 0 ]]; then
    log "2-GPU latest-checkpoint resume finished successfully."
    append_report "- 2-GPU latest-checkpoint resume exit status: 0"
  else
    log "2-GPU latest-checkpoint resume failed/interrupted with status ${status}."
    append_report "- 2-GPU latest-checkpoint resume exit status: ${status}"
  fi
  return "${status}"
}
trap on_exit EXIT

if [[ ! -f "${RESUME_CKPT}" ]]; then
  log "Missing resume checkpoint: ${RESUME_CKPT}"
  exit 2
fi

log "Starting 2-GPU latest-checkpoint resume."
log "Run dir: ${RUN_DIR}"
log "Resume checkpoint: ${RESUME_CKPT}"
log "CUDA devices: 0,1; WORLD_SIZE=2; PER_GPU_BATCH=21; NUM_WORKERS=1"
append_report "### 2026-07-07 2-GPU latest-checkpoint resume"
append_report "- Run dir: \`${RUN_DIR}\`"
append_report "- Resume checkpoint: \`${RESUME_CKPT}\`"
append_report "- Start: $(date '+%Y-%m-%d %H:%M:%S')"
append_report "- Policy: reduce from 3 GPU to 2 GPU after repeated server reboots; continue Task2 from saved latest checkpoint."
append_report "- Runtime: CUDA_DEVICES=0,1, WORLD_SIZE=2, PER_GPU_BATCH=21, NUM_WORKERS=1."

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

CUDA_DEVICES=0,1 \
WORLD_SIZE=2 \
PER_GPU_BATCH=21 \
NUM_WORKERS=1 \
PORT="${PORT:-12372}" \
SEED=140 \
SPLIT_SEED=140 \
SPLIT_MODE=random \
START_TASK=1 \
RUN_DIR="${RUN_DIR}" \
PRETRAINED=checkpoints/detr-r50-hicodet.pth \
RESUME="${RESUME_CKPT}" \
RGIP_LAMBDA_ATTN=0.15 \
RGIP_MAX_ATTN_PAIRS=16 \
SKIP_INITIAL_EVAL=1 \
bash scripts/run_rgip_experiments.sh full-rd-tuned

task2_triplet="$(task_triplet "${RUN_DIR}" 2)"
task3_triplet="$(task_triplet "${RUN_DIR}" 3)"
task4_triplet="$(task_triplet "${RUN_DIR}" 4)"
log "Task2 result: ${task2_triplet}"
log "Task3 result: ${task3_triplet}"
log "Task4 result: ${task4_triplet}"
append_report "- End: $(date '+%Y-%m-%d %H:%M:%S')"
append_report "- Task2 result: \`${task2_triplet}\` (mAP rare non-rare)"
append_report "- Task3 result: \`${task3_triplet}\` (mAP rare non-rare)"
append_report "- Task4 result: \`${task4_triplet}\` (mAP rare non-rare)"
