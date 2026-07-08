#!/usr/bin/env bash
set -euo pipefail

cd /data/hujm/pvic

STAMP="$(date +%Y%m%d-%H%M%S)"
ROOT="outputs-rgip/multiseed-20260708"
RUN_NAME="replay-distill-splitrandom-splitseed141-trainseed141-${STAMP}"
RUN_DIR="${ROOT}/${RUN_NAME}"
LOG="${ROOT}/managed_${RUN_NAME}.log"
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

extract_task_triplet() {
  local run_dir="$1"
  local task_id="$2"
  python - "$run_dir" "$task_id" <<'PY'
import json, os, sys
run_dir, task_id = sys.argv[1], sys.argv[2]
p = os.path.join(run_dir, "metrics", f"task{task_id}_eval.json")
if not os.path.exists(p):
    print("nan nan nan")
else:
    r = json.load(open(p))
    print(r.get("mAP", "nan"), r.get("rare_mAP", "nan"), r.get("non_rare_mAP", "nan"))
PY
}

on_exit() {
  local status=$?
  if [[ "${status}" -eq 0 ]]; then
    log "Replay-distill seed141 finished successfully."
  else
    log "Replay-distill seed141 failed/interrupted with status ${status}."
  fi
  for task_id in 1 2 3 4; do
    triplet="$(extract_task_triplet "${RUN_DIR}" "${task_id}")"
    log "Task${task_id} result: ${triplet}"
    append_report "- Task${task_id}: \`${triplet}\` (mAP rare non-rare)"
  done
  append_report "- End: $(date '+%Y-%m-%d %H:%M:%S'), status=${status}, run=\`${RUN_DIR}\`"
  return "${status}"
}
trap on_exit EXIT

log "Starting replay-distill seed141 on current server."
log "Run dir: ${RUN_DIR}"
log "GPUs: 0,1,2; WORLD_SIZE=3; PER_GPU_BATCH=21"
append_report "### Replay-distill seed141, current server"
append_report "- Start: $(date '+%Y-%m-%d %H:%M:%S')"
append_report "- Run dir: \`${RUN_DIR}\`"
append_report "- GPUs: 0,1,2; WORLD_SIZE=3; PER_GPU_BATCH=21"
append_report "- Purpose: paired baseline for RGIP seed141 running on serverB."

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

CUDA_DEVICES=0,1,2 \
WORLD_SIZE=3 \
PER_GPU_BATCH=21 \
NUM_WORKERS=1 \
PORT=12382 \
SEED=141 \
SPLIT_SEED=141 \
SPLIT_MODE=random \
OUTPUT_ROOT="${ROOT}" \
RUN_NAME="${RUN_NAME}" \
RUN_DIR="${RUN_DIR}" \
PRETRAINED=checkpoints/detr-r50-hicodet.pth \
SKIP_INITIAL_EVAL=1 \
bash scripts/run_rgip_experiments.sh replay-distill
