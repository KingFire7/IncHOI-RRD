#!/usr/bin/env bash
set -euo pipefail

cd /data/hujm/pvic

ROOT="outputs-rgip/attnfix-fullrun-20260706-030113"
OLD_RUN="${ROOT}/full-rd-tuned-attnfix-clean"
RESUME_RUN="${ROOT}/full-rd-tuned-attnfix-ddpfix-resume-task2"
LOG="${ROOT}/managed_ddpfix_resume.log"
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

rgip_attention_summary() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
import json, os, sys
run_dir = sys.argv[1]
p = os.path.join(run_dir, 'metrics', 'rgip_iteration_stats.jsonl')
if not os.path.exists(p):
    print('missing 0 0 0 0')
    raise SystemExit(1)
rows = [json.loads(l) for l in open(p) if l.strip()]
mod = sum(float(r.get('rgip/modulated_pair_count', 0) or 0) for r in rows)
ctx = sum(float(r.get('rgip/context_modulate_pairs', 0) or 0) for r in rows)
bias_vals = [float(r['rgip/attn_bias_abs_mean']) for r in rows if 'rgip/attn_bias_abs_mean' in r]
fallback = sum(float(r.get('rgip/attn_alignment_fallback', 0) or 0) for r in rows)
print('ok', mod, ctx, (sum(bias_vals) / len(bias_vals) if bias_vals else 0.0), fallback)
PY
}

on_exit() {
  local status=$?
  if [[ "${status}" -eq 0 ]]; then
    log "DDP empty-batch fixed resume finished successfully."
    append_report "- DDP-fix resume exit status: 0"
  else
    log "DDP empty-batch fixed resume failed with status ${status}."
    append_report "- DDP-fix resume exit status: ${status}"
  fi
  return "${status}"
}
trap on_exit EXIT

mkdir -p "${RESUME_RUN}/checkpoints" "${RESUME_RUN}/metrics" "${RESUME_RUN}/analysis" "${RESUME_RUN}/logs" "${RESUME_RUN}/configs"

if [[ ! -f "${OLD_RUN}/checkpoints/checkpoint_task1.pth" ]]; then
  log "Missing ${OLD_RUN}/checkpoints/checkpoint_task1.pth; cannot resume from Task2."
  exit 2
fi

cp "${OLD_RUN}/checkpoints/checkpoint_task1.pth" "${RESUME_RUN}/checkpoints/"
if [[ -f "${OLD_RUN}/metrics/task1_eval.json" ]]; then
  cp "${OLD_RUN}/metrics/task1_eval.json" "${RESUME_RUN}/metrics/"
fi
if [[ -f "${OLD_RUN}/metrics/task_eval_summary.jsonl" ]]; then
  cp "${OLD_RUN}/metrics/task_eval_summary.jsonl" "${RESUME_RUN}/metrics/"
fi

log "Starting DDP empty-batch fixed resume."
log "Old run: ${OLD_RUN}"
log "Resume run: ${RESUME_RUN}"
log "Task1 checkpoint copied; START_TASK=1 will train Task2-Task4."
append_report "### 2026-07-06 DDP-fix resume run"
append_report "- Run dir: \`${RESUME_RUN}\`"
append_report "- Start: $(date '+%Y-%m-%d %H:%M:%S')"
append_report "- Resume policy: reuse \`${OLD_RUN}/checkpoints/checkpoint_task1.pth\`, start from Task2 with \`START_TASK=1\`."

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

CUDA_DEVICES=0,1,2 \
WORLD_SIZE=3 \
PER_GPU_BATCH=21 \
PORT=12369 \
SEED=140 \
SPLIT_SEED=140 \
SPLIT_MODE=random \
START_TASK=1 \
RUN_DIR="${RESUME_RUN}" \
PRETRAINED=checkpoints/detr-r50-hicodet.pth \
RGIP_LAMBDA_ATTN=0.15 \
RGIP_MAX_ATTN_PAIRS=16 \
SKIP_INITIAL_EVAL=1 \
bash scripts/run_rgip_experiments.sh full-rd-tuned

task2_triplet="$(task_triplet "${RESUME_RUN}" 2)"
task3_triplet="$(task_triplet "${RESUME_RUN}" 3)"
task4_triplet="$(task_triplet "${RESUME_RUN}" 4)"
attention_summary="$(rgip_attention_summary "${RESUME_RUN}" || true)"

log "Task2 result: ${task2_triplet}"
log "Task3 result: ${task3_triplet}"
log "Task4 result: ${task4_triplet}"
log "Attention summary: ${attention_summary}"
append_report "- End: $(date '+%Y-%m-%d %H:%M:%S')"
append_report "- Task2 result: \`${task2_triplet}\` (mAP rare non-rare)"
append_report "- Task3 result: \`${task3_triplet}\` (mAP rare non-rare)"
append_report "- Task4 result: \`${task4_triplet}\` (mAP rare non-rare)"
append_report "- Attention summary: \`${attention_summary}\`"
