#!/usr/bin/env bash
set -euo pipefail

cd /data/hujm/pvic

STAMP="$(date +%Y%m%d-%H%M%S)"
ROOT="outputs-rgip/attnfix-fullrun-${STAMP}"
LOG="${ROOT}/managed_attnfix_fullrun.log"
REPORT="${ROOT}/PROGRESS.md"
BASELINE_RUN="outputs-rgip/replay-distill-splitrandom-splitseed140-trainseed140-20260702-193603"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_PER_GPU_BATCH="${SMOKE_PER_GPU_BATCH:-8}"
FULL_PER_GPU_BATCH="${FULL_PER_GPU_BATCH:-21}"
RGIP_LAMBDA_ATTN_RUN="${RGIP_LAMBDA_ATTN_RUN:-0.15}"
RGIP_MAX_ATTN_PAIRS_RUN="${RGIP_MAX_ATTN_PAIRS_RUN:-16}"
SMOKE_RUN="${ROOT}/smoke-task4-e${SMOKE_EPOCHS}"
FULL_RUN="${ROOT}/full-rd-tuned-attnfix-clean"

mkdir -p "${ROOT}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG}"
}

append_report() {
  {
    echo
    echo "$*"
  } >> "${REPORT}"
}

prepare_task4_run() {
  local run_dir="$1"
  mkdir -p "${run_dir}/checkpoints"
  cp "${BASELINE_RUN}/checkpoints/checkpoint_task1.pth" "${run_dir}/checkpoints/"
  cp "${BASELINE_RUN}/checkpoints/checkpoint_task2.pth" "${run_dir}/checkpoints/"
  cp "${BASELINE_RUN}/checkpoints/checkpoint_task3.pth" "${run_dir}/checkpoints/"
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

task4_triplet() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
import json, os, sys
p = os.path.join(sys.argv[1], 'metrics', 'task4_eval.json')
if not os.path.exists(p):
    print('nan nan nan')
else:
    r = json.load(open(p))
    print(r.get('mAP', 'nan'), r.get('rare_mAP', 'nan'), r.get('non_rare_mAP', 'nan'))
PY
}

cat > "${REPORT}" <<EOF
# attention 修复后完整训练托管日志

- 启动时间：$(date '+%Y-%m-%d %H:%M:%S')
- GPU: 0,1,2；GPU 3 预留
- 配置：full-rd-tuned + detector eval + attention context fallback + top-SIS limited attention
- Attention: lambda=${RGIP_LAMBDA_ATTN_RUN}, max_attn_pairs=${RGIP_MAX_ATTN_PAIRS_RUN}
- 策略：先做 Task4-only ${SMOKE_EPOCHS} epoch 冒烟验证；若真实数据中 attention modulation 非零，再启动完整四阶段 clean run。

EOF

log "Managed attnfix full-run started. Root=${ROOT}"
log "Smoke run: ${SMOKE_RUN}"

prepare_task4_run "${SMOKE_RUN}"
append_report "## 1. Smoke test: Task4-only ${SMOKE_EPOCHS} epoch"
append_report "- Run dir: \`${SMOKE_RUN}\`"
append_report "- Start: $(date '+%Y-%m-%d %H:%M:%S')"

CUDA_DEVICES=0,1,2 \
WORLD_SIZE=3 \
PER_GPU_BATCH="${SMOKE_PER_GPU_BATCH}" \
PORT=12367 \
SEED=140 \
SPLIT_SEED=140 \
SPLIT_MODE=random \
START_TASK=3 \
EPOCHS="${SMOKE_EPOCHS}" \
RUN_DIR="${SMOKE_RUN}" \
PRETRAINED=checkpoints/detr-r50-hicodet.pth \
RGIP_LAMBDA_ATTN="${RGIP_LAMBDA_ATTN_RUN}" \
RGIP_MAX_ATTN_PAIRS="${RGIP_MAX_ATTN_PAIRS_RUN}" \
PRINT_INTERVAL=10 \
SKIP_INITIAL_EVAL=1 \
bash scripts/run_rgip_experiments.sh full-rd-tuned

smoke_triplet="$(task4_triplet "${SMOKE_RUN}")"
if ! attention_summary="$(rgip_attention_summary "${SMOKE_RUN}")"; then
  attention_summary="failed 0 0 0 0"
fi
log "Smoke result: ${smoke_triplet}"
log "Smoke attention summary: ${attention_summary}"
append_report "- End: $(date '+%Y-%m-%d %H:%M:%S')"
append_report "- Task4 smoke result: \`${smoke_triplet}\` (mAP rare non-rare)"
append_report "- Attention summary: \`${attention_summary}\`"

if ! python - "$attention_summary" <<'PY'
import sys
parts = sys.argv[1].split()
if len(parts) < 5 or parts[0] != 'ok':
    raise SystemExit(1)
modulated = float(parts[1])
bias_mean = float(parts[3])
raise SystemExit(0 if modulated > 0 and bias_mean > 0 else 1)
PY
then
  log "Smoke failed: attention modulation is not active enough; stopping before full run."
  append_report "- Decision: smoke failed; full clean run was not launched."
  append_report "## 2. 结束"
  append_report "- 总日志：\`${LOG}\`"
  exit 1
fi

log "Smoke passed: attention modulation is active. Starting full clean run."
append_report "## 2. Full clean run"
append_report "- Decision: smoke passed; launching full four-task training."
append_report "- Run dir: \`${FULL_RUN}\`"
append_report "- Start: $(date '+%Y-%m-%d %H:%M:%S')"

CUDA_DEVICES=0,1,2 \
WORLD_SIZE=3 \
PER_GPU_BATCH="${FULL_PER_GPU_BATCH}" \
PORT=12368 \
SEED=140 \
SPLIT_SEED=140 \
SPLIT_MODE=random \
RUN_DIR="${FULL_RUN}" \
PRETRAINED=checkpoints/detr-r50-hicodet.pth \
RGIP_LAMBDA_ATTN="${RGIP_LAMBDA_ATTN_RUN}" \
RGIP_MAX_ATTN_PAIRS="${RGIP_MAX_ATTN_PAIRS_RUN}" \
SKIP_INITIAL_EVAL=1 \
bash scripts/run_rgip_experiments.sh full-rd-tuned

final_triplet="$(task4_triplet "${FULL_RUN}")"
log "Full clean run finished: ${final_triplet}"
append_report "- End: $(date '+%Y-%m-%d %H:%M:%S')"
append_report "- Final Task4 result: \`${final_triplet}\` (mAP rare non-rare)"
append_report "## 3. 结束"
append_report "- 总日志：\`${LOG}\`"
log "Managed attnfix full-run finished. Report=${REPORT}"
