#!/usr/bin/env bash
set -euo pipefail

cd /data/hujm/pvic

SESSION_NAME="hjm2"
STAMP="$(date +%Y%m%d-%H%M%S)"
ROOT="outputs-rgip/managed-tuning-${STAMP}"
LOG="${ROOT}/managed_tuning.log"
REPORT="${ROOT}/PROGRESS.md"
BASELINE_RUN="outputs-rgip/replay-distill-splitrandom-splitseed140-trainseed140-20260702-193603"
BASELINE_EPOCH_REF=12
BASELINE_FINAL_MAP=0.31689739502178005
BASELINE_EPOCH_REF_MAP="$(python - <<'PY'
import json
p='outputs-rgip/replay-distill-splitrandom-splitseed140-trainseed140-20260702-193603/metrics/epoch_metrics.jsonl'
target=12
rows=[json.loads(l) for l in open(p)]
vals=[r for r in rows if r.get('task_idx')==4 and r.get('epoch')==target]
print(vals[-1]['mAP'] if vals else 0.0)
PY
)"

mkdir -p "${ROOT}"
LAST_RUN_DIR=""

log() {
  local msg="$*"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}" | tee -a "${LOG}"
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

extract_task4_map() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
import json, os, sys
run_dir = sys.argv[1]
p = os.path.join(run_dir, 'metrics', 'task4_eval.json')
if not os.path.exists(p):
    print('nan')
else:
    r = json.load(open(p))
    print(r.get('mAP', 'nan'))
PY
}

extract_task4_triplet() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
import json, os, sys
run_dir = sys.argv[1]
p = os.path.join(run_dir, 'metrics', 'task4_eval.json')
if not os.path.exists(p):
    print('nan nan nan')
else:
    r = json.load(open(p))
    print(r.get('mAP', 'nan'), r.get('rare_mAP', 'nan'), r.get('non_rare_mAP', 'nan'))
PY
}

extract_rgip_debug_tail() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
import json, os, sys
run_dir = sys.argv[1]
p = os.path.join(run_dir, 'metrics', 'rgip_iteration_stats.jsonl')
if not os.path.exists(p):
    print('No rgip_iteration_stats.jsonl')
    raise SystemExit
rows = [json.loads(l) for l in open(p) if l.strip()]
if not rows:
    print('Empty rgip_iteration_stats.jsonl')
    raise SystemExit
last = rows[-1]
keys = [
    'task_idx', 'epoch', 'iteration', 'loss',
    'rgip/int_loss', 'rgip/prototype_count',
    'rgip/context_modulate_pairs', 'rgip/modulated_pair_count',
    'rgip/attn_bias_abs_mean',
    'distill/replay_loss', 'distill/common_pairs',
]
print({k: last.get(k) for k in keys if k in last})
PY
}

run_candidate() {
  local label="$1"
  local epochs="$2"
  local port="$3"
  shift 3
  local run_dir="${ROOT}/${label}-task4-e${epochs}"
  prepare_task4_run "${run_dir}"
  log "Starting candidate ${label}, epochs=${epochs}, run_dir=${run_dir}"
  append_report "## Candidate ${label} (${epochs} epochs)"
  append_report "- Run dir: \`${run_dir}\`"
  append_report "- Start time: $(date '+%Y-%m-%d %H:%M:%S')"

  env \
    CUDA_DEVICES=0,1,2 \
    WORLD_SIZE=3 \
    PER_GPU_BATCH=21 \
    PORT="${port}" \
    SEED=140 \
    SPLIT_SEED=140 \
    SPLIT_MODE=random \
    START_TASK=3 \
    EPOCHS="${epochs}" \
    RUN_DIR="${run_dir}" \
    PRETRAINED=checkpoints/detr-r50-hicodet.pth \
    "$@" \
    bash scripts/run_rgip_experiments.sh full-rd-tuned

  local triplet
  triplet="$(extract_task4_triplet "${run_dir}")"
  local debug_tail
  debug_tail="$(extract_rgip_debug_tail "${run_dir}")"
  log "Finished candidate ${label}: ${triplet}"
  log "Debug tail ${label}: ${debug_tail}"
  append_report "- End time: $(date '+%Y-%m-%d %H:%M:%S')"
  append_report "- Task4 result: \`${triplet}\` (mAP rare non-rare)"
  append_report "- Debug tail: \`${debug_tail}\`"
  LAST_RUN_DIR="${run_dir}"
}

cat > "${REPORT}" <<EOF
# RGIP 托管调参进展日志

- 启动时间：$(date '+%Y-%m-%d %H:%M:%S')
- screen session: ${SESSION_NAME}
- GPU: 0,1,2；GPU 3 预留
- 目标：在不削弱 baseline 的前提下，将 RGIP 改成 replay-distill 强基线上的轻量增强。
- 当前 replay-distill Task4 final mAP: ${BASELINE_FINAL_MAP}
- 当前 replay-distill Task4 epoch${BASELINE_EPOCH_REF} mAP: ${BASELINE_EPOCH_REF_MAP}

## 策略

1. 先跑低成本 Task4-only 12 epoch 候选。
2. 若候选超过 replay-distill 同 epoch 参考值，则自动扩展跑 30 epoch。
3. 若没有超过，则记录诊断信息，等待下一步进一步降权/修 attention。
EOF

log "Managed RGIP tuning started. Root=${ROOT}"
log "Baseline final=${BASELINE_FINAL_MAP}; baseline epoch${BASELINE_EPOCH_REF}=${BASELINE_EPOCH_REF_MAP}"

run_candidate \
  a_full_rd_tuned \
  12 \
  12362
run_a="${LAST_RUN_DIR}"
map_a="$(extract_task4_map "${run_a}")"

best_label="a_full_rd_tuned"
best_run="${run_a}"
best_map="${map_a}"

if ! python - "$map_a" "$BASELINE_EPOCH_REF_MAP" <<'PY'
import math, sys
val=float(sys.argv[1])
ref=float(sys.argv[2])
raise SystemExit(0 if val >= ref else 1)
PY
then
  log "Candidate A did not beat epoch reference. Running candidate B: replay-distill + SIS only."
  run_candidate \
    b_rd_sis_only \
    12 \
    12364 \
    RGIP_INT_LOSS_WEIGHT=0 \
    RGIP_LAMBDA_ATTN=0 \
    RGIP_GAMMA=0.5
  run_b="${LAST_RUN_DIR}"
  map_b="$(extract_task4_map "${run_b}")"
  if python - "$map_b" "$best_map" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
  then
    best_label="b_rd_sis_only"
    best_run="${run_b}"
    best_map="${map_b}"
  fi
fi

log "Best short candidate=${best_label}, mAP=${best_map}, run=${best_run}"
append_report "## Short-test decision"
append_report "- Best short candidate: **${best_label}**"
append_report "- Best short mAP: **${best_map}**"
append_report "- Best short run: \`${best_run}\`"

if python - "$best_map" "$BASELINE_EPOCH_REF_MAP" <<'PY'
import sys
best=float(sys.argv[1])
ref=float(sys.argv[2])
raise SystemExit(0 if best >= ref else 1)
PY
then
  log "Best short candidate is promising. Starting 30-epoch validation."
  append_report "## 30-epoch validation"
  append_report "- Decision: short test beat/equals epoch reference; launching 30-epoch Task4-only validation."
  if [[ "${best_label}" == "b_rd_sis_only" ]]; then
    run_candidate \
      b_rd_sis_only \
      30 \
      12365 \
      RGIP_INT_LOSS_WEIGHT=0 \
      RGIP_LAMBDA_ATTN=0 \
      RGIP_GAMMA=0.5
  else
    run_candidate \
      a_full_rd_tuned \
      30 \
      12365
  fi
  run_30="${LAST_RUN_DIR}"
  map_30="$(extract_task4_map "${run_30}")"
  log "30-epoch validation finished: ${run_30}, mAP=${map_30}"
  append_report "- 30-epoch run: \`${run_30}\`"
  append_report "- 30-epoch Task4 mAP: **${map_30}**"
  if python - "$map_30" "$BASELINE_FINAL_MAP" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)
PY
  then
    log "SUCCESS: tuned RGIP beats replay-distill final."
    append_report "- 结论：当前 tuned RGIP Task4-only 已超过 replay-distill final，可进一步跑完整四阶段 clean run。"
    append_report "- 建议下一步命令：\`CUDA_DEVICES=0,1,2 WORLD_SIZE=3 PER_GPU_BATCH=21 PORT=12366 SEED=140 SPLIT_SEED=140 SPLIT_MODE=random PRETRAINED=checkpoints/detr-r50-hicodet.pth bash scripts/run_rgip_experiments.sh full-rd-tuned\`"
  else
    log "Tuned RGIP did not beat replay-distill final. Need further tuning."
    append_report "- 结论：30 epoch 仍未超过 replay-distill final，建议进一步降低 RGIP_INT_LOSS_WEIGHT 或排查 attention modulation。"
  fi
else
  log "No short candidate beat epoch reference. Stop after short tests."
  append_report "## Final decision"
  append_report "- 结论：12 epoch 候选未超过 replay-distill 同 epoch 参考值，暂不继续长跑。"
  append_report "- 建议：进一步将 RGIP_INT_LOSS_WEIGHT 降到 0.05 / 关闭 hardneg / 修复 attention modulation 后再测。"
fi

append_report "## 结束"
append_report "- 结束时间：$(date '+%Y-%m-%d %H:%M:%S')"
append_report "- 总日志：\`${LOG}\`"
log "Managed tuning finished. Report=${REPORT}"
