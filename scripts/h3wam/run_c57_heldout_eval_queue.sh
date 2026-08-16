#!/usr/bin/env bash
set -euo pipefail

# This queue is intentionally assigned to an idle GPU/node.  It never touches
# the C57 long-training process and it never selects an intermediate checkpoint
# for promotion; step5000 is pre-registered as the decision checkpoint.
C57_EVAL_PYTHON=${C57_EVAL_PYTHON:?set C57_EVAL_PYTHON}
C57_EVAL_PLAN=${C57_EVAL_PLAN:?set C57_EVAL_PLAN}
C57_CHECKPOINT_DIR=${C57_CHECKPOINT_DIR:?set C57_CHECKPOINT_DIR}
C57_EVAL_CACHE_ROOT=${C57_EVAL_CACHE_ROOT:?set C57_EVAL_CACHE_ROOT}
C57_EVAL_CACHE_SOURCE=${C57_EVAL_CACHE_SOURCE:?set C57_EVAL_CACHE_SOURCE}
C57_EVAL_OUTPUT_DIR=${C57_EVAL_OUTPUT_DIR:?set C57_EVAL_OUTPUT_DIR}
C57_EVAL_DEVICE=${C57_EVAL_DEVICE:-cuda:0}
C57_EVAL_PHYSICAL_GPU=${C57_EVAL_PHYSICAL_GPU:-0}
C57_EVAL_KV_SUBDIR=${C57_EVAL_KV_SUBDIR:-h3_int8_dreamwam_kv_5x32_dense_v1}
C57_EVAL_IDLE_CONFIRM_SECONDS=${C57_EVAL_IDLE_CONFIRM_SECONDS:-30}
C57_C56_GO_LONG=${C57_C56_GO_LONG:-/mnt/h3-wam/outputs/c56b-fact-online-v1/optimizer-canary10-v1/GO_LONG.json}
C57_C56_PARENT_CHECKPOINT=${C57_C56_PARENT_CHECKPOINT:-/mnt/h3-wam/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt}
C57_C56_PARENT_READY=${C57_C56_PARENT_READY:-/mnt/h3-wam/outputs/c58b-fastwam-layerwise-v1/online-long10000/READY.json}
C57_C56_FINAL_CHECKPOINT=${C57_C56_FINAL_CHECKPOINT:-/mnt/h3-wam/outputs/c56b-fact-online-v1/online-long10000-v1/checkpoints/c56b_online_s10000.pt}
C57_C56_FINAL_RESTORE=${C57_C56_FINAL_RESTORE:-/mnt/h3-wam/outputs/c56b-fact-online-v1/online-long10000-v1/restore/restore_s10000.json}
C57_EVAL_PREEMPT_POLL_SECONDS=${C57_EVAL_PREEMPT_POLL_SECONDS:-5}

if ! [[ "${C57_EVAL_PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
  echo "C57_EVAL_PHYSICAL_GPU must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "${C57_EVAL_IDLE_CONFIRM_SECONDS}" =~ ^[0-9]+$ ]] || \
   [[ "${C57_EVAL_IDLE_CONFIRM_SECONDS}" -le 0 ]]; then
  echo "C57_EVAL_IDLE_CONFIRM_SECONDS must be positive" >&2
  exit 2
fi
if ! [[ "${C57_EVAL_PREEMPT_POLL_SECONDS}" =~ ^[0-9]+$ ]] || \
   [[ "${C57_EVAL_PREEMPT_POLL_SECONDS}" -le 0 ]]; then
  echo "C57_EVAL_PREEMPT_POLL_SECONDS must be positive" >&2
  exit 2
fi

c56_parent_ready() {
  [[ -s "${C57_C56_GO_LONG}" && \
     -s "${C57_C56_PARENT_CHECKPOINT}" && \
     -s "${C57_C56_PARENT_READY}" ]]
}

c56_final_complete() {
  [[ -s "${C57_C56_FINAL_CHECKPOINT}" && -s "${C57_C56_FINAL_RESTORE}" ]] || return 1
  "${C57_EVAL_PYTHON}" - "${C57_C56_FINAL_CHECKPOINT}" "${C57_C56_FINAL_RESTORE}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()
report = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
valid = (
    report.get("status") == "PASS_C56B_STRICT_RESTORE"
    and report.get("restore_max_abs") == 0.0
    and Path(report.get("checkpoint", "")).resolve() == checkpoint
)
raise SystemExit(0 if valid else 1)
PY
}

higher_priority_reserved() {
  # GO_LONG alone is not a reservation. C56 cannot start until its C58b
  # s10000 parent and final strict-restore READY are both published.  Once
  # C56 itself has completed its bit-exact s10000 restore, the reservation is
  # released; otherwise a completed C56 would starve the C57 queue forever.
  pgrep -f '[t]rain_c56b_fact_online.py' >/dev/null 2>&1 || \
    { c56_parent_ready && ! c56_final_complete; }
}

gpu_compute_pids() {
  nvidia-smi -i "${C57_EVAL_PHYSICAL_GPU}" \
    --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d '[:space:]'
}

wait_for_idle_gpu() {
  while true; do
    if higher_priority_reserved; then
      sleep 30
    elif [[ -z "$(gpu_compute_pids)" ]]; then
      sleep "${C57_EVAL_IDLE_CONFIRM_SECONDS}"
      if ! higher_priority_reserved && [[ -z "$(gpu_compute_pids)" ]]; then
        return
      fi
    else
      sleep 30
    fi
  done
}

mkdir -p "${C57_EVAL_OUTPUT_DIR}"
for C57_EVAL_STEP in $(seq 200 200 5000); do
  C57_EVAL_CHECKPOINT=$(printf '%s/c57_step%05d.pt' "${C57_CHECKPOINT_DIR}" "${C57_EVAL_STEP}")
  C57_EVAL_REPORT=$(printf '%s/step%05d_paired.json' "${C57_EVAL_OUTPUT_DIR}" "${C57_EVAL_STEP}")
  while [[ ! -s "${C57_EVAL_CHECKPOINT}" ]]; do
    sleep 30
  done
  if [[ -s "${C57_EVAL_REPORT}" ]]; then
    continue
  fi
  while [[ ! -s "${C57_EVAL_REPORT}" ]]; do
    # The queue may sleep for many minutes between C57 checkpoints. Re-check
    # both the physical GPU and the C56 high-priority reservation immediately
    # before every restore/eval.
    wait_for_idle_gpu
    "${C57_EVAL_PYTHON}" scripts/h3wam/evaluate_c57_heldout_paired.py \
      --checkpoint "${C57_EVAL_CHECKPOINT}" \
      --plan "${C57_EVAL_PLAN}" \
      --cache-root "${C57_EVAL_CACHE_ROOT}" \
      --cache-source-manifest "${C57_EVAL_CACHE_SOURCE}" \
      --kv-subdir "${C57_EVAL_KV_SUBDIR}" \
      --device "${C57_EVAL_DEVICE}" \
      --output "${C57_EVAL_REPORT}" &
    C57_EVAL_CHILD_PID=$!
    C57_EVAL_PREEMPTED=0
    while kill -0 "${C57_EVAL_CHILD_PID}" 2>/dev/null; do
      if higher_priority_reserved; then
        echo "C56 priority reservation appeared; preempting C57 step ${C57_EVAL_STEP}" >&2
        kill -TERM "${C57_EVAL_CHILD_PID}" 2>/dev/null || true
        C57_EVAL_PREEMPTED=1
        break
      fi
      sleep "${C57_EVAL_PREEMPT_POLL_SECONDS}"
    done
    if [[ "${C57_EVAL_PREEMPTED}" -eq 1 ]]; then
      wait "${C57_EVAL_CHILD_PID}" 2>/dev/null || true
      continue
    fi
    if ! wait "${C57_EVAL_CHILD_PID}"; then
      echo "C57 evaluator failed at step ${C57_EVAL_STEP}" >&2
      exit 4
    fi
  done
done
