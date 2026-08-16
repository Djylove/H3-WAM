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

if ! [[ "${C57_EVAL_PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
  echo "C57_EVAL_PHYSICAL_GPU must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "${C57_EVAL_IDLE_CONFIRM_SECONDS}" =~ ^[0-9]+$ ]] || \
   [[ "${C57_EVAL_IDLE_CONFIRM_SECONDS}" -le 0 ]]; then
  echo "C57_EVAL_IDLE_CONFIRM_SECONDS must be positive" >&2
  exit 2
fi

gpu_compute_pids() {
  nvidia-smi -i "${C57_EVAL_PHYSICAL_GPU}" \
    --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d '[:space:]'
}

wait_for_idle_gpu() {
  while true; do
    if [[ -z "$(gpu_compute_pids)" ]]; then
      sleep "${C57_EVAL_IDLE_CONFIRM_SECONDS}"
      if [[ -z "$(gpu_compute_pids)" ]]; then
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
  # The queue may sleep for many minutes between C57 checkpoints.  Re-check
  # the physical GPU immediately before every restore/eval so a newly launched
  # C56/C58 job cannot be silently colocated.
  wait_for_idle_gpu
  "${C57_EVAL_PYTHON}" scripts/h3wam/evaluate_c57_heldout_paired.py \
    --checkpoint "${C57_EVAL_CHECKPOINT}" \
    --plan "${C57_EVAL_PLAN}" \
    --cache-root "${C57_EVAL_CACHE_ROOT}" \
    --cache-source-manifest "${C57_EVAL_CACHE_SOURCE}" \
    --kv-subdir "${C57_EVAL_KV_SUBDIR}" \
    --device "${C57_EVAL_DEVICE}" \
    --output "${C57_EVAL_REPORT}"
done
