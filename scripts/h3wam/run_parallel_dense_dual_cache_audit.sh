#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/candidate-d0-rollout-96976ce/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/h3-int8-native/bin/python}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${H3_WORKSPACE}/data/v7_multisuite_dense_candidate}"
CACHE_ROOT="${CACHE_ROOT:-${H3_WORKSPACE}/data/v7_dense_h3_cache}"
KV_SUBDIR="${KV_SUBDIR:-h3_int8_dreamwam_kv_5x32_dense_v1}"
FEATURE_SUBDIR="${FEATURE_SUBDIR:-h3_int8_starwam_last32_dense_v1}"
AUDIT_ROOT="${AUDIT_ROOT:-${H3_WORKSPACE}/dense-d0-v1-96976ce/cache_generation/full_audit}"
AUDIT_NUM_SHARDS="${AUDIT_NUM_SHARDS:-32}"
PRODUCER_SHARDS="${PRODUCER_SHARDS:-32}"
H3_CHECKPOINT="${H3_CHECKPOINT:-${H3_WORKSPACE}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
H3_CHECKPOINT_SHA256="${H3_CHECKPOINT_SHA256:-e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a}"

if [[ ! "${AUDIT_NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AUDIT_NUM_SHARDS must be a positive integer" >&2
  exit 2
fi
MANIFEST="${CANDIDATE_ROOT}/manifest_all.jsonl"
REPORT_ROOT="${AUDIT_ROOT}/parallel_shards"
LOG_ROOT="${AUDIT_ROOT}/parallel_logs"
mkdir -p "${REPORT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

pids=()
for mode in kv star; do
  if [[ "${mode}" == "kv" ]]; then
    subdir="${KV_SUBDIR}"
  else
    subdir="${FEATURE_SUBDIR}"
  fi
  for ((index = 0; index < AUDIT_NUM_SHARDS; index++)); do
    "${PYTHON_BIN}" scripts/h3wam/audit_h3_dense_cache_shard.py \
      "${mode}" "${MANIFEST}" \
      --cache-root "${CACHE_ROOT}" --subdir "${subdir}" \
      --output "${REPORT_ROOT}/${mode}_shard$(printf '%02d' "${index}").json" \
      --audit-num-shards "${AUDIT_NUM_SHARDS}" --audit-shard-index "${index}" \
      --producer-num-shards "${PRODUCER_SHARDS}" \
      --expected-checkpoint "${H3_CHECKPOINT}" \
      --expected-checkpoint-sha256 "${H3_CHECKPOINT_SHA256}" \
      > "${LOG_ROOT}/${mode}_shard$(printf '%02d' "${index}").log" 2>&1 &
    pids+=("$!")
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=$((failed + 1))
  fi
done
if (( failed > 0 )); then
  echo "${failed} cache audit shard processes failed" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/h3wam/reduce_h3_dense_cache_audits.py \
  "${MANIFEST}" --cache-root "${CACHE_ROOT}" \
  --kv-subdir "${KV_SUBDIR}" --feature-subdir "${FEATURE_SUBDIR}" \
  --report-root "${REPORT_ROOT}" --audit-num-shards "${AUDIT_NUM_SHARDS}" \
  --expected-checkpoint "${H3_CHECKPOINT}" \
  --expected-checkpoint-sha256 "${H3_CHECKPOINT_SHA256}" \
  --output "${AUDIT_ROOT}/dense_dual_cache_full.json" \
  --ready-marker "${AUDIT_ROOT}/DUAL_CACHE_AUDIT_READY.json"
