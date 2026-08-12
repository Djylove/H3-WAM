#!/usr/bin/env bash
set -Eeuo pipefail

# Finalize only after all independent cache shards have populated the shared
# window directory.  This is intentionally a count gate rather than a torch
# distributed barrier so nodes may finish or restart independently.
H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
BASE_ROOT="${H3_WORKSPACE}/data/v7_multisuite_dense_base"
CACHE_ROOT="${H3_WORKSPACE}/data/v7_dense_h3_cache"
CANDIDATE_ROOT="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate"
STAGING_ROOT="${CANDIDATE_ROOT}.staging"
LOG_ROOT="${H3_WORKSPACE}/logs/dense-cache-shards"
EXPECTED=222929

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
mkdir -p "${LOG_ROOT}"

while true; do
  available=$(find "${CACHE_ROOT}/windows" -maxdepth 1 -type f -name '*.pt' | wc -l)
  printf '{"event":"dense_cache_wait","available":%s,"expected":%s}\n' \
    "${available}" "${EXPECTED}"
  (( available == EXPECTED )) && break
  (( available < EXPECTED )) || { echo "unexpected extra cache files" >&2; exit 1; }
  sleep 60
done

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3wam/precompute_libero_official_h3.py" \
  stats "${BASE_ROOT}/manifest_train.jsonl" --cache-root "${CACHE_ROOT}" \
  > "${LOG_ROOT}/dense_h3_stats.log" 2>&1

if [[ ! -s "${CANDIDATE_ROOT}/candidate_report.json" ]]; then
  if [[ -e "${STAGING_ROOT}" ]]; then
    mv "${STAGING_ROOT}" "${STAGING_ROOT}.abandoned.$(date +%s)"
  fi
  "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/h3dreamwam/prepare_multisuite_training_candidate.py" \
    --base-candidate "${BASE_ROOT}" --cache-root "${CACHE_ROOT}" \
    --output-dir "${STAGING_ROOT}" --target-total-repeats 1
  mv "${STAGING_ROOT}" "${CANDIDATE_ROOT}"
fi

if [[ ! -s "${CANDIDATE_ROOT}/manifest_val_stratified40.jsonl" ]]; then
  "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/h3dreamwam/build_stratified_eval_manifest.py" \
    "${CANDIDATE_ROOT}/manifest_val.jsonl" \
    "${CANDIDATE_ROOT}/manifest_val_stratified40.jsonl" --per-task 1
fi

printf '{"event":"dense_cache_complete","windows":%s}\n' "${EXPECTED}"
