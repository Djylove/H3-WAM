#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
BASE_ROOT="${H3_WORKSPACE}/data/v8_multisuite_frameindexed_base"
CACHE_ROOT="${H3_WORKSPACE}/data/v8_frameindexed_h3_cache"
CANDIDATE_ROOT="${H3_WORKSPACE}/data/v8_multisuite_frameindexed_candidate"
STAGING_ROOT="${CANDIDATE_ROOT}.staging"
LOG_ROOT="${H3_WORKSPACE}/logs/frameindexed-cache-shards"
EXPECTED=277713

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
mkdir -p "${LOG_ROOT}"
until [[ -e "${CACHE_ROOT}/.seeded_from_v7" ]]; do sleep 30; done
while true; do
  available=$(find "${CACHE_ROOT}/windows" -maxdepth 1 -type f -name '*.pt' | wc -l)
  printf '{"event":"frameindexed_cache_wait","available":%s,"expected":%s}\n' \
    "${available}" "${EXPECTED}"
  (( available == EXPECTED )) && break
  (( available < EXPECTED )) || { echo "unexpected extra cache files" >&2; exit 1; }
  sleep 60
done

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3wam/precompute_libero_official_h3.py" \
  stats "${BASE_ROOT}/manifest_train.jsonl" --cache-root "${CACHE_ROOT}" \
  > "${LOG_ROOT}/frameindexed_h3_stats.log" 2>&1

if [[ ! -s "${CANDIDATE_ROOT}/candidate_report.json" ]]; then
  if [[ -e "${STAGING_ROOT}" ]]; then
    mv "${STAGING_ROOT}" "${STAGING_ROOT}.abandoned.$(date +%s)"
  fi
  "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/h3dreamwam/prepare_multisuite_training_candidate.py" \
    --base-candidate "${BASE_ROOT}" --cache-root "${CACHE_ROOT}" \
    --output-dir "${STAGING_ROOT}" --target-total-repeats 1
  "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/h3dreamwam/build_stratified_eval_manifest.py" \
    "${STAGING_ROOT}/manifest_train_uniform.jsonl" \
    "${STAGING_ROOT}/manifest_eval_stratified40.jsonl" --per-task 1
  mv "${STAGING_ROOT}" "${CANDIDATE_ROOT}"
fi

"${PYTHON_BIN}" - "${BASE_ROOT}" "${CANDIDATE_ROOT}" <<'PY'
import json, pathlib, sys
base, candidate = map(pathlib.Path, sys.argv[1:])
base_report = json.loads((base / "candidate_report.json").read_text())
report = json.loads((candidate / "candidate_report.json").read_text())
rows = sum(1 for line in (candidate / "manifest_train_uniform.jsonl").open() if line.strip())
assert base_report["window_sampling"] == "frame_indexed_padded"
assert base_report["all_train"] and base_report["windows"] == 277713
assert report["contract"]["window_sampling"] == "frame_indexed_padded"
assert rows == 277713
print(json.dumps({"event": "frameindexed_cache_complete", "windows": rows}))
PY
