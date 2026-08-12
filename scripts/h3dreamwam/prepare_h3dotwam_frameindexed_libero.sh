#!/usr/bin/env bash
set -Eeuo pipefail

# Exact public FastWAM/DreamWAM LIBERO sampling population: every raw frame is
# a start, tail data are repeated only for tensor construction and ignored by
# action/video masks, and all 1,712 demonstrations participate in training.
H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
EXTRACTED_ROOT="${H3_WORKSPACE}/data/libero_fastwam_extracted"
DENSE_CACHE="${H3_WORKSPACE}/data/v7_dense_h3_cache"
DENSE_CANDIDATE="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate"
BASE_ROOT="${H3_WORKSPACE}/data/v8_multisuite_frameindexed_base"
CACHE_ROOT="${H3_WORKSPACE}/data/v8_frameindexed_h3_cache"
CANDIDATE_ROOT="${H3_WORKSPACE}/data/v8_multisuite_frameindexed_candidate"
STAGING_ROOT="${CANDIDATE_ROOT}.staging"
LOG_ROOT="${H3_WORKSPACE}/logs/cluster-30234"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-30234-frameindexed-cache"

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="${H3_WORKSPACE}/cache"
export HF_HOME="${H3_WORKSPACE}/cache/huggingface"
export TORCH_HOME="${H3_WORKSPACE}/cache/torch"
export TMPDIR="${TMP_ROOT}"

mkdir -p "${BASE_ROOT}" "${CACHE_ROOT}/windows" "${CACHE_ROOT}/contexts" \
  "${LOG_ROOT}" "${TMP_ROOT}"
cd "${PROJECT_ROOT}"
if [[ ! -s "${BASE_ROOT}/candidate_report.json" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3wam/prepare_libero_full_candidate.py" \
    --dataset "libero_10=${EXTRACTED_ROOT}/libero_10_no_noops_lerobot" \
    --dataset "libero_goal=${EXTRACTED_ROOT}/libero_goal_no_noops_lerobot" \
    --dataset "libero_object=${EXTRACTED_ROOT}/libero_object_no_noops_lerobot" \
    --dataset "libero_spatial=${EXTRACTED_ROOT}/libero_spatial_no_noops_lerobot" \
    --output-dir "${BASE_ROOT}" --frame-indexed --all-train
fi

# Wait for the no-tail cache, then hard-link all 222,929 shared window IDs.
until [[ -s "${DENSE_CANDIDATE}/candidate_report.json" ]]; do sleep 30; done
cp -aln "${DENSE_CACHE}/windows/." "${CACHE_ROOT}/windows/"
cp -aln "${DENSE_CACHE}/contexts/." "${CACHE_ROOT}/contexts/"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3wam/precompute_libero_official_h3.py" \
  vae "${BASE_ROOT}/manifest_all.jsonl" --cache-root "${CACHE_ROOT}" \
  --model "${MODEL_ROOT}" --world-size 8 --vae-batch-size 4 \
  --progress-every 250 > "${LOG_ROOT}/frameindexed_h3_vae.log" 2>&1

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3wam/precompute_libero_official_h3.py" \
  stats "${BASE_ROOT}/manifest_train.jsonl" --cache-root "${CACHE_ROOT}" \
  > "${LOG_ROOT}/frameindexed_h3_stats.log" 2>&1

if [[ ! -s "${CANDIDATE_ROOT}/candidate_report.json" ]]; then
  rm -rf "${STAGING_ROOT}"
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

"${PYTHON_BIN}" - "${BASE_ROOT}" "${CACHE_ROOT}" <<'PY'
import json, pathlib, sys
base, cache = map(pathlib.Path, sys.argv[1:])
report = json.loads((base / "candidate_report.json").read_text())
available = sum(1 for _ in (cache / "windows").glob("*.pt"))
assert report["window_sampling"] == "frame_indexed_padded"
assert report["all_train"] and report["windows"] == 277713
assert available == 277713, (available, report["windows"])
print(json.dumps({"event": "frameindexed_cache_complete", "windows": available}))
PY
