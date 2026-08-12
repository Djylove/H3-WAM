#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
BASE_ROOT="${H3_WORKSPACE}/data/v7_multisuite_dense_base"
DENSE_CACHE="${H3_WORKSPACE}/data/v7_dense_h3_cache"
SPARSE_CACHE="${H3_WORKSPACE}/data/v2_full_cache"
CANARY_CACHE="${H3_WORKSPACE}/data/v7_dense_canary_cache"
CANARY_CANDIDATE="${H3_WORKSPACE}/data/v7_dense_canary_candidate"

until "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3dreamwam/build_cached_dense_canary.py" \
  --base-root "${BASE_ROOT}" --cache-root "${DENSE_CACHE}" \
  --sparse-cache-root "${SPARSE_CACHE}" --output-dir "${CANARY_CANDIDATE}" \
  --per-task 128 --allow-sparse-fill; do
  sleep 30
done

mkdir -p "${CANARY_CACHE}"
ln -sfn "${DENSE_CACHE}/windows" "${CANARY_CACHE}/windows"
ln -sfn "${DENSE_CACHE}/contexts" "${CANARY_CACHE}/contexts"
cp "${SPARSE_CACHE}/stats.pt" "${CANARY_CACHE}/stats.pt"

exec bash "${PROJECT_ROOT}/scripts/h3dreamwam/launch_h3dotwam_dense_canary.sh"
