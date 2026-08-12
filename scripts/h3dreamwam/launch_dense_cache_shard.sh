#!/usr/bin/env bash
set -Eeuo pipefail

# Independently encodes one node's eight episode shards into a shared cache.
# Existing windows are skipped, so this is safe to resume after interruption.
H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
RANK_OFFSET="${1:?usage: launch_dense_cache_shard.sh RANK_OFFSET [LOG_TAG]}"
LOG_TAG="${2:-offset${RANK_OFFSET}}"
WORLD_SIZE="${CACHE_WORLD_SIZE:-16}"
MANIFEST="${CACHE_MANIFEST:-${H3_WORKSPACE}/data/v7_multisuite_dense_base/manifest_all.jsonl}"
CACHE_ROOT="${CACHE_ROOT_OVERRIDE:-${H3_WORKSPACE}/data/v7_dense_h3_cache}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
LOG_ROOT="${H3_WORKSPACE}/logs/dense-cache-shards"
TMP_ROOT="${H3_WORKSPACE}/tmp/dense-cache-${LOG_TAG}"

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="${H3_WORKSPACE}/cache"
export HF_HOME="${H3_WORKSPACE}/cache/huggingface"
export TORCH_HOME="${H3_WORKSPACE}/cache/torch"
export TMPDIR="${TMP_ROOT}"

mkdir -p "${CACHE_ROOT}/windows" "${CACHE_ROOT}/contexts" "${LOG_ROOT}" "${TMP_ROOT}"
cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3wam/precompute_libero_official_h3.py" \
  vae "${MANIFEST}" --cache-root "${CACHE_ROOT}" --model "${MODEL_ROOT}" \
  --world-size "${WORLD_SIZE}" --rank-offset "${RANK_OFFSET}" \
  --vae-batch-size 4 --progress-every 250
