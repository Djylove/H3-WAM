#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
OFFSET="${1:?usage: chain_frameindexed_cache_shard.sh OFFSET TAG}"
TAG="${2:?usage: chain_frameindexed_cache_shard.sh OFFSET TAG}"
DENSE_CANDIDATE="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate/candidate_report.json"
BASE_ROOT="${H3_WORKSPACE}/data/v8_multisuite_frameindexed_base"
CACHE_ROOT="${H3_WORKSPACE}/data/v8_frameindexed_h3_cache"

until [[ -s "${DENSE_CANDIDATE}" && -e "${CACHE_ROOT}/.seeded_from_v7" ]]; do
  sleep 30
done
export CACHE_WORLD_SIZE="${FRAMEINDEXED_CACHE_WORLD_SIZE:-32}"
export CACHE_MANIFEST="${BASE_ROOT}/manifest_all.jsonl"
export CACHE_ROOT_OVERRIDE="${CACHE_ROOT}"
exec bash "${PROJECT_ROOT}/scripts/h3dreamwam/launch_dense_cache_shard.sh" \
  "${OFFSET}" "${TAG}"
