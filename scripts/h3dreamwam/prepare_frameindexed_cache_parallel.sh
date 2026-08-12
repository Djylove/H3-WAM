#!/usr/bin/env bash
set -Eeuo pipefail

# Seed the all-frame cache from the complete no-tail cache, then encode shard
# ranks 0..7.  Other nodes wait for the seed marker before taking ranks 8..23.
H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
DENSE_CANDIDATE="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate/candidate_report.json"
DENSE_CACHE="${H3_WORKSPACE}/data/v7_dense_h3_cache"
BASE_ROOT="${H3_WORKSPACE}/data/v8_multisuite_frameindexed_base"
CACHE_ROOT="${H3_WORKSPACE}/data/v8_frameindexed_h3_cache"
MARKER="${CACHE_ROOT}/.seeded_from_v7"

until [[ -s "${DENSE_CANDIDATE}" ]]; do sleep 30; done
mkdir -p "${CACHE_ROOT}/windows" "${CACHE_ROOT}/contexts"
if [[ ! -e "${MARKER}" ]]; then
  cp -aln "${DENSE_CACHE}/windows/." "${CACHE_ROOT}/windows/"
  cp -aln "${DENSE_CACHE}/contexts/." "${CACHE_ROOT}/contexts/"
  touch "${MARKER}"
fi

export CACHE_WORLD_SIZE="${FRAMEINDEXED_CACHE_WORLD_SIZE:-32}"
export CACHE_MANIFEST="${BASE_ROOT}/manifest_all.jsonl"
export CACHE_ROOT_OVERRIDE="${CACHE_ROOT}"
exec bash "${PROJECT_ROOT}/scripts/h3dreamwam/launch_dense_cache_shard.sh" \
  0 frameindexed-node-d-offset0
