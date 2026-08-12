#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
CANARY="${H3_WORKSPACE}/outputs/h3dotwam-dense/m10_dense_canary_head_gb128_s80.pt"

until [[ -s "${CANARY}" ]]; do sleep 15; done
exec bash "${PROJECT_ROOT}/scripts/h3dreamwam/launch_dense_cache_shard.sh" 8 node-c-offset8
