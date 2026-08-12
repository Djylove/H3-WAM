#!/usr/bin/env bash
set -Eeuo pipefail

# First run that actually consumes the complete episode-disjoint dense LIBERO
# population. It starts from the proven mid256 head instead of replaying the
# original sparse initialization.
H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
export RUN_NAME="m13_dense_full_head_gb128_s1569"
export INITIAL_STAGE="${H3_WORKSPACE}/outputs/h3dotwam-dense/m12_dense_mid256_head_gb128_s160.pt"
exec bash "${PROJECT_ROOT}/scripts/h3dreamwam/launch_h3dotwam_dense_head_epoch.sh"
