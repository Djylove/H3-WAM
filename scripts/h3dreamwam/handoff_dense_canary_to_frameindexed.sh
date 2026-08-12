#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
CANARY="${H3_WORKSPACE}/outputs/h3dotwam-dense/m10_dense_canary_head_gb128_s80.pt"

until [[ -s "${CANARY}" ]]; do sleep 30; done
# Remove only the superseded no-tail one-epoch waiter if an older canary shell
# launched it.  The cache producer and all completed artifacts remain intact.
old=$(pgrep -f "bash ${PROJECT_ROOT}/scripts/h3dreamwam/launch_h3dotwam_dense_head_epoch.sh" || true)
[[ -z "${old}" ]] || kill -TERM ${old}
exec bash "${PROJECT_ROOT}/scripts/h3dreamwam/launch_h3dotwam_frameindexed_head_epoch.sh"
