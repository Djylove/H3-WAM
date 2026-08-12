#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
CANDIDATE="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate/candidate_report.json"
STATS="${H3_WORKSPACE}/data/v7_dense_h3_cache/stats.pt"

until [[ -s "${CANDIDATE}" && -s "${STATS}" ]]; do sleep 15; done

# These are only B-node data preparation helpers. A/C/D continue the padded
# tail cache independently; do not terminate any remote producer.
gap=$(pgrep -f "^/mnt/h3-wam/runtime/conda-py311/bin/python -m torch.distributed.run .*precompute_libero_official_h3.py vae /mnt/h3-wam/data/v7_multisuite_dense_base/manifest_all.jsonl .*--world-size 8 --rank-offset 0" || true)
[[ -z "${gap}" ]] || kill -TERM ${gap}
tail_waiter=$(pgrep -f "^bash ${PROJECT_ROOT}/scripts/h3dreamwam/chain_frameindexed_cache_shard.sh 24 node-b-frameindexed$" || true)
[[ -z "${tail_waiter}" ]] || kill -TERM ${tail_waiter}

exec bash "${PROJECT_ROOT}/scripts/h3dreamwam/launch_m13_dense_full_epoch.sh"
