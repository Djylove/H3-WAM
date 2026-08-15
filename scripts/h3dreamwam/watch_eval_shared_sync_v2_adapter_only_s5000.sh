#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
export RUN_NAME="shared_sync_v2_adapter_only_s5000_fresh"
export STAGE_ROOT="${H3_WORKSPACE}/outputs/h3-lingbot-shared-sync-v2-adapter-only-s5000-fresh"
export RESULT_ROOT="${H3_WORKSPACE}/outputs/eval-h3-lingbot-shared-sync-v2-adapter-only-s5000-fresh"
export FREEZE_SHARED_BLOCKS=1
export EVAL_STEPS="500 1000 1500 2000 2500 3000 3500 4000 4500"
export FINAL_STEP=5000
exec bash scripts/h3dreamwam/watch_eval_shared_sync_v2_s1000.sh
