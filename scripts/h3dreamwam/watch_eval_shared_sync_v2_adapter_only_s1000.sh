#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
export RUN_NAME="shared_sync_v2_adapter_only_s1000"
export STAGE_ROOT="${H3_WORKSPACE}/outputs/h3-lingbot-shared-sync-v2-adapter-only"
export RESULT_ROOT="${H3_WORKSPACE}/outputs/eval-h3-lingbot-shared-sync-v2-adapter-only"
export FREEZE_SHARED_BLOCKS=1
exec bash scripts/h3dreamwam/watch_eval_shared_sync_v2_s1000.sh
