#!/usr/bin/env bash
set -euo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/home/h3wam_finetune}"
export H3_WORKSPACE
export H3_RUN_NAME="${H3_RUN_NAME:-v1_libero_goal_last2_500}"
export H3_STEPS="${H3_STEPS:-500}"
export H3_LAST_BLOCKS="${H3_LAST_BLOCKS:-2}"
export H3_LR="${H3_LR:-1e-6}"
export H3_SEED="${H3_SEED:-2026}"
export H3_DATA_ROOT="${H3_DATA_ROOT:-${H3_WORKSPACE}/data/v1/cache}"
export H3_MANIFEST="${H3_MANIFEST:-${H3_WORKSPACE}/data/v1/candidate/manifest_train.jsonl}"
export H3_VALIDATION_MANIFEST="${H3_VALIDATION_MANIFEST:-${H3_WORKSPACE}/data/v1/candidate/manifest_val.jsonl}"
export H3_VALIDATION_EVERY="${H3_VALIDATION_EVERY:-100}"
export H3_VALIDATION_BATCHES_PER_RANK="${H3_VALIDATION_BATCHES_PER_RANK:-1}"
export H3_CHECKPOINT_EVERY="${H3_CHECKPOINT_EVERY:-100}"
export H3_LOG_EVERY="${H3_LOG_EVERY:-10}"

exec bash "${H3_WORKSPACE}/project/scripts/h3wam/launch_h3_bf16_v0.sh"
