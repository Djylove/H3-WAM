#!/usr/bin/env bash
set -Eeuo pipefail

# Matched C56+C61 arm: all optimizer/model/parent/seed/rank-mixture fields are
# inherited byte-for-byte from the C56 launcher.  The only changed input is the
# causal_failure pool, released by the strict C61 final COMPLETED.json.
workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
c61_ready="${C61_COMPLETED:-${workspace}/eval/c61-finalized-fact-failure-dataset-v1/COMPLETED.json}"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c56b-fact-online-v1/online-long10000-c61-matched-v1}"
arm_root="${ARM_ROOT:-${workspace}/outputs/c56b-fact-online-v1/arm-c61-matched-v1}"

cd "${project}"
exec env CAUSAL_FAILURE_READY="${c61_ready}" OUTPUT_ROOT="${output_root}" \
  ARM_ROOT="${arm_root}" bash scripts/h3wam/watch_and_launch_c56b_after_c58b_final.sh
