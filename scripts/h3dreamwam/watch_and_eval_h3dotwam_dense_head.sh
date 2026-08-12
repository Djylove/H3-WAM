#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
RUN_NAME="m10_dense_head_gb128_s1569"
STAGE_ROOT="${H3_WORKSPACE}/outputs/h3dotwam-dense"
LOG_ROOT="${H3_WORKSPACE}/logs/cluster-30907"
mkdir -p "${LOG_ROOT}"

# Early checkpoints are enough to expose whether dense supervision translates
# to control.  The final checkpoint is evaluated by the training launcher.
for step in 200 400 800 1200; do
  stage="${STAGE_ROOT}/${RUN_NAME}_step$(printf '%06d' "${step}").pt"
  until [[ -s "${stage}" ]]; do sleep 30; done
  bash "${PROJECT_ROOT}/scripts/h3dreamwam/eval_h3dotwam_dense_head_checkpoint.sh" \
    "${stage}" "step$(printf '%04d' "${step}")" \
    > "${LOG_ROOT}/dense_head_step$(printf '%04d' "${step}")_eval.log" 2>&1
done
