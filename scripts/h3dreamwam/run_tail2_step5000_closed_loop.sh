#!/usr/bin/env bash
set -euo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${H3_WORKSPACE}/project"
PYTHON_BIN="${H3_WORKSPACE}/runtime/conda-py311/bin/python"
STAGE="${H3_WORKSPACE}/outputs/h3-lingbot-shared/quantile_flowweight_lr1e5_tail2_s10000_step005000.pt"
OUTPUT_DIR="${H3_WORKSPACE}/outputs/eval-lingbot-shared/scale-s10000/tail2_step05000_goal_task3_trial0_replan16"

cd "${PROJECT_ROOT}"
test -s "${STAGE}"
if "${PYTHON_BIN}" scripts/h3dreamwam/check_completed_rollout.py "${OUTPUT_DIR}/results.json" 2>/dev/null; then
  exit 0
fi

export PYTHON_BIN
export SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES:-/tmp/h3-wam-libero-site}"
bash scripts/h3wam/run_cloud_libero.sh \
  "${PYTHON_BIN}" scripts/h3dreamwam/rollout_h3dreamwam_fsdp.py \
  --lingbot-shared \
  --lingbot-shared-stage "${STAGE}" \
  --model "${H3_WORKSPACE}/models/MiniMax-H3" \
  --cache-root "${H3_WORKSPACE}/data/v7_dense_h3_cache" \
  --manifest "${H3_WORKSPACE}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl" \
  --torchrun scripts/h3dreamwam/torchrun_shared.sh \
  --suite libero_goal \
  --task-ids 3 \
  --trial-indices 0 \
  --max-steps 80 \
  --wait-steps 30 \
  --replan-steps 16 \
  --action-horizon 32 \
  --sample-steps 4 \
  --video-sample-steps 4 \
  --action-sample-steps 4 \
  --output-dir "${OUTPUT_DIR}" \
  --save-video \
  --save-trajectories
