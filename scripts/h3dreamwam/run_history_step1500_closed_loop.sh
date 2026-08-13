#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${H3_WORKSPACE}/project"
PYTHON_BIN="${H3_WORKSPACE}/runtime/conda-py311/bin/python"
STAGE="${H3_WORKSPACE}/outputs/h3-lingbot-history/history16_from_s5000_s3000_step001500.pt"
OUTPUT_DIR="${H3_WORKSPACE}/outputs/eval-lingbot-history/history_step001500_goal_task3_trial0_replan16"
EVAL_LOCK="${H3_WORKSPACE}/tmp/h3-wam-eval-gpu.lock"

cd "${PROJECT_ROOT}"
test -s "${STAGE}"
if "${PYTHON_BIN}" scripts/h3dreamwam/check_completed_rollout.py "${OUTPUT_DIR}/results.json" 2>/dev/null; then
  exit 0
fi

mkdir -p "${OUTPUT_DIR}" "${H3_WORKSPACE}/tmp"
export PYTHON_BIN
export SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES:-/tmp/h3-wam-libero-site}"

exec 9>"${EVAL_LOCK}"
flock 9
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
