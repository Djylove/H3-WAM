#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${H3_WORKSPACE}/project-adapter-sync-v2"
PYTHON_BIN="${H3_WORKSPACE}/runtime/conda-py311/bin/python"
STAGE="${H3_WORKSPACE}/outputs/h3-lingbot-shared-sync-v2-adapter-only-s5000-fresh/shared_sync_v2_adapter_only_s5000_fresh_step002000.pt"
OUTPUT_DIR="${H3_WORKSPACE}/outputs/eval-h3-lingbot-shared-sync-v2-adapter-only-s5000-fresh/step002000_goal_task3_trial0_full400_replan32"
EVAL_LOCK="/tmp/h3-wam-eval-gpu.lock"

cd "${PROJECT_ROOT}"
test -s "${STAGE}"
if "${PYTHON_BIN}" scripts/h3dreamwam/check_completed_rollout.py "${OUTPUT_DIR}/results.json" 2>/dev/null; then
  exit 0
fi

mkdir -p "${OUTPUT_DIR}" "${H3_WORKSPACE}/tmp"
export PYTHON_BIN
export SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES:-/tmp/h3-wam-libero-site}"
export DIFFUSERS_H3_ROOT="${DIFFUSERS_H3_ROOT:-${H3_WORKSPACE}/project/third_party/diffusers_h3}"

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
  --max-steps 400 \
  --wait-steps 30 \
  --replan-steps 32 \
  --action-horizon 32 \
  --actions-per-chunk 4 \
  --target-latent-frames 12 \
  --last-trainable-layers 2 \
  --sample-steps 4 \
  --video-sample-steps 4 \
  --action-sample-steps 4 \
  --clip-normalized-actions \
  --output-dir "${OUTPUT_DIR}" \
  --save-video \
  --save-trajectories
