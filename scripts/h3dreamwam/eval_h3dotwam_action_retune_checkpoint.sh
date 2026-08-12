#!/usr/bin/env bash
set -euo pipefail

STEP="${1:?usage: eval_h3dotwam_action_retune_checkpoint.sh STEP}"
H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v2_full_cache"
CANDIDATE_ROOT="${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate"
MOTION_STAGE="${H3_WORKSPACE}/outputs/h3dotwam-motion/m4_motion_paperio_full50_gb128_s60_joint"
RUN_NAME="m9_motion_frozen_actionlr1e4_s60"
ACTION_STAGE="${H3_WORKSPACE}/outputs/h3dotwam-action-ablation/${RUN_NAME}_step$(printf '%06d' "${STEP}").pt"
# The training entry point writes periodic checkpoints before the final step,
# then writes the requested final stage without a step suffix.  Make the
# evaluator accept that canonical final artifact for the terminal checkpoint.
if [[ ! -s "${ACTION_STAGE}" && "${STEP}" == "60" ]]; then
  ACTION_STAGE="${H3_WORKSPACE}/outputs/h3dotwam-action-ablation/${RUN_NAME}.pt"
fi
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/eval-action-ablation/${RUN_NAME}/step$(printf '%03d' "${STEP}")"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-32409-action-eval"

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TMPDIR="${TMP_ROOT}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

test -s "${ACTION_STAGE}"
test -s "${MOTION_STAGE}/joint_stage.json"
mkdir -p "${OUTPUT_ROOT}" "${TMP_ROOT}"

if [[ ! -s "${OUTPUT_ROOT}/val40.json" ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
    --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
    --manifest "${CANDIDATE_ROOT}/manifest_val_stratified40.jsonl" \
    --output "${OUTPUT_ROOT}/val40.json" \
    --load-joint-stage "${MOTION_STAGE}" --load-stage "${ACTION_STAGE}" \
    --eval-only --steps 5 --sample-steps 10 --action-horizon 32 \
    --require-text-only-context --log-every 1 \
    > "${OUTPUT_ROOT}/val40.log" 2>&1
fi

if [[ ! -s "${OUTPUT_ROOT}/libero_goal_canary/results.json" ]]; then
  SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES:-/tmp/h3-wam-libero-site}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${PROJECT_ROOT}/scripts/h3wam/run_cloud_libero.sh" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3dreamwam/rollout_h3dreamwam_fsdp.py" \
    --dot --model "${MODEL_ROOT}" --action-stage "${ACTION_STAGE}" \
    --h3-joint-stage "${MOTION_STAGE}" --cache-root "${DATA_ROOT}" \
    --manifest "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
    --torchrun "${PROJECT_ROOT}/scripts/h3dreamwam/torchrun_shared.sh" \
    --suite libero_goal --task-ids 0 3 7 8 --trial-indices 0 \
    --max-steps 400 --wait-steps 30 --replan-steps 10 \
    --action-horizon 32 --sample-steps 10 \
    --output-dir "${OUTPUT_ROOT}/libero_goal_canary" \
    --save-video --save-trajectories --require-text-only-context \
    > "${OUTPUT_ROOT}/libero_goal_canary.log" 2>&1
fi
