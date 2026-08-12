#!/usr/bin/env bash
set -euo pipefail

# Freeze the learned motion/RGB H3 world model and retune only DoT action-side
# modules.  This isolates whether the current bottleneck is action optimization
# rather than world representation.  Periodic checkpoints are action-only and
# therefore small; the H3 shards remain those of MOTION_STAGE.
H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v2_full_cache"
CANDIDATE_ROOT="${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate"
TRAIN_MANIFEST="${CANDIDATE_ROOT}/manifest_train_uniform.jsonl"
VAL_MANIFEST="${CANDIDATE_ROOT}/manifest_val_stratified40.jsonl"
MOTION_STAGE="${H3_WORKSPACE}/outputs/h3dotwam-motion/m4_motion_paperio_full50_gb128_s60_joint"
RUN_NAME="${RUN_NAME:-m9_motion_frozen_actionlr1e4_s60}"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3dotwam-action-ablation"
EVAL_ROOT="${H3_WORKSPACE}/outputs/eval-action-ablation/${RUN_NAME}"
LOG_ROOT="${H3_WORKSPACE}/logs/cluster-30234"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-30234"
FINAL_STAGE="${OUTPUT_ROOT}/${RUN_NAME}.pt"

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="${H3_WORKSPACE}/cache"
export HF_HOME="${H3_WORKSPACE}/cache/huggingface"
export TORCH_HOME="${H3_WORKSPACE}/cache/torch"
export TMPDIR="${TMP_ROOT}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

test -x "${PYTHON_BIN}"
test -s "${MOTION_STAGE}/joint_stage.json"
test -s "${MOTION_STAGE}/action_stage.pt"
test ! -e "${FINAL_STAGE}"
mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}" "${LOG_ROOT}" "${TMP_ROOT}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
  --manifest "${TRAIN_MANIFEST}" \
  --output "${OUTPUT_ROOT}/${RUN_NAME}.json" \
  --load-joint-stage "${MOTION_STAGE}" --save-stage "${FINAL_STAGE}" \
  --checkpoint-every 20 --steps 60 --gradient-accumulation-steps 16 \
  --action-horizon 32 --learning-rate 1e-4 --h3-learning-rate 1e-6 \
  --last-h3-blocks 0 --video-loss-weight 1.0 --language-ranking-weight 0 \
  --lr-schedule constant --require-text-only-context --log-every 1 \
  > "${LOG_ROOT}/${RUN_NAME}.log" 2>&1

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
  --manifest "${VAL_MANIFEST}" --output "${EVAL_ROOT}/val40.json" \
  --load-joint-stage "${MOTION_STAGE}" --load-stage "${FINAL_STAGE}" \
  --eval-only --steps 5 --sample-steps 10 --action-horizon 32 \
  --require-text-only-context --log-every 1 \
  > "${EVAL_ROOT}/val40.log" 2>&1

SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES:-/tmp/h3-wam-libero-site}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${PROJECT_ROOT}/scripts/h3wam/run_cloud_libero.sh" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3dreamwam/rollout_h3dreamwam_fsdp.py" \
  --dot --model "${MODEL_ROOT}" --action-stage "${FINAL_STAGE}" \
  --h3-joint-stage "${MOTION_STAGE}" --cache-root "${DATA_ROOT}" \
  --manifest "${TRAIN_MANIFEST}" \
  --torchrun "${PROJECT_ROOT}/scripts/h3dreamwam/torchrun_shared.sh" \
  --suite libero_goal --task-ids 0 3 7 8 --trial-indices 0 \
  --max-steps 400 --wait-steps 30 --replan-steps 10 \
  --action-horizon 32 --sample-steps 10 \
  --output-dir "${EVAL_ROOT}/libero_goal_canary" \
  --save-video --save-trajectories --require-text-only-context \
  > "${EVAL_ROOT}/libero_goal_canary.log" 2>&1
