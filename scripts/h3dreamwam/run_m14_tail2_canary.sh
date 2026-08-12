#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v7_dense_h3_cache"
CANDIDATE_ROOT="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate"
BASE_STAGE="${H3_WORKSPACE}/outputs/h3dotwam-dense/m13_dense_full_head_gb128_s1569_step000400.pt"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3dotwam-tail2"
RUN_NAME="m14_tail2_from_m13s400_gb128_s40"
JOINT_STAGE="${OUTPUT_ROOT}/${RUN_NAME}_joint"
EVAL_ROOT="${H3_WORKSPACE}/outputs/eval-tail2-dot/${RUN_NAME}"
TMP_ROOT="${H3_WORKSPACE}/tmp/${RUN_NAME}"

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="${H3_WORKSPACE}/cache"
export HF_HOME="${H3_WORKSPACE}/cache/huggingface"
export TORCH_HOME="${H3_WORKSPACE}/cache/torch"
export TMPDIR="${TMP_ROOT}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

test -s "${BASE_STAGE}"
test ! -e "${JOINT_STAGE}"
mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}" "${TMP_ROOT}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
  --manifest "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
  --output "${OUTPUT_ROOT}/${RUN_NAME}.json" --load-stage "${BASE_STAGE}" \
  --save-joint-stage "${JOINT_STAGE}" --steps 40 \
  --sample-offset 51200 --gradient-accumulation-steps 16 --action-horizon 32 \
  --learning-rate 1e-5 --h3-learning-rate 2e-6 --last-h3-blocks 2 \
  --video-loss-weight 1.0 --language-ranking-weight 0 \
  --lr-schedule cosine --require-text-only-context --log-every 1 \
  > "${OUTPUT_ROOT}/${RUN_NAME}.log" 2>&1

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
  --manifest "${CANDIDATE_ROOT}/manifest_val_stratified40.jsonl" \
  --output "${EVAL_ROOT}/val40.json" --load-joint-stage "${JOINT_STAGE}" \
  --eval-only --steps 5 --sample-steps 10 --action-horizon 32 \
  --last-h3-blocks 2 --require-text-only-context --log-every 1 \
  > "${EVAL_ROOT}/val40.log" 2>&1

# Only spend simulator time if the tail adaptation beats its step-400 parent.
"${PYTHON_BIN}" - "${EVAL_ROOT}/val40.json" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1]))["mean_action_loss"]
if value >= 0.12231154786422849:
    raise SystemExit(10)
PY

SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES:-/tmp/h3-wam-libero-site}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${PROJECT_ROOT}/scripts/h3wam/run_cloud_libero.sh" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3dreamwam/rollout_h3dreamwam_fsdp.py" \
  --dot --model "${MODEL_ROOT}" --action-stage "${JOINT_STAGE}/action_stage.pt" \
  --h3-joint-stage "${JOINT_STAGE}" --cache-root "${DATA_ROOT}" \
  --manifest "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
  --torchrun "${PROJECT_ROOT}/scripts/h3dreamwam/torchrun_shared.sh" \
  --suite libero_goal --task-ids 3 --trial-indices 0 1 2 \
  --max-steps 400 --wait-steps 30 --replan-steps 5 --action-horizon 32 \
  --sample-steps 10 --fixed-noise-seed 42 \
  --output-dir "${EVAL_ROOT}/libero_goal_task3" \
  --save-video --save-trajectories --require-text-only-context \
  > "${EVAL_ROOT}/libero_goal_task3.log" 2>&1
