#!/usr/bin/env bash
set -euo pipefail

H3_ROOT="/home/h3wam_finetune"
PROJECT_ROOT="${H3_ROOT}/project"
RUN_NAME="m2_language_rank_full50_gb128_s5"
TRAIN_PID_FILE="${H3_ROOT}/logs/h3dotwam/${RUN_NAME}.pid"
TRAIN_REPORT="${H3_ROOT}/outputs/h3dotwam/${RUN_NAME}.json"
JOINT_STAGE="${H3_ROOT}/outputs/h3dotwam/${RUN_NAME}_joint"
POST_ROOT="${H3_ROOT}/outputs/h3dotwam/${RUN_NAME}_posttrain"
MODEL_ROOT="${H3_ROOT}/models/MiniMax-H3"
DATA_ROOT="${H3_ROOT}/data/v2_full_cache"
VAL_MANIFEST="${H3_ROOT}/data/v4_multisuite_uniform_candidate/manifest_val_stratified40.jsonl"
TRAIN_MANIFEST="${H3_ROOT}/data/v4_multisuite_uniform_candidate/manifest_train_uniform.jsonl"
COUNTERFACTUAL_MANIFEST="${PROJECT_ROOT}/experiments/h3dotwam/counterfactual_task0_val.jsonl"
TORCHRUN="${H3_ROOT}/.venv/bin/torchrun"

mkdir -p "${POST_ROOT}"
TRAIN_PID="$(cat "${TRAIN_PID_FILE}")"
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep 30
done
test -s "${TRAIN_REPORT}"
test -s "${JOINT_STAGE}/joint_stage.json"
test -s "${JOINT_STAGE}/action_stage.pt"

PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" "${TORCHRUN}" \
  --standalone --nproc-per-node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
  --manifest "${VAL_MANIFEST}" --output "${POST_ROOT}/val40.json" \
  --load-joint-stage "${JOINT_STAGE}" \
  --eval-only --steps 5 --sample-steps 10 --action-horizon 32 \
  --require-text-only-context --log-every 1 \
  > "${POST_ROOT}/val40.log" 2>&1

for SPEC in correct: stove:task_c0d1b2f3264d13ce; do
  NAME="${SPEC%%:*}"
  CONTEXT_ID="${SPEC#*:}"
  EXTRA=()
  if [[ -n "${CONTEXT_ID}" ]]; then
    EXTRA=(--context-override-id "${CONTEXT_ID}")
  fi
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" "${TORCHRUN}" \
    --standalone --nproc-per-node=8 \
    "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
    --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
    --manifest "${COUNTERFACTUAL_MANIFEST}" \
    --output "${POST_ROOT}/counterfactual_${NAME}.json" \
    --load-joint-stage "${JOINT_STAGE}" \
    --eval-only --steps 1 --sample-steps 10 --action-horizon 32 \
    --require-text-only-context --record-sampled-actions "${EXTRA[@]}" \
    > "${POST_ROOT}/counterfactual_${NAME}.log" 2>&1
done

"${H3_ROOT}/.venv/bin/python" - "${POST_ROOT}" <<'PY'
import json
import pathlib
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
validation = json.loads((root / "val40.json").read_text())
correct = np.asarray(
    json.loads((root / "counterfactual_correct.json").read_text())["history"][0]["sampled_actions"],
    dtype=np.float64,
)
stove = np.asarray(
    json.loads((root / "counterfactual_stove.json").read_text())["history"][0]["sampled_actions"],
    dtype=np.float64,
)
cosine = float(
    np.dot(correct.ravel(), stove.ravel())
    / (np.linalg.norm(correct) * np.linalg.norm(stove))
)
(root / "analysis.json").write_text(
    json.dumps(
        {
            "head_only_val40_mse": 0.21044312305748464,
            "m1_joint_val40_mse": 0.20943731348961592,
            "m2_language_rank_val40_mse": validation["mean_action_loss"],
            "counterfactual_cosine": cosine,
            "counterfactual_rms": float(np.sqrt(np.mean((correct - stove) ** 2))),
            "head_only_counterfactual_cosine": 0.9938990212759606,
            "m1_joint_counterfactual_cosine": 0.9936799449296866,
        },
        indent=2,
    )
)
PY

"${PROJECT_ROOT}/scripts/h3wam/run_cloud_libero.sh" \
  "${H3_ROOT}/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/h3dreamwam/rollout_h3dreamwam_fsdp.py" \
  --dot --model "${MODEL_ROOT}" \
  --action-stage "${JOINT_STAGE}/action_stage.pt" \
  --h3-joint-stage "${JOINT_STAGE}" \
  --cache-root "${DATA_ROOT}" --manifest "${TRAIN_MANIFEST}" \
  --torchrun "${TORCHRUN}" \
  --suite libero_goal --task-ids 0 --trial-indices 0 \
  --max-steps 100 --wait-steps 10 --replan-steps 10 \
  --action-horizon 32 --sample-steps 10 \
  --output-dir "${POST_ROOT}/libero_goal_task0_canary100" \
  --save-video --save-trajectories --require-text-only-context \
  > "${POST_ROOT}/libero_goal_task0_canary100.log" 2>&1
