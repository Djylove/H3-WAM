#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v7_dense_h3_cache"
CANDIDATE_ROOT="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate"
PARENT_JOINT="${H3_WORKSPACE}/outputs/h3dotwam-tail2/m14_tail2_from_m13s400_gb128_s40_joint"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3dotwam-language"
RUN_NAME="m15_dense_language_rank_tail2_from_m14_s100"
JOINT_STAGE="${OUTPUT_ROOT}/${RUN_NAME}_joint"
EVAL_ROOT="${H3_WORKSPACE}/outputs/eval-language-dot/${RUN_NAME}"
TMP_ROOT="${H3_WORKSPACE}/tmp/${RUN_NAME}"
COUNTERFACTUAL_MANIFEST="${EVAL_ROOT}/counterfactual.jsonl"

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="${H3_WORKSPACE}/cache"
export HF_HOME="${H3_WORKSPACE}/cache/huggingface"
export TORCH_HOME="${H3_WORKSPACE}/cache/torch"
export TMPDIR="${TMP_ROOT}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

test -s "${PARENT_JOINT}/joint_stage.json"
test -s "${PARENT_JOINT}/action_stage.pt"
test ! -e "${JOINT_STAGE}"
mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}" "${TMP_ROOT}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
  --manifest "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
  --output "${OUTPUT_ROOT}/${RUN_NAME}.json" \
  --load-joint-stage "${PARENT_JOINT}" --save-joint-stage "${JOINT_STAGE}" \
  --steps 100 --sample-offset 56320 --gradient-accumulation-steps 16 \
  --action-horizon 32 --learning-rate 1e-5 --h3-learning-rate 1e-6 \
  --last-h3-blocks 2 --video-loss-weight 1.0 \
  --language-ranking-weight 0.5 --language-ranking-margin 0.05 \
  --language-ranking-every 1 --lr-schedule cosine \
  --require-text-only-context --log-every 1 \
  > "${OUTPUT_ROOT}/${RUN_NAME}.log" 2>&1

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
  --manifest "${CANDIDATE_ROOT}/manifest_val_stratified40.jsonl" \
  --output "${EVAL_ROOT}/val40.json" --load-joint-stage "${JOINT_STAGE}" \
  --eval-only --steps 5 --sample-steps 10 --action-horizon 32 \
  --last-h3-blocks 2 --require-text-only-context --log-every 1 \
  > "${EVAL_ROOT}/val40.log" 2>&1

# Build a cloud-valid one-window language counterfactual from the dense validation
# manifest. The visual/state/action input stays fixed; only the context ID changes.
head -n 1 "${CANDIDATE_ROOT}/manifest_val_stratified40.jsonl" > "${COUNTERFACTUAL_MANIFEST}"
WRONG_CONTEXT_ID="$(sed -n '2p' "${CANDIDATE_ROOT}/manifest_val_stratified40.jsonl" | \
  "${PYTHON_BIN}" -c 'import json,sys; print(json.load(sys.stdin)["context_id"])')"

for spec in correct: wrong:"${WRONG_CONTEXT_ID}"; do
  name="${spec%%:*}"
  context_id="${spec#*:}"
  extra=()
  if [[ -n "${context_id}" ]]; then
    extra=(--context-override-id "${context_id}")
  fi
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
    --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
    --manifest "${COUNTERFACTUAL_MANIFEST}" \
    --output "${EVAL_ROOT}/counterfactual_${name}.json" \
    --load-joint-stage "${JOINT_STAGE}" --eval-only --steps 1 \
    --sample-steps 10 --action-horizon 32 --last-h3-blocks 2 \
    --require-text-only-context --record-sampled-actions "${extra[@]}" \
    > "${EVAL_ROOT}/counterfactual_${name}.log" 2>&1
done

"${PYTHON_BIN}" - "${EVAL_ROOT}" <<'PY'
import json
import pathlib
import sys

import numpy as np

root = pathlib.Path(sys.argv[1])
correct = np.asarray(json.loads((root / "counterfactual_correct.json").read_text())["history"][0]["sampled_actions"])
wrong = np.asarray(json.loads((root / "counterfactual_wrong.json").read_text())["history"][0]["sampled_actions"])
payload = {
    "val40_mse": json.loads((root / "val40.json").read_text())["mean_action_loss"],
    "parent_val40_mse": 0.1193679756950587,
    "counterfactual_cosine": float(np.dot(correct.ravel(), wrong.ravel()) / (np.linalg.norm(correct) * np.linalg.norm(wrong))),
    "counterfactual_rms": float(np.sqrt(np.mean((correct - wrong) ** 2))),
}
(root / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
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
