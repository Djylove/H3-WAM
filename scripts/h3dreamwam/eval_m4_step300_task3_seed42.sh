#!/usr/bin/env bash
set -euo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v2_full_cache"
MANIFEST="${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate/manifest_train_uniform.jsonl"
STAGE="${H3_WORKSPACE}/outputs/h3dotwam/m4_paper_joint_full40_10ep_joint_step000300"
OUTPUT_DIR="${H3_WORKSPACE}/outputs/eval-rgb-dot/m4_step300/libero_goal_task3_seed42_trials0_9"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-30907"
LOCK_DIR="/tmp/h3-wam-cluster-30907-eval.lock"

mkdir -p "${OUTPUT_DIR}" "${TMP_ROOT}"
while pgrep -f '^bash /mnt/h3-wam/project/scripts/h3dreamwam/eval_m4_multisuite_canary.sh 300$' >/dev/null; do
  sleep 30
done
while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
  sleep 10
done
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT
"${PYTHON_BIN}" scripts/h3dreamwam/check_completed_rollout.py "${OUTPUT_DIR}/results.json" 2>/dev/null && exit 0
export TMPDIR="${TMP_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES:-/tmp/h3-wam-libero-site}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${PROJECT_ROOT}/scripts/h3wam/run_cloud_libero.sh" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3dreamwam/rollout_h3dreamwam_fsdp.py" \
  --dot --model "${MODEL_ROOT}" --action-stage "${STAGE}/action_stage.pt" \
  --h3-joint-stage "${STAGE}" --cache-root "${DATA_ROOT}" --manifest "${MANIFEST}" \
  --torchrun "${PROJECT_ROOT}/scripts/h3dreamwam/torchrun_shared.sh" \
  --suite libero_goal --task-ids 3 --trial-indices 0 1 2 3 4 5 6 7 8 9 \
  --seed 42 --max-steps 400 --wait-steps 30 --replan-steps 10 \
  --action-horizon 32 --sample-steps 10 --output-dir "${OUTPUT_DIR}" \
  --save-video --save-trajectories --require-text-only-context \
  > "${OUTPUT_DIR}.log" 2>&1
