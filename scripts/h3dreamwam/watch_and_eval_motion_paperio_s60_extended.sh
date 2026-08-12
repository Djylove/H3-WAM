#!/usr/bin/env bash
set -euo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v2_full_cache"
TRAIN_MANIFEST="${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate/manifest_train_uniform.jsonl"
VAL_MANIFEST="${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate/manifest_val.jsonl"
STAGE="${H3_WORKSPACE}/outputs/h3dotwam-motion/m4_motion_paperio_full50_gb128_s60_joint"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/eval-motion-dot/m4_motion_paperio_s60"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-30907"
LOCK_DIR="/tmp/h3-wam-cluster-30907-eval.lock"

mkdir -p "${OUTPUT_ROOT}" "${TMP_ROOT}"
while [[ ! -s "${STAGE}/joint_stage.json" || ! -s "${STAGE}/action_stage.pt" ]]; do
  sleep 30
done
for rank in $(seq 0 7); do
  while [[ ! -s "${STAGE}/h3_rank$(printf '%05d' "${rank}").pt" ]]; do
    sleep 10
  done
done
while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
  sleep 10
done
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT
while pgrep -f '[s]erve_h3dotwam_fsdp.py' >/dev/null \
  || pgrep -f '[t]rain_h3dotwam_fsdp.py' >/dev/null; do
  sleep 10
done
export TMPDIR="${TMP_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"

if [[ ! -s "${OUTPUT_ROOT}/val850.json" ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
    --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
    --manifest "${VAL_MANIFEST}" --output "${OUTPUT_ROOT}/val850.json" \
    --load-joint-stage "${STAGE}" --eval-only --steps 107 \
    --sample-steps 10 --action-horizon 32 --require-text-only-context --log-every 20 \
    > "${OUTPUT_ROOT}/val850.log" 2>&1
fi

TASK3_OUTPUT="${OUTPUT_ROOT}/libero_goal_task3_seed42_trials0_9"
if [[ ! -s "${TASK3_OUTPUT}/results.json" ]]; then
  SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES:-/tmp/h3-wam-libero-site}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${PROJECT_ROOT}/scripts/h3wam/run_cloud_libero.sh" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3dreamwam/rollout_h3dreamwam_fsdp.py" \
    --dot --model "${MODEL_ROOT}" --action-stage "${STAGE}/action_stage.pt" \
    --h3-joint-stage "${STAGE}" --cache-root "${DATA_ROOT}" \
    --manifest "${TRAIN_MANIFEST}" \
    --torchrun "${PROJECT_ROOT}/scripts/h3dreamwam/torchrun_shared.sh" \
    --suite libero_goal --task-ids 3 --trial-indices 0 1 2 3 4 5 6 7 8 9 \
    --seed 42 --max-steps 400 --wait-steps 30 --replan-steps 10 \
    --action-horizon 32 --sample-steps 10 --output-dir "${TASK3_OUTPUT}" \
    --save-video --save-trajectories --require-text-only-context \
    > "${TASK3_OUTPUT}.log" 2>&1
fi
