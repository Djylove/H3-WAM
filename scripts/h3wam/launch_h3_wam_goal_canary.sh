#!/usr/bin/env bash
set -euo pipefail

# Reproducible small closed-loop gate for dense50 joint checkpoints.  Keep this
# deliberately separate from the full 10-task x 5-trial benchmark: candidates
# must first preserve known successes and show signal on weak/failing tasks.
H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a complete joint checkpoint directory}"
MODEL="${MODEL:-${H3_WORKSPACE}/models/MiniMax-H3}"
CACHE_ROOT="${CACHE_ROOT:-${H3_WORKSPACE}/data/v5_dense50_action_cache}"
TASK_CONTEXTS="${TASK_CONTEXTS:-${H3_WORKSPACE}/data/v5_multisuite_dense50_candidate/task_contexts.json}"
TASK_IDS="${TASK_IDS:-0 3 7 8}"
TRIAL_INDICES="${TRIAL_INDICES:-0}"
MAX_STEPS="${MAX_STEPS:-400}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
HISTORY_ADAPTER_SCALE="${HISTORY_ADAPTER_SCALE:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to a new rollout directory}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-${PROJECT_ROOT}/scripts/h3wam/torchrun_venv.sh}"
if [[ -z "${SIM_SITE_PACKAGES:-}" ]]; then
  for candidate in \
    "${H3_WORKSPACE}/runtime/libero_site" \
    /tmp/h3-wam-libero-site; do
    if [[ -d "${candidate}/robosuite" ]]; then
      SIM_SITE_PACKAGES="${candidate}"
      break
    fi
  done
fi

for path in "${CHECKPOINT}" "${MODEL}" "${CACHE_ROOT}" "${TASK_CONTEXTS}"; do
  test -e "${path}"
done
test -x "${PYTHON_BIN}"
test -x "${TORCHRUN}"
test -d "${SIM_SITE_PACKAGES:?LIBERO robosuite site-packages not found}"
export SIM_SITE_PACKAGES
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to reuse rollout output: ${OUTPUT_DIR}" >&2
  exit 2
fi

read -r -a task_ids <<<"${TASK_IDS}"
read -r -a trial_indices <<<"${TRIAL_INDICES}"

cd "${PROJECT_ROOT}"
exec scripts/h3wam/run_cloud_libero.sh \
  "${PYTHON_BIN}" scripts/h3wam/rollout_h3_wam_joint_fsdp.py \
  --model "${MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  --cache-root "${CACHE_ROOT}" \
  --task-contexts "${TASK_CONTEXTS}" \
  --torchrun "${TORCHRUN}" \
  --nproc-per-node 8 \
  --suite libero_goal \
  --task-ids "${task_ids[@]}" \
  --trial-indices "${trial_indices[@]}" \
  --max-steps "${MAX_STEPS}" \
  --wait-steps 30 \
  --replan-steps "${REPLAN_STEPS}" \
  --action-horizon 32 \
  --flow-steps 2 \
  --history-adapter-scale "${HISTORY_ADAPTER_SCALE}" \
  --seed 0 \
  --output-dir "${OUTPUT_DIR}" \
  --save-video
