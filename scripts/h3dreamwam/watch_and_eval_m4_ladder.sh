#!/usr/bin/env bash
set -euo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v2_full_cache"
TRAIN_MANIFEST="${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate/manifest_train_uniform.jsonl"
VAL40_MANIFEST="${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate/manifest_val_stratified40.jsonl"
VAL850_MANIFEST="${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate/manifest_val.jsonl"
STAGE_PREFIX="${H3_WORKSPACE}/outputs/h3dotwam/m4_paper_joint_full40_10ep_joint"
OUTPUT_BASE="${H3_WORKSPACE}/outputs/eval-rgb-dot"
LOG_ROOT="${H3_WORKSPACE}/logs/cluster-30907"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-30907"

mkdir -p "${OUTPUT_BASE}" "${LOG_ROOT}" "${TMP_ROOT}"
export TMPDIR="${TMP_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"

wait_for_stage() {
  local stage="$1"
  while [[ ! -s "${stage}/joint_stage.json" || ! -s "${stage}/action_stage.pt" ]]; do
    sleep 30
  done
  local rank
  for rank in $(seq 0 7); do
    while [[ ! -s "${stage}/h3_rank$(printf '%05d' "${rank}").pt" ]]; do
      sleep 10
    done
  done
}

wait_for_local_gpu() {
  while pgrep -f '[s]erve_h3dotwam_fsdp.py' >/dev/null \
    || pgrep -f '[t]rain_h3dotwam_fsdp.py' >/dev/null; do
    sleep 10
  done
}

run_validation() {
  local stage="$1"
  local manifest="$2"
  local steps="$3"
  local output="$4"
  local log="$5"
  [[ -s "${output}" ]] && return 0
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
    --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
    --manifest "${manifest}" --output "${output}" \
    --load-joint-stage "${stage}" \
    --eval-only --steps "${steps}" --sample-steps 10 --action-horizon 32 \
    --require-text-only-context --log-every 20 > "${log}" 2>&1
}

run_rollout() {
  local stage="$1"
  local output_dir="$2"
  local log="$3"
  [[ -s "${output_dir}/results.json" ]] && return 0
  SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES:-/tmp/h3-wam-libero-site}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${PROJECT_ROOT}/scripts/h3wam/run_cloud_libero.sh" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3dreamwam/rollout_h3dreamwam_fsdp.py" \
    --dot --model "${MODEL_ROOT}" \
    --action-stage "${stage}/action_stage.pt" --h3-joint-stage "${stage}" \
    --cache-root "${DATA_ROOT}" --manifest "${TRAIN_MANIFEST}" \
    --torchrun "${PROJECT_ROOT}/scripts/h3dreamwam/torchrun_shared.sh" \
    --suite libero_goal --task-ids 0 3 7 8 --trial-indices 0 \
    --max-steps 400 --wait-steps 30 --replan-steps 10 \
    --action-horizon 32 --sample-steps 10 --output-dir "${output_dir}" \
    --save-video --save-trajectories --require-text-only-context > "${log}" 2>&1
}

for step in 120 180 240 300 360 420 480 540 600 602; do
  if [[ "${step}" == "602" ]]; then
    stage="${STAGE_PREFIX}"
  else
    stage="${STAGE_PREFIX}_step$(printf '%06d' "${step}")"
  fi
  output_root="${OUTPUT_BASE}/m4_step${step}"
  mkdir -p "${output_root}"
  wait_for_stage "${stage}"
  wait_for_local_gpu
  run_validation "${stage}" "${VAL40_MANIFEST}" 5 \
    "${output_root}/val40.json" "${output_root}/val40.log"

  case "${step}" in
    120|180|300|420|540|602)
      wait_for_local_gpu
      run_validation "${stage}" "${VAL850_MANIFEST}" 107 \
        "${output_root}/val850.json" "${output_root}/val850.log"
      wait_for_local_gpu
      run_rollout "${stage}" "${output_root}/libero_goal_canary" \
        "${output_root}/libero_goal_canary.log"
      ;;
  esac
done
