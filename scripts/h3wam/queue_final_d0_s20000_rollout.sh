#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
timeout_seconds="${WAIT_TIMEOUT_SECONDS:-14400}"
poll_seconds="${POLL_SECONDS:-30}"
log_dir="${workspace}/eval/dense-d0-s20000-paired-v1"
h32="${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s20000.pt"
h8="${workspace}/outputs/dense-carrier-d0-h8-s20000-v1/checkpoints/d0_h8_s20000.pt"
rollout_script="${project}/scripts/h3wam/run_dense_d0_milestone_rollout.sh"
deadline=$(( $(date +%s) + timeout_seconds ))

mkdir -p "${log_dir}"
test -x "${rollout_script}"

wait_for_idle_and_checkpoints() {
  while pgrep -f "${project}/scripts/h3wam/rollout_libero.py" >/dev/null || \
        [[ ! -f "${h32}" ]] || [[ ! -f "${h8}" ]]; do
    if (( $(date +%s) >= deadline )); then
      echo "timed out waiting for rollout node/checkpoints" >&2
      return 124
    fi
    sleep "${poll_seconds}"
  done
}

launch_one() {
  local suite="$1"
  local mode="$2"
  local checkpoint="$3"
  local gpu="$4"
  local task="$5"
  local trial="$6"
  local log="${log_dir}/${suite}_task${task}_trial${trial}_${mode}.log"
  local checkpoint_name
  local suite_slug="${suite#libero_}"
  local output_root
  checkpoint_name="$(basename "${checkpoint}" .pt)"
  output_root="${workspace}/outputs/eval-dense-d0-long/${checkpoint_name}_${suite_slug}_task${task}_trial${trial}_replan8"
  if [[ -f "${output_root}/results.json" ]]; then
    echo "skip completed ${output_root}"
    return 0
  fi
  if [[ -e "${output_root}" ]]; then
    echo "refusing partial pre-existing output ${output_root}" >&2
    return 1
  fi
  env SUITE="${suite}" REPLAN_STEPS_OVERRIDE=8 bash "${rollout_script}" \
    "${mode}" "${checkpoint}" "${gpu}" "${task}" "${trial}" >"${log}" 2>&1 &
  wave_pids+=("$!")
}

run_wave() {
  local trial_a="$1"
  local trial_b="$2"
  local failed=0
  wave_pids=()

  launch_one libero_goal H32 "${h32}" 0 5 "${trial_a}"
  launch_one libero_goal H8 "${h8}" 1 5 "${trial_a}"
  launch_one libero_object H32 "${h32}" 2 0 "${trial_a}"
  launch_one libero_object H8 "${h8}" 3 0 "${trial_a}"
  launch_one libero_goal H32 "${h32}" 4 5 "${trial_b}"
  launch_one libero_goal H8 "${h8}" 5 5 "${trial_b}"
  launch_one libero_object H32 "${h32}" 6 0 "${trial_b}"
  launch_one libero_object H8 "${h8}" 7 0 "${trial_b}"

  for pid in "${wave_pids[@]}"; do
    wait "${pid}" || failed=1
  done
  return "${failed}"
}

wait_for_idle_and_checkpoints
run_wave 0 1
run_wave 2 3
