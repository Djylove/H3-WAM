#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
checkpoint="${CHECKPOINT:-${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt}"
trial="${TRIAL_INDEX:-0}"
log_dir="${workspace}/eval/champion-h32-s14000-replan8-all40-trial${trial}-v1"
rollout_script="${project}/scripts/h3wam/run_dense_d0_milestone_rollout.sh"
checkpoint_name="$(basename "${checkpoint}" .pt)"
suites=(libero_goal libero_object libero_spatial libero_10)

[[ "${trial}" =~ ^[0-9]+$ ]] || { echo "TRIAL_INDEX must be non-negative" >&2; exit 2; }
test -f "${checkpoint}"
test -x "${rollout_script}"
mkdir -p "${log_dir}"

output_root_for() {
  local suite="$1"
  local task="$2"
  printf '%s/outputs/eval-dense-d0-long/%s_%s_task%s_trial%s_replan8' \
    "${workspace}" "${checkpoint_name}" "${suite#libero_}" "${task}" "${trial}"
}

for suite in "${suites[@]}"; do
  for task in {0..9}; do
    output_root="$(output_root_for "${suite}" "${task}")"
    if [[ -e "${output_root}" && ! -f "${output_root}/results.json" ]]; then
      echo "refusing partial pre-existing output ${output_root}" >&2
      exit 1
    fi
  done
done

wait_wave() {
  local failed=0
  for pid in "${wave_pids[@]}"; do
    wait "${pid}" || failed=1
  done
  wave_pids=()
  return "${failed}"
}

wave_pids=()
gpu=0
for suite in "${suites[@]}"; do
  for task in {0..9}; do
    output_root="$(output_root_for "${suite}" "${task}")"
    if [[ -f "${output_root}/results.json" ]]; then
      echo "skip completed ${output_root}"
      continue
    fi
    log="${log_dir}/${suite}_task${task}_trial${trial}.log"
    env SUITE="${suite}" REPLAN_STEPS_OVERRIDE=8 \
      bash "${rollout_script}" H32 "${checkpoint}" "${gpu}" "${task}" "${trial}" \
      >"${log}" 2>&1 &
    wave_pids+=("$!")
    gpu=$((gpu + 1))
    if (( gpu == 8 )); then
      wait_wave
      gpu=0
    fi
  done
done
if (( ${#wave_pids[@]} > 0 )); then
  wait_wave
fi
