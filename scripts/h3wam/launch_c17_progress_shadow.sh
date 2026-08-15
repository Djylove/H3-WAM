#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
checkpoint="${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"
experiment_id="${PROGRESS_EXPERIMENT_ID:-C17}"
probe="${PROGRESS_PROBE:-${workspace}/eval/c17-frozen-h3-progress-probe-v1/probe.pt}"
output_root="${PROGRESS_SHADOW_OUTPUT_ROOT:-${workspace}/eval/c17-progress-shadow-v1}"
log_root="${PROGRESS_SHADOW_LOG_ROOT:-${workspace}/logs/c17-progress-shadow-v1}"

test -f "${checkpoint}"
test -f "${probe}"
test -x "${project}/scripts/h3wam/run_dense_d0_milestone_rollout.sh"
test ! -e "${output_root}"
mkdir -p "${output_root}/runs" "${log_root}"

# Frozen before launch from the incumbent's 160-result benchmark: the first two
# successes and first two failures in lexical (task,trial) order per suite.
selection=(
  "libero_goal 2 2 success" "libero_goal 2 3 success"
  "libero_goal 0 0 failure" "libero_goal 0 1 failure"
  "libero_object 0 0 success" "libero_object 0 1 success"
  "libero_object 1 3 failure" "libero_object 2 1 failure"
  "libero_spatial 0 3 success" "libero_spatial 1 1 success"
  "libero_spatial 0 0 failure" "libero_spatial 0 1 failure"
  "libero_10 3 1 success" "libero_10 5 0 success"
  "libero_10 0 0 failure" "libero_10 0 1 failure"
)

printf '%s\n' "${selection[@]}" > "${output_root}/selection.txt"

run_wave() {
  local wave="$1"
  local pids=()
  local gpu suite task trial expected slug run_root
  for gpu in {0..7}; do
    read -r suite task trial expected <<< "${selection[$((wave * 8 + gpu))]}"
    slug="${suite#libero_}"
    run_root="${output_root}/runs/${slug}_task${task}_trial${trial}"
    env \
      H3_WORKSPACE="${workspace}" \
      PROJECT_ROOT="${project}" \
      SUITE="${suite}" \
      REPLAN_STEPS_OVERRIDE=8 \
      H3_PROGRESS_PROBE="${probe}" \
      OUTPUT_ROOT="${run_root}" \
      "${project}/scripts/h3wam/run_dense_d0_milestone_rollout.sh" \
      H32 "${checkpoint}" "${gpu}" "${task}" "${trial}" \
      >"${log_root}/${slug}_task${task}_trial${trial}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  (( failed == 0 )) || return 1
}

started="$(date +%s)"
run_wave 0
run_wave 1
duration="$(( $(date +%s) - started ))"
printf '{"status":"ROLLOUT_COMPLETE","episodes":16,"duration_seconds":%s}\n' \
  "${duration}" > "${output_root}/ROLLOUT_COMPLETE"

cd "${project}"
"${workspace}/runtime/conda-py311/bin/python" \
  scripts/h3wam/evaluate_c17_progress_shadow.py \
  --shadow-root "${output_root}" \
  --incumbent-root "${workspace}/outputs/eval-dense-d0-long" \
  --experiment-id "${experiment_id}" \
  --output "${output_root}/report.json"
