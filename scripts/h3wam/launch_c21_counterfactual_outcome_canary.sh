#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
checkpoint="${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"
source_root="${workspace}/outputs/eval-dense-d0-long"
output_root="${workspace}/eval/c21-counterfactual-outcome-canary-v1"
log_root="${workspace}/logs/c21-counterfactual-outcome-canary-v1"
test ! -e "${output_root}"
mkdir -p "${output_root}/runs" "${log_root}"

# suite task trial start_index original_policy_seed
groups=(
  "libero_goal 2 2 10 202052"
  "libero_object 0 1 14 1056"
  "libero_spatial 0 3 7 3049"
  "libero_10 3 1 41 301083"
)
offsets=(0 1000000 2000000 3000000)
selection=()
for group in "${groups[@]}"; do
  for offset in "${offsets[@]}"; do selection+=("${group} ${offset}"); done
done
printf '%s\n' "${selection[@]}" > "${output_root}/selection.txt"

run_wave() {
  local wave="$1" pids=() gpu suite task trial index base offset slug trajectory run_root
  for gpu in {0..7}; do
    read -r suite task trial index base offset <<< "${selection[$((wave * 8 + gpu))]}"
    slug="${suite#libero_}"
    trajectory="${source_root}/d0_h32_s14000_${slug}_task${task}_trial${trial}_replan8/task$(printf '%02d' "${task}")_trial$(printf '%02d' "${trial}")_trajectory.npz"
    run_root="${output_root}/runs/${slug}_task${task}_trial${trial}_index${index}_offset${offset}"
    env H3_WORKSPACE="${workspace}" PROJECT_ROOT="${project}" SUITE="${suite}" \
      REPLAN_STEPS_OVERRIDE=8 BRANCH_TRAJECTORY="${trajectory}" BRANCH_INDEX="${index}" \
      ENVIRONMENT_SEED=42 POLICY_NOISE_SEED_BASE="$((base + offset))" OUTPUT_ROOT="${run_root}" \
      "${project}/scripts/h3wam/run_dense_d0_milestone_rollout.sh" \
      H32 "${checkpoint}" "${gpu}" "${task}" "${trial}" \
      >"${log_root}/${slug}_task${task}_trial${trial}_index${index}_offset${offset}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 ))
}

started="$(date +%s)"
run_wave 0
run_wave 1
"${workspace}/runtime/conda-py311/bin/python" \
  "${project}/scripts/h3wam/evaluate_c21_counterfactual_outcomes.py" \
  --root "${output_root}" --output "${output_root}/COMPLETED"
printf '{"duration_seconds":%s}\n' "$(( $(date +%s) - started ))" > "${output_root}/duration.json"
