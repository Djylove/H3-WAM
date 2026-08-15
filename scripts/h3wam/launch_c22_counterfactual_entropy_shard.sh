#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
output_root="${C22_OUTPUT_ROOT:-${workspace}/eval/c22-counterfactual-entropy-sweep-v1}"
checkpoint="${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"
shard="${C22_SHARD:?set C22_SHARD to 0, 1, 2, or 3}"
[[ "${shard}" =~ ^[0-3]$ ]]
test -f "${output_root}/selection.jsonl"
test ! -e "${output_root}/shard${shard}.COMPLETED"

mapfile -t selection < <(
  "${workspace}/runtime/conda-py311/bin/python" - "${output_root}/selection.jsonl" "${shard}" <<'PY'
import json,sys
for line in open(sys.argv[1]):
    row=json.loads(line)
    if row["ordinal"] % 4 == int(sys.argv[2]):
        print("\t".join(str(row[key]) for key in (
            "ordinal", "suite", "task", "trial", "index", "distance_replans",
            "policy_noise_seed", "noise_offset", "trajectory"
        )))
PY
)
(( ${#selection[@]} == 24 ))

run_wave() {
  local wave="$1" pids=() gpu row ordinal suite task trial index distance seed offset trajectory slug run_root
  for gpu in {0..7}; do
    row="${selection[$((wave * 8 + gpu))]}"
    IFS=$'\t' read -r ordinal suite task trial index distance seed offset trajectory <<< "${row}"
    slug="${suite#libero_}"
    run_root="${output_root}/runs/${ordinal}_${slug}_task${task}_trial${trial}_d${distance}_offset${offset}"
    env H3_WORKSPACE="${workspace}" PROJECT_ROOT="${project}" SUITE="${suite}" \
      REPLAN_STEPS_OVERRIDE=8 BRANCH_TRAJECTORY="${trajectory}" BRANCH_INDEX="${index}" \
      ENVIRONMENT_SEED=42 POLICY_NOISE_SEED_BASE="${seed}" OUTPUT_ROOT="${run_root}" \
      "${project}/scripts/h3wam/run_dense_d0_milestone_rollout.sh" \
      H32 "${checkpoint}" "${gpu}" "${task}" "${trial}" \
      >"${output_root}/logs/${ordinal}_${slug}_task${task}_trial${trial}_d${distance}_offset${offset}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 ))
}

started="$(date +%s)"
run_wave 0
run_wave 1
run_wave 2
printf '{"shard":%s,"branches":24,"duration_seconds":%s}\n' \
  "${shard}" "$(( $(date +%s) - started ))" > "${output_root}/shard${shard}.COMPLETED"
