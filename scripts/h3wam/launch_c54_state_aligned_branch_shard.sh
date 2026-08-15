#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
root="${C54_OUTPUT_ROOT:-${workspace}/eval/c54-fresh-state-aligned-replication-v1}"
checkpoint="${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"
value_checkpoint="${workspace}/outputs/c50-dense-future-value-v1/joint_seed8675309/checkpoints/step10000.pt"
value_report="${workspace}/outputs/c50-dense-future-value-v1/FINAL_REPORT.json"
shard="${C54_SHARD:?set C54_SHARD to 0, 1, 2, or 3}"
[[ "${shard}" =~ ^[0-3]$ ]]
test -f "${root}/BRANCH_SELECTION.json"
test ! -e "${root}/shard${shard}.COMPLETED"

mapfile -t jobs < <(
  "${workspace}/runtime/conda-py311/bin/python" - "${root}/selection.jsonl" "${shard}" <<'PY'
import json,sys
for line in open(sys.argv[1]):
 row=json.loads(line)
 if row["ordinal"]%4==int(sys.argv[2]): print("\t".join(str(row[k]) for k in ("ordinal","group_id","suite","task","trial","distance_replans","trajectory","index","start_step","first_policy_noise_seed","continuation_policy_noise_seed_base","arm")))
PY
)
(( ${#jobs[@]} >= 20 ))

worker() {
  local gpu="$1" i row ordinal group suite task trial distance trajectory index start_step first_seed continuation arm run_root log
  for ((i=gpu;i<${#jobs[@]};i+=8)); do
    row="${jobs[$i]}"; IFS=$'\t' read -r ordinal group suite task trial distance trajectory index start_step first_seed continuation arm <<<"${row}"
    run_root="${root}/runs/${ordinal}_g${group}_${arm}_${suite#libero_}_task${task}_trial${trial}_d${distance}"
    log="${root}/logs/${ordinal}_g${group}_${arm}_${suite#libero_}_task${task}_trial${trial}_d${distance}.log"
    if [[ -s "${run_root}/results.json" ]] && compgen -G "${run_root}/*trajectory.npz" >/dev/null; then continue; fi
    [[ ! -e "${run_root}" ]] || { echo "partial run requires incident recovery: ${run_root}" >&2; return 2; }
    common=(H3_WORKSPACE="${workspace}" PROJECT_ROOT="${project}" SUITE="${suite}" REPLAN_STEPS_OVERRIDE=8 OUTPUT_ROOT="${run_root}" BRANCH_TRAJECTORY="${trajectory}" BRANCH_INDEX="${index}" ENVIRONMENT_SEED=42 FIRST_POLICY_NOISE_SEED="${first_seed}" CONTINUATION_POLICY_NOISE_SEED_BASE="${continuation}" FIRST_REPLAN_STEPS=32)
    if [[ "${arm}" == dense_bestof4 ]]; then
      env "${common[@]}" DENSE_VALUE_CHECKPOINT="${value_checkpoint}" DENSE_VALUE_FINAL_REPORT="${value_report}" CONSEQUENCE_BEST_OF_N=4 CONSEQUENCE_CANDIDATE_SEED_OFFSETS="0:1000000:2000000:3000000" CONSEQUENCE_SELECTION_MIN_STEP="${start_step}" CONSEQUENCE_SELECTION_MAX_STEP="$((start_step+1))" "${project}/scripts/h3wam/run_dense_d0_milestone_rollout.sh" H32 "${checkpoint}" "${gpu}" "${task}" "${trial}" >"${log}" 2>&1
    else
      env "${common[@]}" "${project}/scripts/h3wam/run_dense_d0_milestone_rollout.sh" H32 "${checkpoint}" "${gpu}" "${task}" "${trial}" >"${log}" 2>&1
    fi
  done
}

started="$(date +%s)"; pids=()
for gpu in {0..7}; do worker "${gpu}" & pids+=("$!"); done
failed=0; for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 ))
printf '{"shard":%s,"jobs":%s,"duration_seconds":%s}\n' "${shard}" "${#jobs[@]}" "$(( $(date +%s)-started ))" >"${root}/shard${shard}.COMPLETED"
