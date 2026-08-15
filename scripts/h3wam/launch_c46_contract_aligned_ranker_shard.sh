#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
root="${C46_OUTPUT_ROOT:-${workspace}/eval/c46-contract-aligned-ranker-v1}"
checkpoint="${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"
ranker="${workspace}/outputs/c44-powered-consequence-ranking-v1/ranker.pt"
models="${workspace}/outputs/c38-temporal-paired-null-replication-v1/temporal_seed161803/checkpoints/temporal_seed161803_step10000.pt:${workspace}/outputs/c38-temporal-paired-null-replication-v1/temporal_seed271828/checkpoints/temporal_seed271828_step10000.pt:${workspace}/outputs/c38-temporal-paired-null-replication-v1/temporal_seed8675309/checkpoints/temporal_seed8675309_step10000.pt:${workspace}/outputs/c38-temporal-paired-null-replication-v1/temporal_seed20260815/checkpoints/temporal_seed20260815_step10000.pt"
shard="${C46_SHARD:?set C46_SHARD to 0, 1, 2, or 3}"
[[ "${shard}" =~ ^[0-3]$ ]]
test ! -e "${root}/shard${shard}.COMPLETED"
mkdir -p "${root}/runs" "${root}/logs"

jobs=(); ordinal=0
for suite in libero_goal libero_object libero_spatial libero_10; do
  for task in {0..4}; do
    for arm in control bestof4; do
      (( ordinal % 4 != shard )) || jobs+=("${ordinal}|${suite}|${task}|${arm}")
      ordinal=$((ordinal + 1))
    done
  done
done
(( ${#jobs[@]} == 10 ))

run_wave() {
  local start="$1" pids=() gpu job ordinal suite task arm run_root log
  for gpu in {0..7}; do
    (( start + gpu < ${#jobs[@]} )) || break
    job="${jobs[$((start + gpu))]}"; IFS='|' read -r ordinal suite task arm <<<"${job}"
    run_root="${root}/runs/${ordinal}_${arm}_${suite#libero_}_task${task}_trial23"
    log="${root}/logs/${ordinal}_${arm}_${suite#libero_}_task${task}_trial23.log"
    [[ ! -s "${run_root}/results.json" ]] || continue
    [[ ! -e "${run_root}" ]] || { echo "partial run requires incident recovery: ${run_root}" >&2; return 2; }
    common=(H3_WORKSPACE="${workspace}" PROJECT_ROOT="${project}" SUITE="${suite}"
      REPLAN_STEPS_OVERRIDE=8 OUTPUT_ROOT="${run_root}"
      SCHEDULED_LONG_REPLAN_STEP=80 SCHEDULED_LONG_REPLAN_STEPS=32)
    if [[ "${arm}" == bestof4 ]]; then
      env "${common[@]}" CONSEQUENCE_RANKER_CHECKPOINT="${ranker}" \
        CONSEQUENCE_MODEL_CHECKPOINTS="${models}" CONSEQUENCE_BEST_OF_N=4 \
        CONSEQUENCE_CANDIDATE_SEED_OFFSETS="0:1000000:2000000:3000000" \
        CONSEQUENCE_SELECTION_MIN_STEP=80 CONSEQUENCE_SELECTION_MAX_STEP=81 \
        "${project}/scripts/h3wam/run_dense_d0_milestone_rollout.sh" \
        H32 "${checkpoint}" "${gpu}" "${task}" 23 >"${log}" 2>&1 &
    else
      env "${common[@]}" "${project}/scripts/h3wam/run_dense_d0_milestone_rollout.sh" \
        H32 "${checkpoint}" "${gpu}" "${task}" 23 >"${log}" 2>&1 &
    fi
    pids+=("$!")
  done
  local failed=0; for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 ))
}

started="$(date +%s)"
for (( start=0; start<${#jobs[@]}; start+=8 )); do run_wave "${start}"; done
printf '{"shard":%s,"jobs":%s,"duration_seconds":%s}\n' \
  "${shard}" "${#jobs[@]}" "$(( $(date +%s) - started ))" >"${root}/shard${shard}.COMPLETED"
