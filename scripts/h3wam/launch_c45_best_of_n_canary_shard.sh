#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
root="${C45_OUTPUT_ROOT:-${workspace}/eval/c45-best-of-n-canary-v1}"
checkpoint="${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"
ranker="${workspace}/outputs/c44-powered-consequence-ranking-v1/ranker.pt"
consequence_models="${workspace}/outputs/c38-temporal-paired-null-replication-v1/temporal_seed161803/checkpoints/temporal_seed161803_step10000.pt:${workspace}/outputs/c38-temporal-paired-null-replication-v1/temporal_seed271828/checkpoints/temporal_seed271828_step10000.pt:${workspace}/outputs/c38-temporal-paired-null-replication-v1/temporal_seed8675309/checkpoints/temporal_seed8675309_step10000.pt:${workspace}/outputs/c38-temporal-paired-null-replication-v1/temporal_seed20260815/checkpoints/temporal_seed20260815_step10000.pt"
shard="${C45_SHARD:?set C45_SHARD to 0, 1, 2, or 3}"
[[ "${shard}" =~ ^[0-3]$ ]]
test -f "${checkpoint}"
test -f "${ranker}"
test ! -e "${root}/shard${shard}.COMPLETED"
mkdir -p "${root}/runs" "${root}/logs"

jobs=()
ordinal=0
for suite in libero_goal libero_object libero_spatial libero_10; do
  for task in {0..4}; do
    for arm in parent bestof4; do
      if (( ordinal % 4 == shard )); then
        jobs+=("${ordinal}|${suite}|${task}|${arm}")
      fi
      ordinal=$((ordinal + 1))
    done
  done
done
(( ${#jobs[@]} == 10 ))

run_wave() {
  local start="$1" pids=() gpu job ordinal suite task arm run_root log
  for gpu in {0..7}; do
    (( start + gpu < ${#jobs[@]} )) || break
    job="${jobs[$((start + gpu))]}"
    IFS='|' read -r ordinal suite task arm <<<"${job}"
    run_root="${root}/runs/${ordinal}_${arm}_${suite#libero_}_task${task}_trial22"
    log="${root}/logs/${ordinal}_${arm}_${suite#libero_}_task${task}_trial22.log"
    if [[ -s "${run_root}/results.json" ]]; then
      continue
    fi
    if [[ -e "${run_root}" ]]; then
      echo "incomplete run directory requires incident recovery: ${run_root}" >&2
      return 2
    fi
    if [[ "${arm}" == bestof4 ]]; then
      env H3_WORKSPACE="${workspace}" PROJECT_ROOT="${project}" SUITE="${suite}" \
        REPLAN_STEPS_OVERRIDE=8 OUTPUT_ROOT="${run_root}" \
        CONSEQUENCE_RANKER_CHECKPOINT="${ranker}" \
        CONSEQUENCE_MODEL_CHECKPOINTS="${consequence_models}" \
        CONSEQUENCE_BEST_OF_N=4 \
        "${project}/scripts/h3wam/run_dense_d0_milestone_rollout.sh" \
        H32 "${checkpoint}" "${gpu}" "${task}" 22 >"${log}" 2>&1 &
    else
      env H3_WORKSPACE="${workspace}" PROJECT_ROOT="${project}" SUITE="${suite}" \
        REPLAN_STEPS_OVERRIDE=8 OUTPUT_ROOT="${run_root}" \
        "${project}/scripts/h3wam/run_dense_d0_milestone_rollout.sh" \
        H32 "${checkpoint}" "${gpu}" "${task}" 22 >"${log}" 2>&1 &
    fi
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 ))
}

started="$(date +%s)"
for (( start=0; start<${#jobs[@]}; start+=8 )); do run_wave "${start}"; done
printf '{"shard":%s,"jobs":%s,"duration_seconds":%s}\n' \
  "${shard}" "${#jobs[@]}" "$(( $(date +%s) - started ))" >"${root}/shard${shard}.COMPLETED"
