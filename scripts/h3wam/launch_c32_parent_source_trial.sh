#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
trial="${C32_TRIAL:?set C32_TRIAL to 8, 9, 10, or 11}"
output_root="${C32_OUTPUT_ROOT:-${workspace}/eval/c32-fresh-parent-sources-v1}"

[[ "${trial}" =~ ^(8|9|10|11)$ ]] || { echo "C32_TRIAL must be 8..11" >&2; exit 2; }
test -x "${project}/scripts/h3wam/run_champion_all_tasks_trial0.sh"
test ! -e "${output_root}/trial${trial}.COMPLETED"
mkdir -p "${output_root}"

started="$(date +%s)"
env H3_WORKSPACE="${workspace}" PROJECT_ROOT="${project}" TRIAL_INDEX="${trial}" \
  bash "${project}/scripts/h3wam/run_champion_all_tasks_trial0.sh"
printf '{"trial":%s,"episodes":40,"duration_seconds":%s}\n' \
  "${trial}" "$(( $(date +%s) - started ))" >"${output_root}/trial${trial}.COMPLETED"
