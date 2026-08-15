#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
trial="${C29_TRIAL:?set C29_TRIAL to 4, 5, 6, or 7}"
output_root="${C29_OUTPUT_ROOT:-${workspace}/eval/c29-fresh-parent-sources-v1}"

[[ "${trial}" =~ ^[4-7]$ ]] || { echo "C29_TRIAL must be 4..7" >&2; exit 2; }
test -x "${project}/scripts/h3wam/run_champion_all_tasks_trial0.sh"
test ! -e "${output_root}/trial${trial}.COMPLETED"
mkdir -p "${output_root}"

started="$(date +%s)"
env H3_WORKSPACE="${workspace}" PROJECT_ROOT="${project}" TRIAL_INDEX="${trial}" \
  bash "${project}/scripts/h3wam/run_champion_all_tasks_trial0.sh"
printf '{"trial":%s,"episodes":40,"duration_seconds":%s}\n' \
  "${trial}" "$(( $(date +%s) - started ))" >"${output_root}/trial${trial}.COMPLETED"
