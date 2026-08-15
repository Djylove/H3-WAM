#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
trial="${C41_TRIAL:?set C41_TRIAL to 12, 13, 14, or 15}"
output_root="${C41_OUTPUT_ROOT:-${workspace}/eval/c41-fresh-parent-sources-v1}"
[[ "${trial}" =~ ^(12|13|14|15)$ ]] || { echo "C41_TRIAL must be 12..15" >&2; exit 2; }
test -x "${project}/scripts/h3wam/run_champion_all_tasks_trial0.sh"
test ! -e "${output_root}/trial${trial}.COMPLETED"
mkdir -p "${output_root}"
started="$(date +%s)"
env H3_WORKSPACE="${workspace}" PROJECT_ROOT="${project}" TRIAL_INDEX="${trial}" \
  bash "${project}/scripts/h3wam/run_champion_all_tasks_trial0.sh"
printf '{"trial":%s,"episodes":40,"duration_seconds":%s}\n' \
  "${trial}" "$(( $(date +%s) - started ))" >"${output_root}/trial${trial}.COMPLETED"
