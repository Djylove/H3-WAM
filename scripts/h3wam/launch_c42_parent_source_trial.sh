#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
trial="${C42_TRIAL:?set C42_TRIAL to 16, 17, 18, 19, 20, or 21}"
output_root="${C42_OUTPUT_ROOT:-${workspace}/eval/c42-fresh-parent-sources-v1}"
[[ "${trial}" =~ ^(16|17|18|19|20|21)$ ]] || {
  echo "C42_TRIAL must be 16..21" >&2
  exit 2
}
test -x "${project}/scripts/h3wam/run_champion_all_tasks_trial0.sh"
test ! -e "${output_root}/trial${trial}.COMPLETED"
mkdir -p "${output_root}"

started="$(date +%s)"
env H3_WORKSPACE="${workspace}" PROJECT_ROOT="${project}" TRIAL_INDEX="${trial}" \
  bash "${project}/scripts/h3wam/run_champion_all_tasks_trial0.sh"
printf '{"trial":%s,"episodes":40,"duration_seconds":%s}\n' \
  "${trial}" "$(( $(date +%s) - started ))" >"${output_root}/trial${trial}.COMPLETED"
