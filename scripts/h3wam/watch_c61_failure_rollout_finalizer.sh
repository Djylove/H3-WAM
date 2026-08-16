#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
source_root="${C61_SOURCE_ROOT:-${workspace}/eval/c61-failure-rollout-expansion-v1}"
final_root="${C61_FINAL_ROOT:-${workspace}/eval/c61-finalized-fact-failure-dataset-v1}"
marker="${source_root}/node0-of-1.COMPLETED"
expected_jobs="${C61_EXPECTED_JOBS:-1128}"
poll_seconds="${C61_POLL_SECONDS:-30}"

[[ "${expected_jobs}" =~ ^[1-9][0-9]*$ ]]
[[ "${poll_seconds}" =~ ^[1-9][0-9]*$ ]]
test -f "${source_root}/FROZEN.json"
test -f "${source_root}/jobs.jsonl"
test -x "${project}/scripts/h3wam/finalize_c61_failure_rollout_dataset.sh"

if [[ -s "${final_root}/COMPLETED.json" ]]; then
  echo "C61 finalized dataset already complete: ${final_root}/COMPLETED.json"
  exit 0
fi
[[ ! -e "${final_root}" ]] || {
  echo "refusing pre-existing incomplete C61 final root: ${final_root}" >&2
  exit 2
}

while [[ ! -s "${marker}" ]]; do
  sleep "${poll_seconds}"
  if ! pgrep -f 'launch_c61_failure_rollout_node\.sh' >/dev/null; then
    # Avoid racing the launcher's final wait/atomic marker write.
    sleep "${poll_seconds}"
    if [[ ! -s "${marker}" ]] \
      && ! pgrep -f 'launch_c61_failure_rollout_node\.sh' >/dev/null; then
      echo "C61 collector exited without its strict completion marker" >&2
      exit 3
    fi
  fi
done

results="$(find "${source_root}/runs" -mindepth 2 -maxdepth 2 -type f -name results.json | wc -l)"
trajectories="$(find "${source_root}/runs" -mindepth 2 -maxdepth 2 -type f -name '*_trajectory.npz' | wc -l)"
if [[ "${results}" -ne "${expected_jobs}" || "${trajectories}" -ne "${expected_jobs}" ]]; then
  echo "C61 marker exists but coverage is not exact: results=${results}, trajectories=${trajectories}, expected=${expected_jobs}" >&2
  exit 4
fi

H3_WORKSPACE="${workspace}" PROJECT_ROOT="${project}" \
  C61_SOURCE_ROOT="${source_root}" C61_FINAL_ROOT="${final_root}" \
  bash "${project}/scripts/h3wam/finalize_c61_failure_rollout_dataset.sh"

test -s "${final_root}/COMPLETED.json"
echo "C61 strict finalization complete: ${final_root}/COMPLETED.json"
