#!/usr/bin/env bash
set -Eeuo pipefail
root="${C69_C58B_ROLLOUT_ROOT:?C69_C58B_ROLLOUT_ROOT is required}"
sim_python="${SIM_PYTHON:-/mnt/h3-wam/runtime/conda-py311/bin/python}"
aggregator="${AGGREGATOR:?AGGREGATOR is required}"
base_script="${BASE_SCRIPT:-/mnt/h3-wam/code-snapshots/h3-wam-8518821-rollout-v1/scripts/h3wam/aggregate_c58b_expanded_paired_eval.py}"
while [[ ! -e "${root}/COMPLETED.json" ]]; do
  markers=$(find "${root}" -maxdepth 1 -name 'SHARD_*_COMPLETE.json' | wc -l)
  launchers=$(find "${root}" -maxdepth 1 -name 'launcher-node*.pid' | wc -l)
  if [[ "${markers}" -eq 5 ]]; then
    "${sim_python}" "${aggregator}" --root "${root}" --base-script "${base_script}" --workers 32 >"${root}/aggregate-direct.log" 2>&1
    exit 0
  fi
  [[ "${launchers}" -eq 5 ]] || { echo "launcher PID set incomplete" >&2; exit 2; }
  sleep 30
done
