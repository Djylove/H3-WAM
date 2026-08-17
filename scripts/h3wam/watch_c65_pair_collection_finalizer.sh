#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:?PROJECT_ROOT must be an immutable C65 finalizer snapshot}"
root="${C65_OUTPUT_ROOT:-${workspace}/eval/c65-c60-deployment-pair-collection-v1}"
log="${root}/finalizer.log"

while [[ ! -s "${root}/node-n1-spatial-object.COMPLETED" || \
         ! -s "${root}/node-n2-goal-10.COMPLETED" ]]; do
  if [[ -e "${root}/DATA_GATE.json" ]]; then exit 0; fi
  sleep 60
done

[[ ! -e "${root}/DATA_GATE.json" ]] || exit 0
"${workspace}/runtime/conda-py311/bin/python" \
  "${project}/scripts/h3wam/finalize_c65_c60_pair_collection.py" \
  --root "${root}" >"${log}" 2>&1
