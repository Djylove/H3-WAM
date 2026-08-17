#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:?PROJECT_ROOT must be an immutable read-only C65 score snapshot}"
collection="${C65_COLLECTION_ROOT:-${workspace}/eval/c65-c60-deployment-pair-collection-v1}"
root="${C65_SCORE_ROOT:?C65_SCORE_ROOT is required}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
data_gate="${collection}/DATA_GATE.json"

while [[ ! -s "${data_gate}" ]]; do sleep 60; done
permission="$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["permission"])' "${data_gate}")"
if [[ "${permission}" != "GO_SCORE_C65" ]]; then
  echo "C65 data gate did not authorize scoring: ${permission}"
  exit 0
fi
while [[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)" -ne 0 ]]; do
  sleep 30
done
exec env H3_WORKSPACE="${workspace}" PROJECT_ROOT="${project}" \
  C65_COLLECTION_ROOT="${collection}" C65_SCORE_ROOT="${root}" \
  bash "${project}/scripts/h3wam/launch_c65_fact_stage2_score_8gpu.sh"
