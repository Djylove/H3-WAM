#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
c58_report="${C58_REPORT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/probe10/train_s10.json}"
c58_checkpoint="${C58_CHECKPOINT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/probe10/c58b_s10.pt}"
c56_output="${C56_OUTPUT:-${workspace}/outputs/c56b-fact-layerwise-v1/mechanical-probe-8gpu-v1}"

cd "${project}"
while [[ ! -f "${c58_report}" || ! -f "${c58_checkpoint}" ]]; do
  sleep 10
done
"${python_bin}" - "${c58_report}" "${c58_checkpoint}" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
checkpoint = Path(sys.argv[2])
if (
    report.get("status") != "mechanical_probe_not_effectiveness_evidence"
    or report.get("completed_steps") != 10
    or report.get("contract", {}).get("candidate") != "C58B_FASTWAM_FULL30_H3_LAYERWISE"
    or report.get("saved_checkpoint") != str(checkpoint)
    or not checkpoint.is_file()
):
    raise SystemExit("C58b completion identity failed; C56b will not start")
PY

# The JSON can be visible just before torchrun tears down.  Do not overlap the
# two 8-GPU allocations.
while pgrep -f '[t]rain_h3_fastwam_full_tower.py' >/dev/null; do
  sleep 5
done
[[ ! -e "${c56_output}" ]] || {
  echo "C56b output already exists; refusing ambiguous automatic resume" >&2
  exit 2
}

OUTPUT_ROOT="${c56_output}" STEPS=1 \
  bash scripts/h3wam/launch_c56b_fact_layerwise_probe.sh

# Formal C56b training no longer creates an H3 K/V cache.  Online frozen INT8
# H3 integration is released separately after its C58 interface commit is
# fixed.  This legacy watcher intentionally ends after the bounded probe.
