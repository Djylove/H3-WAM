#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/conda-py311/bin/python}"
root="${workspace}/outputs/c55-fact-joint-action-long-v2"
output="${root}/offline_final.json"
while [[ ! -f "${root}/evaluations/action_only/EVALUATION_COMPLETED" \
      || ! -f "${root}/evaluations/joint_aux/EVALUATION_COMPLETED" \
      || ! -f "${root}/mechanism/EVALUATION_COMPLETED" ]]; do
  sleep 30
done
[[ ! -e "${output}" ]] || exit 0
cd "${project}"
PYTHONPATH=src "${python_bin}" scripts/h3wam/finalize_c55_long.py \
  --root "${root}" \
  --parent-evaluation "${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/evaluations/d0_h32_s14000_balanced80.json" \
  --output "${output}"
