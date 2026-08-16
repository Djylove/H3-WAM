#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${C58B_EXPANDED_ROOT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/expanded-paired-trials34-49-v2}"
completed="${root}/COMPLETED.json"
output="${root}/FINAL.json"
d0_ready="${workspace}/eval/c58b-expanded-d0-control-v1/READY.json"
trial33="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-final-eval-v1/fresh-libero-trial33/RESULTS.json"
candidate="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt"
control="${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"
python_bin="${workspace}/runtime/conda-py311/bin/python"

for path in "${root}/PREPARED.json" "${root}/jobs.jsonl" "${d0_ready}" \
  "${trial33}" "${candidate}" "${control}" "${python_bin}" \
  "${script_root}/aggregate_c58b_expanded_paired_eval.py"; do
  [[ -e "${path}" ]] || { echo "missing expanded finalizer input: ${path}" >&2; exit 2; }
done
[[ ! -e "${output}" ]] || { echo "expanded final already exists"; exit 0; }
lock="${root}/.finalizer.lock"
mkdir "${lock}" 2>/dev/null || { echo "another expanded finalizer owns ${lock}" >&2; exit 75; }
trap 'rmdir "${lock}" 2>/dev/null || true' EXIT

while [[ ! -s "${completed}" ]]; do
  sleep 30
done

"${python_bin}" "${script_root}/aggregate_c58b_expanded_paired_eval.py" \
  --root "${root}" --d0-ready "${d0_ready}" --trial33-results "${trial33}" \
  --candidate-checkpoint "${candidate}" --d0-checkpoint "${control}" \
  --workers 8 --output "${output}" >"${root}/aggregate.log" 2>&1
