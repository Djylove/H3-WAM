#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${C58_FULL50_ROOT:?C58_FULL50_ROOT is required}"
candidate="${root}/candidate_c58b.COMPLETED.json"
control="${root}/control_d0.COMPLETED.json"
output="${root}/FINAL_DESCRIPTIVE.json"
python_bin="${workspace}/runtime/conda-py311/bin/python"

for path in "${root}/PREPARED.json" "${root}/jobs.jsonl" "${python_bin}" \
  "${script_root}/aggregate_c58b_full50_descriptive_eval.py"; do
  [[ -e "${path}" ]] || { echo "missing full50 finalizer input: ${path}" >&2; exit 2; }
done
[[ ! -e "${output}" ]] || { echo "full50 descriptive final already exists"; exit 0; }
lock="${root}/.finalizer.lock"
mkdir "${lock}" 2>/dev/null || { echo "another full50 finalizer owns ${lock}" >&2; exit 75; }
trap 'rmdir "${lock}" 2>/dev/null || true' EXIT

while [[ ! -s "${candidate}" || ! -s "${control}" ]]; do sleep 30; done

"${python_bin}" "${script_root}/aggregate_c58b_full50_descriptive_eval.py" \
  --root "${root}" --workers 8 --output "${output}" \
  >"${root}/aggregate_descriptive.log" 2>&1
