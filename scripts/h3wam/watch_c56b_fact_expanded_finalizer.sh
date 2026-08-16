#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:?PROJECT_ROOT is required}"
c60_root="${C56B_EXPANDED_ROOT:?C56B_EXPANDED_ROOT is required}"
c58_root="${C58B_EXPANDED_ROOT:?C58B_EXPANDED_ROOT is required}"
output="${c60_root}/RESULTS.json"
log="${c60_root}/finalizer-aggregate.log"
python_bin="${workspace}/runtime/conda-py311/bin/python"
trial33="${workspace}/outputs/c56b-fact-online-v1/paired-final-eval-v2/fresh-execution-libero-trial33/RESULTS.json"
c60_checkpoint="${workspace}/outputs/c56b-fact-online-v1/online-long10000-v1/checkpoints/c56b_online_s10000.pt"
c58_checkpoint="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt"

for path in "${project}/scripts/h3wam/aggregate_c56b_fact_expanded_paired_eval.py" \
  "${project}/scripts/h3wam/aggregate_c58b_expanded_paired_eval.py" \
  "${trial33}" "${c60_checkpoint}" "${c58_checkpoint}"; do
  [[ -e "${path}" ]] || { echo "missing C60 finalizer input: ${path}" >&2; exit 2; }
done
[[ ! -e "${output}" ]] || { echo "C60 expanded final already exists"; exit 0; }
lock="${c60_root}/.finalizer.lock"
mkdir "${lock}" 2>/dev/null || { echo "another C60 finalizer owns ${lock}" >&2; exit 75; }
trap 'rmdir "${lock}" 2>/dev/null || true' EXIT

# File-existence-only wait: no per-episode effect payload is opened early.
while [[ ! -s "${c60_root}/COMPLETED.json" || ! -s "${c58_root}/COMPLETED.json" ]]; do
  if [[ -e "${c60_root}/INVALID.json" || -e "${c58_root}/INVALID.json" ]]; then
    echo "an expanded arm is marked INVALID" >&2; exit 2
  fi
  sleep 60
done

cd "${project}/scripts/h3wam"
"${python_bin}" aggregate_c56b_fact_expanded_paired_eval.py \
  --c60-root "${c60_root}" --c58-root "${c58_root}" \
  --trial33-results "${trial33}" --c60-checkpoint "${c60_checkpoint}" \
  --c58-checkpoint "${c58_checkpoint}" --workers 8 --output "${output}" \
  >"${log}" 2>&1
