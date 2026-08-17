#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C67_FINAL_AUDIT_SOURCE_SNAPSHOT:?C67 final audit requires an immutable snapshot}"
freeze_sha="${C67_FINAL_AUDIT_SOURCE_FREEZE_SHA256:?Set reviewed SOURCE_FREEZE SHA256}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
train_root="${C67_TRAIN_ROOT:?Set the immutable completed C67 train root}"
preview_root="${C67_PREVIEW_ROOT:?Set the original C67 preview root}"
sealed_root="${C67_SEALED_ROOT:-${preview_root}/sealed}"
c58_ready="${C58_PARENT_READY:?Set the fixed C58 READY evidence}"
output_root="${C67_FINAL_AUDIT_ROOT:?Set a new independent audit output root}"
verifier="${project}/scripts/h3wam/freeze_c67_rollout_source.py"
auditor="${project}/scripts/h3wam/audit_c67_final_evidence.py"

[[ ! -e "${output_root}" ]] || {
  echo "refusing existing C67 final audit root: ${output_root}" >&2
  exit 2
}
for path in "${python_bin}" "${project}/SOURCE_FREEZE.json" "${verifier}" \
  "${auditor}" "${train_root}" "${preview_root}" "${c58_ready}"; do
  [[ -e "${path}" ]] || { echo "missing C67 final audit input: ${path}" >&2; exit 2; }
done
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${verifier}" --verify --snapshot "${project}" \
  --expected-manifest-sha256 "${freeze_sha}"

training_complete="${train_root}/TRAINING_COMPLETE.json"
sealed_manifest="${sealed_root}/SEALED.json"
results="${sealed_root}/RESULTS.json"
while [[ ! -s "${training_complete}" || ! -s "${sealed_manifest}" \
  || ! -s "${results}" ]]; do
  sleep 30
done

mkdir "${output_root}"
"${python_bin}" "${auditor}" \
  --train-root "${train_root}" --preview-root "${preview_root}" \
  --sealed-root "${sealed_root}" --c58-ready "${c58_ready}" \
  --output "${output_root}/AUDIT.json" \
  >"${output_root}/audit.log" 2>&1
