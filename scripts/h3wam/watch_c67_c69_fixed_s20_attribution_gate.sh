#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C67_C69_ATTRIBUTION_SOURCE_SNAPSHOT:?Attribution watcher requires an immutable snapshot}"
freeze_sha="${C67_C69_ATTRIBUTION_SOURCE_FREEZE_SHA256:?Set the reviewed SOURCE_FREEZE SHA256}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
c67_train_root="${C67_TRAIN_ROOT:?Set the C67 train root}"
c67_sealed_root="${C67_SEALED_ROOT:?Set the C67 sealed preview root}"
c69_train_root="${C69_TRAIN_ROOT:?Set the C69 train root}"
c69_preview_root="${C69_PREVIEW_ROOT:?Set the C69 preview root}"
output_root="${C67_C69_ATTRIBUTION_ROOT:?Set a new attribution output root}"
verifier="${project}/scripts/h3wam/freeze_c67_rollout_source.py"
launcher="${project}/scripts/h3wam/launch_c67_c69_fixed_s20_attribution_gate.sh"

[[ ! -e "${output_root}" ]] || {
  echo "refusing existing C67/C69 attribution root: ${output_root}" >&2
  exit 2
}
for path in "${python_bin}" "${project}/SOURCE_FREEZE.json" "${verifier}" \
  "${launcher}" "${c67_train_root}" "${c69_train_root}" "${c69_preview_root}"; do
  [[ -e "${path}" ]] || { echo "missing attribution watcher input: ${path}" >&2; exit 2; }
done
"${python_bin}" "${verifier}" --verify --snapshot "${project}" \
  --expected-manifest-sha256 "${freeze_sha}"

c67_complete="${C67_TRAINING_COMPLETE:-${c67_train_root}/TRAINING_COMPLETE.json}"
c69_complete="${C69_TRAINING_COMPLETE:-${c69_train_root}/TRAINING_COMPLETE.json}"
while [[ ! -s "${c67_complete}" \
  || ! -s "${c67_sealed_root}/SEALED.json" \
  || ! -s "${c69_complete}" \
  || ! -s "${c69_preview_root}/PREVIEWS_COMPLETE.json" ]]; do
  sleep 30
done

exec env \
  H3_WORKSPACE="${workspace}" \
  PYTHON_BIN="${python_bin}" \
  C67_C69_ATTRIBUTION_SOURCE_SNAPSHOT="${project}" \
  C67_C69_ATTRIBUTION_SOURCE_FREEZE_SHA256="${freeze_sha}" \
  C67_TRAIN_ROOT="${c67_train_root}" \
  C67_TRAINING_COMPLETE="${c67_complete}" \
  C67_SEALED_ROOT="${c67_sealed_root}" \
  C69_TRAIN_ROOT="${c69_train_root}" \
  C69_TRAINING_COMPLETE="${c69_complete}" \
  C69_PREVIEW_ROOT="${c69_preview_root}" \
  C67_C69_ATTRIBUTION_ROOT="${output_root}" \
  bash "${launcher}"
