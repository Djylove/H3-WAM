#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C70_FINAL_SOURCE_SNAPSHOT:?C70 final watcher requires an immutable snapshot}"
freeze_sha="${C70_FINAL_SOURCE_FREEZE_SHA256:?Set reviewed C70 final SOURCE_FREEZE SHA256}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
train_root="${C70_TRAIN_ROOT:-${workspace}/outputs/c70-sampler-coverage-v1/online-long20000-v1}"
preview_root="${C70_PREVIEW_ROOT:-${workspace}/outputs/c70-sampler-coverage-v1/milestone-preview-be5867a-v1}"
sealed_root="${C70_SEALED_ROOT:-${preview_root}/sealed}"
result_root="${C70_RESULT_ROOT:?C70_RESULT_ROOT is required}"
c67_report="${C67_FIXED_S20_REPORT:-${workspace}/outputs/c67-c60-budget-ablation-v1/milestone-preview-3682e18-v1/sealed/reports/s20000.json}"
verifier="${project}/scripts/h3wam/freeze_c67_rollout_source.py"
sealer="${project}/scripts/h3wam/seal_c70_milestone_previews.py"
aggregator="${project}/scripts/h3wam/aggregate_c70_c67_fixed_s20.py"

for path in "${python_bin}" "${project}/SOURCE_FREEZE.json" "${verifier}" \
  "${sealer}" "${aggregator}" "${c67_report}"; do
  [[ -e "${path}" ]] || { echo "missing C70 final input: ${path}" >&2; exit 2; }
done
"${python_bin}" "${verifier}" --verify --snapshot "${project}" \
  --expected-manifest-sha256 "${freeze_sha}"
[[ ! -e "${sealed_root}" ]] || { echo "refusing existing C70 sealed root" >&2; exit 2; }
[[ ! -e "${result_root}" ]] || { echo "refusing existing C70 result root" >&2; exit 2; }

training_complete="${train_root}/TRAINING_COMPLETE.json"
previews_complete="${preview_root}/PREVIEWS_COMPLETE.json"
while [[ ! -s "${training_complete}" || ! -s "${previews_complete}" ]]; do
  sleep 30
done

export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${sealer}" --preview-root "${preview_root}" \
  --train-root "${train_root}" --training-complete "${training_complete}" \
  --output-root "${sealed_root}"
mkdir -p "${result_root}"
"${python_bin}" "${aggregator}" --c67-report "${c67_report}" \
  --c70-report "${sealed_root}/reports/s20000.json" \
  --c70-sealed "${sealed_root}/SEALED.json" --output "${result_root}/RESULTS.json"
