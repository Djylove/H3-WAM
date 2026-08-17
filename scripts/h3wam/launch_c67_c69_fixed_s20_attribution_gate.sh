#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C67_C69_ATTRIBUTION_SOURCE_SNAPSHOT:?Attribution gate requires an immutable snapshot}"
freeze_sha="${C67_C69_ATTRIBUTION_SOURCE_FREEZE_SHA256:?Set the reviewed SOURCE_FREEZE SHA256}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
c67_train_root="${C67_TRAIN_ROOT:?Set the completed C67 train root}"
c67_complete="${C67_TRAINING_COMPLETE:-${c67_train_root}/TRAINING_COMPLETE.json}"
c67_sealed_root="${C67_SEALED_ROOT:?Set the already sealed C67 preview root}"
c69_train_root="${C69_TRAIN_ROOT:?Set the completed C69 train root}"
c69_complete="${C69_TRAINING_COMPLETE:-${c69_train_root}/TRAINING_COMPLETE.json}"
c69_preview_root="${C69_PREVIEW_ROOT:?Set the complete C69 preview root}"
output_root="${C67_C69_ATTRIBUTION_ROOT:?Set a new attribution output root}"
verifier="${project}/scripts/h3wam/freeze_c67_rollout_source.py"
sealer="${project}/scripts/h3wam/seal_c69_milestone_previews.py"
aggregator="${project}/scripts/h3wam/aggregate_c67_c69_fixed_s20_attribution.py"

[[ ! -e "${output_root}" ]] || {
  echo "refusing existing C67/C69 attribution root: ${output_root}" >&2
  exit 2
}
for path in "${python_bin}" "${project}/SOURCE_FREEZE.json" "${verifier}" \
  "${sealer}" "${aggregator}" "${c67_complete}" "${c69_complete}" \
  "${c67_sealed_root}/SEALED.json" "${c69_preview_root}/PREVIEWS_COMPLETE.json"; do
  [[ -e "${path}" ]] || { echo "missing C67/C69 attribution input: ${path}" >&2; exit 2; }
done

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${verifier}" --verify --snapshot "${project}" \
  --expected-manifest-sha256 "${freeze_sha}"

mkdir -p "${output_root}"
"${python_bin}" "${sealer}" \
  --preview-root "${c69_preview_root}" \
  --train-root "${c69_train_root}" \
  --training-complete "${c69_complete}" \
  --output-root "${output_root}/c69-sealed" \
  >"${output_root}/seal-c69.log" 2>&1

"${python_bin}" "${aggregator}" \
  --c67-root "${c67_sealed_root}" \
  --c67-train-root "${c67_train_root}" \
  --c67-training-complete "${c67_complete}" \
  --c69-root "${output_root}/c69-sealed" \
  --c69-train-root "${c69_train_root}" \
  --c69-training-complete "${c69_complete}" \
  --output "${output_root}/RESULTS.json" \
  >"${output_root}/aggregate.log" 2>&1

"${python_bin}" - "${output_root}/RESULTS.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
print(json.dumps({
    "status": result["status"],
    "permission": result["permission"],
    "effect_status": result["effect_status"],
}, indent=2))
PY
