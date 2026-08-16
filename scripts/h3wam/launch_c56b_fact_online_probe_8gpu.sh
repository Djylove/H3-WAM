#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c56b-fact-online-v1/mechanical-one-step-8gpu-v1}"
c58_report="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-one-step/report.json"
c58_report_sha="84a1a541dcfbfb1a083af0cfd5de79b6c1c0b2d5f0ba0279d9a82890a968a1fc"
c60_dataset="${workspace}/eval/c60-counterfactual-failure-dataset-v1/dataset.pt"
c60_observations="${workspace}/eval/c60-counterfactual-failure-dataset-v1/observations.jsonl"

for path in "${python_bin}" "${source_root}/action_dit.py" "${c58_report}" \
  "${c60_dataset}" "${c60_observations}" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
  "${workspace}/data/v7_dense_h3_cache/stats.pt" \
  "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  "${workspace}/models/MiniMax-H3/vae/config.json" \
  "${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"; do
  [[ -e "${path}" ]] || { echo "missing C56b online probe input: ${path}" >&2; exit 2; }
done
[[ "$(sha256sum "${c58_report}" | awk '{print $1}')" == "${c58_report_sha}" ]]
"${python_bin}" - "${c58_report}" <<'PY'
import json
import sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
if (
    report.get("status") != "PASS_ONLINE_FROZEN_H3_ONE_STEP"
    or not report.get("parity", {}).get("kv_all_exact")
    or not report.get("parity", {}).get("action_exact")
    or not report.get("training", {}).get("all_30_blocks_nonzero_gradient")
):
    raise SystemExit("C58 online parent gate is not PASS")
PY
[[ ! -e "${output_root}" ]] || { echo "refusing existing C56b online output" >&2; exit 2; }
if pgrep -f '[p]recompute_c(55|49).*c60-counterfactual' >/dev/null; then
  echo "NO_GO: an obsolete C60 cache process is running" >&2
  exit 64
fi

export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}${PYTHONPATH:+:${PYTHONPATH}}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
cu13_lib="$(${python_bin} - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib")
PY
)"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${workspace}/tmp/c56b-online-probe"
mkdir -p "${TMPDIR}" "$(dirname "${output_root}")"
cd "${project}"

"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/h3wam/probe_c56b_fact_online.py \
  --source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
  --prepared-data-root "${workspace}/data/v7_dense_h3_cache" \
  --c60-dataset "${c60_dataset}" --c60-observations "${c60_observations}" \
  --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  --h3-model "${workspace}/models/MiniMax-H3" \
  --d0-parent-checkpoint "${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt" \
  --output-root "${output_root}"
