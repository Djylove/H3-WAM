#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c56b-fact-online-v1/target-norm-train512-v1}"

for path in "${python_bin}" "${source_root}/action_dit.py" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
  "${workspace}/data/v7_dense_h3_cache/stats.pt" \
  "${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt" \
  "${workspace}/eval/c48-fact-dense-value-dataset-v1/observations.jsonl" \
  "${workspace}/eval/c59-fact-failure-active-overlay-v1/COMPLETED.json" \
  "${workspace}/eval/c60-counterfactual-failure-dataset-v1/dataset.pt" \
  "${workspace}/eval/c60-counterfactual-failure-dataset-v1/observations.jsonl" \
  "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  "${workspace}/models/MiniMax-H3/vae/config.json"; do
  [[ -e "${path}" ]] || { echo "missing C56b target-norm input: ${path}" >&2; exit 2; }
done
[[ ! -e "${output_root}" ]] || { echo "refusing existing C56b target norm output" >&2; exit 2; }

export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}${PYTHONPATH:+:${PYTHONPATH}}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
cu13_lib="$(${python_bin} - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib")
PY
)"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${workspace}/tmp/c56b-online-target-norm"
mkdir -p "${TMPDIR}" "$(dirname "${output_root}")"
cd "${project}"

exec "${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/h3wam/fit_c56b_fact_online_target_norm.py \
  --demo-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl" \
  --source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
  --demo-cache-root "${workspace}/data/v7_dense_h3_cache" \
  --c48-dataset "${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt" \
  --c48-observations "${workspace}/eval/c48-fact-dense-value-dataset-v1/observations.jsonl" \
  --c59-overlay-root "${workspace}/eval/c59-fact-failure-active-overlay-v1" \
  --c60-dataset "${workspace}/eval/c60-counterfactual-failure-dataset-v1/dataset.pt" \
  --c60-observations "${workspace}/eval/c60-counterfactual-failure-dataset-v1/observations.jsonl" \
  --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  --h3-model "${workspace}/models/MiniMax-H3" \
  --output-root "${output_root}" --sample-count 512 "$@"
