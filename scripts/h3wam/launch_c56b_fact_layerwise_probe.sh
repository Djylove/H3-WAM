#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"
cache_root="${CACHE_ROOT:-${workspace}/data/v7_dense_h3_cache}"
kv_subdir="${KV_SUBDIR:-h3_int8_fastwam_kv_30x32_dense_v1}"
cache_ready="${C58B_CACHE_READY:-${workspace}/outputs/c58b-fastwam-layerwise-v1/cache-canary80/READY.json}"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c56b-fact-layerwise-v1/mechanical-probe-8gpu-v1}"
parent="${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"
parent_sha="36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"

[[ -f "${cache_ready}" ]] || { echo "C58b layerwise cache READY is missing" >&2; exit 2; }
[[ ! -e "${output_root}" ]] || { echo "refusing existing C56b output" >&2; exit 2; }
[[ "$(sha256sum "${parent}" | awk '{print $1}')" == "${parent_sha}" ]]

export PYTHONPATH="${project}/src:${project}${PYTHONPATH:+:${PYTHONPATH}}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
cu13_lib="$(${python_bin} - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib")
PY
)"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${workspace}/tmp/c56b-probe"
mkdir -p "${TMPDIR}" "$(dirname "${output_root}")"
cd "${project}"

"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/h3wam/probe_c56b_fact_layerwise_tower.py \
  --manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl" \
  --source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
  --cache-root "${cache_root}" --kv-subdir "${kv_subdir}" \
  --sample-offset 112000 \
  --d0-parent-checkpoint "${parent}" --expected-parent-sha256 "${parent_sha}" \
  --steps "${STEPS:-1}" --learning-rate "${LEARNING_RATE:-2e-5}" \
  --output-root "${output_root}"
