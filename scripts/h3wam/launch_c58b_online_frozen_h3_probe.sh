#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/h3-wam/candidate-d0-rollout-96976ce/project}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/h3-wam/runtime/h3-int8-native/bin/python}"
SOURCE_ROOT="${H3WAM_FASTWAM_SOURCE_ROOT:-/mnt/h3-wam/upstream-readonly/FastWAM-45d8e145/wan22}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/h3-wam/outputs/c58b-fastwam-layerwise-v1/online-one-step}"
MANIFEST="${MANIFEST:-/mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_all.jsonl}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/h3-wam/data/v7_dense_h3_cache}"
KV_SUBDIR="${KV_SUBDIR:-h3_int8_fastwam_kv_30x32_dense_v1}"
H3_CHECKPOINT="${H3_CHECKPOINT:-/mnt/h3-wam/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
D0_PARENT="${D0_PARENT:-/mnt/h3-wam/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-112000}"
GPU="${GPU:-0}"

for path in "${PYTHON_BIN}" "${MANIFEST}" "${SOURCE_MANIFEST}" \
  "${H3_CHECKPOINT}" "${D0_PARENT}" "${SOURCE_ROOT}/action_dit.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing C58b online probe input: ${path}" >&2
    exit 1
  fi
done
if [[ -e "${OUTPUT_ROOT}/report.json" ]]; then
  echo "refusing to overwrite existing C58b online report: ${OUTPUT_ROOT}/report.json" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
cuda13_lib="$(${PYTHON_BIN} -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["purelib"])/"nvidia"/"cu13"/"lib")')"
export LD_LIBRARY_PATH="${cuda13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="${PROJECT_ROOT}/src"
export H3WAM_FASTWAM_SOURCE_ROOT="${SOURCE_ROOT}"
cd "${PROJECT_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" \
  scripts/h3wam/probe_c58b_online_frozen_h3.py \
  "${MANIFEST}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --cache-root "${CACHE_ROOT}" \
  --kv-subdir "${KV_SUBDIR}" \
  --h3-checkpoint "${H3_CHECKPOINT}" \
  --d0-parent-checkpoint "${D0_PARENT}" \
  --sample-offset "${SAMPLE_OFFSET}" \
  --output "${OUTPUT_ROOT}/report.json" 2>&1 | tee "${OUTPUT_ROOT}/probe.log"

echo "[C58b] online frozen-H3 one-step gate complete"
