#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/h3-wam/candidate-d0-rollout-96976ce/project}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/h3-wam/runtime/h3-int8-native/bin/python}"
SOURCE_ROOT="${H3WAM_FASTWAM_SOURCE_ROOT:-/mnt/h3-wam/upstream-readonly/FastWAM-45d8e145/wan22}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/h3-wam/outputs/c58b-fastwam-layerwise-v1/probe10}"
MANIFEST="${MANIFEST:-/mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_all.jsonl}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/h3-wam/data/v7_dense_h3_cache}"
KV_SUBDIR="${KV_SUBDIR:-h3_int8_fastwam_kv_30x32_dense_v1}"
D0_PARENT="${D0_PARENT:-/mnt/h3-wam/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt}"
CACHE_READY="${CACHE_READY:-/mnt/h3-wam/outputs/c58b-fastwam-layerwise-v1/cache-canary80/READY.json}"

for path in "${PYTHON_BIN}" "${MANIFEST}" "${SOURCE_MANIFEST}" "${D0_PARENT}" \
  "${CACHE_READY}" "${SOURCE_ROOT}/action_dit.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing C58b probe input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}"
cuda13_lib="$(${PYTHON_BIN} -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["purelib"])/"nvidia"/"cu13"/"lib")')"
export LD_LIBRARY_PATH="${cuda13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="${PROJECT_ROOT}/src"
export H3WAM_FASTWAM_SOURCE_ROOT="${SOURCE_ROOT}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=8 \
  scripts/h3wam/train_h3_fastwam_full_tower.py \
  "${MANIFEST}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --cache-root "${CACHE_ROOT}" \
  --kv-subdir "${KV_SUBDIR}" \
  --d0-parent-checkpoint "${D0_PARENT}" \
  --carrier-mode uniform_h3_50_to_action30 \
  --verify-h3-checkpoint-sha256 \
  --steps 10 \
  --sample-offset 112000 \
  --limit 80 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-workers 0 \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --warmup-steps 1000 \
  --scheduler-horizon 10000 \
  --min-learning-rate 1e-6 \
  --action-horizon 32 \
  --action-shift 5 \
  --save-checkpoint "${OUTPUT_ROOT}/c58b_s10.pt" \
  --output "${OUTPUT_ROOT}/train_s10.json" 2>&1 | tee "${OUTPUT_ROOT}/train_s10.log"

echo "[C58b] layer-wise full30 10-step probe complete"
