#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/h3-wam/candidate-d0-rollout-96976ce/project}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/h3-wam/runtime/h3-int8-native/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/h3-wam/data/v7_dense_h3_cache}"
MANIFEST="${MANIFEST:-/mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_all.jsonl}"
H3_CHECKPOINT="${H3_CHECKPOINT:-/mnt/h3-wam/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
KV_SUBDIR="${KV_SUBDIR:-h3_int8_fastwam_kv_30x32_dense_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/h3-wam/outputs/c58b-fastwam-layerwise-v1/cache-canary80}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-112000}"
LIMIT="${LIMIT:-80}"
NUM_SHARDS="${NUM_SHARDS:-8}"

layers=(0 2 3 5 7 8 10 12 14 15 17 19 20 22 24 25 27 29 30 32 34 35 37 39 41 42 44 46 47 49)

for path in "${PYTHON_BIN}" "${MANIFEST}" "${SOURCE_MANIFEST}" "${H3_CHECKPOINT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing C58b cache input: ${path}" >&2
    exit 1
  fi
done
if (( NUM_SHARDS <= 0 || LIMIT <= 0 )); then
  echo "NUM_SHARDS and LIMIT must be positive" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs" "${CACHE_ROOT}/${KV_SUBDIR}"
cuda13_lib="$(${PYTHON_BIN} -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["purelib"])/"nvidia"/"cu13"/"lib")')"
export LD_LIBRARY_PATH="${cuda13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="${PROJECT_ROOT}/src"
cd "${PROJECT_ROOT}"

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  log="${OUTPUT_ROOT}/logs/cache_shard${shard}.log"
  CUDA_VISIBLE_DEVICES="${shard}" "${PYTHON_BIN}" \
    scripts/h3wam/precompute_h3_int8_features.py \
    "${MANIFEST}" \
    --source-manifest "${SOURCE_MANIFEST}" \
    --cache-root "${CACHE_ROOT}" \
    --h3-checkpoint "${H3_CHECKPOINT}" \
    --dreamwam-kv-carrier \
    --dreamwam-kv-output-subdir "${KV_SUBDIR}" \
    --dreamwam-kv-layers "${layers[@]}" \
    --capture-token-count 32 \
    --action-horizon 32 \
    --target-latent-frames 12 \
    --timestep 1 \
    --condition-video-timestep 1 \
    --sample-offset "${SAMPLE_OFFSET}" \
    --limit "${LIMIT}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard}" \
    --device cuda:0 \
    --progress-every 1 >"${log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if (( status != 0 )); then
  echo "one or more C58b cache shards failed" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${MANIFEST}" "${CACHE_ROOT}/${KV_SUBDIR}" \
  "${SAMPLE_OFFSET}" "${LIMIT}" "${OUTPUT_ROOT}/READY.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

manifest, root, offset, limit, output = sys.argv[1:]
rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
selected = rows[int(offset):int(offset) + int(limit)]
root = Path(root)
layers = (0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24,
          25, 27, 29, 30, 32, 34, 35, 37, 39, 41, 42, 44, 46, 47, 49)
paths = [root / f"{row['id']}.pt" for row in selected]
missing = [str(path) for path in paths if not path.is_file()]
if missing:
    raise SystemExit(f"missing {len(missing)} C58b cache files")
total = 0
for path in paths:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(payload.get("layers", ())) != layers:
        raise SystemExit(f"layer contract mismatch: {path}")
    cache = payload.get("video_kv_cache", {})
    if set(cache) != set(layers):
        raise SystemExit(f"cache keys mismatch: {path}")
    tensors = [tensor for layer in layers for tensor in cache[layer].values()]
    if any(tuple(tensor.shape) != (32, 56, 128) for tensor in tensors):
        raise SystemExit(f"cache tensor shape mismatch: {path}")
    total += path.stat().st_size
report = {
    "event": "c58b_layerwise_h3_30x32_cache_canary",
    "status": "READY",
    "sample_offset": int(offset),
    "items": len(paths),
    "layers": list(layers),
    "tensor_payload_bytes_per_sample": 30 * 2 * 32 * 56 * 128 * 2,
    "file_bytes": total,
    "estimated_80000_tensor_payload_bytes": 80000 * 30 * 2 * 32 * 56 * 128 * 2,
}
temporary = Path(output).with_suffix(".partial")
temporary.write_text(json.dumps(report, indent=2) + "\n")
os.replace(temporary, output)
PY

echo "[C58b] layer-wise H3 cache canary READY: ${OUTPUT_ROOT}/READY.json"
