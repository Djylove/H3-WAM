#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 1 )) || [[ ! "$1" =~ ^[0-3]$ ]]; then
  echo "usage: $0 NODE_INDEX(0..3)" >&2
  exit 2
fi
node_index="$1"

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
root="${OUTPUT_ROOT:-${workspace}/eval/c55-fact-joint-action-v1/kv-full-v1}"
node_marker="${root}/node${node_index}.COMPLETED"
smoke_marker="${workspace}/eval/c55-fact-joint-action-v1/kv-smoke/markers/shard0.json"

[[ -f "${smoke_marker}" ]] || { echo "C55 K/V smoke gate is missing" >&2; exit 2; }
[[ ! -e "${node_marker}" ]] || { echo "node marker already exists" >&2; exit 1; }
mkdir -p "${root}/logs"

export PYTHONPATH="${project}/src:${project}${PYTHONPATH:+:${PYTHONPATH}}"
pytorch_cu13_lib="$("${python_bin}" - <<'PY'
import sysconfig
from pathlib import Path

path = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib"
if not (path / "libnvJitLink.so.13").is_file():
    raise SystemExit(f"missing PyTorch cu13 runtime: {path}")
print(path)
PY
)"
export LD_LIBRARY_PATH="${pytorch_cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${workspace}/tmp/c55-kv-node${node_index}"
mkdir -p "${TMPDIR}"
cd "${project}"

pids=()
for local_gpu in $(seq 0 7); do
  shard=$((node_index * 8 + local_gpu))
  log="${root}/logs/shard${shard}.log"
  CUDA_VISIBLE_DEVICES="${local_gpu}" "${python_bin}" \
    scripts/h3wam/precompute_c55_rollout_kv_shard.py \
    --dataset "${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt" \
    --observations "${workspace}/eval/c48-fact-dense-value-dataset-v1/observations.jsonl" \
    --cache-root "${workspace}/data/v7_dense_h3_cache" \
    --source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
    --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
    --h3-model "${workspace}/models/MiniMax-H3" \
    --output-root "${root}" --splits train validation \
    --shard "${shard}" --num-shards 32 --device cuda:0 \
    >"${log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
(( status == 0 )) || { echo "one or more C55 K/V shards failed" >&2; exit 1; }

"${python_bin}" - "${root}" "${node_index}" >"${node_marker}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
node = int(sys.argv[2])
shards = list(range(node * 8, node * 8 + 8))
markers = [json.loads((root / "markers" / f"shard{i}.json").read_text()) for i in shards]
if [item["shard"] for item in markers] != shards:
    raise SystemExit("C55 node shard marker identity mismatch")
print(json.dumps({"node": node, "shards": shards, "items": sum(x["items"] for x in markers)}, sort_keys=True))
PY
