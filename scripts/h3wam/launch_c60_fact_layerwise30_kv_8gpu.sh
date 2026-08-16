#!/usr/bin/env bash
set -Eeuo pipefail

# C56b requires the exact C58b 30-layer H3 prefix, not C56a's five-layer cache.
workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
asset_root="${ASSET_ROOT:-${workspace}/data/c60-h3-fact-cache-v1}"
root="${OUTPUT_ROOT:-${asset_root}/kv-layerwise30}"
dataset="${workspace}/eval/c60-counterfactual-failure-dataset-v1/dataset.pt"
observations="${workspace}/eval/c60-counterfactual-failure-dataset-v1/observations.jsonl"
dataset_sha="1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4"
observations_sha="b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55"
layers=(0 2 3 5 7 8 10 12 14 15 17 19 20 22 24 25 27 29 30 32 34 35 37 39 41 42 44 46 47 49)

[[ ! -e "${root}/READY.json" ]] || { echo "C60 layerwise30 K/V already READY"; exit 0; }
[[ -f "${asset_root}/READY.json" ]] || {
  echo "C60 five-layer/future asset must be READY before layerwise extension" >&2
  exit 2
}
[[ "$(sha256sum "${dataset}" | awk '{print $1}')" == "${dataset_sha}" ]]
[[ "$(sha256sum "${observations}" | awk '{print $1}')" == "${observations_sha}" ]]
mkdir -p "${root}/logs" "${workspace}/tmp/c60-layerwise30"

export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}${PYTHONPATH:+:${PYTHONPATH}}"
cu13_lib="$(${python_bin} - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib")
PY
)"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${workspace}/tmp/c60-layerwise30"
cd "${project}"

for base in 0 8 16 24; do
  pids=()
  shards=()
  for gpu in 0 1 2 3 4 5 6 7; do
    shard=$((base + gpu))
    marker="${root}/markers/shard${shard}.json"
    if [[ -e "${marker}" ]]; then
      continue
    fi
    log="${root}/logs/shard${shard}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" \
      scripts/h3wam/precompute_c55_rollout_kv_shard.py \
      --dataset "${dataset}" --observations "${observations}" \
      --expected-dataset-sha256 "${dataset_sha}" \
      --expected-observations-sha256 "${observations_sha}" \
      --cache-root "${workspace}/data/v7_dense_h3_cache" \
      --source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
      --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
      --h3-model "${workspace}/models/MiniMax-H3" \
      --output-root "${root}" --splits train validation \
      --shard "${shard}" --num-shards 32 --device cuda:0 \
      --layers "${layers[@]}" >"${log}" 2>&1 &
    pids+=("$!"); shards+=("${shard}")
  done
  failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "C60 layerwise30 worker failed: shard ${shards[$index]}" >&2
      failed=1
    fi
  done
  (( failed == 0 )) || exit 1
done

"${python_bin}" scripts/h3wam/finalize_c55_rollout_kv.py \
  --dataset "${dataset}" --observations "${observations}" \
  --expected-dataset-sha256 "${dataset_sha}" \
  --expected-observations-sha256 "${observations_sha}" \
  --root "${root}" --output "${root}/READY.json" --layers "${layers[@]}"

"${python_bin}" - "${asset_root}" "${dataset_sha}" "${observations_sha}" "${layers[@]}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

asset_root = Path(sys.argv[1])
dataset_sha, observations_sha = sys.argv[2:4]
layers = [int(value) for value in sys.argv[4:]]
kv_path = asset_root / "kv-layerwise30" / "READY.json"
future_path = asset_root / "features" / "report.json"
kv = json.loads(kv_path.read_text())
future = json.loads(future_path.read_text())
partials = list(asset_root.rglob("*.partial"))
if (
    not kv.get("ready")
    or kv.get("dataset_sha256") != dataset_sha
    or kv.get("observations_sha256") != observations_sha
    or kv.get("layers") != layers
    or kv.get("missing") != 0
    or kv.get("extra") != 0
    or kv.get("partials") != 0
    or future.get("status") != "PASS_C49_DENSE_VALUE_H3_FEATURES"
    or future.get("observations_sha256") != observations_sha
    or partials
):
    raise SystemExit("C60 C56b layerwise aggregate audit failed")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

result = {
    "format": "h3wam-c60-fact-layerwise30-cache-ready-v1",
    "ready": True,
    "asset_scope": "C56b shared 30-layer P/A/G/V/I training",
    "dataset_sha256": dataset_sha,
    "observations_sha256": observations_sha,
    "kv_layers": layers,
    "kv_items": kv["items"],
    "future_observations": future["observations"],
    "future_feature_report_sha256": sha256(future_path),
    "missing": 0,
    "extra": 0,
    "partials": 0,
}
temporary = asset_root / f".READY_LAYERWISE30.json.{os.getpid()}.partial"
temporary.write_text(json.dumps(result, indent=2) + "\n")
os.replace(temporary, asset_root / "READY_LAYERWISE30.json")
print(json.dumps(result, sort_keys=True))
PY
