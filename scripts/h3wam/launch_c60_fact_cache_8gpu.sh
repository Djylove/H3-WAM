#!/usr/bin/env bash
set -Eeuo pipefail

# Build both inputs required by the C56 FACT tracks from the immutable C60
# branches.  Four GPUs extract current-observation H3 K/V while four GPUs
# extract layer-49 representations for both current and future observations.

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
root="${OUTPUT_ROOT:-${workspace}/data/c60-h3-fact-cache-v1}"
dataset="${workspace}/eval/c60-counterfactual-failure-dataset-v1/dataset.pt"
observations="${workspace}/eval/c60-counterfactual-failure-dataset-v1/observations.jsonl"
dataset_sha="1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4"
observations_sha="b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55"

[[ ! -e "${root}/READY.json" ]] || { echo "C60 FACT cache already READY"; exit 0; }
[[ "$(sha256sum "${dataset}" | awk '{print $1}')" == "${dataset_sha}" ]]
[[ "$(sha256sum "${observations}" | awk '{print $1}')" == "${observations_sha}" ]]
mkdir -p "${root}/kv/logs" "${root}/features/shards" \
  "${root}/features/logs" "${workspace}/tmp/c60-cache"

export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}${PYTHONPATH:+:${PYTHONPATH}}"
cu13_lib="$(${python_bin} - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib")
PY
)"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${workspace}/tmp/c60-cache"
cd "${project}"

# A manually launched first round may already be active.  Preserve it and
# begin audited resume only after those writers have exited.
while pgrep -f '[p]recompute_c(55|49).*c60-counterfactual' >/dev/null; do
  sleep 10
done

for base in $(seq 0 4 28); do
  pids=()
  labels=()
  for offset in 0 1 2 3; do
    shard=$((base + offset))
    kv_marker="${root}/kv/markers/shard${shard}.json"
    if [[ ! -e "${kv_marker}" ]]; then
      log="${root}/kv/logs/shard${shard}.log"
      CUDA_VISIBLE_DEVICES="${offset}" "${python_bin}" \
        scripts/h3wam/precompute_c55_rollout_kv_shard.py \
        --dataset "${dataset}" --observations "${observations}" \
        --expected-dataset-sha256 "${dataset_sha}" \
        --expected-observations-sha256 "${observations_sha}" \
        --cache-root "${workspace}/data/v7_dense_h3_cache" \
        --source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
        --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
        --h3-model "${workspace}/models/MiniMax-H3" \
        --output-root "${root}/kv" --splits train validation \
        --shard "${shard}" --num-shards 32 --device cuda:0 \
        >"${log}" 2>&1 &
      pids+=("$!"); labels+=("kv:${shard}")
    fi

    feature="${root}/features/shards/shard${shard}.pt"
    if [[ ! -e "${feature}" ]]; then
      log="${root}/features/logs/shard${shard}.log"
      CUDA_VISIBLE_DEVICES="$((offset + 4))" "${python_bin}" \
        scripts/h3wam/precompute_c49_dense_value_h3_shard.py \
        --observations "${observations}" \
        --cache-root "${workspace}/data/v7_dense_h3_cache" \
        --source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
        --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
        --h3-model "${workspace}/models/MiniMax-H3" \
        --shard "${shard}" --num-shards 32 --device cuda:0 \
        --output "${feature}" >"${log}" 2>&1 &
      pids+=("$!"); labels+=("feature:${shard}")
    fi
  done
  failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "C60 cache worker failed: ${labels[$index]}" >&2
      failed=1
    fi
  done
  (( failed == 0 )) || exit 1
done

for node in 0 1 2 3; do
  printf '{"node":%s,"shards":8}\n' "${node}" >"${root}/features/node${node}.COMPLETED"
done

"${python_bin}" scripts/h3wam/finalize_c55_rollout_kv.py \
  --dataset "${dataset}" --observations "${observations}" \
  --expected-dataset-sha256 "${dataset_sha}" \
  --expected-observations-sha256 "${observations_sha}" \
  --root "${root}/kv" --output "${root}/kv/READY.json"

"${python_bin}" scripts/h3wam/finalize_c49_dense_value_h3_features.py \
  --root "${root}/features" --observations "${observations}" \
  --output "${root}/features/report.json" \
  --projected-output "${root}/features/projected_features.pt"

"${python_bin}" - "${root}" "${dataset_sha}" "${observations_sha}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
dataset_sha, observations_sha = sys.argv[2:]
kv = json.loads((root / "kv" / "READY.json").read_text())
features = json.loads((root / "features" / "report.json").read_text())
expected_feature_names = {f"shard{i}.pt" for i in range(32)}
actual_feature_names = {p.name for p in (root / "features" / "shards").glob("*.pt")}
partials = sorted(str(p) for p in root.rglob("*.partial"))
missing_features = sorted(expected_feature_names - actual_feature_names)
extra_features = sorted(actual_feature_names - expected_feature_names)
if (
    not kv.get("ready")
    or kv.get("dataset_sha256") != dataset_sha
    or kv.get("observations_sha256") != observations_sha
    or features.get("status") != "PASS_C49_DENSE_VALUE_H3_FEATURES"
    or features.get("observations_sha256") != observations_sha
    or missing_features
    or extra_features
    or partials
):
    raise SystemExit("C60 FACT aggregate READY audit failed")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

result = {
    "format": "h3wam-c60-fact-cache-ready-v1",
    "ready": True,
    "dataset_sha256": dataset_sha,
    "observations_sha256": observations_sha,
    "kv_items": kv["items"],
    "feature_observations": features["observations"],
    "projected_features_sha256": sha256(root / "features" / "projected_features.pt"),
    "missing": 0,
    "extra": 0,
    "partials": 0,
    "kv_ready": str(root / "kv" / "READY.json"),
    "feature_report": str(root / "features" / "report.json"),
}
temporary = root / f".READY.json.{os.getpid()}.partial"
temporary.write_text(json.dumps(result, indent=2) + "\n")
os.replace(temporary, root / "READY.json")
print(json.dumps(result, sort_keys=True))
PY
