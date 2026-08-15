#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${workspace}/runtime/h3-int8-native/bin/python"
split_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
subdir="h3_int8_dreamwam_kv_5x32_dualviewgrid_stage112k_120k_v1"
checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
log_root="${workspace}/eval/c15-grid-cache-stage8k-v1"

[[ ! -e "${log_root}/STARTED" && ! -e "${log_root}/COMPLETED" ]] || {
  echo "refusing to reuse C15 stage cache run" >&2; exit 1;
}
mkdir -p "${log_root}"
date -Iseconds >"${log_root}/STARTED"
export PYTHONPATH="${project}/src:${project}"
cu13_lib="$(${python_bin} - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13/lib")
PY
)"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

common=(
  "${split_manifest}" --source-manifest "${source_manifest}"
  --cache-root "${cache_root}" --h3-checkpoint "${checkpoint}"
  --dreamwam-kv-carrier --dreamwam-kv-output-subdir "${subdir}"
  --dreamwam-kv-layers 9 19 29 39 49 --capture-token-count 32
  --kv-pool-strategy dual_view_spatial_grid_4x4_each_v1
  --action-horizon 32 --target-latent-frames 12
)

# Preserve the trainer's stable restore probe in addition to the disjoint stage.
CUDA_VISIBLE_DEVICES=0 "${python_bin}" "${project}/scripts/h3wam/precompute_h3_int8_features.py" \
  "${common[@]}" --sample-offset 0 --limit 1 --progress-every 1 \
  >"${log_root}/probe.log" 2>&1

pids=()
for rank in {0..7}; do
  CUDA_VISIBLE_DEVICES="${rank}" "${python_bin}" "${project}/scripts/h3wam/precompute_h3_int8_features.py" \
    "${common[@]}" --sample-offset 112000 --limit 8000 \
    --num-shards 8 --shard-index "${rank}" --progress-every 100 \
    >"${log_root}/worker_${rank}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || exit 1

"${python_bin}" - "${split_manifest}" "${cache_root}/${subdir}" "${log_root}" <<'PY'
import json, os, sys
from pathlib import Path
import torch
manifest, root, output_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
selected = [rows[0], *rows[112000:120000]]
missing, invalid = [], []
for row in selected:
    path = root / f"{row['id']}.pt"
    if not path.is_file():
        missing.append(row["id"]); continue
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensors = [tensor for item in payload["video_kv_cache"].values() for tensor in item.values()]
    if (
        payload.get("capture_token_strategy") != "dual_view_spatial_grid_4x4_each_v1"
        or payload.get("manifest_items") != 222929
        or any(tuple(t.shape) != (32, 56, 128) for t in tensors)
        or not all(torch.isfinite(t.float()).all().item() for t in tensors)
    ):
        invalid.append(row["id"])
report = {
    "format": "h3-c15-grid-cache-stage8k-v1", "probe_samples": 1,
    "stage_sample_offset": 112000, "stage_samples": 8000,
    "global_parallel_workers": 8, "strategy": "dual_view_spatial_grid_4x4_each_v1",
    "missing": missing, "invalid": invalid,
    "status": "READY" if not missing and not invalid else "FAIL",
}
if report["status"] != "READY": raise SystemExit(json.dumps(report))
destination = output_root / "COMPLETED"
temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(report, indent=2) + "\n"); os.replace(temporary, destination)
print(json.dumps(report, sort_keys=True))
PY
