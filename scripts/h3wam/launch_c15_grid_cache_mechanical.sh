#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${workspace}/runtime/h3-int8-native/bin/python"
manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
subdir="h3_int8_dreamwam_kv_5x32_dualviewgrid_mechanical64_v1"
checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
log_root="${workspace}/eval/c15-grid-cache-mechanical-v1"

mkdir -p "${log_root}"
test -x "${python_bin}"
test -f "${manifest}"
test -f "${checkpoint}"
export PYTHONPATH="${project}/src:${project}"
cu13_lib="$(${python_bin} - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13/lib")
PY
)"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

pids=()
for rank in {0..7}; do
  CUDA_VISIBLE_DEVICES="${rank}" "${python_bin}" "${project}/scripts/h3wam/precompute_h3_int8_features.py" \
    "${manifest}" --cache-root "${cache_root}" --h3-checkpoint "${checkpoint}" \
    --dreamwam-kv-carrier --dreamwam-kv-output-subdir "${subdir}" \
    --dreamwam-kv-layers 9 19 29 39 49 --capture-token-count 32 \
    --kv-pool-strategy dual_view_spatial_grid_4x4_each_v1 \
    --action-horizon 32 --target-latent-frames 12 --limit 64 \
    --num-shards 8 --shard-index "${rank}" --progress-every 8 \
    >"${log_root}/worker_${rank}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || exit 1

"${python_bin}" - "${manifest}" "${cache_root}/${subdir}" "${log_root}" <<'PY'
import json, os, sys
from pathlib import Path
import torch
manifest, root, output_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()][:64]
records = []
for row in rows:
    path = root / f"{row['id']}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensors = [tensor for item in payload["video_kv_cache"].values() for tensor in item.values()]
    records.append({
        "sample_id": row["id"], "path": str(path),
        "strategy": payload["capture_token_strategy"],
        "layers": list(payload["layers"]),
        "shapes": [list(shape) for shape in sorted({tuple(tensor.shape) for tensor in tensors})],
        "finite": all(torch.isfinite(tensor.float()).all().item() for tensor in tensors),
        "unique_storage": len({tensor.untyped_storage().data_ptr() for tensor in tensors}) == len(tensors),
    })
if len(records) != 64 or not all(r["finite"] and r["unique_storage"] for r in records):
    raise SystemExit("C15 mechanical cache audit failed")
report = {
    "format": "h3-c15-dual-view-grid-cache-mechanical-v1",
    "samples": 64, "global_parallel_workers": 8,
    "source_grid": [7, 14], "output_grid_per_view": [4, 4],
    "tokens": 32, "strategy": "dual_view_spatial_grid_4x4_each_v1",
    "status": "PASS_MECHANICAL_NOT_EFFECT_EVIDENCE", "records": records,
}
for name in ("audit.json", "COMPLETED"):
    destination = output_root / name
    temporary = destination.with_name(f".{name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, destination)
print(json.dumps({k: v for k, v in report.items() if k != "records"}, sort_keys=True))
PY
