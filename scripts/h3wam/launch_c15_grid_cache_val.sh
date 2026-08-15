#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${workspace}/runtime/h3-int8-native/bin/python"
manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_val.jsonl"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
subdir="h3_int8_dreamwam_kv_5x32_dualviewgrid_stage112k_120k_v1"
checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
log_root="${workspace}/eval/c15-grid-cache-val-v1"

[[ ! -e "${log_root}/STARTED" && ! -e "${log_root}/COMPLETED" ]] || {
  echo "refusing to reuse C15 validation cache run" >&2; exit 1;
}
mkdir -p "${log_root}"; date -Iseconds >"${log_root}/STARTED"
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
    "${manifest}" --source-manifest "${source_manifest}" \
    --cache-root "${cache_root}" --h3-checkpoint "${checkpoint}" \
    --dreamwam-kv-carrier --dreamwam-kv-output-subdir "${subdir}" \
    --dreamwam-kv-layers 9 19 29 39 49 --capture-token-count 32 \
    --kv-pool-strategy dual_view_spatial_grid_4x4_each_v1 \
    --action-horizon 32 --target-latent-frames 12 \
    --num-shards 8 --shard-index "${rank}" --progress-every 250 \
    >"${log_root}/worker_${rank}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || exit 1

"${python_bin}" - "${manifest}" "${cache_root}/${subdir}" "${log_root}" <<'PY'
import json, os, sys
from pathlib import Path
manifest, root, output_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ids = [json.loads(line)["id"] for line in manifest.read_text().splitlines() if line.strip()]
missing = [sample_id for sample_id in ids if not (root / f"{sample_id}.pt").is_file()]
report = {"format": "h3-c15-grid-cache-val-v1", "samples": len(ids), "missing": missing, "status": "READY" if not missing else "FAIL"}
if missing: raise SystemExit(json.dumps(report))
destination = output_root / "COMPLETED"
temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(report, indent=2) + "\n"); os.replace(temporary, destination)
print(json.dumps(report, sort_keys=True))
PY
