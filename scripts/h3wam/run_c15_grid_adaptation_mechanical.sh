#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${workspace}/runtime/h3-int8-native/bin/python"
candidate_root="${workspace}/data/v7_multisuite_dense_candidate"
cache_root="${workspace}/data/v7_dense_h3_cache"
kv_subdir="h3_int8_dreamwam_kv_5x32_dualviewgrid_stage112k_120k_v1"
parent="${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"
output_root="${workspace}/outputs/c15-grid-adaptation-mechanical-v1"

[[ ! -e "${output_root}" ]] || { echo "refusing existing mechanical output" >&2; exit 1; }
mkdir -p "${output_root}"
while ! "${python_bin}" - "${candidate_root}/manifest_train_uniform.jsonl" "${cache_root}/${kv_subdir}" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
required = [rows[0], *rows[112000:112008]]
raise SystemExit(0 if all((Path(sys.argv[2]) / f"{row['id']}.pt").is_file() for row in required) else 1)
PY
do
  echo "$(date -Iseconds) WAIT_FIRST_GRID_BATCH"; sleep 10
done

export PYTHONPATH="${project}/src:${project}"
cu13_lib="$(${python_bin} - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13/lib")
PY
)"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
cd "${project}"
common=(
  "${candidate_root}/manifest_train_uniform.jsonl"
  --source-manifest "${candidate_root}/manifest_all.jsonl"
  --cache-root "${cache_root}" --kv-subdir "${kv_subdir}"
  --enable-dreamwam-kv-carrier --enable-d0-repeat-layer49
  --kv-pool-strategy dual_view_spatial_grid_4x4_each_v1
  --verify-h3-checkpoint-sha256 --action-horizon 32
  --per-device-batch-size 1 --gradient-accumulation-steps 1 --num-workers 0
  --learning-rate 1e-4 --weight-decay 0.01 --warmup-steps 1000
  --scheduler-horizon 21700 --min-learning-rate 1e-6 --action-shift 5 --seed 42
)
checkpoint="${output_root}/d0_grid_h32_s14001.pt"
"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/h3wam/train_h3_int8_dreamwam_kv_carrier.py "${common[@]}" \
  --initialize-carrier-from "${parent}" --steps 1 --sample-offset 112000 --limit 8 \
  --save-checkpoint "${checkpoint}" --output "${output_root}/train.json"
"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/h3wam/train_h3_int8_dreamwam_kv_carrier.py "${common[@]}" \
  --load-checkpoint "${checkpoint}" --restore-check-only \
  --steps 1 --sample-offset 0 --limit 1 --output "${output_root}/restore.json"

"${python_bin}" - "${output_root}" "${checkpoint}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root, checkpoint = Path(sys.argv[1]), Path(sys.argv[2])
train, restore = json.loads((root / "train.json").read_text()), json.loads((root / "restore.json").read_text())
report = {
    "format": "h3-c15-grid-adaptation-mechanical-v1",
    "global_batch": 8, "steps": 1, "samples": 8,
    "completed_steps": train["completed_steps"],
    "restore_max_abs": restore["restore_probe_max_abs"],
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "status": "PASS_MECHANICAL_NOT_EFFECT_EVIDENCE",
}
destination = root / "COMPLETED"; temporary = root / f".COMPLETED.{os.getpid()}.partial"
temporary.write_text(json.dumps(report, indent=2) + "\n"); os.replace(temporary, destination)
print(json.dumps(report, sort_keys=True))
PY
