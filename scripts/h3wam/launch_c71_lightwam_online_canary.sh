#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C71_SOURCE_SNAPSHOT:?C71 canary requires an immutable source snapshot}"
freeze_sha="${C71_SOURCE_FREEZE_SHA256:?Set the reviewed SOURCE_FREEZE SHA256}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c71-lightwam-state-fusion-v1/online-canary10}"
verifier="${project}/scripts/h3wam/freeze_c67_rollout_source.py"

[[ ! -e "${output_root}" ]] || { echo "refusing existing C71 canary output: ${output_root}" >&2; exit 2; }
for path in "${python_bin}" "${verifier}" "${project}/SOURCE_FREEZE.json" \
  "${project}/scripts/h3wam/train_c71_lightwam_online.py" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
  "${workspace}/data/v7_dense_h3_cache/stats.pt" \
  "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"; do
  [[ -e "${path}" ]] || { echo "missing C71 canary input: ${path}" >&2; exit 2; }
done

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${verifier}" --verify --snapshot "${project}" --expected-manifest-sha256 "${freeze_sha}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export H3WAM_LIGHTWAM_SOURCE_ROOT="${project}/third_party/Light-WAM/src/lightwam/models/wan22"
cu13_lib="$(${python_bin} -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["purelib"])/"nvidia"/"cu13"/"lib")')"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export TMPDIR="${workspace}/tmp/c71-lightwam-train"
mkdir -p "${TMPDIR}" "${output_root}/checkpoints" "${output_root}/reports" "${output_root}/logs"
cd "${project}"

common=(
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
  --source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
  --cache-root "${workspace}/data/v7_dense_h3_cache"
  --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  --source-freeze "${project}/SOURCE_FREEZE.json"
  --expected-source-freeze-sha256 "${freeze_sha}"
  --probe-sample-offset 0 --learning-rate 1e-4 --weight-decay 0.0
  --warmup-steps 1000 --scheduler-horizon 10000 --min-learning-rate 0
  --num-workers 0 --action-horizon 32
)

CUDA_VISIBLE_DEVICES="${C71_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  "${python_bin}" -m torch.distributed.run --standalone --nproc-per-node=8 \
  "${project}/scripts/h3wam/train_c71_lightwam_online.py" "${common[@]}" \
  --steps 10 --sample-offset 136000 --limit 80 \
  --save-checkpoint "${output_root}/checkpoints/c71_online_s10.pt" \
  --output "${output_root}/reports/train_s10.json" 2>&1 | tee "${output_root}/logs/train_s10.log"

CUDA_VISIBLE_DEVICES="${C71_RESTORE_CUDA_VISIBLE_DEVICES:-0}" \
  "${python_bin}" "${project}/scripts/h3wam/train_c71_lightwam_online.py" "${common[@]}" \
  --steps 0 --sample-offset 136080 --limit 1 --restore-check-only \
  --load-checkpoint "${output_root}/checkpoints/c71_online_s10.pt" \
  --output "${output_root}/reports/restore_s10.json" 2>&1 | tee "${output_root}/logs/restore_s10.log"

"${python_bin}" - "${output_root}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
train = json.loads((root / "reports/train_s10.json").read_text())
restore = json.loads((root / "reports/restore_s10.json").read_text())
gates = {
    "train_stage": train["status"] == "PASS_C71_CHECKPOINTED_TRAIN_STAGE",
    "steps": train["completed_steps"] == 10 and train["training_samples"] == 80,
    "finite_positive_gradients": all(
        all(value > 0 for value in row["gradient_norms"].values()) for row in train["history"]
    ),
    "strict_restore": restore["status"] == "PASS_C71_STRICT_RESTORE" and restore["restore_probe_max_abs"] == 0.0,
}
checkpoint = root / "checkpoints/c71_online_s10.pt"
digest = hashlib.sha256()
with checkpoint.open("rb") as stream:
    while chunk := stream.read(16 * 1024 * 1024): digest.update(chunk)
ready = {
    "format": "h3wam-c71-lightwam-online-canary-ready-v1",
    "status": "PASS_C71_ONLINE_DDP_CANARY" if all(gates.values()) else "FAIL_C71_ONLINE_DDP_CANARY",
    "permission": "GO_LONG" if all(gates.values()) else "NO_GO",
    "effect_status": "NOT_EVIDENCE_READY",
    "gates": gates,
    "completed_steps": 10,
    "global_batch": 8,
    "training_samples": 80,
    "unique_train_windows": train["unique_train_windows"],
    "effective_epochs": train["cumulative_effective_epochs"],
    "seconds_per_step": train["seconds_per_step"],
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": digest.hexdigest(),
    "claim_boundary": "Mechanical canary only; no effectiveness conclusion.",
}
temporary = root / f".READY.json.{os.getpid()}.partial"
temporary.write_text(json.dumps(ready, indent=2) + "\n")
os.replace(temporary, root / "READY.json")
print(json.dumps(ready, indent=2))
if ready["permission"] != "GO_LONG": raise SystemExit(64)
PY
