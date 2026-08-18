#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C71_SOURCE_SNAPSHOT:?C71 long-run orchestrator requires an immutable source snapshot}"
freeze_sha="${C71_SOURCE_FREEZE_SHA256:?Set the reviewed orchestrator SOURCE_FREEZE SHA256}"
training_project="${C71_TRAINING_SOURCE_SNAPSHOT:-${workspace}/code-snapshots/h3-wam-ccf1e43-c71-canary-v1}"
training_freeze_sha="${C71_TRAINING_SOURCE_FREEZE_SHA256:-40d42a1071f2900dda49357a80fa90d9675e69265c6e1741dfdd901af7d2e7ca}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
parent_root="${C71_PARENT_ROOT:-${workspace}/outputs/c71-lightwam-state-fusion-v1/online-long1000-ccf1e43-v1}"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c71-lightwam-state-fusion-v1/online-long10000-v1}"
trainer="${training_project}/scripts/h3wam/train_c71_lightwam_online.py"
verifier="${project}/scripts/h3wam/freeze_c67_rollout_source.py"
training_verifier="${training_project}/scripts/h3wam/freeze_c67_rollout_source.py"
manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"

for path in "${python_bin}" "${trainer}" "${verifier}" "${training_verifier}" "${project}/SOURCE_FREEZE.json" \
  "${training_project}/SOURCE_FREEZE.json" \
  "${manifest}" "${source_manifest}" "${cache_root}/stats.pt" "${h3_checkpoint}" \
  "${parent_root}/checkpoints/c71_online_s1000.pt" \
  "${parent_root}/reports/restore_s1000.json"; do
  [[ -e "${path}" ]] || { echo "missing C71 long-run input: ${path}" >&2; exit 2; }
done
[[ ! -e "${output_root}" ]] || { echo "refusing existing C71 long output: ${output_root}" >&2; exit 2; }

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${verifier}" --verify --snapshot "${project}" \
  --expected-manifest-sha256 "${freeze_sha}"
"${python_bin}" "${training_verifier}" --verify --snapshot "${training_project}" \
  --expected-manifest-sha256 "${training_freeze_sha}"
export PYTHONPATH="${training_project}/third_party/diffusers_h3/src:${training_project}/src:${training_project}"
export H3WAM_LIGHTWAM_SOURCE_ROOT="${training_project}/third_party/Light-WAM/src/lightwam/models/wan22"
export LD_LIBRARY_PATH="${workspace}/runtime/h3-int8-native/lib/python3.11/site-packages/nvidia/cu13/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export TMPDIR="${workspace}/tmp/c71-lightwam-long"
mkdir -p "${TMPDIR}" "${output_root}/checkpoints" "${output_root}/reports" "${output_root}/logs"
cd "${training_project}"

common=(
  "${manifest}" --source-manifest "${source_manifest}" --cache-root "${cache_root}"
  --h3-checkpoint "${h3_checkpoint}" --source-freeze "${training_project}/SOURCE_FREEZE.json"
  --expected-source-freeze-sha256 "${training_freeze_sha}" --probe-sample-offset 0
  --learning-rate 1e-4 --weight-decay 0 --warmup-steps 1000
  --scheduler-horizon 10000 --min-learning-rate 0 --num-workers 0 --action-horizon 32
)

previous="${parent_root}/checkpoints/c71_online_s1000.pt"
completed=1000
offset=0
while (( completed < 10000 )); do
  target=$((completed + 1000))
  checkpoint="${output_root}/checkpoints/c71_online_s${target}.pt"
  report="${output_root}/reports/train_s${target}.json"
  log="${output_root}/logs/train_s${target}.log"
  CUDA_VISIBLE_DEVICES="${C71_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    "${python_bin}" -m torch.distributed.run --standalone --nproc-per-node=8 \
    "${trainer}" "${common[@]}" --steps 1000 --sample-offset "${offset}" --limit 8000 \
    --load-checkpoint "${previous}" --save-checkpoint "${checkpoint}" \
    --output "${report}" 2>&1 | tee "${log}"
  "${python_bin}" - "${report}" "${target}" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); target = int(sys.argv[2])
assert r["status"] == "PASS_C71_CHECKPOINTED_TRAIN_STAGE"
assert r["completed_steps"] == target and r["training_samples"] == 8000
assert r["restore_probe_max_abs"] == 0.0 and len(r["history"]) == 1000
assert all(all(value > 0 for value in row["gradient_norms"].values()) for row in r["history"])
PY
  if [[ "${target}" == 5000 || "${target}" == 10000 ]]; then
    CUDA_VISIBLE_DEVICES="${C71_RESTORE_CUDA_VISIBLE_DEVICES:-0}" \
      "${python_bin}" "${trainer}" "${common[@]}" --steps 0 \
      --sample-offset 144000 --limit 1 --restore-check-only \
      --load-checkpoint "${checkpoint}" \
      --output "${output_root}/reports/restore_s${target}.json" \
      2>&1 | tee "${output_root}/logs/restore_s${target}.log"
  fi
  previous="${checkpoint}"
  completed="${target}"
  offset=$((offset + 8000))
done

"${python_bin}" - "${output_root}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
checkpoint = root / "checkpoints/c71_online_s10000.pt"
restore = json.loads((root / "reports/restore_s10000.json").read_text())
digest = hashlib.sha256()
with checkpoint.open("rb") as stream:
    while chunk := stream.read(16 * 1024 * 1024): digest.update(chunk)
ready = {
    "format": "h3wam-c71-lightwam-online-long10000-ready-v1",
    "status": "PASS_C71_ONLINE_LONG10000_STRICT_RESTORE" if restore["restore_probe_max_abs"] == 0.0 else "FAIL_C71_ONLINE_LONG10000",
    "effect_status": "NOT_EVIDENCE_READY",
    "completed_steps": 10000,
    "global_batch": 8,
    "training_samples": 80000,
    "unique_train_windows": 200779,
    "effective_epochs": 80000 / 200779,
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": digest.hexdigest(),
    "restore_probe_max_abs": restore["restore_probe_max_abs"],
    "claim_boundary": "Training and restore only; balanced80 and closed-loop are separate gates.",
}
temporary = root / f".READY.json.{os.getpid()}.partial"
temporary.write_text(json.dumps(ready, indent=2) + "\n")
os.replace(temporary, root / "READY.json")
print(json.dumps(ready, indent=2))
if not ready["status"].startswith("PASS_"): raise SystemExit(64)
PY
