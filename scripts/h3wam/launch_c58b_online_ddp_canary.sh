#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-ddp-canary10}"
manifest="${MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}"
source_manifest="${SOURCE_MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl}"
cache_root="${CACHE_ROOT:-${workspace}/data/v7_dense_h3_cache}"
h3_checkpoint="${H3_CHECKPOINT:-${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
d0_parent="${D0_PARENT:-${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt}"
sample_offset="${SAMPLE_OFFSET:-112000}"

for path in "${python_bin}" "${manifest}" "${source_manifest}" \
  "${cache_root}/stats.pt" "${h3_checkpoint}" "${d0_parent}" \
  "${source_root}/action_dit.py"; do
  [[ -e "${path}" ]] || { echo "missing C58b online DDP input: ${path}" >&2; exit 2; }
done
[[ ! -e "${output_root}" ]] || {
  echo "refusing existing C58b online canary output: ${output_root}" >&2
  exit 2
}

mkdir -p "${output_root}"
cuda13_lib="$(${python_bin} -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["purelib"])/"nvidia"/"cu13"/"lib")')"
export LD_LIBRARY_PATH="${cuda13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="${project}/src"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
cd "${project}"

common=(
  scripts/h3wam/train_h3_fastwam_full_tower.py "${manifest}"
  --source-manifest "${source_manifest}"
  --cache-root "${cache_root}"
  --online-h3-checkpoint "${h3_checkpoint}"
  --d0-parent-checkpoint "${d0_parent}"
  --carrier-mode uniform_h3_50_to_action30
  --verify-h3-checkpoint-sha256
  --sample-offset "${sample_offset}"
  --probe-sample-offset "${sample_offset}"
  --per-device-batch-size 1
  --gradient-accumulation-steps 1
  --num-workers 0
  --learning-rate 1e-4
  --weight-decay 0.01
  --warmup-steps 1000
  --scheduler-horizon 10000
  --min-learning-rate 1e-6
  --action-horizon 32
  --action-shift 5
  --use-gradient-checkpointing
)

"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node=8 \
  "${common[@]}" --steps 10 --limit 80 \
  --save-checkpoint "${output_root}/c58b_online_s10.pt" \
  --output "${output_root}/train_s10.json" 2>&1 | tee "${output_root}/train_s10.log"

"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node=8 \
  "${common[@]}" --steps 10 --limit 80 \
  --load-checkpoint "${output_root}/c58b_online_s10.pt" \
  --restore-check-only \
  --output "${output_root}/restore_s10.json" 2>&1 | tee "${output_root}/restore_s10.log"

"${python_bin}" - "${output_root}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root=Path(sys.argv[1]); train=json.loads((root/"train_s10.json").read_text()); restore=json.loads((root/"restore_s10.json").read_text())
audits=train["per_rank_runtime"]
ids=[value for rank in audits for value in rank["sample_ids"]]
seeds=[value for rank in audits for value in rank["flow_seeds"]]
if train["world_size"] != 8 or train["training_samples"] != 80 or len(ids) != 80 or len(set(ids)) != 80:
    raise SystemExit("C58b online DDP sample/order gate failed")
if len(seeds) != 80 or len(set(seeds)) != 80:
    raise SystemExit("C58b online DDP flow seed gate failed")
if any(len(row["block_gradient_norms"]) != 30 or min(row["block_gradient_norms"]) <= 0 for row in train["history"]):
    raise SystemExit("C58b online DDP 30-layer gradient gate failed")
if restore["restore_probe_max_abs"] != 0.0:
    raise SystemExit("C58b online strict restore gate failed")
checkpoint=root/"c58b_online_s10.pt"
digest=hashlib.sha256()
with checkpoint.open("rb") as stream:
    while chunk := stream.read(16*1024*1024): digest.update(chunk)
report={
 "format":"h3wam-c58b-online-ddp-canary-v1", "status":"PASS_ONLINE_DDP_CANARY",
 "world_size":8, "global_batch":8, "steps":10, "unique_samples":80,
 "unique_flow_seeds":80, "all_30_gradients":True, "restore_max_abs":0.0,
 "seconds_per_step":train["seconds_per_step"],
 "per_rank_peak_allocated_gib":[row["peak_allocated_gib"] for row in audits],
 "per_rank_peak_reserved_gib":[row["peak_reserved_gib"] for row in audits],
 "checkpoint_sha256":digest.hexdigest(),
 "checkpoint":str(checkpoint),
 "permission":"GO_ONLINE_10000",
 "claim_boundary":"Mechanical DDP/restore gate only; rollout effectiveness is not yet proven."
}
target=root/"READY.json"; temporary=target.with_suffix(".partial"); temporary.write_text(json.dumps(report,indent=2)+"\n"); os.replace(temporary,target)
print(json.dumps(report,indent=2))
PY

echo "[C58b] online 8-GPU DDP canary and strict restore passed"
