#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"
canary_root="${CANARY_ROOT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-ddp-canary10}"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000}"
manifest="${MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}"
source_manifest="${SOURCE_MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl}"
cache_root="${CACHE_ROOT:-${workspace}/data/v7_dense_h3_cache}"
h3_checkpoint="${H3_CHECKPOINT:-${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
d0_parent="${D0_PARENT:-${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt}"
base_offset="${SAMPLE_OFFSET:-112000}"

[[ -f "${canary_root}/READY.json" && -f "${canary_root}/c58b_online_s10.pt" ]] || {
  echo "C58b online DDP canary is not ready" >&2; exit 65;
}
"${python_bin}" - "${canary_root}/READY.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); assert r["status"]=="PASS_ONLINE_DDP_CANARY" and r["permission"]=="GO_ONLINE_10000"
PY
[[ ! -e "${output_root}" ]] || { echo "refusing existing online long output" >&2; exit 2; }
mkdir -p "${output_root}/checkpoints" "${output_root}/reports" "${output_root}/logs"
cuda13_lib="$(${python_bin} -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["purelib"])/"nvidia"/"cu13"/"lib")')"
export LD_LIBRARY_PATH="${cuda13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="${project}/src"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
cd "${project}"

previous="${canary_root}/c58b_online_s10.pt"
completed=10
consumed=80
while (( completed < 10000 )); do
  target=$(( (completed / 1000 + 1) * 1000 ))
  (( target > 10000 )) && target=10000
  steps=$((target - completed))
  offset=$((base_offset + consumed))
  limit=$((steps * 8))
  checkpoint="${output_root}/checkpoints/c58b_online_s${target}.pt"
  report="${output_root}/reports/train_s${target}.json"
  log="${output_root}/logs/train_s${target}.log"
  "${python_bin}" -m torch.distributed.run --standalone --nproc-per-node=8 \
    scripts/h3wam/train_h3_fastwam_full_tower.py "${manifest}" \
    --source-manifest "${source_manifest}" --cache-root "${cache_root}" \
    --online-h3-checkpoint "${h3_checkpoint}" --d0-parent-checkpoint "${d0_parent}" \
    --carrier-mode uniform_h3_50_to_action30 --verify-h3-checkpoint-sha256 \
    --steps "${steps}" --sample-offset "${offset}" --limit "${limit}" \
    --probe-sample-offset "${base_offset}" --per-device-batch-size 1 \
    --gradient-accumulation-steps 1 --num-workers 0 --learning-rate 1e-4 \
    --weight-decay 0.01 --warmup-steps 1000 --scheduler-horizon 10000 \
    --min-learning-rate 1e-6 --action-horizon 32 --action-shift 5 \
    --use-gradient-checkpointing --load-checkpoint "${previous}" \
    --save-checkpoint "${checkpoint}" --output "${report}" 2>&1 | tee "${log}"
  "${python_bin}" - "${report}" "${target}" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); expected=int(sys.argv[2])
assert r["completed_steps"]==expected and r["world_size"]==8
assert all(min(row["block_gradient_norms"])>0 for row in r["history"])
PY
  previous="${checkpoint}"
  consumed=$((consumed + limit))
  completed="${target}"
done
echo "[C58b] online long training reached step ${completed}; starting independent final restore"
bash scripts/h3wam/finalize_c58b_online_long10000.sh
