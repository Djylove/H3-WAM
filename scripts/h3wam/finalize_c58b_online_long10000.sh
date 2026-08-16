#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"
root="${OUTPUT_ROOT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000}"
manifest="${MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}"
source_manifest="${SOURCE_MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl}"
cache_root="${CACHE_ROOT:-${workspace}/data/v7_dense_h3_cache}"
h3_checkpoint="${H3_CHECKPOINT:-${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
d0_parent="${D0_PARENT:-${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt}"
checkpoint="${root}/checkpoints/c58b_online_s10000.pt"
train_report="${root}/reports/train_s10000.json"
restore_report="${root}/reports/restore_s10000.json"
restore_log="${root}/logs/restore_s10000.log"
ready="${root}/READY.json"
lock="${root}/.finalizer.lock"

for path in "${python_bin}" "${manifest}" "${source_manifest}" "${cache_root}/stats.pt" \
  "${h3_checkpoint}" "${d0_parent}" "${source_root}/action_dit.py"; do
  [[ -e "${path}" ]] || { echo "missing C58b finalizer input: ${path}" >&2; exit 2; }
done
[[ ! -e "${ready}" ]] || { echo "C58b online s10000 already READY"; exit 0; }
mkdir "${lock}" 2>/dev/null || {
  echo "another C58b online finalizer owns ${lock}" >&2; exit 75;
}
trap 'rmdir "${lock}" 2>/dev/null || true' EXIT

echo "[C58b finalizer] waiting for complete s10000 checkpoint and report"
while [[ ! -f "${checkpoint}" || ! -f "${train_report}" ]]; do sleep 30; done
while pgrep -af '[t]rain_h3_fastwam_full_tower.py' | grep -q 'online-long10000'; do sleep 30; done
[[ ! -e "${restore_report}" ]] || {
  echo "refusing pre-existing unfinalized restore report: ${restore_report}" >&2; exit 2;
}

cuda13_lib="$(${python_bin} -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["purelib"])/"nvidia"/"cu13"/"lib")')"
export LD_LIBRARY_PATH="${cuda13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="${project}/src"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
cd "${project}"

"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node=8 \
  scripts/h3wam/train_h3_fastwam_full_tower.py "${manifest}" \
  --source-manifest "${source_manifest}" --cache-root "${cache_root}" \
  --online-h3-checkpoint "${h3_checkpoint}" --d0-parent-checkpoint "${d0_parent}" \
  --carrier-mode uniform_h3_50_to_action30 --verify-h3-checkpoint-sha256 \
  --steps 1000 --sample-offset 184000 --limit 8000 --probe-sample-offset 112000 \
  --per-device-batch-size 1 --gradient-accumulation-steps 1 --num-workers 0 \
  --learning-rate 1e-4 --weight-decay 0.01 --warmup-steps 1000 \
  --scheduler-horizon 10000 --min-learning-rate 1e-6 --action-horizon 32 \
  --action-shift 5 --use-gradient-checkpointing --load-checkpoint "${checkpoint}" \
  --restore-check-only --output "${restore_report}" 2>&1 | tee "${restore_log}"

"${python_bin}" scripts/h3wam/finalize_c58b_online_long10000.py \
  --root "${root}" --output "${ready}"
echo "[C58b finalizer] strict s10000 restore and READY audit passed"
