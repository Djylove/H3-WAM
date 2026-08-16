#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
policy_python="${POLICY_PYTHON:-${workspace}/runtime/h3-int8-native/bin/python}"
sim_python="${SIM_PYTHON:-${workspace}/runtime/conda-py311/bin/python}"
long_root="${C58B_LONG_ROOT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000}"
eval_root="${C58B_FINAL_EVAL_ROOT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-final-eval-v1}"
checkpoint="${long_root}/checkpoints/c58b_online_s10000.pt"
d0_checkpoint="${D0_CHECKPOINT:-${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt}"
long_ready="${long_root}/READY.json"
balanced_report="${eval_root}/balanced80/report.json"
balanced_ready="${eval_root}/balanced80/BALANCED80_READY.json"
rollout_root="${eval_root}/fresh-libero-trial33"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
h3_model="${workspace}/models/MiniMax-H3"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
train_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
val_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_val.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"

mkdir -p "${eval_root}/balanced80" "${rollout_root}"
lock="${eval_root}/.watcher.lock"
mkdir "${lock}" 2>/dev/null || { echo "another C58b final watcher owns ${lock}" >&2; exit 75; }
trap 'rmdir "${lock}" 2>/dev/null || true' EXIT

for path in "${project}" "${policy_python}" "${sim_python}" "${h3_checkpoint}" \
  "${h3_model}" "${source_manifest}" "${train_manifest}" "${val_manifest}" \
  "${cache_root}/stats.pt" "${d0_checkpoint}"; do
  [[ -e "${path}" ]] || { echo "missing C58b final input: ${path}" >&2; exit 2; }
done

while [[ ! -s "${long_ready}" ]]; do
  sleep 30
done

cd "${project}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"

if [[ ! -s "${balanced_report}" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${policy_python}" \
    scripts/h3wam/evaluate_h3_fastwam_full_tower_online.py "${checkpoint}" \
    --ready "${long_ready}" --h3-checkpoint "${h3_checkpoint}" \
    --source-manifest "${source_manifest}" --train-manifest "${train_manifest}" \
    --val-manifest "${val_manifest}" --cache-root "${cache_root}" \
    --device cuda:0 --num-workers 0 --output "${balanced_report}"
fi

"${policy_python}" scripts/h3wam/finalize_c58b_online_balanced80.py \
  --report "${balanced_report}" --output "${balanced_ready}"

run_suite() {
  local arm="$1" suite="$2" gpu="$3"
  local output="${rollout_root}/${arm}/${suite}"
  local policy arm_checkpoint
  local -a gate_args=()
  if [[ "${arm}" == "candidate_c58b" ]]; then
    policy="h3_fastwam_online_int8"
    arm_checkpoint="${checkpoint}"
    gate_args=(--c58b-balanced80-ready "${balanced_ready}")
  elif [[ "${arm}" == "control_d0" ]]; then
    policy="h3_dreamwam_kv_int8"
    arm_checkpoint="${d0_checkpoint}"
  else
    echo "unknown C58b paired arm: ${arm}" >&2
    return 2
  fi
  [[ -s "${output}/results.json" ]] && return 0
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHON_BIN="${sim_python}" SIM_SITE_PACKAGES="/tmp/h3-wam-libero-site" \
  bash "${project}/scripts/h3wam/run_cloud_libero.sh" \
    "${sim_python}" "${project}/scripts/h3wam/rollout_libero.py" \
    --policy "${policy}" --policy-python "${policy_python}" \
    --checkpoint "${arm_checkpoint}" "${gate_args[@]}" \
    --cache-root "${cache_root}" --h3-checkpoint "${h3_checkpoint}" \
    --h3-model "${h3_model}" --dreamwam-source-manifest "${source_manifest}" \
    --device cuda:0 --suite "${suite}" --task-ids 0 1 2 3 4 5 6 7 8 9 \
    --trial-indices 33 --max-steps 400 --wait-steps 0 --replan-steps 8 \
    --action-horizon 32 --h3-feature-audio-horizon 32 --target-latent-frames 12 \
    --model-evaluations 10 --seed 42 --environment-seed 42 \
    --policy-noise-seed-base 330042 --normalized-action-pre-clamp \
    --output-dir "${output}" >"${output}/launcher.log" 2>&1
}

pids=()
index=0
for suite in libero_spatial libero_object libero_goal libero_10; do
  run_suite candidate_c58b "${suite}" "${index}" &
  pids+=("$!")
  index=$((index + 1))
done
for suite in libero_spatial libero_object libero_goal libero_10; do
  run_suite control_d0 "${suite}" "${index}" &
  pids+=("$!")
  index=$((index + 1))
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

"${sim_python}" scripts/h3wam/aggregate_c58b_fresh_libero.py \
  --root "${rollout_root}" --gate "${balanced_ready}" \
  --d0-checkpoint "${d0_checkpoint}" \
  --output "${rollout_root}/RESULTS.json"
