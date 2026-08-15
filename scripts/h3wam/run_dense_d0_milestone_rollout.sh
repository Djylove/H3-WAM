#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 5 )); then
  echo "usage: $0 H8|H32 CHECKPOINT GPU TASK_ID TRIAL_INDEX" >&2
  exit 2
fi

mode="$1"
checkpoint="$2"
gpu="$3"
task_id="$4"
trial_index="$5"

case "${mode}" in
  H8)
    action_horizon=8
    replan_steps=8
    ;;
  H32)
    action_horizon=32
    replan_steps=32
    ;;
  *)
    echo "mode must be H8 or H32" >&2
    exit 2
    ;;
esac
if [[ -n "${REPLAN_STEPS_OVERRIDE:-}" ]]; then
  [[ "${REPLAN_STEPS_OVERRIDE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "REPLAN_STEPS_OVERRIDE must be a positive integer" >&2
    exit 2
  }
  replan_steps="${REPLAN_STEPS_OVERRIDE}"
fi
for value in "${gpu}" "${task_id}" "${trial_index}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "GPU/task/trial must be non-negative integers" >&2; exit 2; }
done

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
suite="${SUITE:-libero_goal}"
case "${suite}" in
  libero_goal|libero_spatial|libero_object|libero_10) ;;
  *) echo "unsupported SUITE=${suite}" >&2; exit 2 ;;
esac
suite_slug="${suite#libero_}"
ensemble_args=()
ensemble_suffix=""
case "${USE_ACTION_ENSEMBLER:-0}" in
  0|false|FALSE|no|NO|"") ;;
  1|true|TRUE|yes|YES)
    ensemble_args+=(--use-action-ensembler)
    ensemble_suffix="_ensemble"
    ;;
  *) echo "USE_ACTION_ENSEMBLER must be boolean" >&2; exit 2 ;;
esac
progress_args=()
progress_suffix=""
if [[ -n "${H3_PROGRESS_PROBE:-}" ]]; then
  test -f "${H3_PROGRESS_PROBE}"
  progress_args+=(--progress-probe "$(realpath "${H3_PROGRESS_PROBE}")")
  progress_suffix="_progress-shadow"
fi
branch_args=()
wait_steps=30
if [[ -n "${BRANCH_TRAJECTORY:-}" || -n "${BRANCH_INDEX:-}" || -n "${POLICY_NOISE_SEED_BASE:-}" || -n "${FIRST_POLICY_NOISE_SEED:-}" || -n "${CONTINUATION_POLICY_NOISE_SEED_BASE:-}" ]]; then
  test -f "${BRANCH_TRAJECTORY:?BRANCH_TRAJECTORY is required}"
  [[ "${BRANCH_INDEX:?BRANCH_INDEX is required}" =~ ^[0-9]+$ ]]
  [[ "${ENVIRONMENT_SEED:-42}" =~ ^[0-9]+$ ]]
  wait_steps=0
  branch_args+=(
    --start-trajectory "$(realpath "${BRANCH_TRAJECTORY}")"
    --start-index "${BRANCH_INDEX}"
    --environment-seed "${ENVIRONMENT_SEED:-42}"
  )
  if [[ -n "${POLICY_NOISE_SEED_BASE:-}" ]]; then
    [[ -z "${FIRST_POLICY_NOISE_SEED:-}" && -z "${CONTINUATION_POLICY_NOISE_SEED_BASE:-}" ]]
    [[ "${POLICY_NOISE_SEED_BASE}" =~ ^[0-9]+$ ]]
    branch_args+=(--policy-noise-seed-base "${POLICY_NOISE_SEED_BASE}")
  else
    [[ "${FIRST_POLICY_NOISE_SEED:?FIRST_POLICY_NOISE_SEED is required}" =~ ^[0-9]+$ ]]
    [[ "${CONTINUATION_POLICY_NOISE_SEED_BASE:?CONTINUATION_POLICY_NOISE_SEED_BASE is required}" =~ ^[0-9]+$ ]]
    branch_args+=(
      --first-policy-noise-seed "${FIRST_POLICY_NOISE_SEED}"
      --continuation-policy-noise-seed-base "${CONTINUATION_POLICY_NOISE_SEED_BASE}"
    )
  fi
fi
checkpoint="$(realpath "${checkpoint}")"
checkpoint_name="$(basename "${checkpoint}" .pt)"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/eval-dense-d0-long/${checkpoint_name}_${suite_slug}_task${task_id}_trial${trial_index}_replan${replan_steps}${ensemble_suffix}${progress_suffix}}"

test -f "${checkpoint}"
test -f "${project}/scripts/h3wam/rollout_libero.py"
test ! -e "${output_root}"
mkdir -p "$(dirname "${output_root}")" "${workspace}/logs/dense-d0-long-rollout"

export CUDA_VISIBLE_DEVICES="${gpu}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export PYTHON_BIN="${workspace}/runtime/conda-py311/bin/python"
export SIM_SITE_PACKAGES="/tmp/h3-wam-libero-site"

exec bash "${project}/scripts/h3wam/run_cloud_libero.sh" \
  "${workspace}/runtime/conda-py311/bin/python" \
  "${project}/scripts/h3wam/rollout_libero.py" \
  --policy h3_dreamwam_kv_int8 \
  --policy-python "${workspace}/runtime/h3-int8-native/bin/python" \
  --checkpoint "${checkpoint}" \
  --cache-root "${workspace}/data/v7_dense_h3_cache" \
  --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  --h3-model "${workspace}/models/MiniMax-H3" \
  --dreamwam-source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
  --device cuda:0 \
  --suite "${suite}" \
  --task-ids "${task_id}" \
  --trial-indices "${trial_index}" \
  --max-steps 400 \
  --wait-steps "${wait_steps}" \
  --replan-steps "${replan_steps}" \
  --action-horizon "${action_horizon}" \
  --h3-feature-audio-horizon 32 \
  --target-latent-frames 12 \
  --model-evaluations 10 \
  --seed 42 \
  --normalized-action-pre-clamp \
  "${branch_args[@]}" \
  "${progress_args[@]}" \
  "${ensemble_args[@]}" \
  --output-dir "${output_root}" \
  --save-video \
  --save-trajectories
