#!/usr/bin/env bash
set -euo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/project}"
task="${1:?usage: $0 task0|task3|push_plate [--execute]}"
mode="${2:---dry-run}"
if [[ "${mode}" != "--dry-run" && "${mode}" != "--execute" ]]; then
  echo "second argument must be --dry-run or --execute" >&2
  exit 2
fi

historical="${workspace}/int8-action/checkpoints/historical"
cache="${workspace}/int8-action/cache/historical_online"
output_root="${workspace}/int8-action/results/historical_online_canary"
read -r -a trial_indices <<< "${H3_TRIAL_INDICES:-0}"
if [[ "${#trial_indices[@]}" -eq 0 ]]; then
  echo "H3_TRIAL_INDICES must contain at least one trial index" >&2
  exit 2
fi
for trial_index in "${trial_indices[@]}"; do
  if [[ ! "${trial_index}" =~ ^[0-9]+$ ]]; then
    echo "invalid trial index: ${trial_index}" >&2
    exit 2
  fi
done
run_name="${H3_RUN_NAME:-${task}_trial0_seed42}"
common=(
  "${workspace}/runtime/conda-py311/bin/python"
  "${project}/scripts/h3wam/rollout_libero.py"
  --policy h3_feature_int8
  --policy-python "${workspace}/runtime/h3-int8-native/bin/python"
  --cache-root "${cache}"
  --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  --h3-model "${workspace}/models/MiniMax-H3"
  --device "${H3_DEVICE:-cuda:0}"
  --suite libero_goal
  --trial-indices "${trial_indices[@]}"
  --max-steps 400
  --wait-steps 30
  --replan-steps 8
  --action-horizon 8
  --h3-feature-audio-horizon 8
  --target-latent-frames 12
  --seed 42
  --save-video
)

case "${task}" in
  task0)
    args=(
      --task-languages "open the middle drawer of the cabinet"
      --checkpoint "${historical}/libero_goal_task0_h3_feature_action_h8_regression_proprio_phase_gripper10_5000.pt"
      --h3-feature-ensemble-checkpoint "${historical}/libero_goal_task0_h3_h8_teacher_rollin_trials1_3_idx08_only_lr1e5_25.pt"
      --h3-feature-ensemble-mode switch
      --h3-feature-switch-step 64
    )
    ;;
  task3)
    args=(
      --task-languages "open the top drawer and put the bowl inside"
      --checkpoint "${historical}/libero_goal_task3_h3_videolora500_feature_action_h8_regression_proprio_phase_gripper10_5000_best.pt"
      --h3-feature-ensemble-checkpoint "${historical}/libero_goal_task3_h3_videolora500_h8_teacher_rollin_trial1_idx09_only_lr1e5_25_best.pt"
      --h3-feature-ensemble-mode learned_switch
      --h3-feature-switch-gate-checkpoint "${historical}/libero_goal_task3_h3_videolora500_h8_switch_gate_step72_1000_best.pt"
      --h3-feature-gate-threshold 0.1
      --h3-video-lora-checkpoint "${historical}/libero_goal_all_h3_video_lora_attn_ffn_r4_last10_500.pt"
    )
    ;;
  push_plate)
    args=(
      --task-languages "push the plate to the front of the stove"
      --checkpoint "${historical}/libero_goal_task2_h3_feature_action_h8_regression_proprio_phase_gripper10_5000.pt"
      --h3-feature-ensemble-mode mean
    )
    ;;
  *)
    echo "unknown task ${task}" >&2
    exit 2
    ;;
esac

command=(
  env
  LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
  PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
  PYTHON_BIN="${workspace}/runtime/conda-py311/bin/python"
  SIM_SITE_PACKAGES="/tmp/h3-wam-libero-site"
  bash "${project}/scripts/h3wam/run_cloud_libero.sh"
  "${common[@]}"
  "${args[@]}"
  --output-dir "${output_root}/${run_name}"
)

if [[ "${mode}" == "--dry-run" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
