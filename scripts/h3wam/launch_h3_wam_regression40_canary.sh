#!/usr/bin/env bash
set -euo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
MODEL_ROOT="${MODEL_ROOT:-${H3_WORKSPACE}/models/MiniMax-H3}"
DATA_ROOT="${DATA_ROOT:-${H3_WORKSPACE}/data/v2_full_cache}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate}"
MANIFEST="${MANIFEST:-${CANDIDATE_ROOT}/manifest_train_uniform.jsonl}"
VALIDATION_MANIFEST="${VALIDATION_MANIFEST:-${CANDIDATE_ROOT}/manifest_val_stratified40.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${H3_WORKSPACE}/outputs/h3wam_regression40_full50_lr3e5_s200}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/.venv/bin/python}"
RESUME_FROM="${RESUME_FROM:-}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-0}"
FROZEN_ACTION_ONLY="${FROZEN_ACTION_ONLY:-0}"
EXPLICIT_LANGUAGE_CONDITIONING="${EXPLICIT_LANGUAGE_CONDITIONING:-0}"
PHASE_CONDITIONING="${PHASE_CONDITIONING:-0}"
PREVIOUS_ACTION_CONDITIONING="${PREVIOUS_ACTION_CONDITIONING:-0}"
PREVIOUS_ACTION_CACHE="${PREVIOUS_ACTION_CACHE:-}"
INITIALIZE_ACTION_FROM="${INITIALIZE_ACTION_FROM:-}"
TRAIN_PREVIOUS_ACTION_PROJECTION_ONLY="${TRAIN_PREVIOUS_ACTION_PROJECTION_ONLY:-0}"
HISTORY_FRAME_CONDITIONING="${HISTORY_FRAME_CONDITIONING:-0}"
HISTORY_FRAME_MAP="${HISTORY_FRAME_MAP:-}"
TRAIN_HISTORY_GATE_ONLY="${TRAIN_HISTORY_GATE_ONLY:-0}"
HISTORY_ADAPTER_RANK="${HISTORY_ADAPTER_RANK:-0}"
TRAIN_HISTORY_ADAPTER_ONLY="${TRAIN_HISTORY_ADAPTER_ONLY:-0}"
STEPS="${STEPS:-200}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-1e-6}"
ACTION_LEARNING_RATE="${ACTION_LEARNING_RATE:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
ACTION_LAYERS="${ACTION_LAYERS:-1}"
LAYER_MIX_INITIALIZATION="${LAYER_MIX_INITIALIZATION:-spaced}"
LAYER_MIX_LEARNING_RATE="${LAYER_MIX_LEARNING_RATE:-${ACTION_LEARNING_RATE}}"
ROUTING_DIAGNOSTICS_EVERY="${ROUTING_DIAGNOSTICS_EVERY:-0}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
VALIDATION_EVERY="${VALIDATION_EVERY:-10}"
VALIDATION_BATCHES_PER_RANK="${VALIDATION_BATCHES_PER_RANK:-5}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-2}"

export PYTHONPATH="${PROJECT_ROOT}/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="${H3_WORKSPACE}/cache"
export HF_HOME="${H3_WORKSPACE}/cache/huggingface"
export TORCH_HOME="${H3_WORKSPACE}/cache/torch"
export TMPDIR="${H3_WORKSPACE}/tmp"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

test -x "${PYTHON_BIN}"
test -f "${MODEL_ROOT}/transformer/diffusion_pytorch_model.safetensors.index.json"
test -f "${DATA_ROOT}/stats.pt"
test -f "${MANIFEST}"
test -f "${VALIDATION_MANIFEST}"
mkdir -p "${OUTPUT_DIR}" "${TMPDIR}"

resume_args=()
if [[ -n "${RESUME_FROM}" ]]; then
  test -f "${RESUME_FROM}/manifest.json"
  resume_args=(--resume-from "${RESUME_FROM}")
fi
backbone_args=()
if [[ "${FREEZE_BACKBONE}" == "1" ]]; then
  backbone_args=(--freeze-backbone)
fi
action_only_args=()
if [[ "${FROZEN_ACTION_ONLY}" == "1" ]]; then
  action_only_args=(--frozen-action-only)
fi
language_args=()
if [[ "${EXPLICIT_LANGUAGE_CONDITIONING}" == "1" ]]; then
  language_args=(--explicit-language-conditioning)
fi
phase_args=()
if [[ "${PHASE_CONDITIONING}" == "1" ]]; then
  phase_args=(--phase-conditioning)
fi
previous_action_args=()
if [[ "${PREVIOUS_ACTION_CONDITIONING}" == "1" ]]; then
  previous_action_args=(--previous-action-conditioning)
  if [[ -n "${PREVIOUS_ACTION_CACHE}" ]]; then
    test -f "${PREVIOUS_ACTION_CACHE}"
    previous_action_args+=(--previous-action-cache "${PREVIOUS_ACTION_CACHE}")
  fi
fi
initialize_action_args=()
if [[ -n "${INITIALIZE_ACTION_FROM}" ]]; then
  test -f "${INITIALIZE_ACTION_FROM}/manifest.json"
  initialize_action_args=(--initialize-action-from "${INITIALIZE_ACTION_FROM}")
fi
projection_only_args=()
if [[ "${TRAIN_PREVIOUS_ACTION_PROJECTION_ONLY}" == "1" ]]; then
  projection_only_args=(--train-previous-action-projection-only)
fi
history_args=()
if [[ "${HISTORY_FRAME_CONDITIONING}" == "1" ]]; then
  test -f "${HISTORY_FRAME_MAP:?set HISTORY_FRAME_MAP}"
  history_args=(--history-frame-conditioning --history-frame-map "${HISTORY_FRAME_MAP}")
fi
history_gate_args=()
if [[ "${TRAIN_HISTORY_GATE_ONLY}" == "1" ]]; then
  history_gate_args=(--train-history-gate-only)
fi
history_adapter_args=(--history-adapter-rank "${HISTORY_ADAPTER_RANK}")
if [[ "${TRAIN_HISTORY_ADAPTER_ONLY}" == "1" ]]; then
  history_adapter_args+=(--train-history-adapter-only)
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/h3wam/train_h3_wam_joint_fsdp.py \
  --model "${MODEL_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --manifest "${MANIFEST}" \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  "${resume_args[@]}" \
  "${backbone_args[@]}" \
  "${action_only_args[@]}" \
  "${language_args[@]}" \
  "${phase_args[@]}" \
  "${previous_action_args[@]}" \
  "${initialize_action_args[@]}" \
  "${projection_only_args[@]}" \
  "${history_args[@]}" \
  "${history_gate_args[@]}" \
  "${history_adapter_args[@]}" \
  --steps "${STEPS}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --last-blocks 50 \
  --capture-layers 4 9 14 19 24 29 34 39 44 49 \
  --backbone-learning-rate "${BACKBONE_LEARNING_RATE}" \
  --action-learning-rate "${ACTION_LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --minimum-lr-ratio 0.2 \
  --video-loss-weight 1.0 \
  --action-loss-weight 1.0 \
  --action-objective regression \
  --action-hidden-dim 1024 \
  --action-layers "${ACTION_LAYERS}" \
  --action-heads 16 \
  --action-ffn-dim 4096 \
  --layer-mix-initialization "${LAYER_MIX_INITIALIZATION}" \
  --layer-mix-learning-rate "${LAYER_MIX_LEARNING_RATE}" \
  --validation-every "${VALIDATION_EVERY}" \
  --validation-batches-per-rank "${VALIDATION_BATCHES_PER_RANK}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --keep-last-checkpoints "${KEEP_LAST_CHECKPOINTS}" \
  --log-every 1 \
  --routing-diagnostics-every "${ROUTING_DIAGNOSTICS_EVERY}" \
  --save-final \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"
