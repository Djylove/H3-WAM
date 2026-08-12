#!/usr/bin/env bash
set -euo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/home/h3wam_finetune}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
MODEL_ROOT="${MODEL_ROOT:-${H3_WORKSPACE}/models/MiniMax-H3}"
DATA_ROOT="${DATA_ROOT:-${H3_WORKSPACE}/data/v2_full_cache}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${H3_WORKSPACE}/data/v2_full_candidate}"
OUTPUT_DIR="${OUTPUT_DIR:-${H3_WORKSPACE}/outputs/v3_full40_joint_h3_10ep}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/.venv/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${H3_WORKSPACE}/.venv/bin/torchrun}"

export PYTHONPATH="${PROJECT_ROOT}/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="${H3_WORKSPACE}/cache"
export HF_HOME="${H3_WORKSPACE}/cache/huggingface"
export TORCH_HOME="${H3_WORKSPACE}/cache/torch"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

test -x "${PYTHON_BIN}"
test -x "${TORCHRUN_BIN}"
test -f "${MODEL_ROOT}/transformer/diffusion_pytorch_model.safetensors.index.json"
test -f "${DATA_ROOT}/stats.pt"
test -f "${CANDIDATE_ROOT}/manifest_train.jsonl"
test -f "${CANDIDATE_ROOT}/manifest_val.jsonl"
mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_ROOT}"
"${TORCHRUN_BIN}" --standalone --nproc_per_node=8 \
  scripts/h3wam/train_h3_wam_joint_fsdp.py \
  --model "${MODEL_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --manifest "${CANDIDATE_ROOT}/manifest_train.jsonl" \
  --validation-manifest "${CANDIDATE_ROOT}/manifest_val.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --steps 600 \
  --gradient-accumulation-steps 16 \
  --last-blocks 50 \
  --capture-layers 4 9 14 19 24 29 34 39 44 49 \
  --backbone-learning-rate 1e-5 \
  --action-learning-rate 1e-4 \
  --weight-decay 0.01 \
  --warmup-steps 30 \
  --minimum-lr-ratio 0.1 \
  --action-flow-shift 5.0 \
  --action-hidden-dim 1024 \
  --action-layers 1 \
  --action-heads 16 \
  --action-ffn-dim 4096 \
  --validation-every 20 \
  --validation-batches-per-rank 5 \
  --checkpoint-every 100 \
  --keep-last-checkpoints 3 \
  --log-every 1 \
  --save-final \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
