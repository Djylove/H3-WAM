#!/usr/bin/env bash
set -euo pipefail

# Paper-aligned Faster-WAM/DoT canary for MiniMax-H3.  The only H3-specific
# adaptation is the lower backbone LR already shown stable by the full-50-layer
# probe.  No project-specific ranking, history, phase, or regression loss is
# enabled here.
H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
MODEL_ROOT="${MODEL_ROOT:-${H3_WORKSPACE}/models/MiniMax-H3}"
DATA_ROOT="${DATA_ROOT:-${H3_WORKSPACE}/data/v2_full_cache}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate}"
MANIFEST="${MANIFEST:-${CANDIDATE_ROOT}/manifest_train_uniform.jsonl}"
INITIAL_ACTION_STAGE="${INITIAL_ACTION_STAGE:-${H3_WORKSPACE}/outputs/h3dotwam/m0v2_h32_gb128_s150_step000125.pt}"
RUN_NAME="${RUN_NAME:-m3_paper_joint_full50_gb128_s10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${H3_WORKSPACE}/outputs/h3dotwam}"
LOG_ROOT="${LOG_ROOT:-${H3_WORKSPACE}/logs/pipeline}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/.venv/bin/python}"

STEPS="${STEPS:-10}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
ACTION_LEARNING_RATE="${ACTION_LEARNING_RATE:-1e-5}"
H3_LEARNING_RATE="${H3_LEARNING_RATE:-1e-6}"
VIDEO_LOSS_WEIGHT="${VIDEO_LOSS_WEIGHT:-1.0}"

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
test -f "${INITIAL_ACTION_STAGE}"
test ! -e "${OUTPUT_ROOT}/${RUN_NAME}_joint"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${TMPDIR}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/h3dreamwam/train_h3dotwam_fsdp.py \
  --model "${MODEL_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --manifest "${MANIFEST}" \
  --output "${OUTPUT_ROOT}/${RUN_NAME}.json" \
  --load-stage "${INITIAL_ACTION_STAGE}" \
  --save-joint-stage "${OUTPUT_ROOT}/${RUN_NAME}_joint" \
  --steps "${STEPS}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --action-horizon 32 \
  --learning-rate "${ACTION_LEARNING_RATE}" \
  --h3-learning-rate "${H3_LEARNING_RATE}" \
  --last-h3-blocks 50 \
  --video-loss-weight "${VIDEO_LOSS_WEIGHT}" \
  --language-ranking-weight 0 \
  --lr-schedule constant \
  --require-text-only-context \
  --log-every 1 \
  2>&1 | tee "${LOG_ROOT}/${RUN_NAME}.log"
