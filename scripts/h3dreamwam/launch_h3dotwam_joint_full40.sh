#!/usr/bin/env bash
set -euo pipefail

# Ten-epoch four-suite LIBERO training using the paper-backed DoT interface and
# joint video/action flow matching.  7710 windows / global batch 128 rounds to
# 602 optimizer steps.  A full joint stage is kept roughly once per epoch so
# closed-loop evaluation can select the best point instead of only the final.
H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
MODEL_ROOT="${MODEL_ROOT:-${H3_WORKSPACE}/models/MiniMax-H3}"
DATA_ROOT="${DATA_ROOT:-${H3_WORKSPACE}/data/v2_full_cache}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate}"
MANIFEST="${MANIFEST:-${CANDIDATE_ROOT}/manifest_train_uniform.jsonl}"
INITIAL_ACTION_STAGE="${INITIAL_ACTION_STAGE:-${H3_WORKSPACE}/outputs/h3dotwam/m0v2_h32_gb128_s150_step000125.pt}"
RUN_NAME="${RUN_NAME:-m4_paper_joint_full40_10ep}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${H3_WORKSPACE}/outputs/h3dotwam}"
LOG_ROOT="${LOG_ROOT:-${H3_WORKSPACE}/logs/pipeline}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/.venv/bin/python}"

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
  --steps 602 \
  --gradient-accumulation-steps 16 \
  --action-horizon 32 \
  --learning-rate 1e-5 \
  --h3-learning-rate 1e-6 \
  --last-h3-blocks 50 \
  --video-loss-weight 1.0 \
  --language-ranking-weight 0 \
  --lr-schedule cosine \
  --require-text-only-context \
  --joint-checkpoint-every 60 \
  --keep-last-joint-checkpoints 10 \
  --log-every 1 \
  2>&1 | tee "${LOG_ROOT}/${RUN_NAME}.log"
