#!/usr/bin/env bash
set -euo pipefail

# Keep every mutable path inside the project-owned cloud workspace.  In
# particular, running this as the server's root login must not populate /root.
H3_WORKSPACE="${H3_WORKSPACE:-/home/h3wam_finetune}"
H3_RUN_NAME="${H3_RUN_NAME:-v0_last2_smoke}"
H3_NPROC="${H3_NPROC:-8}"
H3_STEPS="${H3_STEPS:-1}"
H3_LAST_BLOCKS="${H3_LAST_BLOCKS:-2}"
H3_LR="${H3_LR:-1e-5}"
H3_SEED="${H3_SEED:-2026}"
H3_DATA_ROOT="${H3_DATA_ROOT:-${H3_WORKSPACE}/data/v0}"
H3_MANIFEST="${H3_MANIFEST:-}"
H3_VALIDATION_MANIFEST="${H3_VALIDATION_MANIFEST:-}"
H3_VALIDATION_EVERY="${H3_VALIDATION_EVERY:-0}"
H3_VALIDATION_BATCHES_PER_RANK="${H3_VALIDATION_BATCHES_PER_RANK:-1}"
H3_CHECKPOINT_EVERY="${H3_CHECKPOINT_EVERY:-0}"
H3_LOG_EVERY="${H3_LOG_EVERY:-1}"
H3_RESUME="${H3_RESUME:-}"
H3_FP32_MASTER_WEIGHTS="${H3_FP32_MASTER_WEIGHTS:-1}"
H3_VERIFY_PARAMETER_UPDATE="${H3_VERIFY_PARAMETER_UPDATE:-0}"

case "${H3_WORKSPACE}" in
  /home/h3wam_finetune|/home/h3wam_finetune/*) ;;
  *)
    echo "H3_WORKSPACE must stay under /home/h3wam_finetune" >&2
    exit 2
    ;;
esac

export HOME="${H3_WORKSPACE}"
export XDG_CACHE_HOME="${H3_WORKSPACE}/cache"
export HF_HOME="${H3_WORKSPACE}/cache/huggingface"
export TORCH_HOME="${H3_WORKSPACE}/cache/torch"
export TMPDIR="${H3_WORKSPACE}/tmp"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

PROJECT="${H3_WORKSPACE}/project"
OUTPUT="${H3_WORKSPACE}/outputs/${H3_RUN_NAME}"
LOG="${H3_WORKSPACE}/logs/${H3_RUN_NAME}.log"
mkdir -p "${OUTPUT}" "${H3_WORKSPACE}/logs" "${TMPDIR}"
cd "${PROJECT}"

TRAIN_ARGS=(
  --model "${H3_WORKSPACE}/models/MiniMax-H3"
  --data-root "${H3_DATA_ROOT}"
  --output-dir "${OUTPUT}"
  --steps "${H3_STEPS}"
  --last-blocks "${H3_LAST_BLOCKS}"
  --learning-rate "${H3_LR}"
  --seed "${H3_SEED}"
  --checkpoint-every "${H3_CHECKPOINT_EVERY}"
  --log-every "${H3_LOG_EVERY}"
  --validation-every "${H3_VALIDATION_EVERY}"
  --validation-batches-per-rank "${H3_VALIDATION_BATCHES_PER_RANK}"
)
if [[ -n "${H3_MANIFEST}" ]]; then
  TRAIN_ARGS+=(--manifest "${H3_MANIFEST}")
fi
if [[ -n "${H3_VALIDATION_MANIFEST}" ]]; then
  TRAIN_ARGS+=(--validation-manifest "${H3_VALIDATION_MANIFEST}")
fi
if [[ -n "${H3_RESUME}" ]]; then
  TRAIN_ARGS+=(--resume "${H3_RESUME}")
fi
if [[ "${H3_FP32_MASTER_WEIGHTS}" == "1" ]]; then
  TRAIN_ARGS+=(--fp32-master-weights)
else
  TRAIN_ARGS+=(--no-fp32-master-weights)
fi
if [[ "${H3_VERIFY_PARAMETER_UPDATE}" == "1" ]]; then
  TRAIN_ARGS+=(--verify-parameter-update)
fi

"${H3_WORKSPACE}/.venv/bin/torchrun" \
  --standalone \
  --nproc-per-node="${H3_NPROC}" \
  scripts/h3wam/train_h3_bf16_fsdp.py \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${LOG}"
