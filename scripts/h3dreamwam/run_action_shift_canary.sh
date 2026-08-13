#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ACTION_TRAIN_SHIFT TAG" >&2
  exit 2
fi

ACTION_TRAIN_SHIFT="$1"
TAG="$2"
PROJECT_ROOT="/mnt/h3-wam/project"
PYTHON_BIN="/mnt/h3-wam/runtime/conda-py311/bin/python"
OUTPUT_ROOT="/mnt/h3-wam/outputs/h3-lingbot-shared/action-train-shift-v1"
TRAIN_SCRIPT="scripts/h3dreamwam/verify_h3_lingbot_four_stream_fsdp.py"
MODEL_ROOT="/mnt/h3-wam/models/MiniMax-H3"
DATA_ROOT="/mnt/h3-wam/data/v7_dense_h3_cache"
TRAIN_MANIFEST="/mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
VAL_MANIFEST="/mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_val_stratified40.jsonl"
STATS_JSON="experiments/data/libero_v7_action_quantiles.json"
CHECKPOINT="${OUTPUT_ROOT}/trainshift${TAG}_s100.pt"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:/mnt/h3-wam/.venv/lib/python3.11/site-packages"

COMMON=(
  --shared-backbone
  --model "${MODEL_ROOT}"
  --data-root "${DATA_ROOT}"
  --action-normalization quantile
  --action-stats-json "${STATS_JSON}"
  --flow-match-loss-weighting
  --last-trainable-layers 2
  --action-horizon 32
  --learning-rate 1e-5
  --weight-decay 0.01
  --seed 2026
  --action-train-shift "${ACTION_TRAIN_SHIFT}"
  --action-infer-shift 0.05
)

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  "${TRAIN_SCRIPT}" "${COMMON[@]}" \
  --manifest "${TRAIN_MANIFEST}" \
  --output "${OUTPUT_ROOT}/trainshift${TAG}_s100_train.json" \
  --save-stage "${CHECKPOINT}" \
  --steps 100 \
  --rotate-windows \
  --random-timesteps \
  --warmup-steps 10

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  "${TRAIN_SCRIPT}" "${COMMON[@]}" \
  --manifest "${VAL_MANIFEST}" \
  --load-stage "${CHECKPOINT}" \
  --output "${OUTPUT_ROOT}/trainshift${TAG}_s100_val40.json" \
  --steps 1 \
  --eval-only \
  --eval-all \
  --eval-limit 40

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  "${TRAIN_SCRIPT}" "${COMMON[@]}" \
  --manifest "${VAL_MANIFEST}" \
  --load-stage "${CHECKPOINT}" \
  --output "${OUTPUT_ROOT}/trainshift${TAG}_s100_infer0p05_action4_sample40.json" \
  --steps 1 \
  --eval-only \
  --eval-all \
  --eval-limit 40 \
  --sample-eval \
  --sample-steps 4 \
  --video-sample-steps 4 \
  --action-sample-steps 4

# LingBot LIBERO uses 50 action denoising steps. Keep video at four steps and
# change only the action solver resolution so this does not contaminate the
# training-shift comparison above.
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  "${TRAIN_SCRIPT}" "${COMMON[@]}" \
  --manifest "${VAL_MANIFEST}" \
  --load-stage "${CHECKPOINT}" \
  --output "${OUTPUT_ROOT}/trainshift${TAG}_s100_infer0p05_action50_sample40.json" \
  --steps 1 \
  --eval-only \
  --eval-all \
  --eval-limit 40 \
  --sample-eval \
  --sample-steps 4 \
  --video-sample-steps 4 \
  --action-sample-steps 50
