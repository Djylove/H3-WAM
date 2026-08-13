#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v7_dense_h3_cache"
HISTORY_ROOT="${H3_WORKSPACE}/data/v7_executed_action_history"
MANIFEST="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
PARENT="${H3_WORKSPACE}/outputs/h3-lingbot-history/history16_from_s5000_s100.pt"
FINAL="${H3_WORKSPACE}/outputs/h3-lingbot-history/history16_from_s5000_s3000.pt"
REPORT="${H3_WORKSPACE}/outputs/h3-lingbot-history/history16_from_s5000_s3000_train.json"
LOG="${H3_WORKSPACE}/logs/history16_from_s5000_s3000_train.log"
TMP_ROOT="${H3_WORKSPACE}/tmp/history16-s3000"

test -s "${PARENT}"
mkdir -p "$(dirname "${FINAL}")" "$(dirname "${LOG}")" "${TMP_ROOT}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export TMPDIR="${TMP_ROOT}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/h3dreamwam/verify_h3_lingbot_four_stream_fsdp.py \
  --shared-backbone --model "${MODEL}" --data-root "${DATA_ROOT}" \
  --manifest "${MANIFEST}" --executed-action-history-steps 16 \
  --executed-action-history-root "${HISTORY_ROOT}" \
  --action-normalization quantile \
  --action-stats-json experiments/data/libero_v7_action_quantiles.json \
  --flow-match-loss-weighting --load-stage "${PARENT}" \
  --save-stage "${FINAL}" --output "${REPORT}" \
  --steps 2900 --base-completed-steps 100 --checkpoint-every 500 \
  --sample-offset 5100 --rotate-windows --random-timesteps --warmup-steps 0 \
  --last-trainable-layers 2 --action-horizon 32 \
  --learning-rate 1e-5 --weight-decay 0.01 2>&1 | tee "${LOG}"
