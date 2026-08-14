#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project-shared-sync-v2}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v7_dense_h3_cache"
TRAIN_MANIFEST="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3-lingbot-shared-sync-v2"
LOG_ROOT="${H3_WORKSPACE}/logs/h3-lingbot-shared-sync-v2"
TMP_ROOT="${H3_WORKSPACE}/tmp/h3-lingbot-shared-sync-v2-train"
TRAINER="scripts/h3dreamwam/verify_h3_lingbot_four_stream_fsdp.py"
EXPECTED_TRAINER_SHA256="a6507e9c98b78108fff5c1169a7a2641ad3efa1b71105028d2d894e62f502a3b"
RUN_NAME="shared_sync_v2_clean_s1000"

cd "${PROJECT_ROOT}"
actual_sha256="$(sha256sum "${TRAINER}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_TRAINER_SHA256}" ]]; then
  echo "trainer identity mismatch: ${actual_sha256}" >&2
  exit 2
fi
if [[ $(wc -l < "${TRAIN_MANIFEST}") -ne 200779 ]]; then
  echo "train manifest row count changed" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}/${RUN_NAME}.pt" || -e "${OUTPUT_ROOT}/${RUN_NAME}_train.json" ]]; then
  echo "refusing to overwrite an existing sync-v2 run" >&2
  exit 2
fi
if [[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d' | wc -l) -ne 0 ]]; then
  echo "training GPUs are not idle" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${TMP_ROOT}"
export PYTHONPATH="${H3_WORKSPACE}/project/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="${H3_WORKSPACE}/runtime/gl_root/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TMPDIR="${TMP_ROOT}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  "${TRAINER}" --shared-backbone --model "${MODEL_ROOT}" \
  --data-root "${DATA_ROOT}" --manifest "${TRAIN_MANIFEST}" \
  --action-normalization quantile \
  --action-stats-json experiments/data/libero_v7_action_quantiles.json \
  --flow-match-loss-weighting --rotate-windows --random-timesteps \
  --output "${OUTPUT_ROOT}/${RUN_NAME}_train.json" \
  --save-stage "${OUTPUT_ROOT}/${RUN_NAME}.pt" \
  --checkpoint-every 200 --steps 1000 --warmup-steps 10 \
  --last-trainable-layers 2 --action-horizon 32 \
  --learning-rate 1e-5 --weight-decay 0.01
