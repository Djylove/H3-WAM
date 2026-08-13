#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${H3_WORKSPACE}/project"
PYTHON_BIN="${H3_WORKSPACE}/runtime/conda-py311/bin/python"
MARKER="${H3_WORKSPACE}/outputs/h3-lingbot-shared/quantile_flowweight_lr1e5_tail2_s10000.pt"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3dotwam-depth"
OUTPUT="${OUTPUT_ROOT}/h3dotwam_depth4_random_gb128_s2170.json"
STAGE="${OUTPUT_ROOT}/h3dotwam_depth4_random_gb128_s2170.pt"
LOG="${H3_WORKSPACE}/logs/cluster-30907/h3dotwam_depth4_random_gb128_s2170.log"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-30907-depth4-random"

until [[ -s "${MARKER}" ]]; do sleep 30; done
if [[ -s "${OUTPUT}" && -s "${STAGE}" ]]; then
  exit 0
fi
while [[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) -ne 0 ]]; do
  sleep 30
done

mkdir -p "${OUTPUT_ROOT}" "$(dirname "${LOG}")" "${TMP_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export TMPDIR="${TMP_ROOT}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  scripts/h3dreamwam/train_h3dotwam_fsdp.py \
  --model "${H3_WORKSPACE}/models/MiniMax-H3" \
  --data-root "${H3_WORKSPACE}/data/v8_frameindexed_h3_cache" \
  --manifest "${H3_WORKSPACE}/data/v8_multisuite_frameindexed_candidate/manifest_train_uniform.jsonl" \
  --output "${OUTPUT}" --save-stage "${STAGE}" \
  --checkpoint-every 200 --steps 2170 --action-layers 4 \
  --gradient-accumulation-steps 16 --sample-offset 0 \
  --action-horizon 32 --learning-rate 1e-4 --last-h3-blocks 0 \
  --video-loss-weight 1 --language-ranking-weight 0 --lr-schedule cosine \
  --require-text-only-context --log-every 1 >"${LOG}" 2>&1
