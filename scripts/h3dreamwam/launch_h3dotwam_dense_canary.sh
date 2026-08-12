#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v7_dense_canary_cache"
CANDIDATE_ROOT="${H3_WORKSPACE}/data/v7_dense_canary_candidate"
INITIAL_STAGE="${H3_WORKSPACE}/outputs/h3dotwam/m0v2_h32_gb128_s150_step000125.pt"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3dotwam-dense"
RUN_NAME="m10_dense_canary_head_gb128_s80"
FINAL_STAGE="${OUTPUT_ROOT}/${RUN_NAME}.pt"
LOG_ROOT="${H3_WORKSPACE}/logs/cluster-32409"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-32409-dense-canary"

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="${H3_WORKSPACE}/cache"
export HF_HOME="${H3_WORKSPACE}/cache/huggingface"
export TORCH_HOME="${H3_WORKSPACE}/cache/torch"
export TMPDIR="${TMP_ROOT}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

test -s "${INITIAL_STAGE}"
test -s "${CANDIDATE_ROOT}/candidate_report.json"
test ! -e "${FINAL_STAGE}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${TMP_ROOT}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
  --manifest "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
  --output "${OUTPUT_ROOT}/${RUN_NAME}.json" \
  --load-stage "${INITIAL_STAGE}" --save-stage "${FINAL_STAGE}" \
  --checkpoint-every 40 --steps 80 --gradient-accumulation-steps 16 \
  --action-horizon 32 --learning-rate 1e-4 --h3-learning-rate 1e-6 \
  --last-h3-blocks 0 --video-loss-weight 1.0 --language-ranking-weight 0 \
  --lr-schedule constant --require-text-only-context --log-every 1 \
  > "${LOG_ROOT}/${RUN_NAME}.log" 2>&1

# Continue with the exact public frame-indexed population once its cache is ready.
nohup bash "${PROJECT_ROOT}/scripts/h3dreamwam/launch_h3dotwam_frameindexed_head_epoch.sh" \
  > "${LOG_ROOT}/frameindexed_head_waiter.log" 2>&1 &
