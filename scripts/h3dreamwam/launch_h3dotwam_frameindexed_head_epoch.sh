#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v8_frameindexed_h3_cache"
CANDIDATE_ROOT="${H3_WORKSPACE}/data/v8_multisuite_frameindexed_candidate"
INITIAL_STAGE="${H3_WORKSPACE}/outputs/h3dotwam-dense/m12_dense_mid256_head_gb128_s160.pt"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3dotwam-frameindexed"
RUN_NAME="m11_frameindexed_head_gb128_s2170"
FINAL_STAGE="${OUTPUT_ROOT}/${RUN_NAME}.pt"
LOG_ROOT="${H3_WORKSPACE}/logs/cluster-32409"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-32409-frameindexed-head"

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="${H3_WORKSPACE}/cache"
export HF_HOME="${H3_WORKSPACE}/cache/huggingface"
export TORCH_HOME="${H3_WORKSPACE}/cache/torch"
export TMPDIR="${TMP_ROOT}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${TMP_ROOT}"
until [[ -s "${CANDIDATE_ROOT}/candidate_report.json" \
      && -s "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
      && -s "${DATA_ROOT}/stats.pt" \
      && -s "${INITIAL_STAGE}" ]]; do
  sleep 30
done

"${PYTHON_BIN}" - "${CANDIDATE_ROOT}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = sum(1 for x in (root / "manifest_train_uniform.jsonl").open() if x.strip())
report = json.loads((root / "candidate_report.json").read_text())
assert rows == 277713, rows
assert report["contract"]["window_sampling"] == "frame_indexed_padded"
PY
test -s "${INITIAL_STAGE}"
test ! -e "${FINAL_STAGE}"
LOCK_DIR="${OUTPUT_ROOT}/${RUN_NAME}.launch.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "frame-indexed training already claimed: ${LOCK_DIR}" >&2
  exit 0
fi

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
  --manifest "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
  --output "${OUTPUT_ROOT}/${RUN_NAME}.json" \
  --load-stage "${INITIAL_STAGE}" --save-stage "${FINAL_STAGE}" \
  --checkpoint-every 200 --steps 2170 --gradient-accumulation-steps 16 \
  --action-horizon 32 --learning-rate 1e-4 --h3-learning-rate 1e-6 \
  --last-h3-blocks 0 --video-loss-weight 1.0 --language-ranking-weight 0 \
  --lr-schedule cosine --require-text-only-context --log-every 1 \
  > "${LOG_ROOT}/${RUN_NAME}.log" 2>&1
