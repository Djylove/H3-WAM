#!/usr/bin/env bash
set -Eeuo pipefail

# Usefully occupies the evaluation node while the complete frame-indexed cache
# is still being built.  Every selected item is a newly added dense window;
# none is taken from the old five-start-per-episode population.
H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
BASE_ROOT="${H3_WORKSPACE}/data/v7_multisuite_dense_base"
DENSE_CACHE="${H3_WORKSPACE}/data/v7_dense_h3_cache"
SPARSE_CACHE="${H3_WORKSPACE}/data/v2_full_cache"
DATA_ROOT="${H3_WORKSPACE}/data/v7_dense_canary_cache"
CANDIDATE_ROOT="${H3_WORKSPACE}/data/v7_dense_mid256_candidate"
INITIAL_STAGE="${H3_WORKSPACE}/outputs/h3dotwam-dense/m10_dense_canary_head_gb128_s80.pt"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3dotwam-dense"
RUN_NAME="m12_dense_mid256_head_gb128_s160"
FINAL_STAGE="${OUTPUT_ROOT}/${RUN_NAME}.pt"
EVAL_COMPLETE="${H3_WORKSPACE}/outputs/eval-dense-dot/m10_dense_canary/dense_step80/complete.json"
LOG_ROOT="${H3_WORKSPACE}/logs/cluster-30907"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-30907-dense-mid256"

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="${H3_WORKSPACE}/cache"
export HF_HOME="${H3_WORKSPACE}/cache/huggingface"
export TORCH_HOME="${H3_WORKSPACE}/cache/torch"
export TMPDIR="${TMP_ROOT}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${TMP_ROOT}"
until [[ -s "${EVAL_COMPLETE}" ]]; do sleep 15; done

until "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3dreamwam/build_cached_dense_canary.py" \
  --base-root "${BASE_ROOT}" --cache-root "${DENSE_CACHE}" \
  --sparse-cache-root "${SPARSE_CACHE}" --output-dir "${CANDIDATE_ROOT}" \
  --per-task 256; do
  sleep 60
done

"${PYTHON_BIN}" - "${CANDIDATE_ROOT}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
report = json.loads((root / "candidate_report.json").read_text())
assert report["tasks"] == 40
assert report["train_windows"] == 10240
assert report["train_windows_per_task"] == 256
assert report["new_dense_train_windows"] == 10240
assert report["sampling"] == "task_round_robin_new_dense_only"
print(json.dumps({"event": "dense_mid256_audit", **report}, sort_keys=True))
PY

test -s "${INITIAL_STAGE}"
test ! -e "${FINAL_STAGE}"
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
  --manifest "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
  --output "${OUTPUT_ROOT}/${RUN_NAME}.json" \
  --load-stage "${INITIAL_STAGE}" --save-stage "${FINAL_STAGE}" \
  --checkpoint-every 80 --steps 160 --gradient-accumulation-steps 16 \
  --action-horizon 32 --learning-rate 1e-4 --h3-learning-rate 1e-6 \
  --last-h3-blocks 0 --video-loss-weight 1.0 --language-ranking-weight 0 \
  --lr-schedule cosine --require-text-only-context --log-every 1 \
  > "${LOG_ROOT}/${RUN_NAME}.log" 2>&1

bash "${PROJECT_ROOT}/scripts/h3dreamwam/eval_h3dotwam_dense_canary_checkpoint.sh" \
  "${FINAL_STAGE}" dense_mid256_step160
