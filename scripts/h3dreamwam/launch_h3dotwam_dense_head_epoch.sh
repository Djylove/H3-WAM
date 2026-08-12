#!/usr/bin/env bash
set -Eeuo pipefail

# One dense, episode-disjoint LIBERO epoch with the MiniMax-H3 hub frozen.
# This isolates data coverage and action-side optimization before another
# expensive 32B-parameter joint run.
H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v7_dense_h3_cache"
CANDIDATE_ROOT="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate"
RUN_NAME="${RUN_NAME:-m10_dense_head_gb128_s1569}"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3dotwam-dense"
LOG_ROOT="${H3_WORKSPACE}/logs/cluster-32409"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-32409-dense-head"
FINAL_STAGE="${OUTPUT_ROOT}/${RUN_NAME}.pt"
INITIAL_STAGE="${INITIAL_STAGE:-${H3_WORKSPACE}/outputs/h3dotwam/m0v2_h32_gb128_s150_step000125.pt}"

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
      && -s "${DATA_ROOT}/stats.pt" ]]; do
  sleep 30
done

read -r TRAIN_WINDOWS STEPS < <("${PYTHON_BIN}" - "${CANDIDATE_ROOT}" <<'PY'
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = sum(1 for line in (root / "manifest_train_uniform.jsonl").open() if line.strip())
report = json.loads((root / "candidate_report.json").read_text())
if report["contract"].get("window_sampling") != "dense" or rows != 200779:
    raise SystemExit(f"unexpected dense candidate: rows={rows}, report={report['contract']}")
print(rows, math.ceil(rows / 128))
PY
)
[[ "${STEPS}" == "1569" ]]
test ! -e "${FINAL_STAGE}"
test -s "${INITIAL_STAGE}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
  --manifest "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
  --output "${OUTPUT_ROOT}/${RUN_NAME}.json" \
  --load-stage "${INITIAL_STAGE}" --save-stage "${FINAL_STAGE}" \
  --checkpoint-every 200 --steps "${STEPS}" --gradient-accumulation-steps 16 \
  --action-horizon 32 --learning-rate 1e-4 --h3-learning-rate 1e-6 \
  --last-h3-blocks 0 --video-loss-weight 1.0 --language-ranking-weight 0 \
  --lr-schedule cosine --require-text-only-context --log-every 1 \
  > "${LOG_ROOT}/${RUN_NAME}.log" 2>&1

bash "${PROJECT_ROOT}/scripts/h3dreamwam/eval_h3dotwam_dense_head_checkpoint.sh" \
  "${FINAL_STAGE}" final
