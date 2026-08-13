#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v7_dense_h3_cache"
HISTORY_ROOT="${H3_WORKSPACE}/data/v7_executed_action_history"
VAL_MANIFEST="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate/manifest_val_stratified40.jsonl"
STAGE_ROOT="${H3_WORKSPACE}/outputs/h3-lingbot-history"
RESULT_ROOT="${STAGE_ROOT}/eval-s3000"
TMP_ROOT="${H3_WORKSPACE}/tmp/history16-s3000-eval"
EVAL_LOCK="${H3_WORKSPACE}/tmp/h3-wam-eval-gpu.lock"

mkdir -p "${RESULT_ROOT}" "${TMP_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export TMPDIR="${TMP_ROOT}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${WAIT_FOR_MARKER:-}" ]]; then
  until [[ -e "${WAIT_FOR_MARKER}" ]]; do sleep 30; done
fi

wait_for_idle_gpus() {
  while [[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) -ne 0 ]]; do
    sleep 30
  done
}

evaluate_stage() {
  local step="$1" stage="$2" prefix
  prefix="${RESULT_ROOT}/history_step$(printf '%06d' "${step}")"
  [[ -s "${prefix}_complete.json" ]] && return 0
  until [[ -s "${stage}" ]]; do sleep 30; done
  exec 9>"${EVAL_LOCK}"
  flock 9
  [[ -s "${prefix}_complete.json" ]] && { flock -u 9; exec 9>&-; return 0; }
  wait_for_idle_gpus

  local common=(
    --shared-backbone --model "${MODEL}" --data-root "${DATA_ROOT}"
    --manifest "${VAL_MANIFEST}" --executed-action-history-steps 16
    --executed-action-history-root "${HISTORY_ROOT}"
    --action-normalization quantile
    --action-stats-json experiments/data/libero_v7_action_quantiles.json
    --flow-match-loss-weighting --load-stage "${stage}"
    --steps 1 --eval-only --eval-all --last-trainable-layers 2 --action-horizon 32
  )
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
    scripts/h3dreamwam/verify_h3_lingbot_four_stream_fsdp.py "${common[@]}" \
    --output "${prefix}_val40.json" > "${prefix}_val40.log" 2>&1
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
    scripts/h3dreamwam/verify_h3_lingbot_four_stream_fsdp.py "${common[@]}" \
    --sample-eval --sample-steps 4 --video-sample-steps 4 --action-sample-steps 4 \
    --output "${prefix}_sample40.json" > "${prefix}_sample40.log" 2>&1

  "${PYTHON_BIN}" - "${prefix}" "${step}" "${stage}" <<'PY'
import json
import pathlib
import sys

prefix, step, stage = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
val = json.loads(pathlib.Path(f"{prefix}_val40.json").read_text())
sample = json.loads(pathlib.Path(f"{prefix}_sample40.json").read_text())
record = {
    "family": "executed_history16",
    "cumulative_step": step,
    "stage": stage,
    "teacher_forced_video_mse": val["mean_video_loss"],
    "teacher_forced_action_mse": val["mean_action_loss"],
    "causal_video_mse": sample["mean_video_loss"],
    "causal_action_mse": sample["mean_action_loss"],
}
pathlib.Path(f"{prefix}_complete.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, sort_keys=True))
PY
  flock -u 9
  exec 9>&-
}

for step in 500 1000 1500 2000 2500; do
  evaluate_stage "${step}" \
    "${STAGE_ROOT}/history16_from_s5000_s3000_step$(printf '%06d' "${step}").pt"
done
evaluate_stage 3000 "${STAGE_ROOT}/history16_from_s5000_s3000.pt"
touch "${RESULT_ROOT}/ladder_complete"
