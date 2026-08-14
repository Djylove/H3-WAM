#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project-shared-sync-v2}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v7_dense_h3_cache"
VAL_MANIFEST="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate/manifest_val_stratified40.jsonl"
STAGE_ROOT="${STAGE_ROOT:-${H3_WORKSPACE}/outputs/h3-lingbot-shared-sync-v2}"
RESULT_ROOT="${RESULT_ROOT:-${H3_WORKSPACE}/outputs/eval-h3-lingbot-shared-sync-v2}"
LOG_ROOT="${H3_WORKSPACE}/logs/h3-lingbot-shared-sync-v2"
TMP_ROOT="${H3_WORKSPACE}/tmp/h3-lingbot-shared-sync-v2-eval"
EVAL_LOCK="${H3_WORKSPACE}/tmp/h3-lingbot-shared-sync-v2-eval.lock"
RUN_NAME="${RUN_NAME:-shared_sync_v2_clean_s1000}"
FREEZE_SHARED_BLOCKS="${FREEZE_SHARED_BLOCKS:-0}"
extra_model_args=()
if [[ "${FREEZE_SHARED_BLOCKS}" == "1" ]]; then
  extra_model_args+=(--freeze-shared-blocks)
fi

export PYTHONPATH="${H3_WORKSPACE}/project/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="${H3_WORKSPACE}/runtime/gl_root/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TMPDIR="${TMP_ROOT}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}" "${TMP_ROOT}"
cd "${PROJECT_ROOT}"

wait_for_idle_gpus() {
  while [[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d' | wc -l) -ne 0 ]]; do
    sleep 30
  done
}

evaluate_stage() {
  local step="$1" stage="$2" prefix
  prefix="${RESULT_ROOT}/${RUN_NAME}_step$(printf '%04d' "${step}")"
  [[ -s "${prefix}_complete.json" ]] && return 0
  until [[ -s "${stage}" ]]; do sleep 30; done
  exec 9>"${EVAL_LOCK}"
  flock 9
  [[ -s "${prefix}_complete.json" ]] && { flock -u 9; exec 9>&-; return 0; }
  wait_for_idle_gpus

  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
    scripts/h3dreamwam/verify_h3_lingbot_four_stream_fsdp.py \
    --shared-backbone --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
    --manifest "${VAL_MANIFEST}" --action-normalization quantile \
    --action-stats-json experiments/data/libero_v7_action_quantiles.json \
    --flow-match-loss-weighting --load-stage "${stage}" \
    "${extra_model_args[@]}" \
    --output "${prefix}_val40.json" --steps 1 --eval-only --eval-all \
    --last-trainable-layers 2 --action-horizon 32 \
    > "${prefix}_val40.log" 2>&1

  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
    scripts/h3dreamwam/verify_h3_lingbot_four_stream_fsdp.py \
    --shared-backbone --sample-eval --sample-steps 4 \
    --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
    --manifest "${VAL_MANIFEST}" --action-normalization quantile \
    --action-stats-json experiments/data/libero_v7_action_quantiles.json \
    --flow-match-loss-weighting --load-stage "${stage}" \
    "${extra_model_args[@]}" \
    --output "${prefix}_sample40.json" --steps 1 --eval-only --eval-all \
    --last-trainable-layers 2 --action-horizon 32 \
    > "${prefix}_sample40.log" 2>&1

  "${PYTHON_BIN}" - "${prefix}" "${step}" "${stage}" <<'PY'
import json
import os
import pathlib
import sys

prefix, step, stage = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
val = json.loads(pathlib.Path(f"{prefix}_val40.json").read_text())
sample = json.loads(pathlib.Path(f"{prefix}_sample40.json").read_text())
if val.get("evaluated_samples") != 40 or sample.get("evaluated_samples") != 40:
    raise SystemExit("incomplete val40/sample40 result")
record = {
    "status": "complete",
    "cumulative_step": step,
    "stage": stage,
    "evaluated_samples": 40,
    "teacher_forced_video_mse": val["mean_video_loss"],
    "teacher_forced_action_mse": val["mean_action_loss"],
    "causal_video_mse": sample["mean_video_loss"],
    "causal_action_mse": sample["mean_action_loss"],
}
target = pathlib.Path(f"{prefix}_complete.json")
temporary = target.with_suffix(".partial.json")
temporary.write_text(json.dumps(record, indent=2) + "\n")
os.replace(temporary, target)
print(json.dumps(record, sort_keys=True))
PY
  flock -u 9
  exec 9>&-
}

for step in 200 400 600 800; do
  evaluate_stage "${step}" \
    "${STAGE_ROOT}/${RUN_NAME}_step$(printf '%06d' "${step}").pt"
done
evaluate_stage 1000 "${STAGE_ROOT}/${RUN_NAME}.pt"
