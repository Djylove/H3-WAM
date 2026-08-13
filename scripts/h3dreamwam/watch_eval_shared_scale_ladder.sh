#!/usr/bin/env bash
set -Eeuo pipefail

# Sequentially consume the tail-2/tail-4 shared-H3 scale ladders on a dedicated
# 8-GPU evaluation node. Training nodes only write immutable milestone stages.
H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v7_dense_h3_cache"
VAL_MANIFEST="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate/manifest_val_stratified40.jsonl"
STAGE_ROOT="${H3_WORKSPACE}/outputs/h3-lingbot-shared"
RESULT_ROOT="${H3_WORKSPACE}/outputs/eval-lingbot-shared/scale-ladder"
LOG_ROOT="${H3_WORKSPACE}/logs/cluster-32409"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-32409-shared-scale-eval"

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="${H3_WORKSPACE}/runtime/gl_root/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TMPDIR="${TMP_ROOT}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}" "${TMP_ROOT}"
cd "${PROJECT_ROOT}"

wait_for_idle_gpus() {
  while [[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) -ne 0 ]]; do
    sleep 30
  done
}

evaluate_stage() {
  local family="$1" step="$2" layers="$3" stage="$4"
  local prefix="${RESULT_ROOT}/${family}_step$(printf '%04d' "${step}")"
  [[ -s "${prefix}_complete.json" ]] && return 0
  until [[ -s "${stage}" ]]; do sleep 30; done
  wait_for_idle_gpus

  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
    scripts/h3dreamwam/verify_h3_lingbot_four_stream_fsdp.py \
    --shared-backbone --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
    --manifest "${VAL_MANIFEST}" --action-normalization quantile \
    --action-stats-json experiments/data/libero_v7_action_quantiles.json \
    --flow-match-loss-weighting --load-stage "${stage}" \
    --output "${prefix}_val40.json" --steps 1 --eval-only --eval-all \
    --last-trainable-layers "${layers}" --action-horizon 32 \
    > "${prefix}_val40.log" 2>&1

  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
    scripts/h3dreamwam/verify_h3_lingbot_four_stream_fsdp.py \
    --shared-backbone --sample-eval --sample-steps 4 \
    --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
    --manifest "${VAL_MANIFEST}" --action-normalization quantile \
    --action-stats-json experiments/data/libero_v7_action_quantiles.json \
    --flow-match-loss-weighting --load-stage "${stage}" \
    --output "${prefix}_sample40.json" --steps 1 --eval-only --eval-all \
    --last-trainable-layers "${layers}" --action-horizon 32 \
    > "${prefix}_sample40.log" 2>&1

  "${PYTHON_BIN}" - "${prefix}" "${family}" "${step}" "${stage}" <<'PY'
import json
import pathlib
import sys

prefix, family, step, stage = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
val = json.loads(pathlib.Path(f"{prefix}_val40.json").read_text())
sample = json.loads(pathlib.Path(f"{prefix}_sample40.json").read_text())
record = {
    "family": family,
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
}

for step in 1000 1500 2000; do
  evaluate_stage tail2 "${step}" 2 \
    "${STAGE_ROOT}/quantile_flowweight_lr1e5_tail2_s2500_step$(printf '%06d' "${step}").pt"
done
evaluate_stage tail2 2500 2 "${STAGE_ROOT}/quantile_flowweight_lr1e5_tail2_s2500.pt"

for step in 500 1000 1500 2000; do
  evaluate_stage tail4 "${step}" 4 \
    "${STAGE_ROOT}/quantile_flowweight_lr1e5_tail4_s2500_step$(printf '%06d' "${step}").pt"
done
evaluate_stage tail4 2500 4 "${STAGE_ROOT}/quantile_flowweight_lr1e5_tail4_s2500.pt"

touch "${RESULT_ROOT}/ladder_complete"
