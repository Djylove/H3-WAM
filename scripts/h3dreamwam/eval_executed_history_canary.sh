#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v7_dense_h3_cache"
HISTORY_ROOT="${H3_WORKSPACE}/data/v7_executed_action_history"
VAL_MANIFEST="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate/manifest_val_stratified40.jsonl"
PARENT="${H3_WORKSPACE}/outputs/h3-lingbot-shared/quantile_flowweight_lr1e5_tail2_s10000_step005000.pt"
TRAINED="${H3_WORKSPACE}/outputs/h3-lingbot-history/history16_from_s5000_s100.pt"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3-lingbot-history/eval-s100"
TRAIN_SCRIPT="scripts/h3dreamwam/verify_h3_lingbot_four_stream_fsdp.py"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${H3_WORKSPACE}/tmp/history16-eval"
export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export TMPDIR="${H3_WORKSPACE}/tmp/history16-eval"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

COMMON=(
  --shared-backbone --model "${MODEL}" --data-root "${DATA_ROOT}"
  --manifest "${VAL_MANIFEST}" --executed-action-history-steps 16
  --executed-action-history-root "${HISTORY_ROOT}"
  --action-normalization quantile
  --action-stats-json experiments/data/libero_v7_action_quantiles.json
  --flow-match-loss-weighting --last-trainable-layers 2 --action-horizon 32
  --learning-rate 1e-5 --weight-decay 0.01 --steps 1 --eval-only --eval-all
)

evaluate() {
  local tag="$1" stage="$2" bootstrap="$3"
  local prefix="${OUTPUT_ROOT}/${tag}"
  local bootstrap_args=()
  [[ "${bootstrap}" == "1" ]] && bootstrap_args+=(--allow-history-bootstrap)
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
    "${TRAIN_SCRIPT}" "${COMMON[@]}" "${bootstrap_args[@]}" \
    --load-stage "${stage}" --output "${prefix}_val40.json" \
    > "${prefix}_val40.log" 2>&1
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
    "${TRAIN_SCRIPT}" "${COMMON[@]}" "${bootstrap_args[@]}" \
    --load-stage "${stage}" --sample-eval --sample-steps 4 \
    --video-sample-steps 4 --action-sample-steps 4 \
    --output "${prefix}_sample40.json" > "${prefix}_sample40.log" 2>&1
}

evaluate parent_history_untrained "${PARENT}" 1
evaluate history_s100 "${TRAINED}" 0

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
records = {}
for tag in ("parent_history_untrained", "history_s100"):
    raw = json.loads((root / f"{tag}_val40.json").read_text())
    causal = json.loads((root / f"{tag}_sample40.json").read_text())
    records[tag] = {
        "teacher_forced_video_mse": raw["mean_video_loss"],
        "teacher_forced_action_mse": raw["mean_action_loss"],
        "causal_video_mse": causal["mean_video_loss"],
        "causal_action_mse": causal["mean_action_loss"],
    }
parent = records["parent_history_untrained"]
trained = records["history_s100"]
action_gain = (parent["causal_action_mse"] - trained["causal_action_mse"]) / parent["causal_action_mse"]
video_change = (trained["causal_video_mse"] - parent["causal_video_mse"]) / parent["causal_video_mse"]
record = {
    "experiment": "h3_lingbot_executed_history_s100_v1",
    "records": records,
    "causal_action_relative_improvement": action_gain,
    "causal_video_relative_change": video_change,
    "offline_gate": bool(action_gain >= 0.02 and video_change <= 0.02),
}
(root / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, sort_keys=True))
PY
