#!/usr/bin/env bash
set -euo pipefail

H3_ROOT="/home/h3wam_finetune"
PROJECT_ROOT="${H3_ROOT}/project"
RUN_NAME="m0v2_h32_gb128_s150"
TRAIN_PID_FILE="${H3_ROOT}/logs/h3dotwam/${RUN_NAME}.pid"
TRAIN_REPORT="${H3_ROOT}/outputs/h3dotwam/${RUN_NAME}.json"
POST_LOG_ROOT="${H3_ROOT}/outputs/h3dotwam/${RUN_NAME}_posttrain"
MODEL_ROOT="${H3_ROOT}/models/MiniMax-H3"
DATA_ROOT="${H3_ROOT}/data/v2_full_cache"
VAL_MANIFEST="${H3_ROOT}/data/v4_multisuite_uniform_candidate/manifest_val_stratified40.jsonl"
TRAIN_MANIFEST="${H3_ROOT}/data/v4_multisuite_uniform_candidate/manifest_train_uniform.jsonl"
TORCHRUN="${H3_ROOT}/.venv/bin/torchrun"

mkdir -p "${POST_LOG_ROOT}"
TRAIN_PID="$(cat "${TRAIN_PID_FILE}")"
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep 30
done
test -s "${TRAIN_REPORT}"

for STEP in 25 50 75 100 125 150; do
  if [[ "${STEP}" -eq 150 ]]; then
    STAGE="${H3_ROOT}/outputs/h3dotwam/${RUN_NAME}.pt"
  else
    STAGE="${H3_ROOT}/outputs/h3dotwam/${RUN_NAME}_step$(printf '%06d' "${STEP}").pt"
  fi
  test -s "${STAGE}"
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" "${TORCHRUN}" \
    --standalone --nproc-per-node=8 \
    "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
    --model "${MODEL_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --manifest "${VAL_MANIFEST}" \
    --output "${POST_LOG_ROOT}/val40_step$(printf '%06d' "${STEP}").json" \
    --load-stage "${STAGE}" \
    --eval-only --steps 5 --sample-steps 10 --action-horizon 32 \
    --require-text-only-context --log-every 1 \
    > "${POST_LOG_ROOT}/val40_step$(printf '%06d' "${STEP}").log" 2>&1
done

BEST_STAGE="$(${H3_ROOT}/.venv/bin/python - "${POST_LOG_ROOT}" "${H3_ROOT}/outputs/h3dotwam" <<'PY'
import json
import pathlib
import sys

report_root = pathlib.Path(sys.argv[1])
stage_root = pathlib.Path(sys.argv[2])
reports = sorted(report_root.glob("val40_step*.json"))
if len(reports) != 6:
    raise SystemExit(f"expected 6 validation reports, got {len(reports)}")
scored = [(json.loads(path.read_text())["mean_action_loss"], path) for path in reports]
score, report = min(scored)
step = int(report.stem.rsplit("step", 1)[1])
stage = (
    stage_root / "m0v2_h32_gb128_s150.pt"
    if step == 150
    else stage_root / f"m0v2_h32_gb128_s150_step{step:06d}.pt"
)
(report_root / "selection.json").write_text(
    json.dumps(
        {
            "best_step": step,
            "mean_action_mse": score,
            "checkpoint": str(stage),
            "all": [
                {
                    "step": int(path.stem.rsplit("step", 1)[1]),
                    "mean_action_mse": value,
                }
                for value, path in sorted(scored, key=lambda item: item[1].name)
            ],
        },
        indent=2,
    )
)
print(stage)
PY
)"

"${PROJECT_ROOT}/scripts/h3wam/run_cloud_libero.sh" \
  "${H3_ROOT}/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/h3dreamwam/rollout_h3dreamwam_fsdp.py" \
  --dot \
  --model "${MODEL_ROOT}" \
  --action-stage "${BEST_STAGE}" \
  --cache-root "${DATA_ROOT}" \
  --manifest "${TRAIN_MANIFEST}" \
  --torchrun "${TORCHRUN}" \
  --suite libero_goal --task-ids 0 --trial-indices 0 \
  --max-steps 100 --wait-steps 10 --replan-steps 10 \
  --action-horizon 32 --sample-steps 10 \
  --output-dir "${POST_LOG_ROOT}/libero_goal_task0_best_canary100" \
  --save-video --save-trajectories --require-text-only-context \
  > "${POST_LOG_ROOT}/libero_goal_task0_best_canary100.log" 2>&1
