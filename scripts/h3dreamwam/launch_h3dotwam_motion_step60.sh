#!/usr/bin/env bash
set -euo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MOTION_ROOT="${H3_WORKSPACE}/data/v6_motion_multisuite"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v2_full_cache"
BASE_STAGE="${H3_WORKSPACE}/outputs/h3dotwam/m0v2_h32_gb128_s150_step000125.pt"
TRAIN_MANIFEST="${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate/manifest_train_uniform.jsonl"
VAL_MANIFEST="${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate/manifest_val_stratified40.jsonl"
M1_EVAL_ROOT="${H3_WORKSPACE}/outputs/eval-motion-dot/m1_motion_s10"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3dotwam-motion"
EVAL_ROOT="${H3_WORKSPACE}/outputs/eval-motion-dot/m2_motion_s60"
LOG_ROOT="${H3_WORKSPACE}/logs/cluster-32409"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-32409"
JOINT_STAGE="${OUTPUT_ROOT}/m2_motion_full50_gb128_s60_joint"

mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}" "${LOG_ROOT}" "${TMP_ROOT}"
while pgrep -f '[w]atch_and_eval_motion_m1.sh' >/dev/null; do
  sleep 30
done
test -s "${M1_EVAL_ROOT}/val40.json"
test -s "${M1_EVAL_ROOT}/libero_goal_canary/results.json"
"${PYTHON_BIN}" - "${M1_EVAL_ROOT}" <<'PY'
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
validation = json.loads((root / "val40.json").read_text())
rollout = json.loads((root / "libero_goal_canary/results.json").read_text())
loss = float(validation["mean_action_loss"])
if not math.isfinite(loss) or loss > 1.1 * 0.217331:
    raise SystemExit(f"motion-10 val gate failed: {loss}")
if int(rollout.get("episodes", 0)) != 4:
    raise SystemExit("motion-10 rollout gate is incomplete")
PY
while pgrep -f '[t]rain_h3dotwam_fsdp.py' >/dev/null \
  || pgrep -f '[s]erve_h3dotwam_fsdp.py' >/dev/null; do
  sleep 10
done
test ! -e "${JOINT_STAGE}"

export TMPDIR="${TMP_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
  --motion-root "${MOTION_ROOT}" --manifest "${TRAIN_MANIFEST}" \
  --output "${OUTPUT_ROOT}/m2_motion_full50_gb128_s60.json" \
  --save-joint-stage "${JOINT_STAGE}" --load-stage "${BASE_STAGE}" \
  --steps 60 --gradient-accumulation-steps 16 --action-horizon 32 \
  --learning-rate 1e-5 --h3-learning-rate 1e-6 --last-h3-blocks 50 \
  --video-loss-weight 1.0 --flow-loss-weight 0.5 --train-h3-io \
  --dreamwam-world-weighting --language-ranking-weight 0 --lr-schedule cosine \
  --require-text-only-context --log-every 1 \
  > "${LOG_ROOT}/motion_train_60step.log" 2>&1

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
  --manifest "${VAL_MANIFEST}" --output "${EVAL_ROOT}/val40.json" \
  --load-joint-stage "${JOINT_STAGE}" \
  --eval-only --steps 5 --sample-steps 10 --action-horizon 32 \
  --require-text-only-context --log-every 1 \
  > "${EVAL_ROOT}/val40.log" 2>&1

SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES:-/tmp/h3-wam-libero-site}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${PROJECT_ROOT}/scripts/h3wam/run_cloud_libero.sh" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3dreamwam/rollout_h3dreamwam_fsdp.py" \
  --dot --model "${MODEL_ROOT}" \
  --action-stage "${JOINT_STAGE}/action_stage.pt" \
  --h3-joint-stage "${JOINT_STAGE}" \
  --cache-root "${DATA_ROOT}" --manifest "${TRAIN_MANIFEST}" \
  --torchrun "${PROJECT_ROOT}/scripts/h3dreamwam/torchrun_shared.sh" \
  --suite libero_goal --task-ids 0 3 7 8 --trial-indices 0 \
  --max-steps 400 --wait-steps 30 --replan-steps 10 \
  --action-horizon 32 --sample-steps 10 \
  --output-dir "${EVAL_ROOT}/libero_goal_canary" \
  --save-video --save-trajectories --require-text-only-context \
  > "${EVAL_ROOT}/libero_goal_canary.log" 2>&1
