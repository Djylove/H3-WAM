#!/usr/bin/env bash
set -Eeuo pipefail

ACTION_STAGE="${1:?usage: eval_h3dotwam_dense_canary_checkpoint.sh ACTION_STAGE LABEL}"
LABEL="${2:?usage: eval_h3dotwam_dense_canary_checkpoint.sh ACTION_STAGE LABEL}"
H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v7_dense_canary_cache"
CANDIDATE_ROOT="${H3_WORKSPACE}/data/v7_dense_canary_candidate"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/eval-dense-dot/m10_dense_canary/${LABEL}"
TMP_ROOT="${H3_WORKSPACE}/tmp/dense-canary-eval-${LABEL}"

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="${H3_WORKSPACE}/cache"
export HF_HOME="${H3_WORKSPACE}/cache/huggingface"
export TORCH_HOME="${H3_WORKSPACE}/cache/torch"
export TMPDIR="${TMP_ROOT}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

test -s "${ACTION_STAGE}"
test ! -e "${OUTPUT_ROOT}/complete.json"
mkdir -p "${OUTPUT_ROOT}" "${TMP_ROOT}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
  --manifest "${CANDIDATE_ROOT}/manifest_val_stratified40.jsonl" \
  --output "${OUTPUT_ROOT}/val40.json" --load-stage "${ACTION_STAGE}" \
  --eval-only --steps 5 --sample-steps 10 --action-horizon 32 \
  --require-text-only-context --log-every 1 \
  > "${OUTPUT_ROOT}/val40.log" 2>&1

SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES:-/tmp/h3-wam-libero-site}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${PROJECT_ROOT}/scripts/h3wam/run_cloud_libero.sh" \
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3dreamwam/rollout_h3dreamwam_fsdp.py" \
  --dot --model "${MODEL_ROOT}" --action-stage "${ACTION_STAGE}" \
  --cache-root "${DATA_ROOT}" \
  --manifest "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
  --torchrun "${PROJECT_ROOT}/scripts/h3dreamwam/torchrun_shared.sh" \
  --suite libero_goal --task-ids 3 --trial-indices 0 1 2 3 \
  --max-steps 400 --wait-steps 30 --replan-steps 10 \
  --action-horizon 32 --sample-steps 10 \
  --output-dir "${OUTPUT_ROOT}/libero_goal_task3" \
  --save-video --save-trajectories --require-text-only-context \
  > "${OUTPUT_ROOT}/libero_goal_task3.log" 2>&1

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${ACTION_STAGE}" <<'PY'
import json, pathlib, sys
root, stage = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
val = json.loads((root / "val40.json").read_text())
summaries = list((root / "libero_goal_task3").rglob("results.json"))
if len(summaries) != 1:
    raise SystemExit(f"expected one rollout summary, found {summaries}")
rollout = json.loads(summaries[0].read_text())
record = {
    "action_stage": str(stage.resolve()),
    "val_action_loss": val["mean_action_loss"],
    "rollout_summary": rollout,
}
(root / "complete.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, sort_keys=True))
PY
