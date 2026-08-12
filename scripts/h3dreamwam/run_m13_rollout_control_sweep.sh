#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
STAGE="${H3_WORKSPACE}/outputs/h3dotwam-dense/m13_dense_full_head_gb128_s1569_step000400.pt"
DATA_ROOT="${H3_WORKSPACE}/data/v7_dense_h3_cache"
MANIFEST="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/eval-dense-dot/m13_control_sweep_step0400"

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="${H3_WORKSPACE}/cache"
export HF_HOME="${H3_WORKSPACE}/cache/huggingface"
export TORCH_HOME="${H3_WORKSPACE}/cache/torch"

test -s "${STAGE}"
mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

run_variant() {
  local replan="$1"
  local median="$2"
  local scale="$3"
  local label="replan${replan}_median${median}_scale${scale}"
  local output="${OUTPUT_ROOT}/${label}"
  [[ -s "${output}/results.json" ]] && return 0
  SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES:-/tmp/h3-wam-libero-site}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${PROJECT_ROOT}/scripts/h3wam/run_cloud_libero.sh" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3dreamwam/rollout_h3dreamwam_fsdp.py" \
    --dot --model "${MODEL_ROOT}" --action-stage "${STAGE}" \
    --cache-root "${DATA_ROOT}" --manifest "${MANIFEST}" \
    --torchrun "${PROJECT_ROOT}/scripts/h3dreamwam/torchrun_shared.sh" \
    --suite libero_goal --task-ids 3 --trial-indices 0 \
    --max-steps 400 --wait-steps 30 --replan-steps "${replan}" \
    --action-horizon 32 --sample-steps 10 --action-median-window "${median}" \
    --action-scale "${scale}" --fixed-noise-seed 42 \
    --output-dir "${output}" --save-video --save-trajectories \
    --require-text-only-context > "${OUTPUT_ROOT}/${label}.log" 2>&1
}

# First isolate action-chunk execution length.  The final variant additionally
# tests whether smoothing/scaling is the missing controller-side stabilizer.
run_variant 5 1 1.0
run_variant 2 1 1.0
run_variant 1 1 1.0
run_variant 5 3 0.5

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for result in sorted(root.glob("*/results.json")):
    payload = json.loads(result.read_text())
    episodes = [episode for task in payload["tasks"] for episode in task["episodes"]]
    rows.append({
        "variant": result.parent.name,
        "successes": sum(bool(episode["success"]) for episode in episodes),
        "episodes": len(episodes),
        "mean_steps": sum(episode["steps"] for episode in episodes) / len(episodes),
    })
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows))
PY
