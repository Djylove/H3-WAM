#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
ACTION_STAGE="${H3_WORKSPACE}/outputs/h3dotwam-dense/m12_dense_mid256_head_gb128_s160.pt"
DATA_ROOT="${H3_WORKSPACE}/data/v7_dense_canary_cache"
MANIFEST="${H3_WORKSPACE}/data/v7_dense_mid256_candidate/manifest_train_uniform.jsonl"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/eval-dense-dot/m12_crosssuite"

export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="${H3_WORKSPACE}/cache"
export HF_HOME="${H3_WORKSPACE}/cache/huggingface"
export TORCH_HOME="${H3_WORKSPACE}/cache/torch"

test -s "${ACTION_STAGE}"
mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
for suite in libero_10 libero_goal libero_object libero_spatial; do
  output="${OUTPUT_ROOT}/${suite}"
  [[ -s "${output}/results.json" ]] && continue
  mkdir -p "${output}"
  SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES:-/tmp/h3-wam-libero-site}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${PROJECT_ROOT}/scripts/h3wam/run_cloud_libero.sh" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3dreamwam/rollout_h3dreamwam_fsdp.py" \
    --dot --model "${MODEL_ROOT}" --action-stage "${ACTION_STAGE}" \
    --cache-root "${DATA_ROOT}" --manifest "${MANIFEST}" \
    --torchrun "${PROJECT_ROOT}/scripts/h3dreamwam/torchrun_shared.sh" \
    --suite "${suite}" --task-ids 0 3 7 8 --trial-indices 0 \
    --max-steps 400 --wait-steps 30 --replan-steps 10 \
    --action-horizon 32 --sample-steps 10 --output-dir "${output}" \
    --save-video --save-trajectories --require-text-only-context \
    > "${output}.log" 2>&1
done
