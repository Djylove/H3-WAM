#!/usr/bin/env bash
set -euo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v2_full_cache"
TRAIN_MANIFEST="${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate/manifest_train_uniform.jsonl"
JOINT_STAGE="${H3_WORKSPACE}/outputs/h3dotwam/m4_paper_joint_full40_10ep_joint_step000060"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/eval-rgb-dot/m4_step60"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-30907"

mkdir -p "${OUTPUT_ROOT}" "${TMP_ROOT}"
while pgrep -f '[s]erve_h3dotwam_fsdp.py' >/dev/null \
  || pgrep -f '[t]rain_h3dotwam_fsdp.py' >/dev/null; do
  sleep 10
done
export TMPDIR="${TMP_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"

for suite in libero_spatial libero_object libero_10; do
  output_dir="${OUTPUT_ROOT}/${suite}_canary"
  [[ -s "${output_dir}/results.json" ]] && continue
  SIM_SITE_PACKAGES="${SIM_SITE_PACKAGES:-/tmp/h3-wam-libero-site}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${PROJECT_ROOT}/scripts/h3wam/run_cloud_libero.sh" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/h3dreamwam/rollout_h3dreamwam_fsdp.py" \
    --dot --model "${MODEL_ROOT}" \
    --action-stage "${JOINT_STAGE}/action_stage.pt" --h3-joint-stage "${JOINT_STAGE}" \
    --cache-root "${DATA_ROOT}" --manifest "${TRAIN_MANIFEST}" \
    --torchrun "${PROJECT_ROOT}/scripts/h3dreamwam/torchrun_shared.sh" \
    --suite "${suite}" --task-ids 0 3 7 8 --trial-indices 0 \
    --max-steps 400 --wait-steps 30 --replan-steps 10 \
    --action-horizon 32 --sample-steps 10 --output-dir "${output_dir}" \
    --save-video --save-trajectories --require-text-only-context \
    > "${OUTPUT_ROOT}/${suite}_canary.log" 2>&1
done
