#!/usr/bin/env bash
set -euo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
JOINT_STAGE="${H3_WORKSPACE}/outputs/h3dotwam-motion/m1_motion_full50_gb128_s10_joint"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/eval-motion-dot/m1_motion_s10"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v2_full_cache"
VAL_MANIFEST="${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate/manifest_val.jsonl"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-30907"

mkdir -p "${OUTPUT_ROOT}" "${TMP_ROOT}"
while [[ ! -s "${JOINT_STAGE}/joint_stage.json" || ! -s "${JOINT_STAGE}/action_stage.pt" ]]; do
  sleep 30
done
for rank in $(seq 0 7); do
  test -s "${JOINT_STAGE}/h3_rank$(printf '%05d' "${rank}").pt"
done
while pgrep -f '[s]erve_h3dotwam_fsdp.py' >/dev/null \
  || pgrep -f '[t]rain_h3dotwam_fsdp.py' >/dev/null; do
  sleep 10
done
[[ -s "${OUTPUT_ROOT}/val850.json" ]] && exit 0

export TMPDIR="${TMP_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  --model "${MODEL_ROOT}" --data-root "${DATA_ROOT}" \
  --manifest "${VAL_MANIFEST}" --output "${OUTPUT_ROOT}/val850.json" \
  --load-joint-stage "${JOINT_STAGE}" \
  --eval-only --steps 107 --sample-steps 10 --action-horizon 32 \
  --require-text-only-context --log-every 20 > "${OUTPUT_ROOT}/val850.log" 2>&1
