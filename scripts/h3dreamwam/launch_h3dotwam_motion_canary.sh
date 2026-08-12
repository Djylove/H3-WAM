#!/usr/bin/env bash
set -euo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
MOTION_ROOT="${MOTION_ROOT:-${H3_WORKSPACE}/data/v6_motion_multisuite}"
MANIFEST="${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate/manifest_train_uniform.jsonl"
ALL_MANIFEST="${H3_WORKSPACE}/data/v4_multisuite_uniform_candidate/manifest_all.jsonl"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3dotwam-motion"
LOG_ROOT="${H3_WORKSPACE}/logs/cluster-32409"
TMP_ROOT="${H3_WORKSPACE}/tmp/cluster-32409"
MODEL_ROOT="${H3_WORKSPACE}/models/MiniMax-H3"
DATA_ROOT="${H3_WORKSPACE}/data/v2_full_cache"
BASE_STAGE="${H3_WORKSPACE}/outputs/h3dotwam/m0v2_h32_gb128_s150_step000125.pt"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${TMP_ROOT}"
EXPECTED="$(wc -l < "${ALL_MANIFEST}")"
while [[ "$(find "${MOTION_ROOT}" -maxdepth 1 -name '*.pt' | wc -l)" -lt "${EXPECTED}" ]]; do
  sleep 30
done
while pgrep -f '[p]recompute_h3_motion_latents.py' >/dev/null; do
  sleep 10
done

export TMPDIR="${TMP_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"

COMMON=(
  --model "${MODEL_ROOT}"
  --data-root "${DATA_ROOT}"
  --motion-root "${MOTION_ROOT}"
  --manifest "${MANIFEST}"
  --load-stage "${BASE_STAGE}"
  --action-horizon 32
  --learning-rate 1e-5
  --h3-learning-rate 1e-6
  --last-h3-blocks 50
  --video-loss-weight 1.0
  --flow-loss-weight 0.5
  --train-h3-io
  --dreamwam-world-weighting
  --language-ranking-weight 0
  --require-text-only-context
  --log-every 1
)

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  "${COMMON[@]}" \
  --output "${OUTPUT_ROOT}/m0_motion_full50_1step.json" \
  --steps 1 --gradient-accumulation-steps 1 \
  > "${LOG_ROOT}/motion_train_1step.log" 2>&1

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "${PROJECT_ROOT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
  "${COMMON[@]}" \
  --output "${OUTPUT_ROOT}/m1_motion_full50_gb128_s10.json" \
  --save-joint-stage "${OUTPUT_ROOT}/m1_motion_full50_gb128_s10_joint" \
  --steps 10 --gradient-accumulation-steps 16 --lr-schedule cosine \
  > "${LOG_ROOT}/motion_train_10step.log" 2>&1
