#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/h3-wam/candidate-d0-rollout-96976ce/project}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/h3-wam/runtime/h3-int8-native/bin/python}"
SOURCE_ROOT="${H3WAM_FASTWAM_SOURCE_ROOT:-/mnt/h3-wam/upstream-readonly/FastWAM-45d8e145/wan22}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/h3-wam/outputs/c58-fastwam-full30-v1/long10000}"
MANIFEST="${MANIFEST:-/mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_all.jsonl}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/h3-wam/data/v7_dense_h3_cache}"
KV_SUBDIR="${KV_SUBDIR:-h3_int8_dreamwam_kv_5x32_dense_v1}"
D0_PARENT="${D0_PARENT:-/mnt/h3-wam/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt}"

BASE_OFFSET=112000
STAGE_STEPS=1000
GLOBAL_BATCH=8
STAGE_ROWS=$((STAGE_STEPS * GLOBAL_BATCH))
TOTAL_STAGES=10

for required in \
  "${PYTHON_BIN}" \
  "${MANIFEST}" \
  "${SOURCE_MANIFEST}" \
  "${D0_PARENT}" \
  "${SOURCE_ROOT}/action_dit.py" \
  "${SOURCE_ROOT}/wan_video_dit.py" \
  "${SOURCE_ROOT}/helpers/gradient.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required C58 input: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/reports" "${OUTPUT_ROOT}/logs"

CUDA13_LIB="$(${PYTHON_BIN} -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["purelib"])/"nvidia"/"cu13"/"lib")')"
export LD_LIBRARY_PATH="${CUDA13_LIB}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="${PROJECT_ROOT}/src"
export H3WAM_FASTWAM_SOURCE_ROOT="${SOURCE_ROOT}"

cd "${PROJECT_ROOT}"

previous=""
for stage in $(seq 1 "${TOTAL_STAGES}"); do
  completed=$((stage * STAGE_STEPS))
  offset=$((BASE_OFFSET + (stage - 1) * STAGE_ROWS))
  checkpoint="${OUTPUT_ROOT}/checkpoints/c58_s${completed}.pt"
  report="${OUTPUT_ROOT}/reports/train_s${completed}.json"
  log="${OUTPUT_ROOT}/logs/train_s${completed}.log"

  if [[ -f "${checkpoint}" && -f "${report}" ]]; then
    echo "[C58] stage s${completed} already complete; resuming launcher"
    previous="${checkpoint}"
    continue
  fi
  if [[ -e "${checkpoint}" || -e "${report}" ]]; then
    echo "[C58] partial stage artifacts require audit before restart: s${completed}" >&2
    exit 1
  fi

  command=(
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=8
    scripts/h3wam/train_h3_fastwam_full_tower.py
    "${MANIFEST}"
    --source-manifest "${SOURCE_MANIFEST}"
    --cache-root "${CACHE_ROOT}"
    --kv-subdir "${KV_SUBDIR}"
    --d0-parent-checkpoint "${D0_PARENT}"
    --verify-h3-checkpoint-sha256
    --steps "${STAGE_STEPS}"
    --sample-offset "${offset}"
    --limit "${STAGE_ROWS}"
    --per-device-batch-size 1
    --gradient-accumulation-steps 1
    --num-workers 0
    --learning-rate 1e-4
    --weight-decay 0.01
    --warmup-steps 1000
    --scheduler-horizon 10000
    --min-learning-rate 1e-6
    --action-horizon 32
    --action-shift 5
    --save-checkpoint "${checkpoint}"
    --output "${report}"
  )
  if [[ -n "${previous}" ]]; then
    command+=(--load-checkpoint "${previous}")
  fi

  echo "[C58] launching s${completed}: offset=${offset} rows=${STAGE_ROWS}"
  "${command[@]}" 2>&1 | tee "${log}"
  previous="${checkpoint}"
done

final_report="${OUTPUT_ROOT}/reports/restore_s10000.json"
final_log="${OUTPUT_ROOT}/logs/restore_s10000.log"
if [[ ! -f "${final_report}" ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=8 \
    scripts/h3wam/train_h3_fastwam_full_tower.py \
    "${MANIFEST}" \
    --source-manifest "${SOURCE_MANIFEST}" \
    --cache-root "${CACHE_ROOT}" \
    --kv-subdir "${KV_SUBDIR}" \
    --d0-parent-checkpoint "${D0_PARENT}" \
    --verify-h3-checkpoint-sha256 \
    --steps "${STAGE_STEPS}" \
    --sample-offset $((BASE_OFFSET + (TOTAL_STAGES - 1) * STAGE_ROWS)) \
    --limit "${STAGE_ROWS}" \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --num-workers 0 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --warmup-steps 1000 \
    --scheduler-horizon 10000 \
    --min-learning-rate 1e-6 \
    --action-horizon 32 \
    --action-shift 5 \
    --load-checkpoint "${OUTPUT_ROOT}/checkpoints/c58_s10000.pt" \
    --restore-check-only \
    --output "${final_report}" 2>&1 | tee "${final_log}"
fi

touch "${OUTPUT_ROOT}/COMPLETED"
echo "[C58] completed 10000 steps and final strict restore"
