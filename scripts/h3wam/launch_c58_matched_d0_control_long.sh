#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/h3-wam/candidate-d0-rollout-96976ce/project}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/h3-wam/runtime/h3-int8-native/bin/python}"
SOURCE_ROOT="${H3WAM_FASTWAM_SOURCE_ROOT:-/mnt/h3-wam/upstream-readonly/FastWAM-45d8e145/wan22}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/h3-wam/outputs/c58-matched-d0-control-v1/long10000}"
MANIFEST="${MANIFEST:-/mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_all.jsonl}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/h3-wam/data/v7_dense_h3_cache}"
KV_SUBDIR="${KV_SUBDIR:-h3_int8_dreamwam_kv_5x32_dense_v1}"
D0_PARENT="${D0_PARENT:-/mnt/h3-wam/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt}"
CANARY_READY="${CANARY_READY:-/mnt/h3-wam/outputs/c58-matched-d0-control-v1/probe10/CANARY_READY.json}"

BASE_OFFSET=112000
STAGE_STEPS=1000
GLOBAL_BATCH=8
STAGE_ROWS=$((STAGE_STEPS * GLOBAL_BATCH))
TOTAL_STAGES=10

for path in "${PYTHON_BIN}" "${MANIFEST}" "${SOURCE_MANIFEST}" "${D0_PARENT}" "${CANARY_READY}" \
  "${SOURCE_ROOT}/action_dit.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing matched-control input: ${path}" >&2
    exit 1
  fi
done
"${PYTHON_BIN}" - "${CANARY_READY}" <<'PY'
import json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text())
if x.get("status") != "PASS_C58_MATCHED_CONTROL_CANARY":
    raise SystemExit("matched-control canary did not authorize long training")
PY
mkdir -p "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/reports" "${OUTPUT_ROOT}/logs"

cuda13_lib="$(${PYTHON_BIN} -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["purelib"])/"nvidia"/"cu13"/"lib")')"
export LD_LIBRARY_PATH="${cuda13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="${PROJECT_ROOT}/src"
export H3WAM_FASTWAM_SOURCE_ROOT="${SOURCE_ROOT}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/h3wam/prepare_c58_matched_pair_contract.py \
  --manifest "${MANIFEST}" --d0-parent "${D0_PARENT}" \
  --output "${OUTPUT_ROOT}/PAIR_CONTRACT.json"

previous=""
for stage in $(seq 1 "${TOTAL_STAGES}"); do
  completed=$((stage * STAGE_STEPS))
  offset=$((BASE_OFFSET + (stage - 1) * STAGE_ROWS))
  checkpoint="${OUTPUT_ROOT}/checkpoints/control_s${completed}.pt"
  report="${OUTPUT_ROOT}/reports/train_s${completed}.json"
  restore_report="${OUTPUT_ROOT}/reports/restore_s${completed}.json"
  train_log="${OUTPUT_ROOT}/logs/train_s${completed}.log"
  restore_log="${OUTPUT_ROOT}/logs/restore_s${completed}.log"
  if [[ -f "${checkpoint}" && -f "${report}" && -f "${restore_report}" ]]; then
    previous="${checkpoint}"
    continue
  fi
  if [[ -e "${checkpoint}" || -e "${report}" || -e "${restore_report}" ]]; then
    echo "partial matched-control stage requires audit: s${completed}" >&2
    exit 1
  fi
  common=(
    scripts/h3wam/train_h3_fastwam_full_tower.py "${MANIFEST}"
    --source-manifest "${SOURCE_MANIFEST}"
    --cache-root "${CACHE_ROOT}" --kv-subdir "${KV_SUBDIR}"
    --d0-parent-checkpoint "${D0_PARENT}"
    --matched-d0-control --verify-h3-checkpoint-sha256
    --steps "${STAGE_STEPS}" --sample-offset "${offset}" --limit "${STAGE_ROWS}"
    --per-device-batch-size 1 --gradient-accumulation-steps 1 --num-workers 0
    --learning-rate 1e-4 --weight-decay 0.01
    --warmup-steps 1000 --scheduler-horizon 10000 --min-learning-rate 1e-6
    --action-horizon 32 --action-shift 5
  )
  load=()
  if [[ -n "${previous}" ]]; then
    load=(--load-checkpoint "${previous}")
  fi
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=8 \
    "${common[@]}" "${load[@]}" \
    --save-checkpoint "${checkpoint}" --output "${report}" >"${train_log}" 2>&1
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node=8 \
    "${common[@]}" --load-checkpoint "${checkpoint}" --restore-check-only \
    --output "${restore_report}" >"${restore_log}" 2>&1
  previous="${checkpoint}"
done

touch "${OUTPUT_ROOT}/COMPLETED"
echo "[C58 control] completed 10000 matched steps with per-milestone strict restore"
