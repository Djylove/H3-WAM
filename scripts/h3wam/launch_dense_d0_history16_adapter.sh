#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/candidate-d0-rollout-96976ce/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/h3-int8-native/bin/python}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${H3_WORKSPACE}/data/v7_multisuite_dense_candidate}"
CACHE_ROOT="${CACHE_ROOT:-${H3_WORKSPACE}/data/v7_dense_h3_cache}"
HISTORY_ROOT="${HISTORY_ROOT:-${H3_WORKSPACE}/data/v7_executed_action_history}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${H3_WORKSPACE}/outputs/d0-history16-adapter-s3000-v1}"
PARENT="${PARENT:-${H3_WORKSPACE}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt}"
PARENT_SHA256="36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

for required in \
  "${PROJECT_ROOT}/scripts/h3wam/train_h3_int8_dreamwam_kv_carrier.py" \
  "${CANDIDATE_ROOT}/manifest_all.jsonl" \
  "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
  "${CACHE_ROOT}/stats.pt" \
  "${HISTORY_ROOT}/report.json" \
  "${PARENT}"; do
  [[ -f "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 2; }
done
[[ $(wc -l < "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl") -eq 200779 ]] || {
  echo "training manifest row count changed" >&2
  exit 2
}
actual_parent_sha256="$(sha256sum "${PARENT}" | awk '{print $1}')"
[[ "${actual_parent_sha256}" == "${PARENT_SHA256}" ]] || {
  echo "parent checkpoint identity mismatch: ${actual_parent_sha256}" >&2
  exit 2
}
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  echo "refusing to reuse history output root: ${OUTPUT_ROOT}" >&2
  exit 1
}

mkdir -p "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/reports" \
  "${OUTPUT_ROOT}/logs" "${H3_WORKSPACE}/tmp/d0-history16-adapter-s3000-v1"
printf '%s\n' "$(date -Iseconds)" > "${OUTPUT_ROOT}/STARTED"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${H3_WORKSPACE}/tmp/d0-history16-adapter-s3000-v1"
export LD_LIBRARY_PATH="${H3_WORKSPACE}/runtime/h3-int8-native/lib/python3.11/site-packages/nvidia/cu13/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
cd "${PROJECT_ROOT}"

common_args=(
  "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl"
  --source-manifest "${CANDIDATE_ROOT}/manifest_all.jsonl"
  --cache-root "${CACHE_ROOT}" --kv-subdir h3_int8_dreamwam_kv_5x32_dense_v1
  --enable-dreamwam-kv-carrier --enable-d0-repeat-layer49
  --verify-h3-checkpoint-sha256 --action-horizon 32 --action-shift 5
  --history-action-steps 16 --executed-action-history-root "${HISTORY_ROOT}"
  --train-history-adapter-only
  --per-device-batch-size 1 --gradient-accumulation-steps 1 --num-workers 0
  --learning-rate 1e-4 --weight-decay 0.01 --warmup-steps 100
  --scheduler-horizon 3000 --min-learning-rate 1e-6 --seed 42
)

previous_checkpoint=""
for adapter_step in 500 1000 1500 2000 2500 3000; do
  absolute_step=$((14000 + adapter_step))
  sample_offset=$((112000 + (adapter_step - 500) * 8))
  checkpoint="${OUTPUT_ROOT}/checkpoints/d0_history16_s${absolute_step}.pt"
  report="${OUTPUT_ROOT}/reports/d0_history16_s${absolute_step}_train.json"
  restore_report="${OUTPUT_ROOT}/reports/d0_history16_s${absolute_step}_restore.json"
  log="${OUTPUT_ROOT}/logs/d0_history16_s${absolute_step}.log"
  initialization_args=()
  if [[ -z "${previous_checkpoint}" ]]; then
    initialization_args+=(--initialize-history-from "${PARENT}")
  else
    initialization_args+=(--load-checkpoint "${previous_checkpoint}")
  fi

  "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc-per-node "${NPROC_PER_NODE}" \
    scripts/h3wam/train_h3_int8_dreamwam_kv_carrier.py \
    "${common_args[@]}" "${initialization_args[@]}" \
    --steps 500 --sample-offset "${sample_offset}" --limit 4000 \
    --save-checkpoint "${checkpoint}" --output "${report}" 2>&1 | tee "${log}"

  "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc-per-node "${NPROC_PER_NODE}" \
    scripts/h3wam/train_h3_int8_dreamwam_kv_carrier.py \
    "${common_args[@]}" --load-checkpoint "${checkpoint}" --restore-check-only \
    --steps 1 --sample-offset 0 --limit 1 --output "${restore_report}" \
    >> "${log}" 2>&1
  previous_checkpoint="${checkpoint}"
done

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "completed": True,
    "parent_completed_steps": 14000,
    "adapter_steps": 3000,
    "global_batch": 8,
    "training_samples": 24000,
    "manifest_items": 200779,
    "effective_epochs": 24000 / 200779,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "final_checkpoint": str(root / "checkpoints/d0_history16_s17000.pt"),
}
output = root / "COMPLETED"
temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(temporary, output)
print(json.dumps(payload, sort_keys=True))
PY
