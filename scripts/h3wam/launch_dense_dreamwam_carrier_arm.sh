#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 1 )) || [[ "$1" != "d" && "$1" != "d0" ]]; then
  echo "usage: $0 d|d0" >&2
  exit 2
fi
ARM="$1"

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/candidate-d0-rollout-96976ce/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/h3-int8-native/bin/python}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${H3_WORKSPACE}/data/v7_multisuite_dense_candidate}"
CACHE_ROOT="${CACHE_ROOT:-${H3_WORKSPACE}/data/v7_dense_h3_cache}"
KV_SUBDIR="${KV_SUBDIR:-h3_int8_dreamwam_kv_5x32_dense_v1}"
READY_MARKER="${READY_MARKER:-${H3_WORKSPACE}/dense-d0-v1-96976ce/cache_generation/full_audit/DUAL_CACHE_AUDIT_READY.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${H3_WORKSPACE}/outputs/dense-carrier-tournament-v1/${ARM}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE and WAIT_SECONDS must be positive integers" >&2
  exit 2
fi
for required in \
  "${CANDIDATE_ROOT}/manifest_all.jsonl" \
  "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
  "${PROJECT_ROOT}/scripts/h3wam/train_h3_int8_dreamwam_kv_carrier.py"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done

while [[ ! -f "${READY_MARKER}" ]]; do
  printf '{"time":"%s","arm":"%s","state":"WAIT_CACHE_AUDIT"}\n' "$(date -Iseconds)" "${ARM}"
  sleep "${WAIT_SECONDS}"
done

"${PYTHON_BIN}" - "${READY_MARKER}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("ready") is not True:
    raise SystemExit("cache audit marker is not READY")
if payload.get("manifest_sha256") != "cab8876f067114dce41d16ca52cb0bafddf17da33c92d0adde5f11d7ac9555b9":
    raise SystemExit("cache audit marker has the wrong manifest")
if payload.get("checkpoint_sha256") != "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a":
    raise SystemExit("cache audit marker has the wrong H3 checkpoint")
PY

if [[ -e "${OUTPUT_ROOT}/STARTED" || -e "${OUTPUT_ROOT}/COMPLETED" ]]; then
  echo "refusing to reuse an already-started output root: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/reports" "${OUTPUT_ROOT}/logs"
printf '%s\n' "$(date -Iseconds)" > "${OUTPUT_ROOT}/STARTED"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${H3_WORKSPACE}/tmp/dense-carrier-${ARM}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${TMPDIR}"
cd "${PROJECT_ROOT}"

arm_args=()
if [[ "${ARM}" == "d0" ]]; then
  arm_args+=(--enable-d0-repeat-layer49)
fi

common_args=(
  "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl"
  --source-manifest "${CANDIDATE_ROOT}/manifest_all.jsonl"
  --cache-root "${CACHE_ROOT}"
  --kv-subdir "${KV_SUBDIR}"
  --enable-dreamwam-kv-carrier
  --verify-h3-checkpoint-sha256
  --per-device-batch-size 1
  --gradient-accumulation-steps 1
  --num-workers 0
  --learning-rate 1e-4
  --weight-decay 0.01
  --warmup-steps 1000
  --scheduler-horizon 21700
  --min-learning-rate 1e-6
  --action-shift 5
  --seed 42
)

previous_checkpoint=""
run_stage() {
  local milestone="$1"
  local stage_steps="$2"
  local sample_offset="$3"
  local sample_limit="$4"
  local checkpoint="${OUTPUT_ROOT}/checkpoints/${ARM}_s${milestone}.pt"
  local report="${OUTPUT_ROOT}/reports/${ARM}_s${milestone}_train.json"
  local restore_report="${OUTPUT_ROOT}/reports/${ARM}_s${milestone}_restore.json"
  local log="${OUTPUT_ROOT}/logs/${ARM}_s${milestone}.log"
  local load_args=()
  if [[ -n "${previous_checkpoint}" ]]; then
    load_args+=(--load-checkpoint "${previous_checkpoint}")
  fi

  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node "${NPROC_PER_NODE}" \
    scripts/h3wam/train_h3_int8_dreamwam_kv_carrier.py \
    "${common_args[@]}" "${arm_args[@]}" "${load_args[@]}" \
    --steps "${stage_steps}" --sample-offset "${sample_offset}" --limit "${sample_limit}" \
    --save-checkpoint "${checkpoint}" --output "${report}" 2>&1 | tee "${log}"

  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node "${NPROC_PER_NODE}" \
    scripts/h3wam/train_h3_int8_dreamwam_kv_carrier.py \
    "${common_args[@]}" "${arm_args[@]}" \
    --load-checkpoint "${checkpoint}" --restore-check-only \
    --steps 1 --sample-offset 0 --limit 1 --output "${restore_report}" \
    >> "${log}" 2>&1
  previous_checkpoint="${checkpoint}"
}

run_stage 10 10 0 80
run_stage 50 40 80 320
run_stage 250 200 400 1600
run_stage 500 250 2000 2000
run_stage 963 463 4000 3704

"${PYTHON_BIN}" - "${ARM}" "${OUTPUT_ROOT}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

arm, raw_root = sys.argv[1:]
root = Path(raw_root)
payload = {
    "completed": True,
    "arm": arm,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "milestones": [10, 50, 250, 500, 963],
    "training_samples": 7704,
    "final_checkpoint": str(root / "checkpoints" / f"{arm}_s963.pt"),
}
output = root / "COMPLETED"
temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(temporary, output)
print(json.dumps(payload, sort_keys=True))
PY
