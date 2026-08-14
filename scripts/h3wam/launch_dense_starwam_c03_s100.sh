#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/candidate-d0-rollout-96976ce/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/h3-int8-native/bin/python}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${H3_WORKSPACE}/data/v7_multisuite_dense_candidate}"
CACHE_ROOT="${CACHE_ROOT:-${H3_WORKSPACE}/data/v7_dense_h3_cache}"
FEATURE_SUBDIR="${FEATURE_SUBDIR:-h3_int8_starwam_last32_dense_v1}"
READY_MARKER="${READY_MARKER:-${H3_WORKSPACE}/dense-d0-v1-96976ce/cache_generation/full_audit/DUAL_CACHE_AUDIT_READY.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${H3_WORKSPACE}/outputs/dense-carrier-tournament-v1/c03-starwam}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE and WAIT_SECONDS must be positive integers" >&2
  exit 2
fi
for required in \
  "${CANDIDATE_ROOT}/manifest_all.jsonl" \
  "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
  "${PROJECT_ROOT}/scripts/h3wam/train_h3_int8_starwam_action.py"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done

while [[ ! -f "${READY_MARKER}" ]]; do
  printf '{"time":"%s","arm":"c03-starwam","state":"WAIT_CACHE_AUDIT"}\n' "$(date -Iseconds)"
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
export TMPDIR="${H3_WORKSPACE}/tmp/dense-starwam-c03"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${TMPDIR}"
cd "${PROJECT_ROOT}"

common_args=(
  "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl"
  --source-manifest "${CANDIDATE_ROOT}/manifest_all.jsonl"
  --cache-root "${CACHE_ROOT}"
  --feature-subdir "${FEATURE_SUBDIR}"
  --verify-h3-checkpoint-sha256
  --feature-input-scale 0.009606920816877307
  --action-layers 30
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
  local checkpoint="${OUTPUT_ROOT}/checkpoints/c03_starwam_s${milestone}.pt"
  local report="${OUTPUT_ROOT}/reports/c03_starwam_s${milestone}_train.json"
  local restore_report="${OUTPUT_ROOT}/reports/c03_starwam_s${milestone}_restore.json"
  local log="${OUTPUT_ROOT}/logs/c03_starwam_s${milestone}.log"
  local load_args=()
  if [[ -n "${previous_checkpoint}" ]]; then
    load_args+=(--load-checkpoint "${previous_checkpoint}")
  fi

  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node "${NPROC_PER_NODE}" \
    scripts/h3wam/train_h3_int8_starwam_action.py \
    "${common_args[@]}" "${load_args[@]}" \
    --steps "${stage_steps}" --sample-offset "${sample_offset}" --limit "${sample_limit}" \
    --save-checkpoint "${checkpoint}" --output "${report}" 2>&1 | tee "${log}"

  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node "${NPROC_PER_NODE}" \
    scripts/h3wam/train_h3_int8_starwam_action.py \
    "${common_args[@]}" --load-checkpoint "${checkpoint}" --restore-check-only \
    --steps 1 --sample-offset 0 --limit 1 --output "${restore_report}" \
    >> "${log}" 2>&1
  previous_checkpoint="${checkpoint}"
}

run_stage 1 1 0 8
run_stage 50 49 8 392
run_stage 100 50 400 400

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "completed": True,
    "arm": "c03-starwam",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "milestones": [1, 50, 100],
    "training_samples": 800,
    "final_checkpoint": str(root / "checkpoints" / "c03_starwam_s100.pt"),
    "long_training_permission": "NO_GO",
}
output = root / "COMPLETED"
temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(temporary, output)
print(json.dumps(payload, sort_keys=True))
PY
