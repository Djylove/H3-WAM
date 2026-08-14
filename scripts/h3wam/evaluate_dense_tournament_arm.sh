#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 2 )) || [[ "$1" != "d" && "$1" != "d0" && "$1" != "c03" ]] || [[ ! "$2" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 d|d0|c03 GPU_INDEX" >&2
  exit 2
fi
ARM="$1"
GPU_INDEX="$2"

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/candidate-d0-rollout-96976ce/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/h3-int8-native/bin/python}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${H3_WORKSPACE}/data/v7_multisuite_dense_candidate}"
CACHE_ROOT="${CACHE_ROOT:-${H3_WORKSPACE}/data/v7_dense_h3_cache}"
READY_MARKER="${READY_MARKER:-${H3_WORKSPACE}/dense-d0-v1-96976ce/cache_generation/full_audit/DUAL_CACHE_AUDIT_READY.json}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
SELECTED80_SHA256="26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42"

if [[ ! "${WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WAIT_SECONDS must be a positive integer" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
cd "${PROJECT_ROOT}"

while [[ ! -f "${READY_MARKER}" ]]; do
  printf '{"time":"%s","arm":"%s","state":"WAIT_CACHE_AUDIT","gpu":%s}\n' \
    "$(date -Iseconds)" "${ARM}" "${GPU_INDEX}"
  sleep "${WAIT_SECONDS}"
done

audit_hash="$(${PYTHON_BIN} - "${READY_MARKER}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("ready") is not True:
    raise SystemExit("cache audit marker is not READY")
print(payload["dreamwam_kv_aggregate_sha256"])
PY
)"

wait_for_checkpoint() {
  local checkpoint="$1"
  while [[ ! -f "${checkpoint}" ]]; do
    printf '{"time":"%s","arm":"%s","state":"WAIT_CHECKPOINT","checkpoint":"%s","gpu":%s}\n' \
      "$(date -Iseconds)" "${ARM}" "${checkpoint}" "${GPU_INDEX}"
    sleep "${WAIT_SECONDS}"
  done
}

if [[ "${ARM}" == "c03" ]]; then
  train_root="${H3_WORKSPACE}/outputs/dense-carrier-tournament-v1/c03-starwam"
  mkdir -p "${train_root}/evaluations" "${train_root}/evaluation_logs"
  for milestone in 1 50 100; do
    checkpoint="${train_root}/checkpoints/c03_starwam_s${milestone}.pt"
    output="${train_root}/evaluations/c03_starwam_s${milestone}_balanced80.json"
    log="${train_root}/evaluation_logs/c03_starwam_s${milestone}_balanced80.log"
    wait_for_checkpoint "${checkpoint}"
    if [[ -e "${output}" ]]; then
      echo "refusing to overwrite evaluation: ${output}" >&2
      exit 1
    fi
    "${PYTHON_BIN}" scripts/h3wam/evaluate_h3_int8_starwam_action.py \
      "${checkpoint}" \
      --source-manifest "${CANDIDATE_ROOT}/manifest_all.jsonl" \
      --train-manifest "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
      --val-manifest "${CANDIDATE_ROOT}/manifest_val.jsonl" \
      --cache-root "${CACHE_ROOT}" \
      --feature-subdir h3_int8_starwam_last32_dense_v1 \
      --output "${output}" --device cuda --batch-size 1 --num-workers 0 \
      --samples-per-task 2 --seed 42 --inference-steps 10 --action-shift 5 \
      --language-sensitivity --visual-feature-shuffle > "${log}" 2>&1
  done
  completion="${train_root}/EVALUATION_COMPLETED"
else
  train_root="${H3_WORKSPACE}/outputs/dense-carrier-tournament-v1/${ARM}"
  mkdir -p "${train_root}/evaluations" "${train_root}/evaluation_logs"
  for milestone in 10 50 250 500 963; do
    checkpoint="${train_root}/checkpoints/${ARM}_s${milestone}.pt"
    output="${train_root}/evaluations/${ARM}_s${milestone}_balanced80.json"
    log="${train_root}/evaluation_logs/${ARM}_s${milestone}_balanced80.log"
    wait_for_checkpoint "${checkpoint}"
    if [[ -e "${output}" ]]; then
      echo "refusing to overwrite evaluation: ${output}" >&2
      exit 1
    fi
    "${PYTHON_BIN}" scripts/h3wam/evaluate_h3_dreamwam_kv_carrier.py \
      "${checkpoint}" \
      --source-manifest "${CANDIDATE_ROOT}/manifest_all.jsonl" \
      --train-manifest "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
      --val-manifest "${CANDIDATE_ROOT}/manifest_val.jsonl" \
      --cache-root "${CACHE_ROOT}" \
      --kv-subdir h3_int8_dreamwam_kv_5x32_dense_v1 \
      --output "${output}" --device cuda --num-workers 0 \
      --cache-audit-aggregate-sha256 "${audit_hash}" \
      --expected-selected-ids-sha256 "${SELECTED80_SHA256}" > "${log}" 2>&1
  done
  completion="${train_root}/EVALUATION_COMPLETED"
fi

"${PYTHON_BIN}" - "${ARM}" "${completion}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

arm, raw_output = sys.argv[1:]
output = Path(raw_output)
payload = {
    "completed": True,
    "arm": arm,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "scope": "balanced80_offline_condition_aware",
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(temporary, output)
print(json.dumps(payload, sort_keys=True))
PY
