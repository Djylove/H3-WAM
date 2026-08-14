#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/candidate-d0-rollout-96976ce/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/h3-int8-native/bin/python}"
DATA_ROOT="${DATA_ROOT:-${H3_WORKSPACE}/data}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${DATA_ROOT}/v7_multisuite_dense_candidate}"
CACHE_ROOT="${CACHE_ROOT:-${DATA_ROOT}/v7_dense_h3_cache}"
KV_SUBDIR="${KV_SUBDIR:-h3_int8_dreamwam_kv_5x32_dense_v1}"
FEATURE_SUBDIR="${FEATURE_SUBDIR:-h3_int8_starwam_last32_dense_v1}"
EXPECTED_ITEMS="${EXPECTED_ITEMS:-222929}"
PRODUCER_SHARDS="${PRODUCER_SHARDS:-32}"
POLL_SECONDS="${POLL_SECONDS:-300}"
H3_CHECKPOINT="${H3_CHECKPOINT:-${H3_WORKSPACE}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
H3_CHECKPOINT_SHA256="${H3_CHECKPOINT_SHA256:-e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a}"
AUDIT_ROOT="${AUDIT_ROOT:-${H3_WORKSPACE}/dense-d0-v1-96976ce/cache_generation/full_audit}"
PRODUCER_LOG_ROOT="${PRODUCER_LOG_ROOT:-${H3_WORKSPACE}/dense-d0-v1-96976ce/cache_generation/dual_logs}"

if [[ ! "${EXPECTED_ITEMS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_ITEMS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${PRODUCER_SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "PRODUCER_SHARDS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

MANIFEST="${CANDIDATE_ROOT}/manifest_all.jsonl"
KV_ROOT="${CACHE_ROOT}/${KV_SUBDIR}"
FEATURE_ROOT="${CACHE_ROOT}/${FEATURE_SUBDIR}"
READY_MARKER="${AUDIT_ROOT}/DUAL_CACHE_AUDIT_READY.json"
mkdir -p "${AUDIT_ROOT}"

for required in \
  "${MANIFEST}" \
  "${H3_CHECKPOINT}" \
  "${PROJECT_ROOT}/scripts/h3wam/audit_h3_dreamwam_kv_cache.py" \
  "${PROJECT_ROOT}/scripts/h3wam/audit_h3_starwam_feature_cache.py"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required file: ${required}" >&2
    exit 2
  fi
done

count_pt() {
  find "$1" -maxdepth 1 -type f -name '*.pt' | wc -l
}

stable_complete=0
while (( stable_complete < 2 )); do
  kv_count="$(count_pt "${KV_ROOT}")"
  feature_count="$(count_pt "${FEATURE_ROOT}")"
  temporary_count="$({
    find "${KV_ROOT}" -maxdepth 1 -type f -name '*.tmp'
    find "${FEATURE_ROOT}" -maxdepth 1 -type f -name '*.tmp'
  } | wc -l)"
  producer_errors="$({
    grep -Eil 'traceback|runtimeerror|out of memory|exception' "${PRODUCER_LOG_ROOT}"/shard*.log 2>/dev/null || true
  } | wc -l)"
  printf '{"time":"%s","kv":%s,"starwam":%s,"temporary":%s,"error_logs":%s,"stable_checks":%s}\n' \
    "$(date -Iseconds)" "${kv_count}" "${feature_count}" "${temporary_count}" "${producer_errors}" "${stable_complete}"

  if (( producer_errors > 0 )); then
    echo "producer error detected; refusing cache audit" >&2
    exit 1
  fi
  if (( kv_count > EXPECTED_ITEMS || feature_count > EXPECTED_ITEMS )); then
    echo "cache contains more .pt files than the fixed manifest" >&2
    exit 1
  fi
  if (( kv_count == EXPECTED_ITEMS && feature_count == EXPECTED_ITEMS && temporary_count == 0 )); then
    stable_complete=$((stable_complete + 1))
  else
    stable_complete=0
  fi
  if (( stable_complete < 2 )); then
    sleep "${POLL_SECONDS}"
  fi
done

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/h3wam/audit_h3_dreamwam_kv_cache.py \
  "${MANIFEST}" \
  --cache-root "${CACHE_ROOT}" \
  --kv-subdir "${KV_SUBDIR}" \
  --output "${AUDIT_ROOT}/dreamwam_kv_full.json" \
  --limit "${EXPECTED_ITEMS}" \
  --num-shards "${PRODUCER_SHARDS}" \
  --expected-checkpoint "${H3_CHECKPOINT}"

"${PYTHON_BIN}" scripts/h3wam/audit_h3_starwam_feature_cache.py \
  "${MANIFEST}" \
  --cache-root "${CACHE_ROOT}" \
  --feature-subdir "${FEATURE_SUBDIR}" \
  --output "${AUDIT_ROOT}/starwam_feature_full.json" \
  --producer-num-shards "${PRODUCER_SHARDS}" \
  --expected-checkpoint "${H3_CHECKPOINT}" \
  --expected-checkpoint-sha256 "${H3_CHECKPOINT_SHA256}"

"${PYTHON_BIN}" - "${AUDIT_ROOT}/dreamwam_kv_full.json" \
  "${AUDIT_ROOT}/starwam_feature_full.json" "${READY_MARKER}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

kv_path, feature_path, output_path = map(Path, sys.argv[1:])
kv = json.loads(kv_path.read_text())
feature = json.loads(feature_path.read_text())
if kv.get("valid") is not True or feature.get("valid") is not True:
    raise SystemExit("both full audits must be valid before READY")
payload = {
    "ready": True,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "dreamwam_kv_audit": str(kv_path),
    "dreamwam_kv_aggregate_sha256": kv["aggregate_cache_sha256"],
    "starwam_feature_audit": str(feature_path),
    "starwam_feature_aggregate_sha256": feature["aggregate_cache_sha256"],
    "manifest_sha256": kv["manifest_sha256"],
    "checkpoint_sha256": feature["checkpoint_sha256"],
}
temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(temporary, output_path)
print(json.dumps(payload, sort_keys=True))
PY
