#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 1 )) || [[ "$1" != "h32_resume" && "$1" != "h8_fresh" && "$1" != "d_h32_resume" ]]; then
  echo "usage: $0 h32_resume|h8_fresh|d_h32_resume" >&2
  exit 2
fi
MODE="$1"

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/candidate-d0-rollout-96976ce/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/h3-int8-native/bin/python}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-${H3_WORKSPACE}/data/v7_multisuite_dense_candidate}"
CACHE_ROOT="${CACHE_ROOT:-${H3_WORKSPACE}/data/v7_dense_h3_cache}"
KV_SUBDIR="${KV_SUBDIR:-h3_int8_dreamwam_kv_5x32_dense_v1}"
READY_MARKER="${READY_MARKER:-${H3_WORKSPACE}/dense-d0-v1-96976ce/cache_generation/full_audit/DUAL_CACHE_AUDIT_READY.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RESUME_FROM_MILESTONE="${RESUME_FROM_MILESTONE:-0}"
[[ "${RESUME_FROM_MILESTONE}" =~ ^[0-9]+$ ]] || {
  echo "RESUME_FROM_MILESTONE must be a non-negative integer" >&2
  exit 2
}
SELECTED80_SHA256="26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42"
PARENT_S963_SHA256="0a72d829a70f6b408f9aedeeae4dac6734e1c67cc29bf3fca85a8fd1f5234cc5"
PARENT_D_S963_SHA256="b9084e294e86e63756b6a13b99dac1bf817f9e2e9513026dcf03b382b79bad25"

if [[ "${MODE}" == "h32_resume" ]]; then
  ACTION_HORIZON=32
  ARM=d0
  FINAL_MILESTONE=20000
  OUTPUT_ROOT="${OUTPUT_ROOT:-${H3_WORKSPACE}/outputs/dense-carrier-d0-h32-s20000-v1}"
  previous_checkpoint="${H3_WORKSPACE}/outputs/dense-carrier-tournament-v1/d0/checkpoints/d0_s963.pt"
  previous_milestone=963
  expected_parent_sha256="${PARENT_S963_SHA256}"
elif [[ "${MODE}" == "d_h32_resume" ]]; then
  ACTION_HORIZON=32
  ARM=d
  FINAL_MILESTONE=14000
  OUTPUT_ROOT="${OUTPUT_ROOT:-${H3_WORKSPACE}/outputs/dense-carrier-d-h32-s14000-v1}"
  previous_checkpoint="${H3_WORKSPACE}/outputs/dense-carrier-tournament-v1/d/checkpoints/d_s963.pt"
  previous_milestone=963
  expected_parent_sha256="${PARENT_D_S963_SHA256}"
else
  ACTION_HORIZON=8
  ARM=d0
  FINAL_MILESTONE=20000
  OUTPUT_ROOT="${OUTPUT_ROOT:-${H3_WORKSPACE}/outputs/dense-carrier-d0-h8-s20000-v1}"
  previous_checkpoint=""
  previous_milestone=0
  expected_parent_sha256=""
fi

for required in \
  "${READY_MARKER}" \
  "${CANDIDATE_ROOT}/manifest_all.jsonl" \
  "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
  "${CANDIDATE_ROOT}/manifest_val.jsonl" \
  "${PROJECT_ROOT}/scripts/h3wam/train_h3_int8_dreamwam_kv_carrier.py" \
  "${PROJECT_ROOT}/scripts/h3wam/evaluate_h3_dreamwam_kv_carrier.py"; do
  [[ -f "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 2; }
done
[[ $(wc -l < "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl") -eq 200779 ]] || {
  echo "train manifest row count changed" >&2
  exit 2
}
if [[ -n "${previous_checkpoint}" ]]; then
  [[ -f "${previous_checkpoint}" ]] || { echo "missing parent: ${previous_checkpoint}" >&2; exit 2; }
  actual_parent_sha256="$(sha256sum "${previous_checkpoint}" | awk '{print $1}')"
  [[ "${actual_parent_sha256}" == "${expected_parent_sha256}" ]] || {
    echo "parent s963 identity mismatch: ${actual_parent_sha256}" >&2
    exit 2
  }
fi
if (( RESUME_FROM_MILESTONE > 0 )); then
  [[ -d "${OUTPUT_ROOT}" ]] || { echo "resume output root is missing" >&2; exit 2; }
  previous_checkpoint="${OUTPUT_ROOT}/checkpoints/${ARM}_h${ACTION_HORIZON}_s${RESUME_FROM_MILESTONE}.pt"
  [[ -f "${previous_checkpoint}" ]] || { echo "resume checkpoint is missing" >&2; exit 2; }
  previous_milestone="${RESUME_FROM_MILESTONE}"
  printf '%s milestone=%s\n' "$(date -Iseconds)" "${RESUME_FROM_MILESTONE}" >> "${OUTPUT_ROOT}/RESUMED"
else
  [[ ! -e "${OUTPUT_ROOT}" ]] || { echo "refusing to reuse output root: ${OUTPUT_ROOT}" >&2; exit 1; }
  mkdir -p "${OUTPUT_ROOT}/checkpoints" "${OUTPUT_ROOT}/reports" \
    "${OUTPUT_ROOT}/evaluations" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/evaluation_logs"
  printf '%s\n' "$(date -Iseconds)" > "${OUTPUT_ROOT}/STARTED"
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTORCH_CU13_LIB="$("${PYTHON_BIN}" - <<'PY'
import sysconfig
from pathlib import Path

path = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib"
if not (path / "libnvJitLink.so.13").is_file():
    raise SystemExit(f"missing PyTorch-bundled cu13 runtime: {path}")
print(path)
PY
)"
export LD_LIBRARY_PATH="${PYTORCH_CU13_LIB}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${H3_WORKSPACE}/tmp/dense-d0-${MODE}-s20000"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${TMPDIR}"
cd "${PROJECT_ROOT}"

audit_hash="$("${PYTHON_BIN}" - "${READY_MARKER}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("ready") is not True:
    raise SystemExit("cache audit marker is not READY")
if payload.get("manifest_sha256") != "cab8876f067114dce41d16ca52cb0bafddf17da33c92d0adde5f11d7ac9555b9":
    raise SystemExit("cache audit manifest identity changed")
print(payload["dreamwam_kv_aggregate_sha256"])
PY
)"

common_args=(
  "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl"
  --source-manifest "${CANDIDATE_ROOT}/manifest_all.jsonl"
  --cache-root "${CACHE_ROOT}" --kv-subdir "${KV_SUBDIR}"
  --enable-dreamwam-kv-carrier
  --verify-h3-checkpoint-sha256
  --action-horizon "${ACTION_HORIZON}"
  --per-device-batch-size 1 --gradient-accumulation-steps 1 --num-workers 0
  --learning-rate 1e-4 --weight-decay 0.01 --warmup-steps 1000
  --scheduler-horizon 21700 --min-learning-rate 1e-6
  --action-shift 5 --seed 42
)
if [[ "${ARM}" == "d0" ]]; then
  common_args+=(--enable-d0-repeat-layer49)
fi

for milestone in $(seq 1000 1000 "${FINAL_MILESTONE}"); do
  if (( milestone <= previous_milestone )); then
    continue
  fi
  stage_steps=$((milestone - previous_milestone))
  sample_offset=$((previous_milestone * 8))
  sample_limit=$((stage_steps * 8))
  checkpoint="${OUTPUT_ROOT}/checkpoints/${ARM}_h${ACTION_HORIZON}_s${milestone}.pt"
  report="${OUTPUT_ROOT}/reports/${ARM}_h${ACTION_HORIZON}_s${milestone}_train.json"
  restore_report="${OUTPUT_ROOT}/reports/${ARM}_h${ACTION_HORIZON}_s${milestone}_restore.json"
  evaluation="${OUTPUT_ROOT}/evaluations/${ARM}_h${ACTION_HORIZON}_s${milestone}_balanced80.json"
  train_log="${OUTPUT_ROOT}/logs/${ARM}_h${ACTION_HORIZON}_s${milestone}.log"
  evaluation_log="${OUTPUT_ROOT}/evaluation_logs/${ARM}_h${ACTION_HORIZON}_s${milestone}_balanced80.log"
  load_args=()
  if [[ -n "${previous_checkpoint}" ]]; then
    load_args+=(--load-checkpoint "${previous_checkpoint}")
  fi

  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node "${NPROC_PER_NODE}" \
    scripts/h3wam/train_h3_int8_dreamwam_kv_carrier.py \
    "${common_args[@]}" "${load_args[@]}" \
    --steps "${stage_steps}" --sample-offset "${sample_offset}" --limit "${sample_limit}" \
    --save-checkpoint "${checkpoint}" --output "${report}" 2>&1 | tee "${train_log}"

  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node "${NPROC_PER_NODE}" \
    scripts/h3wam/train_h3_int8_dreamwam_kv_carrier.py \
    "${common_args[@]}" --load-checkpoint "${checkpoint}" --restore-check-only \
    --steps 1 --sample-offset 0 --limit 1 --output "${restore_report}" \
    >> "${train_log}" 2>&1

  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" scripts/h3wam/evaluate_h3_dreamwam_kv_carrier.py \
    "${checkpoint}" \
    --source-manifest "${CANDIDATE_ROOT}/manifest_all.jsonl" \
    --train-manifest "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl" \
    --val-manifest "${CANDIDATE_ROOT}/manifest_val.jsonl" \
    --cache-root "${CACHE_ROOT}" --kv-subdir "${KV_SUBDIR}" \
    --output "${evaluation}" --device cuda --num-workers 0 \
    --cache-audit-aggregate-sha256 "${audit_hash}" \
    --expected-selected-ids-sha256 "${SELECTED80_SHA256}" \
    > "${evaluation_log}" 2>&1

  previous_checkpoint="${checkpoint}"
  previous_milestone="${milestone}"
done

"${PYTHON_BIN}" - "${MODE}" "${ARM}" "${ACTION_HORIZON}" "${FINAL_MILESTONE}" "${OUTPUT_ROOT}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

mode, arm, horizon, final_milestone, raw_root = sys.argv[1:]
root = Path(raw_root)
output = root / "COMPLETED"
payload = {
    "completed": True,
    "mode": mode,
    "action_horizon": int(horizon),
    "arm": arm,
    "completed_steps": int(final_milestone),
    "training_samples": int(final_milestone) * 8,
    "effective_epochs": int(final_milestone) * 8 / 200779,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "final_checkpoint": str(root / "checkpoints" / f"{arm}_h{horizon}_s{final_milestone}.pt"),
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(temporary, output)
print(json.dumps(payload, sort_keys=True))
PY
