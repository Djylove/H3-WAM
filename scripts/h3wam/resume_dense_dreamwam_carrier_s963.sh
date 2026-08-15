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
OUTPUT_ROOT="${OUTPUT_ROOT:-${H3_WORKSPACE}/outputs/dense-carrier-tournament-v1/${ARM}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

previous="${OUTPUT_ROOT}/checkpoints/${ARM}_s500.pt"
checkpoint="${OUTPUT_ROOT}/checkpoints/${ARM}_s963.pt"
report="${OUTPUT_ROOT}/reports/${ARM}_s963_train.json"
restore_report="${OUTPUT_ROOT}/reports/${ARM}_s963_restore.json"
log="${OUTPUT_ROOT}/logs/${ARM}_s963.log"
for required in \
  "${OUTPUT_ROOT}/STARTED" \
  "${OUTPUT_ROOT}/reports/${ARM}_s500_restore.json" \
  "${previous}" \
  "${CANDIDATE_ROOT}/manifest_all.jsonl" \
  "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl"; do
  [[ -f "${required}" ]] || { echo "missing required resume input: ${required}" >&2; exit 2; }
done
for forbidden in "${checkpoint}" "${report}" "${restore_report}" "${OUTPUT_ROOT}/COMPLETED"; do
  [[ ! -e "${forbidden}" ]] || { echo "refusing to overwrite resume output: ${forbidden}" >&2; exit 1; }
done

"${PYTHON_BIN}" - "${ARM}" "${OUTPUT_ROOT}/reports/${ARM}_s500_restore.json" <<'PY'
import json
import sys
from pathlib import Path

arm, raw_report = sys.argv[1:]
report = json.loads(Path(raw_report).read_text())
expected_candidate = arm.upper()
if report.get("candidate") != expected_candidate:
    raise SystemExit("s500 restore report has the wrong candidate")
if report.get("completed_steps") != 500:
    raise SystemExit("s500 restore report does not restore global step 500")
if report.get("restore_probe_max_abs") != 0.0:
    raise SystemExit("s500 restore probe is not bitwise exact")
PY

PYTORCH_CU13_LIB="$("${PYTHON_BIN}" - <<'PY'
import sysconfig
from pathlib import Path

path = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib"
if not (path / "libnvJitLink.so.13").is_file():
    raise SystemExit(f"missing PyTorch-bundled cu13 runtime: {path}")
print(path)
PY
)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${PYTORCH_CU13_LIB}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TMPDIR="${H3_WORKSPACE}/tmp/dense-carrier-${ARM}-resume-s963"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${TMPDIR}"
cd "${PROJECT_ROOT}"

arm_args=()
[[ "${ARM}" != "d0" ]] || arm_args+=(--enable-d0-repeat-layer49)
common_args=(
  "${CANDIDATE_ROOT}/manifest_train_uniform.jsonl"
  --source-manifest "${CANDIDATE_ROOT}/manifest_all.jsonl"
  --cache-root "${CACHE_ROOT}" --kv-subdir "${KV_SUBDIR}"
  --enable-dreamwam-kv-carrier --verify-h3-checkpoint-sha256
  --per-device-batch-size 1 --gradient-accumulation-steps 1 --num-workers 0
  --learning-rate 1e-4 --weight-decay 0.01 --warmup-steps 1000
  --scheduler-horizon 21700 --min-learning-rate 1e-6
  --action-shift 5 --seed 42
)

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node "${NPROC_PER_NODE}" \
  scripts/h3wam/train_h3_int8_dreamwam_kv_carrier.py \
  "${common_args[@]}" "${arm_args[@]}" --load-checkpoint "${previous}" \
  --steps 463 --sample-offset 4000 --limit 3704 \
  --save-checkpoint "${checkpoint}" --output "${report}" 2>&1 | tee "${log}"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node "${NPROC_PER_NODE}" \
  scripts/h3wam/train_h3_int8_dreamwam_kv_carrier.py \
  "${common_args[@]}" "${arm_args[@]}" --load-checkpoint "${checkpoint}" \
  --restore-check-only --steps 1 --sample-offset 0 --limit 1 \
  --output "${restore_report}" >> "${log}" 2>&1

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
    "resume_source": str(root / "checkpoints" / f"{arm}_s500.pt"),
}
output = root / "COMPLETED"
temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(temporary, output)
print(json.dumps(payload, sort_keys=True))
PY
