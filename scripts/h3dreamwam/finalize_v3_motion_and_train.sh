#!/usr/bin/env bash
set -euo pipefail

M3_ROOT=${M3_ROOT:-/home/h3wam_finetune}
M3_MOTION_PID=${M3_MOTION_PID:?set M3_MOTION_PID to the active torchrun launcher}
M3_PROJECT=${M3_ROOT}/project
M3_CANDIDATE=${M3_ROOT}/data/v4_multisuite_uniform_candidate
M3_CACHE=${M3_ROOT}/data/v3_multisuite_cache
M3_MOTION=${M3_ROOT}/data/v3_motion_multisuite
M3_AUDIT=${M3_CANDIDATE}/full_audit.json

while kill -0 "${M3_MOTION_PID}" 2>/dev/null; do
  sleep 30
done

M3_EXPECTED=$(wc -l < "${M3_CANDIDATE}/manifest_all.jsonl")
M3_AVAILABLE=$(find "${M3_MOTION}" -maxdepth 1 -type f -name '*.pt' | wc -l)
if [[ "${M3_AVAILABLE}" -ne "${M3_EXPECTED}" ]]; then
  echo "motion precompute exited incomplete: ${M3_AVAILABLE}/${M3_EXPECTED}" >&2
  exit 3
fi

cd "${M3_PROJECT}"
M3_AUDIT_PARTIAL=${M3_AUDIT}.partial
"${M3_ROOT}/.venv/bin/python" \
  scripts/h3dreamwam/audit_multisuite_training_candidate.py \
  --candidate "${M3_CANDIDATE}" \
  --cache-root "${M3_CACHE}" \
  --motion-root "${M3_MOTION}" \
  --require-complete-motion \
  --full-cache-audit >"${M3_AUDIT_PARTIAL}"
mv "${M3_AUDIT_PARTIAL}" "${M3_AUDIT}"

scripts/h3dreamwam/run_v4_uniform_head_stage.sh

M3_HEAD_REPORT=${M3_ROOT}/outputs/h3dreamwam_m3/multisuite_uniform_head_epoch1.json
M3_HEAD_CHECKPOINT=${M3_ROOT}/outputs/h3dreamwam_m3/multisuite_uniform_head_epoch1.pt
"${M3_ROOT}/.venv/bin/python" - "${M3_HEAD_REPORT}" <<'PY'
import json
import math
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
history = report["history"]
if len(history) != 964:
    raise SystemExit(f"uniform head stage has incomplete history: {len(history)}/964")
losses = [float(row["action_loss"]) for row in history]
if not all(math.isfinite(value) for value in losses):
    raise SystemExit("uniform head stage contains non-finite action loss")
span = 50
first = sum(losses[:span]) / span
last = sum(losses[-span:]) / span
print(json.dumps({"event": "uniform_head_gate", "first_50": first, "last_50": last}))
if last >= first:
    raise SystemExit("uniform head stage did not improve; joint training blocked")
PY

exec env \
  M3_LOAD_STAGE="${M3_HEAD_CHECKPOINT}" \
  M3_TAG=multisuite_uniform_joint100 \
  scripts/h3dreamwam/run_v3_multisuite_stage.sh
