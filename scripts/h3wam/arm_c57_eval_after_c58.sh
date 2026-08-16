#!/usr/bin/env bash
set -euo pipefail

# Arm, but do not occupy, the n2 C57 queue.  Completion of C58 s10000 is only
# the release condition; run_c57_heldout_eval_queue.sh still proves the target
# GPU idle twice before every individual restore/evaluation.
C57_C58_FINAL_CHECKPOINT=${C57_C58_FINAL_CHECKPOINT:?set C57_C58_FINAL_CHECKPOINT}
C57_C58_FINAL_REPORT=${C57_C58_FINAL_REPORT:?set C57_C58_FINAL_REPORT}
C57_EVAL_PYTHON=${C57_EVAL_PYTHON:?set C57_EVAL_PYTHON}
C57_EVAL_ARM_POLL_SECONDS=${C57_EVAL_ARM_POLL_SECONDS:-30}

if ! [[ "${C57_EVAL_ARM_POLL_SECONDS}" =~ ^[0-9]+$ ]] || \
   [[ "${C57_EVAL_ARM_POLL_SECONDS}" -le 0 ]]; then
  echo "C57_EVAL_ARM_POLL_SECONDS must be positive" >&2
  exit 2
fi

while [[ ! -s "${C57_C58_FINAL_CHECKPOINT}" || ! -s "${C57_C58_FINAL_REPORT}" ]]; do
  sleep "${C57_EVAL_ARM_POLL_SECONDS}"
done

# Do not trust file presence alone: the segmented C58 launcher may still be
# finalizing or may have failed before publishing a PASS report.
"${C57_EVAL_PYTHON}" -c '
import json, pathlib, sys
report_path, checkpoint_path = map(pathlib.Path, sys.argv[1:])
report = json.load(report_path.open())
expected_status = "mechanical_probe_not_effectiveness_evidence"
checks = {
    "status": report.get("status") == expected_status,
    "completed_steps": int(report.get("completed_steps", -1)) == 10000,
    "restore_probe": float(report.get("restore_probe_max_abs", -1.0)) == 0.0,
    "checkpoint_identity": pathlib.Path(report.get("saved_checkpoint", "")).resolve()
        == checkpoint_path.resolve(),
    "checkpoint_size": int(report.get("checkpoint_file_size_bytes", 0))
        == checkpoint_path.stat().st_size,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("C58 s10000 release gate failed: " + ",".join(failed))
' "${C57_C58_FINAL_REPORT}" "${C57_C58_FINAL_CHECKPOINT}"

exec scripts/h3wam/run_c57_heldout_eval_queue.sh
