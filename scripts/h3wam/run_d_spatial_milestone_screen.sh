#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 1 )); then
  echo "usage: $0 CHECKPOINT" >&2
  exit 2
fi

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
checkpoint="$(realpath "$1")"
checkpoint_name="$(basename "${checkpoint}" .pt)"
rollout_script="${project}/scripts/h3wam/run_dense_d0_milestone_rollout.sh"
trial="${TRIAL_INDEX:-0}"
log_dir="${workspace}/eval/d-spatial-milestone/${checkpoint_name}_trial${trial}"

[[ "${trial}" =~ ^[0-9]+$ ]] || { echo "TRIAL_INDEX must be non-negative" >&2; exit 2; }
test -f "${checkpoint}"
test -x "${rollout_script}"
mkdir -p "${log_dir}"

wave_pids=()
for task in {0..7}; do
  output_root="${workspace}/outputs/eval-d-spatial/${checkpoint_name}_spatial_task${task}_trial${trial}_replan8"
  if [[ -f "${output_root}/results.json" ]]; then
    echo "skip completed ${output_root}"
    continue
  fi
  if [[ -e "${output_root}" ]]; then
    echo "refusing partial pre-existing output ${output_root}" >&2
    exit 1
  fi
  env SUITE=libero_spatial REPLAN_STEPS_OVERRIDE=8 OUTPUT_ROOT="${output_root}" \
    bash "${rollout_script}" H32 "${checkpoint}" "${task}" "${task}" "${trial}" \
    >"${log_dir}/spatial_task${task}_trial${trial}.log" 2>&1 &
  wave_pids+=("$!")
done

failed=0
for pid in "${wave_pids[@]}"; do
  wait "${pid}" || failed=1
done
(( failed == 0 )) || exit 1

"${workspace}/runtime/conda-py311/bin/python" - "${workspace}" "${checkpoint_name}" "${trial}" <<'PY'
import json
import sys
from pathlib import Path

workspace, checkpoint_name, trial = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rows = []
for task in range(8):
    root = workspace / "outputs" / "eval-d-spatial" / (
        f"{checkpoint_name}_spatial_task{task}_trial{trial}_replan8"
    )
    payload = json.loads((root / "results.json").read_text())
    episode = payload["tasks"][0]["episodes"][0]
    rows.append({"task_id": task, "trial": trial, "success": bool(episode["success"]), "steps": int(episode["steps"])})
report = {
    "format": "h3-d-aligned-spatial-milestone-screen-v1",
    "checkpoint": checkpoint_name,
    "tasks": list(range(8)),
    "trial": trial,
    "replan_steps": 8,
    "successes": sum(row["success"] for row in rows),
    "episodes": len(rows),
    "records": rows,
    "evidence_boundary": "Early milestone diagnostic only; not the pre-registered final paired 20-episode promotion gate.",
}
output = workspace / "eval" / "d-spatial-milestone" / f"{checkpoint_name}_trial{trial}_summary.json"
tmp = output.with_name(f".{output.name}.partial")
tmp.write_text(json.dumps(report, indent=2) + "\n")
tmp.replace(output)
print(json.dumps(report, indent=2))
PY
