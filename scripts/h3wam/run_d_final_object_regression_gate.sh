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
log_dir="${workspace}/eval/d-final-object-gate-v1/${checkpoint_name}"
mkdir -p "${log_dir}"
test -f "${checkpoint}"
test -x "${rollout_script}"

pids=()
gpu=0
for trial in 0 1; do
  for task in 0 1 5 9; do
    output_root="${workspace}/outputs/eval-d-object/${checkpoint_name}_object_task${task}_trial${trial}_replan8"
    [[ ! -e "${output_root}" ]] || { echo "refusing pre-existing output ${output_root}" >&2; exit 1; }
    env SUITE=libero_object REPLAN_STEPS_OVERRIDE=8 OUTPUT_ROOT="${output_root}" \
      bash "${rollout_script}" H32 "${checkpoint}" "${gpu}" "${task}" "${trial}" \
      >"${log_dir}/object_task${task}_trial${trial}.log" 2>&1 &
    pids+=("$!"); gpu=$((gpu + 1))
  done
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || exit 1

"${workspace}/runtime/conda-py311/bin/python" - "${workspace}" "${checkpoint_name}" <<'PY'
import json, sys
from pathlib import Path
workspace, name = Path(sys.argv[1]), sys.argv[2]
rows = []
for trial in (0, 1):
    for task in (0, 1, 5, 9):
        root = workspace / "outputs/eval-d-object" / f"{name}_object_task{task}_trial{trial}_replan8"
        episode = json.loads((root / "results.json").read_text())["tasks"][0]["episodes"][0]
        rows.append({"task_id": task, "trial": trial, "success": bool(episode["success"]), "steps": int(episode["steps"]), "result": str(root / "results.json")})
report = {
    "format": "h3-d-final-object-regression-gate-v1", "checkpoint": name,
    "tasks": [0, 1, 5, 9], "trials": [0, 1], "replan_steps": 8,
    "successes": sum(row["success"] for row in rows), "episodes": len(rows),
    "fixed_parent": "5/8", "promotion": "candidate must retain at least 4/8",
    "records": rows,
}
output = workspace / "eval/d-final-object-gate-v1" / f"{name}_summary.json"
tmp = output.with_name(f".{output.name}.partial")
tmp.write_text(json.dumps(report, indent=2) + "\n"); tmp.replace(output)
print(json.dumps(report, indent=2))
PY
