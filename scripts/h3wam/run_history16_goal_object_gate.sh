#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
checkpoint="${CHECKPOINT:-${workspace}/outputs/d0-history16-adapter-s3000-v1/checkpoints/d0_history16_s17000.pt}"
rollout_script="${project}/scripts/h3wam/run_dense_d0_milestone_rollout.sh"
checkpoint_name="$(basename "${checkpoint}" .pt)"
trial_spec="${TRIALS:-0 1}"
summary_tag="${SUMMARY_TAG:-}"
read -r -a trials <<<"${trial_spec}"
log_dir="${workspace}/eval/history16-goal-object-gate-v1/${checkpoint_name}${summary_tag:+_${summary_tag}}"

test -f "${checkpoint}"
test -x "${rollout_script}"
mkdir -p "${log_dir}"

jobs=()
for trial in "${trials[@]}"; do
  jobs+=("libero_goal:3:${trial}")
  for task in 0 1 5 9; do
    jobs+=("libero_object:${task}:${trial}")
  done
done

wait_wave() {
  local failed=0
  for pid in "${wave_pids[@]}"; do
    wait "${pid}" || failed=1
  done
  wave_pids=()
  return "${failed}"
}

wave_pids=()
gpu=0
for job in "${jobs[@]}"; do
  IFS=: read -r suite task trial <<<"${job}"
  suite_slug="${suite#libero_}"
  output_root="${workspace}/outputs/eval-history16/${checkpoint_name}_${suite_slug}_task${task}_trial${trial}_replan8"
  if [[ -f "${output_root}/results.json" ]]; then
    echo "skip completed ${output_root}"
  else
    if [[ -e "${output_root}" ]]; then
      echo "refusing partial pre-existing output ${output_root}" >&2
      exit 1
    fi
    env SUITE="${suite}" REPLAN_STEPS_OVERRIDE=8 OUTPUT_ROOT="${output_root}" \
      bash "${rollout_script}" H32 "${checkpoint}" "${gpu}" "${task}" "${trial}" \
      >"${log_dir}/${suite_slug}_task${task}_trial${trial}.log" 2>&1 &
    wave_pids+=("$!")
  fi
  gpu=$((gpu + 1))
  if (( gpu == 8 )); then
    wait_wave
    gpu=0
  fi
done
if (( ${#wave_pids[@]} > 0 )); then
  wait_wave
fi

"${workspace}/runtime/conda-py311/bin/python" - "${workspace}" "${checkpoint_name}" "${trial_spec}" "${summary_tag}" <<'PY'
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
checkpoint_name = sys.argv[2]
trials = [int(value) for value in sys.argv[3].split()]
summary_tag = sys.argv[4]
jobs = [
    (suite, task, trial)
    for trial in trials
    for suite, task in (("goal", 3), ("object", 0), ("object", 1), ("object", 5), ("object", 9))
]
rows = []
for suite, task, trial in jobs:
    root = workspace / "outputs" / "eval-history16" / (
        f"{checkpoint_name}_{suite}_task{task}_trial{trial}_replan8"
    )
    payload = json.loads((root / "results.json").read_text())
    result = payload["tasks"][0]["episodes"][0]
    rows.append(
        {
            "suite": suite,
            "task_id": task,
            "trial": trial,
            "success": bool(result["success"]),
            "steps": int(result["steps"]),
            "result": str(root / "results.json"),
        }
    )
summary = {
    "format": "h3-history16-goal-object-rollout-gate-v1",
    "checkpoint": checkpoint_name,
    "trials": trials,
    "replan_steps": 8,
    "successes": sum(row["success"] for row in rows),
    "episodes": len(rows),
    "by_suite": {
        suite: {
            "successes": sum(row["success"] for row in rows if row["suite"] == suite),
            "episodes": sum(1 for row in rows if row["suite"] == suite),
        }
        for suite in ("goal", "object")
    },
    "records": rows,
    "fixed_parent": (
        {"goal": "0/2", "object": "5/8"}
        if trials == [0, 1]
        else "Read from the frozen parent 160-episode artifact for these exact trials."
    ),
    "evidence_boundary": "Paired candidate slice; promotion also consumes the separate LIBERO-10 long-composition gate.",
}
suffix = f"_{summary_tag}" if summary_tag else ""
output = workspace / "eval" / "history16-goal-object-gate-v1" / f"{checkpoint_name}{suffix}_summary.json"
tmp = output.with_name(".summary.json.partial")
tmp.write_text(json.dumps(summary, indent=2) + "\n")
tmp.replace(output)
print(json.dumps(summary, indent=2))
PY
