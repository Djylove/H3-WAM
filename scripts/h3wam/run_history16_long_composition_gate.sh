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
log_dir="${workspace}/eval/history16-long-composition-gate-v1/${checkpoint_name}${summary_tag:+_${summary_tag}}"

test -f "${checkpoint}"
test -x "${rollout_script}"
mkdir -p "${log_dir}"

# Pre-registered long-composition slice.  The unchanged D0-s14000 parent has
# existing results for the same tasks, trials, replan horizon, and seed.
tasks=(0 3 7 9)

wait_wave() {
  local failed=0
  for pid in "${wave_pids[@]}"; do
    wait "${pid}" || failed=1
  done
  return "${failed}"
}

wave_pids=()
gpu=0
for trial in "${trials[@]}"; do
  for task in "${tasks[@]}"; do
    output_root="${workspace}/outputs/eval-history16/${checkpoint_name}_10_task${task}_trial${trial}_replan8"
    if [[ -f "${output_root}/results.json" ]]; then
      echo "skip completed ${output_root}"
      gpu=$((gpu + 1))
      continue
    fi
    if [[ -e "${output_root}" ]]; then
      echo "refusing partial pre-existing output ${output_root}" >&2
      exit 1
    fi
    env SUITE=libero_10 REPLAN_STEPS_OVERRIDE=8 OUTPUT_ROOT="${output_root}" \
      bash "${rollout_script}" H32 "${checkpoint}" "${gpu}" "${task}" "${trial}" \
      >"${log_dir}/10_task${task}_trial${trial}.log" 2>&1 &
    wave_pids+=("$!")
    gpu=$((gpu + 1))
  done
done

wait_wave

"${workspace}/runtime/conda-py311/bin/python" - "${workspace}" "${checkpoint_name}" "${trial_spec}" "${summary_tag}" <<'PY'
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
checkpoint_name = sys.argv[2]
trials = [int(value) for value in sys.argv[3].split()]
summary_tag = sys.argv[4]
rows = []
for trial in trials:
    for task in (0, 3, 7, 9):
        root = workspace / "outputs" / "eval-history16" / (
            f"{checkpoint_name}_10_task{task}_trial{trial}_replan8"
        )
        payload = json.loads((root / "results.json").read_text())
        result = payload["tasks"][0]["episodes"][0]
        rows.append(
            {
                "task_id": task,
                "trial": trial,
                "success": bool(result["success"]),
                "steps": int(result["steps"]),
                "result": str(root / "results.json"),
            }
        )
summary = {
    "format": "h3-history16-long-composition-rollout-gate-v1",
    "checkpoint": checkpoint_name,
    "suite": "libero_10",
    "tasks": [0, 3, 7, 9],
    "trials": trials,
    "replan_steps": 8,
    "successes": sum(row["success"] for row in rows),
    "episodes": len(rows),
    "records": rows,
    "evidence_boundary": "Candidate-only results; compare against the fixed D0-s14000 parent artifacts before promotion.",
}
suffix = f"_{summary_tag}" if summary_tag else ""
output = workspace / "eval" / "history16-long-composition-gate-v1" / f"{checkpoint_name}{suffix}_summary.json"
tmp = output.with_name(".summary.json.partial")
tmp.write_text(json.dumps(summary, indent=2) + "\n")
tmp.replace(output)
print(json.dumps(summary, indent=2))
PY
