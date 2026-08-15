#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
checkpoint="${workspace}/outputs/c15-d0-grid-adaptation-s1000-v1/checkpoints/d0_grid_h32_s15000.pt"
output_root="${workspace}/eval/c15-grid-closed-loop-gate-v1"

mkdir -p "${output_root}"
[[ ! -e "${output_root}/COMPLETED" ]] || { echo "refusing completed C15 closed-loop gate" >&2; exit 1; }
while [[ ! -f "${checkpoint}" ]]; do
  echo "$(date -Iseconds) WAIT_C15_CHECKPOINT"; sleep 30
done

cd "${project}"
bash scripts/h3wam/run_d_final_spatial_gate.sh "${checkpoint}" \
  >"${output_root}/spatial.log" 2>&1
bash scripts/h3wam/run_d_final_object_regression_gate.sh "${checkpoint}" \
  >"${output_root}/object.log" 2>&1

"${workspace}/runtime/conda-py311/bin/python" - "${workspace}" "${output_root}/COMPLETED" <<'PY'
import json, os, sys
from pathlib import Path
workspace, destination = map(Path, sys.argv[1:])
name = "d0_grid_h32_s15000"
spatial = json.loads((workspace / "eval/d-final-spatial-gate-v1" / f"{name}_summary.json").read_text())
objects = json.loads((workspace / "eval/d-final-object-gate-v1" / f"{name}_summary.json").read_text())
report = {
    "format": "h3-c15-grid-closed-loop-gate-v1",
    "spatial": {"successes": spatial["successes"], "episodes": spatial["episodes"], "fixed_parent": "7/20"},
    "object": {"successes": objects["successes"], "episodes": objects["episodes"], "fixed_parent": "5/8"},
    "promotion_pass": spatial["successes"] > 7 and objects["successes"] >= 4,
    "status": "PASS_PAIRED_GATE" if spatial["successes"] > 7 and objects["successes"] >= 4 else "FAIL_PAIRED_GATE",
}
temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(report, indent=2) + "\n"); os.replace(temporary, destination)
print(json.dumps(report, sort_keys=True))
PY
