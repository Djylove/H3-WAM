#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
result_root="${workspace}/outputs/eval-dense-d0-long"
output_root="${C20_OUTPUT_ROOT:-${workspace}/eval/c20-libero-branch-repeatability-v1}"
python_bin="${workspace}/runtime/conda-py311/bin/python"
test ! -e "${output_root}"
mkdir -p "${output_root}/logs"
cases=(
  "libero_goal goal 0 0" "libero_object object 1 3"
  "libero_spatial spatial 0 0" "libero_10 10 0 0"
)
for item in "${cases[@]}"; do
  read -r suite slug task trial <<< "${item}"
  trajectory="${result_root}/d0_h32_s14000_${slug}_task${task}_trial${trial}_replan8/task$(printf '%02d' "${task}")_trial$(printf '%02d' "${trial}")_trajectory.npz"
  env H3_WORKSPACE="${workspace}" PROJECT_ROOT="${project}" \
    PYTHON_BIN="${python_bin}" SIM_SITE_PACKAGES="/tmp/h3-wam-libero-site" \
    bash "${project}/scripts/h3wam/run_cloud_libero.sh" \
    "${python_bin}" "${project}/scripts/h3wam/audit_libero_branch_repeatability.py" \
    --trajectory "${trajectory}" --suite "${suite}" --task-id "${task}" \
    --execution-horizon 8 --output "${output_root}/${slug}.json" \
    >"${output_root}/logs/${slug}.log" 2>&1
done

"${python_bin}" - "${output_root}" <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
items = [json.loads(path.read_text()) for path in sorted(root.glob("*.json"))]
passed = all(item["status"] == "PASS_PAIRED_BRANCH_REPEATABILITY_GATE" for item in items)
report = {
    "format": "h3wam-c20-multisuite-branch-repeatability-gate-v1",
    "cases": len(items), "states": sum(len(item["rows"]) for item in items),
    "status": "PASS_MULTISUITE_BRANCH_REPEATABILITY_GATE" if passed else "FAIL_MULTISUITE_BRANCH_REPEATABILITY_GATE",
    "items": items,
}
temporary = root / f".COMPLETED.{os.getpid()}.partial"
temporary.write_text(json.dumps(report, indent=2) + "\n")
os.replace(temporary, root / "COMPLETED")
print(json.dumps(report, indent=2))
PY
