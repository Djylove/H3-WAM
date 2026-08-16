#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 3 )) || [[ ! "$2" =~ ^[0-9]+$ ]] || [[ ! "$3" =~ ^[0-7]$ ]]; then
  echo "usage: $0 STAGE_ROOT WORKER_ID GPU_INDEX" >&2
  exit 2
fi
stage_root="$(realpath "$1")"
worker_id="$2"
gpu="$3"
total_workers="${TOTAL_WORKERS:-32}"
[[ "${total_workers}" =~ ^[1-9][0-9]*$ ]] || exit 2
(( worker_id < total_workers )) || exit 2

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${workspace}/runtime/conda-py311/bin/python"
manifest="${stage_root}/jobs.jsonl"
[[ -f "${manifest}" && -f "${stage_root}/PREPARED.json" ]] || exit 2
expected_workers="$(${python_bin} - "${stage_root}/PREPARED.json" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text()).get("total_workers", 32))
PY
)"
[[ "${total_workers}" == "${expected_workers}" ]] || {
  echo "TOTAL_WORKERS=${total_workers} does not match frozen ${expected_workers}" >&2
  exit 2
}
mkdir -p "${stage_root}/logs" "${stage_root}/workers"
jobs="$(${python_bin} - "${manifest}" <<'PY'
import sys
from pathlib import Path
print(sum(1 for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()))
PY
)"

completed=0
for ((index=worker_id; index<jobs; index+=total_workers)); do
  IFS=$'\t' read -r job_id arm suite task trial checkpoint output < <(
    "${python_bin}" - "${manifest}" "${index}" <<'PY'
import json, sys
from pathlib import Path
row=json.loads(Path(sys.argv[1]).read_text().splitlines()[int(sys.argv[2])])
print("\t".join(str(row[k]) for k in ("job_id","arm","suite","task","trial","checkpoint","output")))
PY
  )
  [[ "${job_id}" == "${index}" ]] || exit 2
  if [[ -f "${output}/results.json" ]]; then
    completed=$((completed + 1))
    continue
  fi
  [[ ! -e "${output}" ]] || {
    echo "ambiguous partial C55 rollout output: ${output}" >&2
    exit 1
  }
  log="${stage_root}/logs/job$(printf '%04d' "${job_id}")_${arm}_${suite}_task${task}_trial${trial}.log"
  SUITE="${suite}" OUTPUT_ROOT="${output}" REPLAN_STEPS_OVERRIDE=8 \
    bash "${project}/scripts/h3wam/run_dense_d0_milestone_rollout.sh" \
    H32 "${checkpoint}" "${gpu}" "${task}" "${trial}" >"${log}" 2>&1
  [[ -f "${output}/results.json" ]] || exit 1
  completed=$((completed + 1))
done
"${python_bin}" - "${stage_root}/workers/worker$(printf '%02d' "${worker_id}").COMPLETED" "${worker_id}" "${completed}" <<'PY'
import json, os, sys
from pathlib import Path
path=Path(sys.argv[1]); payload={"worker_id":int(sys.argv[2]),"completed_or_reused_jobs":int(sys.argv[3]),"status":"COMPLETED"}
temporary=path.with_name(f".{path.name}.{os.getpid()}.partial"); temporary.write_text(json.dumps(payload,sort_keys=True)+"\n"); os.replace(temporary,path)
PY
