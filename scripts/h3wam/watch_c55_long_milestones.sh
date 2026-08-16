#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 2 )) || [[ "$1" != "action_only" && "$1" != "joint_aux" ]] || [[ ! "$2" =~ ^[0-7]$ ]]; then
  echo "usage: $0 action_only|joint_aux GPU_INDEX" >&2
  exit 2
fi
arm="$1"
gpu="$2"

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
train_root="${workspace}/outputs/c55-fact-joint-action-long-v2/${arm}"
eval_root="${workspace}/outputs/c55-fact-joint-action-long-v2/evaluations/${arm}"
temporary_root="${workspace}/tmp/c55-long-milestone-exports-v2"
parent="${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"
audit_sha="0ffd6c99a7ceb7269221afb31ee51a453da19ebe2676a421ffc1939c35a0cbc6"
selection_sha="26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42"
mkdir -p "${eval_root}/logs" "${temporary_root}"

export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${project}/src:${project}${PYTHONPATH:+:${PYTHONPATH}}"
pytorch_cu13_lib="$("${python_bin}" - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib")
PY
)"
export LD_LIBRARY_PATH="${pytorch_cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${project}"

for step in 1000 2000 3000 4000 5000 6000; do
  source_checkpoint="${train_root}/checkpoints/step${step}.pt"
  source_report="${train_root}/reports/step${step}.json"
  output="${eval_root}/step${step}.balanced80.json"
  export_report="${eval_root}/step${step}.export.json"
  log="${eval_root}/logs/step${step}.log"
  while [[ ! -f "${source_checkpoint}" || ! -f "${source_report}" ]]; do
    sleep 30
  done
  if [[ -f "${output}" && -f "${export_report}" ]]; then
    continue
  fi
  if [[ -e "${output}" || -e "${export_report}" ]]; then
    echo "incomplete C55 evaluation pair at step ${step}" >&2
    exit 1
  fi
  exported="${temporary_root}/${arm}-step${step}.d0.pt"
  [[ ! -e "${exported}" ]] || { echo "stale C55 temporary export: ${exported}" >&2; exit 1; }
  "${python_bin}" scripts/h3wam/export_c55_deployment_checkpoint.py \
    --c55-checkpoint "${source_checkpoint}" --parent-checkpoint "${parent}" \
    --output "${exported}" --report "${export_report}"
  "${python_bin}" scripts/h3wam/evaluate_h3_dreamwam_kv_carrier.py \
    "${exported}" \
    --source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
    --train-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl" \
    --val-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_val.jsonl" \
    --cache-root "${workspace}/data/v7_dense_h3_cache" \
    --kv-subdir h3_int8_dreamwam_kv_5x32_dense_v1 \
    --output "${output}" --device cuda --num-workers 0 \
    --cache-audit-aggregate-sha256 "${audit_sha}" \
    --expected-selected-ids-sha256 "${selection_sha}" >"${log}" 2>&1
  rm -f -- "${exported}"
done

"${python_bin}" - "${eval_root}/EVALUATION_COMPLETED" "${arm}" <<'PY'
import json
import os
import sys
from pathlib import Path
output = Path(sys.argv[1])
payload = {"arm": sys.argv[2], "milestones": [1000, 2000, 3000, 4000, 5000, 6000], "status": "OFFLINE_EVALUATED_NOT_CLOSED_LOOP"}
temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
os.replace(temporary, output)
PY
