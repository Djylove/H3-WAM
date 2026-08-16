#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 1 )) || [[ ! "$1" =~ ^[0-7]$ ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi
gpu="$1"
workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
train_root="${workspace}/outputs/c55-fact-joint-action-long-v2/joint_aux"
output_root="${workspace}/outputs/c55-fact-joint-action-long-v2/mechanism"
mkdir -p "${output_root}"

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
  checkpoint="${train_root}/checkpoints/step${step}.pt"
  report="${train_root}/reports/step${step}.json"
  output="${output_root}/step${step}.json"
  log="${output_root}/step${step}.log"
  while [[ ! -f "${checkpoint}" || ! -f "${report}" ]]; do
    sleep 30
  done
  [[ ! -e "${output}" ]] || continue
  "${python_bin}" scripts/h3wam/evaluate_c55_joint_mechanism.py "${checkpoint}" \
    --rollout-dataset "${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt" \
    --rollout-projected-features "${workspace}/eval/c49-dense-value-h3-features-v1/projected_features.pt" \
    --rollout-kv-root "${workspace}/eval/c55-fact-joint-action-v1/kv-full-v1" \
    --demo-cache-root "${workspace}/data/v7_dense_h3_cache" \
    --per-outcome 128 --output "${output}" >"${log}" 2>&1
done

"${python_bin}" - "${output_root}/EVALUATION_COMPLETED" <<'PY'
import json
import os
import sys
from pathlib import Path
output = Path(sys.argv[1])
temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps({"milestones": [1000, 2000, 3000, 4000, 5000, 6000], "status": "MECHANISM_EVALUATED_NOT_CLOSED_LOOP"}, sort_keys=True) + "\n")
os.replace(temporary, output)
PY
