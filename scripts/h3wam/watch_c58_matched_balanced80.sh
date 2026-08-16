#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )) || [[ ! "$1" =~ ^[0-7]$ ]]; then
  echo "usage: $0 GPU_INDEX" >&2
  exit 2
fi
gpu="$1"
workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
candidate_root="${C58_ROOT:-${workspace}/outputs/c58-fastwam-full30-v1/long10000}"
control_root="${CONTROL_ROOT:-${workspace}/outputs/c58-matched-d0-control-v1/long10000}"
eval_root="${EVAL_ROOT:-${workspace}/outputs/c58-matched-depth-balanced80-v1}"
candidate_data="${workspace}/data/v7_multisuite_dense_candidate"
cache_root="${workspace}/data/v7_dense_h3_cache"
ready="${workspace}/dense-d0-v1-96976ce/cache_generation/full_audit/DUAL_CACHE_AUDIT_READY.json"
wait_seconds="${WAIT_SECONDS:-30}"
selected_sha="26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42"

[[ "${wait_seconds}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid WAIT_SECONDS" >&2; exit 2; }
for path in "${python_bin}" "${ready}" "${candidate_data}/manifest_all.jsonl" \
  "${candidate_data}/manifest_train_uniform.jsonl" "${candidate_data}/manifest_val.jsonl"; do
  [[ -f "${path}" ]] || { echo "missing balanced80 input: ${path}" >&2; exit 1; }
done
mkdir -p "${eval_root}/candidate" "${eval_root}/control" "${eval_root}/logs"
export CUDA_VISIBLE_DEVICES="${gpu}"
export PYTHONPATH="${project}/src"
export H3WAM_FASTWAM_SOURCE_ROOT="${workspace}/upstream-readonly/FastWAM-45d8e145/wan22"
cuda13_lib="$(${python_bin} -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["purelib"])/"nvidia"/"cu13"/"lib")')"
export LD_LIBRARY_PATH="${cuda13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
cd "${project}"
audit_hash="$(${python_bin} - "${ready}" <<'PY'
import json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text())
if x.get("ready") is not True: raise SystemExit("cache audit is not READY")
print(x["dreamwam_kv_aggregate_sha256"])
PY
)"

for step in $(seq 1000 1000 10000); do
  candidate="${candidate_root}/checkpoints/c58_s${step}.pt"
  control="${control_root}/checkpoints/control_s${step}.pt"
  while [[ ! -f "${candidate}" || ! -f "${control}" ]]; do
    printf '{"state":"WAIT_PAIRED_CHECKPOINT","step":%s}\n' "${step}"
    sleep "${wait_seconds}"
  done
  for arm in candidate control; do
    checkpoint="${candidate}"
    [[ "${arm}" == control ]] && checkpoint="${control}"
    output="${eval_root}/${arm}/step${step}.balanced80.json"
    log="${eval_root}/logs/${arm}_step${step}.log"
    if [[ -f "${output}" ]]; then
      continue
    fi
    [[ ! -e "${output}" ]] || { echo "partial evaluation exists: ${output}" >&2; exit 1; }
    "${python_bin}" scripts/h3wam/evaluate_h3_fastwam_full_tower.py \
      "${checkpoint}" \
      --source-manifest "${candidate_data}/manifest_all.jsonl" \
      --train-manifest "${candidate_data}/manifest_train_uniform.jsonl" \
      --val-manifest "${candidate_data}/manifest_val.jsonl" \
      --cache-root "${cache_root}" --kv-subdir h3_int8_dreamwam_kv_5x32_dense_v1 \
      --output "${output}" --device cuda --num-workers 0 \
      --cache-audit-aggregate-sha256 "${audit_hash}" \
      --expected-selected-ids-sha256 "${selected_sha}" >"${log}" 2>&1
  done
  "${python_bin}" scripts/h3wam/finalize_c58_matched_balanced80.py \
    --root "${eval_root}" --output "${eval_root}/SUMMARY.json"
done

echo "[C58 pair] all ten balanced-80 milestone pairs evaluated and gated"
