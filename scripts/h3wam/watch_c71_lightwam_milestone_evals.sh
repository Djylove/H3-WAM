#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C71_SOURCE_SNAPSHOT:?C71 milestone watcher requires an immutable source snapshot}"
freeze_sha="${C71_SOURCE_FREEZE_SHA256:?Set the reviewed SOURCE_FREEZE SHA256}"
train_root="${C71_TRAIN_ROOT:?Set the immutable C71 long-run output root}"
eval_root="${C71_EVAL_ROOT:?Set a new C71 milestone evaluation root}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
evaluator="${project}/scripts/h3wam/evaluate_c71_lightwam_balanced80.py"
verifier="${project}/scripts/h3wam/freeze_c67_rollout_source.py"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
data_root="${workspace}/data/v7_multisuite_dense_candidate"
cache_root="${workspace}/data/v7_dense_h3_cache"

for path in "${python_bin}" "${evaluator}" "${verifier}" "${project}/SOURCE_FREEZE.json" \
  "${h3_checkpoint}" "${cache_root}/stats.pt" "${data_root}/manifest_all.jsonl" \
  "${data_root}/manifest_train_uniform.jsonl" "${data_root}/manifest_val.jsonl"; do
  [[ -e "${path}" ]] || { echo "missing C71 milestone input: ${path}" >&2; exit 2; }
done
[[ ! -e "${eval_root}" ]] || { echo "refusing existing C71 milestone root: ${eval_root}" >&2; exit 2; }
mkdir -p "${eval_root}/logs"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export H3WAM_LIGHTWAM_SOURCE_ROOT="${project}/third_party/Light-WAM/src/lightwam/models/wan22"
export LD_LIBRARY_PATH="${workspace}/runtime/h3-int8-native/lib/python3.11/site-packages/nvidia/cu13/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export TMPDIR="${workspace}/tmp/c71-lightwam-milestone-eval"
mkdir -p "${TMPDIR}"
"${python_bin}" "${verifier}" --verify --snapshot "${project}" \
  --expected-manifest-sha256 "${freeze_sha}"

for step in 5000 10000; do
  checkpoint="${train_root}/checkpoints/c71_online_s${step}.pt"
  restore="${train_root}/reports/restore_s${step}.json"
  while [[ ! -s "${checkpoint}" || ! -s "${restore}" ]]; do
    sleep 15
  done
  output="${eval_root}/balanced80_s${step}.json"
  CUDA_VISIBLE_DEVICES="${C71_EVAL_CUDA_VISIBLE_DEVICES:-0}" \
    "${python_bin}" "${evaluator}" "${checkpoint}" \
    --restore-report "${restore}" --h3-checkpoint "${h3_checkpoint}" \
    --source-manifest "${data_root}/manifest_all.jsonl" \
    --train-manifest "${data_root}/manifest_train_uniform.jsonl" \
    --val-manifest "${data_root}/manifest_val.jsonl" --cache-root "${cache_root}" \
    --expected-steps "${step}" --device cuda:0 --num-workers 0 --output "${output}" \
    > "${eval_root}/logs/balanced80_s${step}.log" 2>&1
  "${python_bin}" - "${output}" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = Path(sys.argv[1]); report = json.loads(p.read_text())
assert report["status"] == "PASS_C71_LIGHTWAM_BALANCED80"
assert report["checkpoint"]["fresh_evaluator_restore_max_abs"] == 0.0
print(json.dumps({
    "step": report["checkpoint"]["completed_steps"],
    "report": str(p),
    "report_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    "normalized_mse": report["metrics"]["normalized_clip5_model_domain"]["action_mse"],
    "physical_mse": report["metrics"]["denormalized_official_minmax_clamp"]["action_mse"],
    "gripper_macro_f1": report["metrics"]["gripper_sign"]["macro_f1"],
}, indent=2))
PY
done
