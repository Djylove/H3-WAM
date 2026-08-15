#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${workspace}/runtime/h3-int8-native/bin/python"
candidate_root="${workspace}/data/v7_multisuite_dense_candidate"
cache_root="${workspace}/data/v7_dense_h3_cache"
kv_subdir="h3_int8_dreamwam_kv_5x32_dualviewgrid_stage112k_120k_v1"
checkpoint="${workspace}/outputs/c15-d0-grid-adaptation-s1000-v1/checkpoints/d0_grid_h32_s15000.pt"
sibling="${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/evaluations/d0_h32_s15000_balanced80.json"
cache_ready="${workspace}/eval/c15-grid-cache-val-v1/COMPLETED"
output_root="${workspace}/eval/c15-grid-balanced80-gate-v1"
evaluation="${output_root}/d0_grid_h32_s15000_balanced80.json"

mkdir -p "${output_root}"
[[ ! -e "${output_root}/COMPLETED" ]] || { echo "refusing completed C15 balanced80 gate" >&2; exit 1; }
while [[ ! -f "${checkpoint}" || ! -f "${cache_ready}" ]]; do
  echo "$(date -Iseconds) WAIT_C15_CHECKPOINT_OR_VAL_CACHE"; sleep 30
done
[[ ! -e "${evaluation}" ]] || { echo "refusing existing evaluation" >&2; exit 1; }

export PYTHONPATH="${project}/src:${project}"
cd "${project}"
CUDA_VISIBLE_DEVICES=0 "${python_bin}" scripts/h3wam/evaluate_h3_dreamwam_kv_carrier.py \
  "${checkpoint}" \
  --source-manifest "${candidate_root}/manifest_all.jsonl" \
  --train-manifest "${candidate_root}/manifest_train_uniform.jsonl" \
  --val-manifest "${candidate_root}/manifest_val.jsonl" \
  --cache-root "${cache_root}" --kv-subdir "${kv_subdir}" \
  --output "${evaluation}" --device cuda --num-workers 0 \
  --expected-selected-ids-sha256 b507e1ff6031f01c88cd6181aaeb4cba33b76e2c67737a986bf764c76be87519 \
  >"${output_root}/evaluation.log" 2>&1

"${python_bin}" - "${evaluation}" "${sibling}" "${output_root}/COMPLETED" <<'PY'
import json, os, sys
from pathlib import Path
candidate_path, sibling_path, destination = map(Path, sys.argv[1:])
candidate, sibling = json.loads(candidate_path.read_text()), json.loads(sibling_path.read_text())
def metrics(report):
    values = report["metrics"]
    return {
        "normalized_action_mse": values["normalized_clip5_model_domain"]["action_mse"],
        "physical_action_mse": values["denormalized_official_minmax_clamp"]["action_mse"],
        "gripper_macro_f1": values["gripper_sign"]["macro_f1"],
        "language_delta": values["language_replacement_sensitivity"]["mean_abs_prediction_delta"],
        "visual_shuffle_action_delta": values["visual_feature_shuffle"]["baseline_vs_shuffle_action_delta"]["normalized_model_domain"]["action_mae"],
    }
cand, sib = metrics(candidate), metrics(sibling)
report = {
    "format": "h3-c15-grid-balanced80-gate-v1",
    "candidate": cand, "fixed_sibling_d0_s15000": sib,
    "normalized_mse_ratio": cand["normalized_action_mse"] / sib["normalized_action_mse"],
    "offline_win": cand["normalized_action_mse"] < sib["normalized_action_mse"],
    "material_regression": cand["normalized_action_mse"] > 1.10 * sib["normalized_action_mse"],
    "status": "EFFECT_EVIDENCE_PENDING_CLOSED_LOOP",
}
temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(report, indent=2) + "\n"); os.replace(temporary, destination)
print(json.dumps(report, sort_keys=True))
PY
