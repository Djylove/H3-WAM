#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:?PROJECT_ROOT must be the reviewed immutable C67 preview snapshot}"
train_root="${C67_TRAIN_ROOT:-${workspace}/outputs/c67-c60-budget-ablation-v1/online-long20000-v1}"
root="${C67_PREVIEW_ROOT:?C67_PREVIEW_ROOT is required}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
training_complete="${train_root}/TRAINING_COMPLETE.json"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
train_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
val_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_val.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"
auditor="${project}/scripts/h3wam/prepare_c67_milestone_preview_audit.py"
evaluator="${project}/scripts/h3wam/evaluate_c67_fact_milestone_balanced80.py"
sealer="${project}/scripts/h3wam/seal_c67_milestone_previews.py"
aggregator="${project}/scripts/h3wam/aggregate_c67_fact_milestone_balanced80.py"

[[ ! -e "${root}" ]] || { echo "refusing existing C67 preview root: ${root}" >&2; exit 2; }
for path in "${auditor}" "${evaluator}" "${sealer}" "${aggregator}"; do
  [[ -f "${path}" && "$(stat -c '%A' "${path}")" != *w* ]] || {
    echo "C67 preview execution source is missing or writable: ${path}" >&2
    exit 2
  }
done
for path in "${python_bin}" "${h3_checkpoint}" "${source_manifest}" \
  "${train_manifest}" "${val_manifest}" "${cache_root}/stats.pt" \
  "${source_root}/action_dit.py" "${source_root}/wan_video_dit.py" \
  "${source_root}/helpers/gradient.py"; do
  [[ -e "${path}" ]] || { echo "missing C67 preview input: ${path}" >&2; exit 2; }
done

mkdir -p "${root}/preview-audit" "${root}/reports" "${root}/logs"
cd "${project}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${workspace}/tmp/c67-balanced80-preview"
mkdir -p "${TMPDIR}"

run_gpu() {
  local gpu="$1" milestone checkpoint train_report restore_report audit output
  for milestone in $(seq 1000 1000 20000); do
    (( (milestone / 1000 - 1) % 8 == gpu )) || continue
    checkpoint="${train_root}/checkpoints/c67_online_s${milestone}.pt"
    train_report="${train_root}/reports/train_s${milestone}.json"
    restore_report="${train_root}/restore/restore_s${milestone}.json"
    audit="${root}/preview-audit/s${milestone}.json"
    output="${root}/reports/s${milestone}.json"
    while [[ ! -s "${checkpoint}" || ! -s "${train_report}" || ! -s "${restore_report}" ]]; do
      sleep 30
    done
    "${python_bin}" "${auditor}" \
      --train-root "${train_root}" --milestone "${milestone}" --output "${audit}" \
      >"${root}/logs/audit_s${milestone}.log" 2>&1
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" "${evaluator}" \
      --checkpoint "${checkpoint}" --preview-audit "${audit}" \
      --milestone "${milestone}" --h3-checkpoint "${h3_checkpoint}" \
      --source-manifest "${source_manifest}" --train-manifest "${train_manifest}" \
      --val-manifest "${val_manifest}" --cache-root "${cache_root}" \
      --device cuda:0 --output "${output}" \
      >"${root}/logs/eval_s${milestone}.log" 2>&1
  done
}

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  run_gpu "${gpu}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "one or more C67 preview workers failed" >&2; exit 1; }

while [[ ! -s "${training_complete}" ]]; do sleep 30; done
"${python_bin}" "${sealer}" \
  --preview-root "${root}" --train-root "${train_root}" \
  --training-complete "${training_complete}" --output-root "${root}/sealed" \
  >"${root}/seal.log" 2>&1
"${python_bin}" "${aggregator}" \
  --root "${root}/sealed" --training-complete "${training_complete}" \
  --output "${root}/sealed/RESULTS.json" >"${root}/aggregate.log" 2>&1
