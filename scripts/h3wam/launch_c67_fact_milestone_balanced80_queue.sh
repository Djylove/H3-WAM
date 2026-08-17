#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:?PROJECT_ROOT must be the reviewed immutable C67 snapshot}"
train_root="${C67_TRAIN_ROOT:-${workspace}/outputs/c67-c60-budget-ablation-v1/online-long20000-v1}"
root="${C67_OFFLINE_ROOT:?C67_OFFLINE_ROOT is required}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
training_complete="${train_root}/TRAINING_COMPLETE.json"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
train_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
val_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_val.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"
evaluator="${project}/scripts/h3wam/evaluate_c67_fact_milestone_balanced80.py"
aggregator="${project}/scripts/h3wam/aggregate_c67_fact_milestone_balanced80.py"

[[ ! -e "${root}" ]] || { echo "refusing existing C67 offline root: ${root}" >&2; exit 2; }
for source in "${evaluator}" "${aggregator}"; do
  [[ -f "${source}" && "$(stat -c '%A' "${source}")" != *w* ]] || {
    echo "C67 offline execution source is missing or writable: ${source}" >&2
    exit 2
  }
done
for path in "${python_bin}" "${training_complete}" "${h3_checkpoint}" \
  "${source_manifest}" "${train_manifest}" "${val_manifest}" \
  "${cache_root}/stats.pt" "${source_root}/action_dit.py" \
  "${source_root}/wan_video_dit.py" "${source_root}/helpers/gradient.py"; do
  [[ -e "${path}" ]] || { echo "missing C67 offline input: ${path}" >&2; exit 2; }
done
for milestone in $(seq 1000 1000 20000); do
  [[ -f "${train_root}/checkpoints/c67_online_s${milestone}.pt" ]] || {
    echo "missing C67 checkpoint s${milestone}" >&2; exit 2;
  }
  [[ -f "${train_root}/milestone-audit/s${milestone}.json" ]] || {
    echo "missing C67 restore audit s${milestone}" >&2; exit 2;
  }
done

mkdir -p "${root}/reports" "${root}/logs"
cd "${project}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export TMPDIR="${workspace}/tmp/c67-balanced80"
mkdir -p "${TMPDIR}"

run_gpu() {
  local gpu="$1" milestone checkpoint audit output
  for milestone in $(seq 1000 1000 20000); do
    (( (milestone / 1000 - 1) % 8 == gpu )) || continue
    checkpoint="${train_root}/checkpoints/c67_online_s${milestone}.pt"
    audit="${train_root}/milestone-audit/s${milestone}.json"
    output="${root}/reports/s${milestone}.json"
    [[ ! -e "${output}" ]] || { echo "refusing reused C67 report s${milestone}" >&2; return 2; }
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" \
      scripts/h3wam/evaluate_c67_fact_milestone_balanced80.py \
      --checkpoint "${checkpoint}" --restore-audit "${audit}" \
      --training-complete "${training_complete}" --milestone "${milestone}" \
      --h3-checkpoint "${h3_checkpoint}" --source-manifest "${source_manifest}" \
      --train-manifest "${train_manifest}" --val-manifest "${val_manifest}" \
      --cache-root "${cache_root}" --device cuda:0 --output "${output}" \
      >"${root}/logs/s${milestone}.log" 2>&1
  done
}

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  run_gpu "${gpu}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "one or more C67 milestone evaluators failed" >&2; exit 1; }

"${python_bin}" scripts/h3wam/aggregate_c67_fact_milestone_balanced80.py \
  --root "${root}" --training-complete "${training_complete}" \
  --output "${root}/RESULTS.json" >"${root}/aggregate.log" 2>&1
