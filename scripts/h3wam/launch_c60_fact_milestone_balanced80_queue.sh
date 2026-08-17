#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:?PROJECT_ROOT must be an immutable read-only snapshot}"
root="${C60_MILESTONE_EVAL_ROOT:?C60_MILESTONE_EVAL_ROOT is required}"
python_bin="${workspace}/runtime/h3-int8-native/bin/python"
train_root="${workspace}/outputs/c56b-fact-online-v1/online-long10000-v1"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
train_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
val_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_val.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
source_root="${workspace}/upstream-readonly/FastWAM-45d8e145/wan22"

[[ ! -e "${root}" ]] || { echo "refusing existing C60 milestone eval root" >&2; exit 2; }
[[ "$(stat -c '%A' "${project}/scripts/h3wam/evaluate_c60_fact_milestone_balanced80.py")" != *w* ]] || {
  echo "PROJECT_ROOT is not read-only" >&2; exit 2;
}
for path in "${python_bin}" "${h3_checkpoint}" "${source_manifest}" \
  "${train_manifest}" "${val_manifest}" "${cache_root}/stats.pt" \
  "${source_root}/action_dit.py" "${source_root}/wan_video_dit.py" \
  "${source_root}/helpers/gradient.py"; do
  [[ -e "${path}" ]] || { echo "missing C60 milestone input: ${path}" >&2; exit 2; }
done
mkdir -p "${root}/reports" "${root}/logs"

cd "${project}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

run_gpu() {
  local gpu="$1" step checkpoint audit output
  for step in $(seq 1000 1000 10000); do
    (( (step / 1000 - 1) % 8 == gpu )) || continue
    checkpoint="${train_root}/checkpoints/c56b_online_s${step}.pt"
    audit="${train_root}/milestone-audit/s${step}.json"
    output="${root}/reports/s${step}.json"
    [[ -f "${checkpoint}" && -f "${audit}" && ! -e "${output}" ]] || {
      echo "missing/reused milestone input: s${step}" >&2; return 2;
    }
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" \
      scripts/h3wam/evaluate_c60_fact_milestone_balanced80.py \
      --checkpoint "${checkpoint}" --restore-audit "${audit}" --milestone "${step}" \
      --h3-checkpoint "${h3_checkpoint}" --source-manifest "${source_manifest}" \
      --train-manifest "${train_manifest}" --val-manifest "${val_manifest}" \
      --cache-root "${cache_root}" --device cuda:0 --output "${output}" \
      >"${root}/logs/s${step}.log" 2>&1
  done
}

pids=()
for gpu in 0 1 2 3 4 5 6 7; do run_gpu "${gpu}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "one or more C60 milestone evaluators failed" >&2; exit 1; }

"${python_bin}" scripts/h3wam/aggregate_c60_fact_milestone_balanced80.py \
  --root "${root}" --training-root "${train_root}" --output "${root}/RESULTS.json" \
  >"${root}/aggregate.log" 2>&1
