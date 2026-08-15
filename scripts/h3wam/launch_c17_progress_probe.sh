#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${workspace}/runtime/conda-py311/bin/python"
target_root="${workspace}/eval/expert-progress-targets-v1"
cache_root="${workspace}/data/v7_dense_h3_cache"
kv_subdir="h3_int8_dreamwam_kv_5x32_dense_v1"
output_root="${workspace}/eval/c17-frozen-h3-progress-probe-v1"

[[ ! -e "${output_root}" ]] || { echo "refusing existing C17 output root" >&2; exit 1; }
mkdir -p "${output_root}/features" "${output_root}/logs"
cd "${project}"

run_split() {
  local split="$1" per_suite="$2"
  local pids=()
  for shard in 0 1; do
    nice -n 10 "${python_bin}" scripts/h3wam/precompute_h3_progress_probe_features.py \
      "${target_root}/${split}.jsonl" --cache-root "${cache_root}" --kv-subdir "${kv_subdir}" \
      --per-suite "${per_suite}" --num-shards 2 --shard-index "${shard}" \
      --output "${output_root}/features/${split}_shard_${shard}.pt" \
      >"${output_root}/logs/${split}_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
  (( failed == 0 )) || return 1
}

run_split train 1000
run_split val 500
"${python_bin}" scripts/h3wam/evaluate_h3_progress_probe.py \
  --train "${output_root}/features/train_shard_0.pt" "${output_root}/features/train_shard_1.pt" \
  --val "${output_root}/features/val_shard_0.pt" "${output_root}/features/val_shard_1.pt" \
  --output "${output_root}/COMPLETED" >"${output_root}/logs/evaluation.log" 2>&1
cat "${output_root}/COMPLETED"
