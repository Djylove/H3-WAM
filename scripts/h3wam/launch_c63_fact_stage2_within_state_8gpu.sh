#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:?PROJECT_ROOT must be an immutable read-only snapshot}"
root="${C63_ROOT:?C63_ROOT is required}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"
ready="${workspace}/outputs/c56b-fact-online-v1/online-long10000-v1/READY.json"
dataset="${workspace}/eval/c60-counterfactual-failure-dataset-v1/dataset.pt"
observations="${workspace}/eval/c60-counterfactual-failure-dataset-v1/observations.jsonl"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
h3_model="${workspace}/models/MiniMax-H3"

[[ ! -e "${root}" ]] || { echo "refusing existing C63 root" >&2; exit 2; }
[[ "$(stat -c '%A' "${project}/scripts/h3wam/evaluate_c63_fact_stage2_within_state.py")" != *w* ]] || {
  echo "PROJECT_ROOT is not read-only" >&2; exit 2;
}
for path in "${python_bin}" "${ready}" "${dataset}" "${observations}" \
  "${source_manifest}" "${cache_root}/stats.pt" "${h3_checkpoint}" "${h3_model}"; do
  [[ -e "${path}" ]] || { echo "missing C63 input: ${path}" >&2; exit 2; }
done
mkdir -p "${root}/shards" "${root}/logs"
cd "${project}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

pair_manifest="${root}/PAIRS.json"
"${python_bin}" scripts/h3wam/prepare_c63_fact_stage2_pairs.py \
  --dataset "${dataset}" --observations "${observations}" --output "${pair_manifest}" \
  >"${root}/prepare.log" 2>&1

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" \
    scripts/h3wam/evaluate_c63_fact_stage2_within_state.py \
    --ready "${ready}" --pairs "${pair_manifest}" \
    --c60-dataset "${dataset}" --c60-observations "${observations}" \
    --source-manifest "${source_manifest}" --cache-root "${cache_root}" \
    --h3-checkpoint "${h3_checkpoint}" --h3-model "${h3_model}" \
    --shard "${gpu}" --num-shards 8 --device cuda:0 \
    --output "${root}/shards/shard${gpu}.json" \
    >"${root}/logs/shard${gpu}.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" >"${root}/PIDS"
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "one or more C63 shards failed" >&2; exit 1; }

"${python_bin}" scripts/h3wam/aggregate_c63_fact_stage2_within_state.py \
  --root "${root}" --pairs "${pair_manifest}" --output "${root}/RESULTS.json" \
  >"${root}/aggregate.log" 2>&1
sha256sum "${pair_manifest}" "${root}/RESULTS.json" >"${root}/SHA256SUMS"
