#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:?PROJECT_ROOT must be an immutable read-only C65 score snapshot}"
collection="${C65_COLLECTION_ROOT:-${workspace}/eval/c65-c60-deployment-pair-collection-v1}"
root="${C65_SCORE_ROOT:?C65_SCORE_ROOT is required}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"
ready="${workspace}/outputs/c56b-fact-online-v1/online-long10000-v1/READY.json"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
h3_model="${workspace}/models/MiniMax-H3"
evaluator="${project}/scripts/h3wam/evaluate_c65_fact_stage2_pairs.py"
aggregator="${project}/scripts/h3wam/aggregate_c65_fact_stage2_pairs.py"
data_gate="${collection}/DATA_GATE.json"
pairs="${collection}/PAIRS.json"

[[ ! -e "${root}" ]] || { echo "refusing existing C65 score root" >&2; exit 2; }
for path in "${evaluator}" "${aggregator}"; do
  [[ -f "${path}" && "$(stat -c '%A' "${path}")" != *w* ]] || {
    echo "C65 score source is missing or writable: ${path}" >&2; exit 2;
  }
done
for path in "${python_bin}" "${ready}" "${data_gate}" "${pairs}" \
  "${source_manifest}" "${cache_root}/stats.pt" "${h3_checkpoint}" "${h3_model}"; do
  [[ -e "${path}" ]] || { echo "missing C65 score input: ${path}" >&2; exit 2; }
done

mkdir -p "${root}/shards" "${root}/logs"
cd "${project}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" "${evaluator}" \
    --ready "${ready}" --data-gate "${data_gate}" --pairs "${pairs}" \
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
(( status == 0 )) || { echo "one or more C65 score shards failed" >&2; exit 1; }

"${python_bin}" "${aggregator}" \
  --root "${root}" --data-gate "${data_gate}" --pairs "${pairs}" \
  --output "${root}/RESULTS.json" >"${root}/aggregate.log" 2>&1
sha256sum "${data_gate}" "${pairs}" "${root}/RESULTS.json" >"${root}/SHA256SUMS"
