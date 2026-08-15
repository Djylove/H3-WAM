#!/usr/bin/env bash
set -Eeuo pipefail
workspace="${H3_WORKSPACE:-/mnt/h3-wam}"; project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"; root="${workspace}/eval/c49-dense-value-h3-features-v1"; node="${C49_NODE:?0..3}"; [[ "${node}" =~ ^[0-3]$ ]]
test -f "${workspace}/eval/c48-fact-dense-value-dataset-v1/COMPLETED"; test ! -e "${root}/node${node}.COMPLETED"; mkdir -p "${root}/shards" "${root}/logs"
pids=()
for gpu in {0..7}; do
 shard=$((node*8+gpu)); output="${root}/shards/shard${shard}.pt"; [[ ! -e "${output}" ]] || { echo "refusing existing ${output}" >&2; exit 2; }
 CUDA_VISIBLE_DEVICES="${gpu}" LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}" \
  "${workspace}/runtime/h3-int8-native/bin/python" "${project}/scripts/h3wam/precompute_c49_dense_value_h3_shard.py" \
  --observations "${workspace}/eval/c48-fact-dense-value-dataset-v1/observations.jsonl" --cache-root "${workspace}/data/v7_dense_h3_cache" --source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" --h3-model "${workspace}/models/MiniMax-H3" --shard "${shard}" --num-shards 32 --device cuda:0 --output "${output}" >"${root}/logs/shard${shard}.log" 2>&1 &
 pids+=("$!")
done
failed=0; for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done; ((failed==0)); printf '{"node":%s,"shards":8}\n' "${node}" >"${root}/node${node}.COMPLETED"
