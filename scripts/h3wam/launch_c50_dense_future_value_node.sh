#!/usr/bin/env bash
set -Eeuo pipefail
workspace="${H3_WORKSPACE:-/mnt/h3-wam}"; project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"; root="${workspace}/outputs/c50-dense-future-value-v1"; node="${C50_NODE:?0..3}"; [[ "${node}" =~ ^[0-3]$ ]]
seeds=(161803 271828 8675309 20260815); seed="${seeds[${node}]}"; c38="${workspace}/outputs/c38-temporal-paired-null-replication-v1/temporal_seed${seed}/checkpoints/temporal_seed${seed}_step10000.pt"
test -f "${workspace}/eval/c49-dense-value-h3-features-v1/COMPLETED"; test -f "${workspace}/eval/c49-dense-value-h3-features-v1/projected_features.pt"; test -f "${c38}"; test ! -e "${root}/node${node}.COMPLETED"; mkdir -p "${root}" "${workspace}/logs/c50-dense-future-value-v1"
pids=(); gpu=0
for arm in joint frozen_consequence; do
 out="${root}/${arm}_seed${seed}"; test ! -e "${out}"
 CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${project}/src:${project}" "${workspace}/runtime/conda-py311/bin/python" "${project}/scripts/h3wam/train_c50_dense_future_value_expert.py" --dataset "${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt" --features "${workspace}/eval/c49-dense-value-h3-features-v1/projected_features.pt" --c38-checkpoint "${c38}" --arm "${arm}" --steps 10000 --batch-size 64 --seed "${seed}" --device cuda:0 --output-root "${out}" >"${workspace}/logs/c50-dense-future-value-v1/${arm}_seed${seed}.log" 2>&1 &
 pids+=("$!"); gpu=$((gpu+1))
done
failed=0; for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done; ((failed==0)); printf '{"node":%s,"seed":%s,"arms":2}\n' "${node}" "${seed}" >"${root}/node${node}.COMPLETED"
