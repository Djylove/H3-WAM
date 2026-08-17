#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:?PROJECT_ROOT must be a reviewed immutable source snapshot}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/conda-py311/bin/python}"
supplemental_site="${SUPPLEMENTAL_SITE_PACKAGES:-${workspace}/.venv/lib/python3.11/site-packages}"
int8_runtime_deps="${INT8_RUNTIME_DEPS:-${workspace}/runtime/c66-conda-deps}"
plan_root="${C66_PLAN_ROOT:-${workspace}/data/c66-lingbot-c58-canary-v1}"
canary_root="${C66_CANARY_ROOT:-${workspace}/outputs/c66-lingbot-c58-block-persistent/paired-canary-s100-conda-v3}"
output_root="${C66_DIAGNOSTIC_ROOT:-${workspace}/outputs/c66-lingbot-c58-block-persistent/context-length-diagnostic-v1}"

parent="${C58_CHECKPOINT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt}"
h3="${H3_CHECKPOINT:-${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
dense="${DENSE_MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}"
source="${SOURCE_MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}"
cache="${CACHE_ROOT:-${workspace}/data/v7_dense_h3_cache}"
evaluator="${project}/scripts/h3wam/evaluate_c66_context_length_diagnostic.py"

for path in \
  "${python_bin}" "${supplemental_site}" "${int8_runtime_deps}/comfy_kitchen" \
  "${evaluator}" "${plan_root}/PLAN.json" \
  "${plan_root}/manifest_heldout64.jsonl" "${parent}" "${h3}" \
  "${canary_root}/c66_s00100.pt" "${canary_root}/report.json" \
  "${dense}" "${source}" "${cache}/stats.pt"; do
  [[ -e "${path}" ]] || { echo "missing C66 diagnostic input: ${path}" >&2; exit 2; }
done
[[ "$(stat -c '%A' "${evaluator}")" != *w* ]] || {
  echo "C66 diagnostic evaluator must come from a read-only snapshot" >&2
  exit 2
}
[[ ! -e "${output_root}" ]] || {
  echo "refusing existing C66 diagnostic root: ${output_root}" >&2
  exit 2
}

mkdir -p "${output_root}"
cd "${project}"
export PYTHONPATH="${project}/src:${project}/third_party/diffusers_h3/src:${int8_runtime_deps}:${supplemental_site}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

exec "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/h3wam/evaluate_c66_context_length_diagnostic.py \
  --plan "${plan_root}/PLAN.json" \
  --heldout-manifest "${plan_root}/manifest_heldout64.jsonl" \
  --dense-manifest "${dense}" \
  --source-manifest "${source}" \
  --cache-root "${cache}" \
  --h3-checkpoint "${h3}" \
  --parent-checkpoint "${parent}" \
  --c66-checkpoint "${canary_root}/c66_s00100.pt" \
  --c66-report "${canary_root}/report.json" \
  --output "${output_root}/RESULTS.json" \
  --seed 66017
