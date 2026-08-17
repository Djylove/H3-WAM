#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:?PROJECT_ROOT must be a reviewed immutable source snapshot}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/conda-py311/bin/python}"
supplemental_site="${SUPPLEMENTAL_SITE_PACKAGES:-${workspace}/.venv/lib/python3.11/site-packages}"
int8_runtime_deps="${INT8_RUNTIME_DEPS:-${workspace}/runtime/c66-conda-deps}"
plan_root="${C66_PLAN_ROOT:-${workspace}/data/c66-lingbot-c58-canary-v1}"
c66_root="${C66_CANARY_ROOT:-${workspace}/outputs/c66-lingbot-c58-block-persistent/paired-canary-s100-conda-v3}"
diagnostic="${C66_DIAGNOSTIC:-${workspace}/outputs/c66-lingbot-c58-block-persistent/context-length-diagnostic-v1/RESULTS.json}"
output_root="${C66_K1_OUTPUT_ROOT:-${workspace}/outputs/c66-k1-bounded-mechanism/s100-fresh-v1}"

parent="${C58_CHECKPOINT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt}"
h3="${H3_CHECKPOINT:-${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}"
dense="${DENSE_MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}"
source="${SOURCE_MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}"
cache="${CACHE_ROOT:-${workspace}/data/v7_dense_h3_cache}"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"
trainer="${project}/scripts/h3wam/train_c66_k1_bounded_mechanism_canary.py"
auditor="${project}/scripts/h3wam/audit_c66_k1_single_variable.py"

case "${project}" in
  "${workspace}"/code-snapshots/*) ;;
  *) echo "C66-k1 project must be under ${workspace}/code-snapshots" >&2; exit 2 ;;
esac
[[ -d "${project}" ]] || { echo "missing immutable project: ${project}" >&2; exit 2; }
if find "${project}" -type l -print -quit | grep -q .; then
  echo "C66-k1 source snapshot must not contain symlinks" >&2
  exit 2
fi
if find "${project}" -type f -perm /222 -print -quit | grep -q .; then
  echo "C66-k1 source snapshot contains writable files" >&2
  exit 2
fi

for path in \
  "${python_bin}" "${supplemental_site}" "${int8_runtime_deps}/comfy_kitchen" \
  "${trainer}" "${auditor}" "${plan_root}/PLAN.json" \
  "${plan_root}/manifest_train800.jsonl" "${plan_root}/manifest_heldout64.jsonl" \
  "${parent}" "${h3}" "${c66_root}/report.json" "${c66_root}/c66_s00100.pt" \
  "${diagnostic}" "${dense}" "${source}" "${cache}/stats.pt" \
  "${source_root}/action_dit.py" "${source_root}/wan_video_dit.py" \
  "${source_root}/helpers/gradient.py" \
  "${project}/third_party/diffusers_h3/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py"; do
  [[ -e "${path}" ]] || { echo "missing C66-k1 input: ${path}" >&2; exit 2; }
done
[[ ! -e "${output_root}" ]] || {
  echo "refusing existing C66-k1 output root: ${output_root}" >&2
  exit 2
}

mkdir -p "${output_root}"
cd "${project}"
export PYTHONPATH="${project}/src:${project}/third_party/diffusers_h3/src:${int8_runtime_deps}:${supplemental_site}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

"${python_bin}" "${auditor}" \
  --project "${project}" \
  --plan-root "${plan_root}" \
  --parent-checkpoint "${parent}" \
  --h3-checkpoint "${h3}" \
  --c66-root "${c66_root}" \
  --diagnostic "${diagnostic}" \
  --output "${output_root}/SOURCE_DATA_AUDIT.json"

"${python_bin}" -c 'import json,sys; value=json.load(open(sys.argv[1])); assert value["status"]=="PASS_C66_K1_SOURCE_DATA_GATE" and value["permission"]=="GO_BOUNDED_S100_MECHANISM_CANARY_ONLY"' "${output_root}/SOURCE_DATA_AUDIT.json"

exec "${python_bin}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/h3wam/train_c66_k1_bounded_mechanism_canary.py \
  --plan "${plan_root}/PLAN.json" \
  --train-manifest "${plan_root}/manifest_train800.jsonl" \
  --heldout-manifest "${plan_root}/manifest_heldout64.jsonl" \
  --dense-manifest "${dense}" \
  --source-manifest "${source}" \
  --cache-root "${cache}" \
  --h3-checkpoint "${h3}" \
  --parent-checkpoint "${parent}" \
  --output "${output_root}/report.json" \
  --save-checkpoint "${output_root}/c66_k1_s00100.pt" \
  --steps 100 \
  --learning-rate 1e-5 \
  --weight-decay 0.01 \
  --warmup-steps 10 \
  --max-grad-norm 1.0 \
  --seed 66017 \
  --num-workers 0
