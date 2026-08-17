#!/usr/bin/env bash
set -euo pipefail

workspace=${H3_WORKSPACE:-/mnt/h3-wam}
project_root=${PROJECT_ROOT:-${workspace}/project}
python_bin=${PYTHON_BIN:-${workspace}/runtime/conda-py311/bin/python}
supplemental_site=${SUPPLEMENTAL_SITE_PACKAGES:-${workspace}/.venv/lib/python3.11/site-packages}
int8_runtime_deps=${INT8_RUNTIME_DEPS:-${workspace}/runtime/c66-conda-deps}
output_root=${C66_OUTPUT_ROOT:-${workspace}/outputs/c66-lingbot-c58-block-persistent/paired-canary-s100}
plan_root=${C66_PLAN_ROOT:-${workspace}/data/c66-lingbot-c58-canary-v1}

parent=${C58_CHECKPOINT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt}
h3=${H3_CHECKPOINT:-${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}
mechanical=${C66_MECHANICAL_REPORT:-${workspace}/outputs/c66-lingbot-c58-block-persistent/mechanical-v7-direct/report.json}
sequence=${C57_SEQUENCE_MANIFEST:-${workspace}/data/c57-lingbot-replan8-v1/manifest_train.jsonl}
dense=${DENSE_MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}
source=${SOURCE_MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}
cache=${CACHE_ROOT:-${workspace}/data/v7_dense_h3_cache}
diffusers_h3_root=${DIFFUSERS_H3_ROOT:-${project_root}/third_party/diffusers_h3}

for path in "${python_bin}" "${supplemental_site}" "${int8_runtime_deps}/comfy_kitchen" "${parent}" "${h3}" "${mechanical}" "${sequence}" "${dense}" "${source}" "${cache}/stats.pt" "${diffusers_h3_root}/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "required C66 canary path is missing: ${path}" >&2
    exit 1
  fi
done

mechanical_sha=$(sha256sum "${mechanical}" | awk '{print $1}')
if [[ "${mechanical_sha}" != "dba327cd41f26596cec23228eb5d4be67ff2fa6a4c354b198271ef48cd87468e" ]]; then
  echo "C66 mechanical report identity mismatch: ${mechanical_sha}" >&2
  exit 1
fi
"${python_bin}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="PASS_MECHANICAL_DATA_GATE" and r["optimizer_steps"]==0 and r["training_checkpoints_written"]==0' "${mechanical}"

mkdir -p "${output_root}"
cd "${project_root}"
export PYTHONPATH="${project_root}/src:${diffusers_h3_root}/src:${int8_runtime_deps}:${supplemental_site}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
"${python_bin}" -c 'import importlib.metadata as m, torch, comfy_kitchen; assert torch.__version__.startswith("2.8.0"); assert m.version("comfy-kitchen")=="0.2.26"; print("C66_CONDA_INT8_RUNTIME_OK", torch.__version__, m.version("comfy-kitchen"), flush=True)'

# Split freezing is deterministic and fail-closed. Existing artifacts must
# match the trainer's hashes; the freezer never silently rewrites them.
if [[ ! -e "${plan_root}/PLAN.json" ]]; then
  "${python_bin}" scripts/h3wam/freeze_c66_lingbot_c58_canary.py \
    "${sequence}" \
    --parent-checkpoint "${parent}" \
    --output-dir "${plan_root}" \
    --seed 66017
fi

exec "${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  scripts/h3wam/train_c66_lingbot_c58_persistent_canary.py \
  --plan "${plan_root}/PLAN.json" \
  --train-manifest "${plan_root}/manifest_train800.jsonl" \
  --heldout-manifest "${plan_root}/manifest_heldout64.jsonl" \
  --dense-manifest "${dense}" \
  --source-manifest "${source}" \
  --cache-root "${cache}" \
  --h3-checkpoint "${h3}" \
  --parent-checkpoint "${parent}" \
  --output "${output_root}/report.json" \
  --save-checkpoint "${output_root}/c66_s00100.pt" \
  --steps 100 \
  --learning-rate 1e-5 \
  --weight-decay 0.01 \
  --warmup-steps 10 \
  --max-grad-norm 1.0 \
  --seed 66017 \
  --num-workers 0
