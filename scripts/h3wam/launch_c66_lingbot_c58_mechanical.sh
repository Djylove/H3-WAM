#!/usr/bin/env bash
set -euo pipefail

workspace=${H3_WORKSPACE:-/mnt/h3-wam}
project_root=${PROJECT_ROOT:-${workspace}/project}
python_bin=${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}
output_root=${C66_OUTPUT_ROOT:-${workspace}/outputs/c66-lingbot-c58-block-persistent/mechanical-v1}

checkpoint=${C58_CHECKPOINT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt}
h3_checkpoint=${H3_CHECKPOINT:-${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}
lingbot_source=${LINGBOT_SOURCE:-${project_root}/third_party/code_audit/lingbot-va}
diffusers_h3_root=${DIFFUSERS_H3_ROOT:-${project_root}/third_party/diffusers_h3}
sequence_root=${C57_SEQUENCE_ROOT:-${workspace}/data/c57-lingbot-replan8-v1}
source_manifest=${SOURCE_MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}
cache_root=${CACHE_ROOT:-${workspace}/data/v7_dense_h3_cache}

for path in \
  "${python_bin}" \
  "${checkpoint}" \
  "${h3_checkpoint}" \
  "${lingbot_source}/wan_va/modules/model.py" \
  "${diffusers_h3_root}/src/diffusers/modular_pipelines/minimax_h3/before_denoise.py" \
  "${sequence_root}/manifest_train.jsonl" \
  "${sequence_root}/AUDIT.json" \
  "${source_manifest}" \
  "${cache_root}/stats.pt"; do
  if [[ ! -e "${path}" ]]; then
    echo "required C66 path is missing: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${output_root}"
cd "${project_root}"
# H3's released packed-sequence/layout implementation lives in the pinned
# vendored diffusers fork.  The lean training runtime intentionally does not
# install a site-package copy, so both source roots are execution-critical.
export PYTHONPATH="${project_root}/src:${diffusers_h3_root}/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Fail before hashing the 20+ GiB H3 checkpoint if the runtime cannot import
# the released layout code or initialize cuBLAS. Both BF16 and FP16 are used by
# the online H3 stack on A800.
"${python_bin}" -c 'import torch; from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep; assert torch.cuda.is_available(); device=torch.device("cuda:0"); [(torch.randn(64,64,device=device,dtype=d) @ torch.randn(64,64,device=device,dtype=d)).sum().item() for d in (torch.bfloat16,torch.float16)]; torch.cuda.synchronize(); print("C66_CUDA_DIFFUSERS_PREFLIGHT_PASS", flush=True)'

exec "${python_bin}" scripts/h3wam/probe_c66_lingbot_c58_persistent.py \
  --checkpoint "${checkpoint}" \
  --h3-checkpoint "${h3_checkpoint}" \
  --lingbot-source "${lingbot_source}" \
  --diffusers-h3-source "${diffusers_h3_root}" \
  --sequence-manifest "${sequence_root}/manifest_train.jsonl" \
  --sequence-audit "${sequence_root}/AUDIT.json" \
  --source-manifest "${source_manifest}" \
  --cache-root "${cache_root}" \
  --output "${output_root}/report.json"
