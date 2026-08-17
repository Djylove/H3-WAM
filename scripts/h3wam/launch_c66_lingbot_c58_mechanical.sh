#!/usr/bin/env bash
set -euo pipefail

workspace=${H3_WORKSPACE:-/mnt/h3-wam}
project_root=${PROJECT_ROOT:-${workspace}/project}
python_bin=${PYTHON_BIN:-${workspace}/runtime/conda-py311/bin/python}
output_root=${C66_OUTPUT_ROOT:-${workspace}/outputs/c66-lingbot-c58-block-persistent/mechanical-v1}

checkpoint=${C58_CHECKPOINT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt}
h3_checkpoint=${H3_CHECKPOINT:-${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}
lingbot_source=${LINGBOT_SOURCE:-${project_root}/third_party/code_audit/lingbot-va}
sequence_root=${C57_SEQUENCE_ROOT:-${workspace}/data/c57-lingbot-replan8-v1}
source_manifest=${SOURCE_MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}
cache_root=${CACHE_ROOT:-${workspace}/data/v7_dense_h3_cache}

for path in \
  "${python_bin}" \
  "${checkpoint}" \
  "${h3_checkpoint}" \
  "${lingbot_source}/wan_va/modules/model.py" \
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
export PYTHONPATH="${project_root}/src:${project_root}/third_party/diffusers_h3/src:${PYTHONPATH:-}"
exec "${python_bin}" scripts/h3wam/probe_c66_lingbot_c58_persistent.py \
  --checkpoint "${checkpoint}" \
  --h3-checkpoint "${h3_checkpoint}" \
  --lingbot-source "${lingbot_source}" \
  --sequence-manifest "${sequence_root}/manifest_train.jsonl" \
  --sequence-audit "${sequence_root}/AUDIT.json" \
  --source-manifest "${source_manifest}" \
  --cache-root "${cache_root}" \
  --output "${output_root}/report.json"
