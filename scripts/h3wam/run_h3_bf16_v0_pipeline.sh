#!/usr/bin/env bash
set -euo pipefail

# Download -> verify -> one-step real 33B FSDP smoke.  This wrapper is safe to
# leave running after an SSH session disconnects; all state stays in H3_WORKSPACE.
H3_WORKSPACE="${H3_WORKSPACE:-/home/h3wam_finetune}"
case "${H3_WORKSPACE}" in
  /home/h3wam_finetune|/home/h3wam_finetune/*) ;;
  *)
    echo "H3_WORKSPACE must stay under /home/h3wam_finetune" >&2
    exit 2
    ;;
esac

export HOME="${H3_WORKSPACE}"
export XDG_CACHE_HOME="${H3_WORKSPACE}/cache"
export HF_HOME="${H3_WORKSPACE}/cache/huggingface"
export TORCH_HOME="${H3_WORKSPACE}/cache/torch"
export TMPDIR="${H3_WORKSPACE}/tmp"
export PYTHONNOUSERSITE=1

cd "${H3_WORKSPACE}/project"
"${H3_WORKSPACE}/.venv/bin/python" \
  scripts/h3wam/download_h3_transformer_modelscope.py \
  --output "${H3_WORKSPACE}/models/MiniMax-H3" \
  --workers 14

H3_RUN_NAME="${H3_RUN_NAME:-v0_last2_smoke}" \
H3_STEPS="${H3_STEPS:-1}" \
H3_LAST_BLOCKS="${H3_LAST_BLOCKS:-2}" \
H3_LR="${H3_LR:-1e-5}" \
H3_SEED="${H3_SEED:-2026}" \
bash scripts/h3wam/launch_h3_bf16_v0.sh
