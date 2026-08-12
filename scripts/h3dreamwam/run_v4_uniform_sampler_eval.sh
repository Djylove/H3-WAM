#!/usr/bin/env bash
set -euo pipefail

M4_ROOT=${M4_ROOT:-/home/h3wam_finetune}
M4_EVAL_TAG=${M4_EVAL_TAG:?set M4_EVAL_TAG}
M4_LOAD_STAGE=${M4_LOAD_STAGE:-}
M4_EVAL_STEPS=${M4_EVAL_STEPS:-5}
M4_DISABLE_ADAPTERS=${M4_DISABLE_ADAPTERS:-0}
M4_OVERRIDE_ACTION_IO=${M4_OVERRIDE_ACTION_IO:-}
M4_PROJECT=${M4_ROOT}/project
M4_CANDIDATE=${M4_ROOT}/data/v4_multisuite_uniform_candidate
M4_CACHE=${M4_ROOT}/data/v3_multisuite_cache
M4_OUTPUT_DIR=${M4_ROOT}/outputs/h3dreamwam_m3/eval_uniform_val40
M4_REPORT=${M4_OUTPUT_DIR}/${M4_EVAL_TAG}.json

if [[ -e "${M4_REPORT}" ]]; then
  echo "refusing to overwrite evaluation output: ${M4_REPORT}" >&2
  exit 2
fi
if [[ -n "${M4_LOAD_STAGE}" && ! -f "${M4_LOAD_STAGE}" ]]; then
  echo "missing evaluation checkpoint: ${M4_LOAD_STAGE}" >&2
  exit 2
fi

mkdir -p "${M4_OUTPUT_DIR}"
cd "${M4_PROJECT}"
M4_ARGS=(
  --model "${M4_ROOT}/models/MiniMax-H3"
  --data-root "${M4_CACHE}"
  --output "${M4_REPORT}"
  --manifest "${M4_CANDIDATE}/manifest_val_stratified40.jsonl"
  --rotate-manifest
  --last-h3-blocks 2
  --train-h3-io
  --action-train-stage tail_sharded
  --last-action-blocks 2
  --freeze-action-body
  --learning-rate 1e-7
  --h3-learning-rate 2e-6
  --eval-only
  --joint-sample-steps 10
  --steps "${M4_EVAL_STEPS}"
  --require-text-only-context
)
if [[ -n "${M4_LOAD_STAGE}" ]]; then
  M4_ARGS+=(--load-action-stage "${M4_LOAD_STAGE}")
fi
if [[ "${M4_DISABLE_ADAPTERS}" == "1" ]]; then
  M4_ARGS+=(--disable-video-residual-adapters)
fi
if [[ -n "${M4_OVERRIDE_ACTION_IO}" ]]; then
  M4_ARGS+=(--override-action-io "${M4_OVERRIDE_ACTION_IO}")
fi

exec env \
  HF_HOME="${M4_ROOT}/hf_cache" \
  PYTHONPATH=src \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  "${M4_ROOT}/.venv/bin/torchrun" \
  --standalone \
  --nproc-per-node=8 \
  scripts/h3dreamwam/verify_h3dreamwam_fsdp_real.py \
  "${M4_ARGS[@]}"
