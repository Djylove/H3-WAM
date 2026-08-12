#!/usr/bin/env bash
set -euo pipefail

# One uniform pass over every unique multi-suite training window.  This stage
# starts from H3-derived ActionDiT initialization; it never loads a task-tuned
# checkpoint and updates only the action output head.
M4_ROOT=${M4_ROOT:-/home/h3wam_finetune}
M4_STEPS=${M4_STEPS:-964}
M4_TAG=${M4_TAG:-multisuite_uniform_head_epoch1}
M4_PROJECT=${M4_ROOT}/project
M4_CANDIDATE=${M4_ROOT}/data/v4_multisuite_uniform_candidate
M4_CACHE=${M4_ROOT}/data/v3_multisuite_cache
M4_OUTPUT_DIR=${M4_ROOT}/outputs/h3dreamwam_m3
M4_REPORT=${M4_OUTPUT_DIR}/${M4_TAG}.json
M4_CHECKPOINT=${M4_OUTPUT_DIR}/${M4_TAG}.pt

if [[ -e "${M4_REPORT}" || -e "${M4_CHECKPOINT}" ]]; then
  echo "refusing to overwrite an existing uniform-head output: ${M4_TAG}" >&2
  exit 2
fi

M4_UNIQUE=$(wc -l < "${M4_CANDIDATE}/manifest_train_unique.jsonl")
if [[ "${M4_UNIQUE}" -ne 7710 ]]; then
  echo "unexpected uniform training population: ${M4_UNIQUE}" >&2
  exit 2
fi

mkdir -p "${M4_OUTPUT_DIR}"
cd "${M4_PROJECT}"
exec env \
  HF_HOME="${M4_ROOT}/hf_cache" \
  PYTHONPATH=src \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  "${M4_ROOT}/.venv/bin/torchrun" \
  --standalone \
  --nproc-per-node=8 \
  scripts/h3dreamwam/verify_h3dreamwam_fsdp_real.py \
  --model "${M4_ROOT}/models/MiniMax-H3" \
  --data-root "${M4_CACHE}" \
  --output "${M4_REPORT}" \
  --manifest "${M4_CANDIDATE}/manifest_train_uniform.jsonl" \
  --rotate-manifest \
  --last-h3-blocks 0 \
  --action-train-stage head \
  --last-action-blocks 1 \
  --learning-rate 1e-5 \
  --bf16-model-storage \
  --save-action-stage "${M4_CHECKPOINT}" \
  --dreamwam-action-weighting \
  --steps "${M4_STEPS}" \
  --require-text-only-context
