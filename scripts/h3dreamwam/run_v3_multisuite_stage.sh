#!/usr/bin/env bash
set -euo pipefail

# Tunable lab-stage launcher.  The default 100-step run is a capacity/data
# canary; follow-up stages load its model-only checkpoint and extend the ladder.
M3_ROOT=${M3_ROOT:-/home/h3wam_finetune}
M3_STEPS=${M3_STEPS:-100}
M3_TAG=${M3_TAG:-multisuite_uniform_canary100}
M3_LOAD_STAGE=${M3_LOAD_STAGE:-${M3_ROOT}/outputs/h3dreamwam_m3/multisuite_uniform_head_epoch1.pt}
M3_ACTION_LR=${M3_ACTION_LR:-1e-7}
M3_H3_LR=${M3_H3_LR:-2e-6}

M3_PROJECT=${M3_ROOT}/project
M3_CANDIDATE=${M3_ROOT}/data/v4_multisuite_uniform_candidate
M3_CACHE=${M3_ROOT}/data/v3_multisuite_cache
M3_MOTION=${M3_ROOT}/data/v3_motion_multisuite
M3_MANIFEST=${M3_MANIFEST:-${M3_CANDIDATE}/manifest_train.jsonl}
M3_OUTPUT_DIR=${M3_ROOT}/outputs/h3dreamwam_m3
M3_REPORT=${M3_OUTPUT_DIR}/${M3_TAG}.json
M3_CHECKPOINT=${M3_OUTPUT_DIR}/${M3_TAG}.pt

if [[ -e "${M3_REPORT}" || -e "${M3_CHECKPOINT}" ]]; then
  echo "refusing to overwrite an existing M3 output: ${M3_TAG}" >&2
  exit 2
fi
if [[ ! -f "${M3_LOAD_STAGE}" ]]; then
  echo "missing input checkpoint: ${M3_LOAD_STAGE}" >&2
  exit 2
fi

M3_EXPECTED=$(wc -l < "${M3_CANDIDATE}/manifest_all.jsonl")
M3_AVAILABLE=$(find "${M3_MOTION}" -maxdepth 1 -type f -name '*.pt' | wc -l)
if [[ "${M3_AVAILABLE}" -ne "${M3_EXPECTED}" ]]; then
  echo "motion cache incomplete: ${M3_AVAILABLE}/${M3_EXPECTED}" >&2
  exit 2
fi

mkdir -p "${M3_OUTPUT_DIR}"
cd "${M3_PROJECT}"
exec env \
  HF_HOME="${M3_ROOT}/hf_cache" \
  PYTHONPATH=src \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  "${M3_ROOT}/.venv/bin/torchrun" \
  --standalone \
  --nproc-per-node=8 \
  scripts/h3dreamwam/verify_h3dreamwam_fsdp_real.py \
  --model "${M3_ROOT}/models/MiniMax-H3" \
  --data-root "${M3_CACHE}" \
  --motion-root "${M3_MOTION}" \
  --flow-loss-weight 0.5 \
  --train-h3-io \
  --output "${M3_REPORT}" \
  --manifest "${M3_MANIFEST}" \
  --rotate-manifest \
  --last-h3-blocks 2 \
  --action-train-stage tail_sharded \
  --last-action-blocks 2 \
  --freeze-action-body \
  --separate-expert-clipping \
  --learning-rate "${M3_ACTION_LR}" \
  --h3-learning-rate "${M3_H3_LR}" \
  --load-action-stage "${M3_LOAD_STAGE}" \
  --save-action-stage "${M3_CHECKPOINT}" \
  --dreamwam-action-weighting \
  --dreamwam-world-weighting \
  --steps "${M3_STEPS}" \
  --require-text-only-context
