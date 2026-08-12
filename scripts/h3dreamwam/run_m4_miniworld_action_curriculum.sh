#!/usr/bin/env bash
set -euo pipefail

# MiniWorld-inspired ActionDiT curriculum for the multi-suite H3-DreamWAM.
# Each invocation runs exactly one phase so offline/closed-loop evaluation can
# gate promotion before more compute and checkpoint storage are consumed.
M4_ROOT=${M4_ROOT:-/home/h3wam_finetune}
M4_PHASE=${M4_PHASE:-gate_h8}
M4_PROJECT=${M4_ROOT}/project
M4_CANDIDATE=${M4_ROOT}/data/v4_multisuite_uniform_candidate
M4_CACHE=${M4_ROOT}/data/v3_multisuite_cache
M4_OUTPUT_DIR=${M4_ROOT}/outputs/h3dreamwam_m4
M4_MANIFEST=${M4_MANIFEST:-${M4_CANDIDATE}/manifest_train_uniform.jsonl}
M4_TRAIN_GATE=1
M4_TRAIN_ADAPTER=0
M4_TRAIN_CROSS=0
M4_FREEZE_OUTPUT=0

case "${M4_PHASE}" in
  gate_h8)
    M4_STEPS=${M4_STEPS:-100}
    M4_HORIZON=8
    M4_LAST_ACTION_BLOCKS=2
    M4_FREEZE_BODY=1
    M4_LOAD_STAGE=${M4_LOAD_STAGE:-${M4_ROOT}/outputs/h3dreamwam_m3/multisuite_uniform_joint100.pt}
    M4_TAG=${M4_TAG:-miniworld_gate_h8_s100}
    M4_IO_LR=${M4_IO_LR:-1e-7}
    M4_GATE_LR=${M4_GATE_LR:-1e-5}
    M4_TAIL_LR=${M4_TAIL_LR:-1e-8}
    ;;
  adapter_h8)
    M4_STEPS=${M4_STEPS:-100}
    M4_HORIZON=8
    M4_LAST_ACTION_BLOCKS=2
    M4_FREEZE_BODY=1
    M4_TRAIN_GATE=0
    M4_TRAIN_ADAPTER=1
    M4_LOAD_STAGE=${M4_LOAD_STAGE:-${M4_ROOT}/outputs/h3dreamwam_m3/multisuite_uniform_joint100.pt}
    M4_TAG=${M4_TAG:-miniworld_adapter_h8_s100}
    M4_IO_LR=${M4_IO_LR:-1e-7}
    M4_GATE_LR=${M4_GATE_LR:-1e-6}
    M4_ADAPTER_LR=${M4_ADAPTER_LR:-1e-5}
    M4_TAIL_LR=${M4_TAIL_LR:-1e-8}
    ;;
  crossattn_h8)
    M4_STEPS=${M4_STEPS:-100}
    M4_HORIZON=8
    M4_LAST_ACTION_BLOCKS=2
    M4_FREEZE_BODY=1
    M4_FREEZE_OUTPUT=1
    M4_TRAIN_GATE=0
    M4_TRAIN_ADAPTER=0
    M4_TRAIN_CROSS=1
    M4_LOAD_STAGE=${M4_LOAD_STAGE:-${M4_OUTPUT_DIR}/miniworld_output_h8_s100.pt}
    M4_TAG=${M4_TAG:-language_crossattn_h8_s100}
    M4_IO_LR=${M4_IO_LR:-1e-7}
    M4_GATE_LR=${M4_GATE_LR:-1e-6}
    M4_ADAPTER_LR=${M4_ADAPTER_LR:-1e-6}
    M4_CROSS_LR=${M4_CROSS_LR:-1e-7}
    M4_TAIL_LR=${M4_TAIL_LR:-1e-8}
    ;;
  tail2_h8)
    # Gate-only H8 warmup regressed the fixed val40 sampler by 1.50%. Restart
    # from the accepted joint100 parent and adapt the gate with its ActionDiT
    # tail, checkpointing after 100 steps before spending a longer budget.
    M4_STEPS=${M4_STEPS:-100}
    M4_HORIZON=8
    M4_LAST_ACTION_BLOCKS=2
    M4_FREEZE_BODY=0
    M4_LOAD_STAGE=${M4_LOAD_STAGE:-${M4_ROOT}/outputs/h3dreamwam_m3/multisuite_uniform_joint100.pt}
    M4_TAG=${M4_TAG:-miniworld_tail2_gate_h8_s100}
    M4_IO_LR=${M4_IO_LR:-1e-7}
    M4_GATE_LR=${M4_GATE_LR:-1e-6}
    M4_TAIL_LR=${M4_TAIL_LR:-1e-8}
    M4_NEW_LAYER_LR_SCALE=${M4_NEW_LAYER_LR_SCALE:-0.1}
    ;;
  tail2_h16)
    M4_STEPS=${M4_STEPS:-300}
    M4_HORIZON=16
    M4_LAST_ACTION_BLOCKS=2
    M4_FREEZE_BODY=0
    M4_LOAD_STAGE=${M4_LOAD_STAGE:-${M4_OUTPUT_DIR}/miniworld_tail2_h8_s300.pt}
    M4_TAG=${M4_TAG:-miniworld_tail2_h16_s300}
    M4_IO_LR=${M4_IO_LR:-1e-7}
    M4_GATE_LR=${M4_GATE_LR:-5e-7}
    M4_TAIL_LR=${M4_TAIL_LR:-1e-8}
    M4_NEW_LAYER_LR_SCALE=${M4_NEW_LAYER_LR_SCALE:-0.1}
    ;;
  tail4_h32)
    M4_STEPS=${M4_STEPS:-400}
    M4_HORIZON=32
    M4_LAST_ACTION_BLOCKS=4
    M4_FREEZE_BODY=0
    M4_LOAD_STAGE=${M4_LOAD_STAGE:-${M4_OUTPUT_DIR}/miniworld_tail2_h16_s300.pt}
    M4_TAG=${M4_TAG:-miniworld_tail4_h32_s400}
    M4_IO_LR=${M4_IO_LR:-5e-8}
    M4_GATE_LR=${M4_GATE_LR:-5e-7}
    M4_TAIL_LR=${M4_TAIL_LR:-5e-9}
    M4_NEW_LAYER_LR_SCALE=${M4_NEW_LAYER_LR_SCALE:-0.1}
    ;;
  *)
    echo "unknown M4_PHASE=${M4_PHASE}" >&2
    exit 2
    ;;
esac

M4_NEW_LAYER_LR_SCALE=${M4_NEW_LAYER_LR_SCALE:-1.0}
M4_ADAPTER_LR=${M4_ADAPTER_LR:-1e-6}
M4_CROSS_LR=${M4_CROSS_LR:-1e-7}

M4_REPORT=${M4_OUTPUT_DIR}/${M4_TAG}.json
M4_CHECKPOINT=${M4_OUTPUT_DIR}/${M4_TAG}.pt
if [[ ! -f "${M4_LOAD_STAGE}" ]]; then
  echo "missing input checkpoint: ${M4_LOAD_STAGE}" >&2
  exit 2
fi
if [[ -e "${M4_REPORT}" || -e "${M4_CHECKPOINT}" ]]; then
  echo "refusing to overwrite M4 output: ${M4_TAG}" >&2
  exit 2
fi
if [[ ! -f "${M4_MANIFEST}" ]]; then
  echo "missing uniform manifest: ${M4_MANIFEST}" >&2
  exit 2
fi

mkdir -p "${M4_OUTPUT_DIR}"
cd "${M4_PROJECT}"

M4_EXTRA_ARGS=()
if [[ "${M4_FREEZE_BODY}" == "1" ]]; then
  M4_EXTRA_ARGS+=(--freeze-action-body --freeze-shared-state)
fi
if [[ "${M4_TRAIN_GATE}" == "1" ]]; then
  M4_EXTRA_ARGS+=(--train-video-residual-gates)
fi
if [[ "${M4_TRAIN_ADAPTER}" == "1" ]]; then
  M4_EXTRA_ARGS+=(--train-video-residual-adapters)
fi
if [[ "${M4_TRAIN_CROSS}" == "1" ]]; then
  M4_EXTRA_ARGS+=(--train-cross-attention-output)
fi
if [[ "${M4_FREEZE_OUTPUT}" == "1" ]]; then
  M4_EXTRA_ARGS+=(--freeze-action-output)
fi

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
  --manifest "${M4_MANIFEST}" \
  --rotate-manifest \
  --last-h3-blocks 0 \
  --action-train-stage tail_sharded \
  --last-action-blocks "${M4_LAST_ACTION_BLOCKS}" \
  --action-horizon "${M4_HORIZON}" \
  --separate-expert-clipping \
  --learning-rate "${M4_IO_LR}" \
  --gate-learning-rate "${M4_GATE_LR}" \
  --adapter-learning-rate "${M4_ADAPTER_LR}" \
  --cross-attention-learning-rate "${M4_CROSS_LR}" \
  --tail-learning-rate "${M4_TAIL_LR}" \
  --new-layer-lr-scale "${M4_NEW_LAYER_LR_SCALE}" \
  --load-action-stage "${M4_LOAD_STAGE}" \
  --save-action-stage "${M4_CHECKPOINT}" \
  --dreamwam-action-weighting \
  --steps "${M4_STEPS}" \
  --require-text-only-context \
  "${M4_EXTRA_ARGS[@]}"
