#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
VARIANT="${VARIANT:?set VARIANT=baseline or VARIANT=bidirectional}"
STEPS="${STEPS:-100}"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3-lingbot-port"
LOG_ROOT="${H3_WORKSPACE}/logs/h3-lingbot-port"

COMMON=(
  --model "${H3_WORKSPACE}/models/MiniMax-H3"
  --data-root "${H3_WORKSPACE}/data/v7_dense_h3_cache"
  --manifest "${H3_WORKSPACE}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
  --rotate-manifest
  --steps "${STEPS}"
  --action-horizon 32
  --last-h3-blocks 0
  --last-action-blocks 2
  --learning-rate 1e-5
  --dreamwam-action-weighting
  --dreamwam-world-weighting
  --require-text-only-context
  --dreamwam-exact-action-norm
  --action-init-alpha-scaling
)

case "${VARIANT}" in
  baseline)
    NAME="ab_s${STEPS}_output_only"
    # Match the B arm's FSDP layout so memory behavior is comparable.  The
    # frozen body stays in zero-LR shards; only the shared action output head
    # enters the optimizer, and no reverse-stream gate is enabled.
    EXTRA=(
      --action-train-stage tail_sharded
      --freeze-action-body
      --freeze-shared-state
      --separate-expert-clipping
    )
    ;;
  bidirectional)
    NAME="ab_s${STEPS}_output_plus_bidir_tail2"
    EXTRA=(
      --action-train-stage tail_sharded
      --freeze-action-body
      --freeze-shared-state
      --bidirectional-action-video
      --train-action-to-video-gates
      --gate-learning-rate 1e-4
      --separate-expert-clipping
    )
    ;;
  *)
    echo "VARIANT must be baseline or bidirectional" >&2
    exit 2
    ;;
esac

REPORT="${OUTPUT_ROOT}/${NAME}.json"
STAGE="${OUTPUT_ROOT}/${NAME}.pt"
LOG="${LOG_ROOT}/${NAME}.log"
if [[ -e "${REPORT}" || -e "${STAGE}" ]]; then
  echo "refusing to overwrite existing ${NAME} output" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${H3_WORKSPACE}/tmp"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${H3_WORKSPACE}/tmp"

exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/h3dreamwam/verify_h3dreamwam_fsdp_real.py \
  "${COMMON[@]}" \
  "${EXTRA[@]}" \
  --output "${REPORT}" \
  --save-action-stage "${STAGE}" \
  2>&1 | tee "${LOG}"
