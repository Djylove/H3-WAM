#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
VARIANT="${VARIANT:?set VARIANT=baseline or VARIANT=bidirectional}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3-lingbot-port"
LOG_ROOT="${H3_WORKSPACE}/logs/h3-lingbot-port"

case "${VARIANT}" in
  baseline)
    NAME="ab_s${TRAIN_STEPS}_output_only"
    EXTRA=()
    ;;
  bidirectional)
    NAME="ab_s${TRAIN_STEPS}_output_plus_bidir_tail2"
    EXTRA=(--bidirectional-action-video)
    ;;
  *)
    echo "VARIANT must be baseline or bidirectional" >&2
    exit 2
    ;;
esac

STAGE="${OUTPUT_ROOT}/${NAME}.pt"
REPORT="${OUTPUT_ROOT}/${NAME}_val40.json"
LOG="${LOG_ROOT}/${NAME}_val40.log"
if [[ ! -s "${STAGE}" ]]; then
  echo "missing action stage: ${STAGE}" >&2
  exit 2
fi
if [[ -e "${REPORT}" ]]; then
  echo "refusing to overwrite existing ${REPORT}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${H3_WORKSPACE}/tmp"

exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/h3dreamwam/verify_h3dreamwam_fsdp_real.py \
  --model "${H3_WORKSPACE}/models/MiniMax-H3" \
  --data-root "${H3_WORKSPACE}/data/v7_dense_h3_cache" \
  --manifest "${H3_WORKSPACE}/data/v7_multisuite_dense_candidate/manifest_val_stratified40.jsonl" \
  --rotate-manifest \
  --steps 5 \
  --seed 4242 \
  --action-horizon 32 \
  --last-h3-blocks 0 \
  --last-action-blocks 2 \
  --action-train-stage head \
  --load-action-stage "${STAGE}" \
  --eval-only \
  --dreamwam-action-weighting \
  --dreamwam-world-weighting \
  --require-text-only-context \
  --dreamwam-exact-action-norm \
  --action-init-alpha-scaling \
  "${EXTRA[@]}" \
  --output "${REPORT}" \
  2>&1 | tee "${LOG}"
