#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/h3-int8-native/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TMP_ROOT="${TMP_ROOT:-${H3_WORKSPACE}/tmp/h3-int8-starwam-action}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export TMPDIR="${TMP_ROOT}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${TMP_ROOT}"
cd "${PROJECT_ROOT}"

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node "${NPROC_PER_NODE}" \
  scripts/h3wam/train_h3_int8_starwam_action.py "$@"
