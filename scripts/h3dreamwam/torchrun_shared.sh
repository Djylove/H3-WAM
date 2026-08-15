#!/usr/bin/env bash
set -euo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
DIFFUSERS_H3_ROOT="${DIFFUSERS_H3_ROOT:-${PROJECT_ROOT}/third_party/diffusers_h3}"

test -d "${DIFFUSERS_H3_ROOT}/src/diffusers"
export PYTHONPATH="${DIFFUSERS_H3_ROOT}/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m torch.distributed.run "$@"
