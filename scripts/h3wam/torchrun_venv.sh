#!/usr/bin/env bash
set -euo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/.venv/bin/python}"
test -x "${PYTHON_BIN}"
exec "${PYTHON_BIN}" -m torch.distributed.run "$@"
