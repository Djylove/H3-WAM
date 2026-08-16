#!/usr/bin/env bash
set -euo pipefail

# All normal rollout_libero.py arguments are forwarded.  The C57 entry point
# rejects wait/replan/ensemble settings that would violate reset→predict→obs4→commit8.
C57_LIBERO_PYTHON=${C57_LIBERO_PYTHON:?set C57_LIBERO_PYTHON}
exec "${C57_LIBERO_PYTHON}" scripts/h3wam/rollout_c57_lingbot_libero.py "$@"
