#!/usr/bin/env bash
set -euo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
LIBERO_ROOT="${LIBERO_ROOT:-${H3_WORKSPACE}/simulator/LIBERO}"
SIM_PYTHON_BIN="${SIM_PYTHON_BIN:-${H3_WORKSPACE}/.venv-libero/bin/python}"
if [[ -z "${SIM_SITE_PACKAGES:-}" ]]; then
  test -x "${SIM_PYTHON_BIN}"
  SIM_SITE_PACKAGES="$("${SIM_PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
fi
GL_ROOT="${GL_ROOT:-${H3_WORKSPACE}/runtime/gl_root/usr/lib/x86_64-linux-gnu}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/.venv/bin/python}"

export LIBERO_CONFIG_PATH="${H3_WORKSPACE}/config/libero"
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
# LIBERO's trusted init-state files are legacy NumPy pickles. PyTorch 2.6+
# otherwise changes torch.load's implicit default to weights_only=True.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1"
export LD_LIBRARY_PATH="${GL_ROOT}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${LIBERO_ROOT}:${SIM_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"

test -x "${PYTHON_BIN}"
test -f "${LIBERO_CONFIG_PATH}/config.yaml"
test -f "${GL_ROOT}/libOSMesa.so.8"

if [[ $# -eq 0 ]]; then
  exec "${PYTHON_BIN}"
fi
exec "$@"
