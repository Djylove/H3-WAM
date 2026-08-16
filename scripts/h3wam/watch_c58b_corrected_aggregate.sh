#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# != 5 )) || [[ ! "$1" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 WATCHED_PID ROLLOUT_ROOT GATE D0_CHECKPOINT OUTPUT" >&2
  exit 2
fi

watched_pid="$1"
rollout_root="$(realpath "$2")"
gate="$(realpath "$3")"
d0_checkpoint="$(realpath "$4")"
output="$5"
python_bin="${PYTHON_BIN:-/mnt/h3-wam/runtime/conda-py311/bin/python}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while kill -0 "${watched_pid}" 2>/dev/null; do
  sleep 30
done

[[ ! -e "${output}" ]] || {
  echo "refusing to overwrite corrected C58 aggregate: ${output}" >&2
  exit 1
}
exec "${python_bin}" "${script_dir}/aggregate_c58b_fresh_libero.py" \
  --root "${rollout_root}" --gate "${gate}" \
  --d0-checkpoint "${d0_checkpoint}" --output "${output}"
