#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 || $# > 5 )); then
  echo "usage: $0 MANIFEST OUTPUT_SUBDIR LAYERS_CSV CAPTURE_TOKENS [NUM_GPUS]" >&2
  exit 2
fi

manifest=$1
output_subdir=$2
layers_csv=$3
capture_tokens=$4
num_gpus=${5:-8}

h3_workspace=/mnt/h3-wam
project_root=${h3_workspace}/project
python_bin=${h3_workspace}/runtime/h3-int8-native/bin/python
checkpoint=${h3_workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
cache_root=${h3_workspace}/data/v8_frameindexed_h3_cache
log_root=${h3_workspace}/logs/h3-int8-native/${output_subdir}

if [[ ! -x "${python_bin}" || ! -f "${checkpoint}" || ! -f "${manifest}" ]]; then
  echo "missing INT8 runtime, checkpoint, or manifest under ${h3_workspace}" >&2
  exit 1
fi
if (( num_gpus <= 0 )); then
  echo "NUM_GPUS must be positive" >&2
  exit 2
fi

# Some A800 containers append /usr/local/cuda/lib64.  With the pinned
# torch-2.10.0+cu130 environment that path selects an incompatible CUBLAS and
# makes even a BF16 GEMM fail.  Use only the driver libraries supplied by the
# container.  This changes process lookup only and never writes to /usr/local.
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export PYTHONPATH=${project_root}/src

IFS=',' read -r -a layers <<< "${layers_csv}"
mkdir -p "${log_root}"
: > "${log_root}/worker_pids.txt"

worker_pids=()
for ((rank = 0; rank < num_gpus; rank++)); do
  CUDA_VISIBLE_DEVICES=${rank} "${python_bin}" \
    "${project_root}/scripts/h3wam/precompute_h3_int8_features.py" \
    "${manifest}" \
    --cache-root "${cache_root}" \
    --h3-checkpoint "${checkpoint}" \
    --output-subdir "${output_subdir}" \
    --layers "${layers[@]}" \
    --capture-token-count "${capture_tokens}" \
    --num-shards "${num_gpus}" \
    --shard-index "${rank}" \
    --progress-every 100 \
    > "${log_root}/worker_${rank}.log" 2>&1 &
  worker_pids+=("$!")
  echo "$!" >> "${log_root}/worker_pids.txt"
done

status=0
for worker_pid in "${worker_pids[@]}"; do
  if ! wait "${worker_pid}"; then
    status=1
  fi
done
exit "${status}"
