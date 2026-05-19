#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NPROC_PER_NODE=""
NNODES_OVERRIDE=""
NODE_RANK_OVERRIDE=""
MASTER_ADDR_OVERRIDE=""
MASTER_PORT_OVERRIDE=""
WAND_KEY_ARG=""
CONDA_SH="${CONDA_SH:-/workspace/mnt/data/wxy/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-fastwam}"
SKIP_CONDA="${SKIP_CONDA:-0}"
EXTRA_ARGS=()

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_zero1_mobile_multinode.sh [nproc_per_node] [options] [hydra_overrides...]

Options:
  --nnodes <int>          total node count; default: PET_NNODES, then NNODES
  --node-rank <int>       current node rank, 0-based; default: PET_NODE_RANK, then RANK/NODE_RANK
  --master-addr <host>    rank-0 node address; default: PET_MASTER_ADDR, then MASTER_ADDR
  --master-port <int>     rendezvous/master port; default: PET_MASTER_PORT, then MASTER_PORT, then 29500
  --wandb-key <key>       set WANDB_API_KEY for this run
  --conda-sh <path>       conda.sh path
  --conda-env <name>      conda env name
  --skip-conda            skip conda activation
  --help                  show this help

Examples:
  bash scripts/train_zero1_mobile_multinode.sh \
    --wandb-key <key> \
    task=robotwin_hierarchical_3cam_384_1e-4

  bash scripts/train_zero1_mobile_multinode.sh 8 \
    --nnodes 2 \
    --node-rank "${RANK}" \
    --master-addr "${MASTER_ADDR}" \
    task=robotwin_hierarchical_3cam_384_1e-4
EOF
}

while (($#)); do
  case "$1" in
    --nnodes)
      NNODES_OVERRIDE="$2"
      shift 2
      ;;
    --node-rank)
      NODE_RANK_OVERRIDE="$2"
      shift 2
      ;;
    --master-addr)
      MASTER_ADDR_OVERRIDE="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT_OVERRIDE="$2"
      shift 2
      ;;
    --wandb-key)
      WAND_KEY_ARG="$2"
      shift 2
      ;;
    --conda-sh)
      CONDA_SH="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --skip-conda)
      SKIP_CONDA="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      if [[ -z "${NPROC_PER_NODE}" ]] && is_integer "$1"; then
        NPROC_PER_NODE="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -n "${WAND_KEY_ARG}" ]]; then
  export WANDB_API_KEY="${WAND_KEY_ARG}"
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "Error: WANDB_API_KEY is empty. Please pass --wandb-key or set env." >&2
  exit 1
fi

detect_nnodes() {
  if [[ -n "${NNODES_OVERRIDE}" ]]; then
    echo "${NNODES_OVERRIDE}"
    return
  fi
  # Mobile Cloud predefines PET_NNODES as the instance/node count.
  if [[ -n "${PET_NNODES:-}" ]] && is_integer "${PET_NNODES}"; then
    echo "${PET_NNODES}"
    return
  fi
  if [[ -n "${NNODES:-}" ]] && is_integer "${NNODES}"; then
    echo "${NNODES}"
    return
  fi
  # On Mobile Cloud WORLD_SIZE is the total GPU count. Derive node count only
  # when per-node process count is also available.
  if [[ -n "${WORLD_SIZE:-}" ]] && is_integer "${WORLD_SIZE}" && [[ -n "${PET_NPROC_PER_NODE:-}" ]] && is_integer "${PET_NPROC_PER_NODE}" && (( PET_NPROC_PER_NODE > 0 )); then
    echo $((WORLD_SIZE / PET_NPROC_PER_NODE))
    return
  fi
  echo "1"
}

detect_node_rank() {
  if [[ -n "${NODE_RANK_OVERRIDE}" ]]; then
    echo "${NODE_RANK_OVERRIDE}"
    return
  fi
  # Mobile Cloud predefines PET_NODE_RANK as the instance/node rank.
  if [[ -n "${PET_NODE_RANK:-}" ]] && is_integer "${PET_NODE_RANK}"; then
    echo "${PET_NODE_RANK}"
    return
  fi
  if [[ -n "${RANK:-}" ]] && is_integer "${RANK}"; then
    echo "${RANK}"
    return
  fi
  if [[ -n "${NODE_RANK:-}" ]] && is_integer "${NODE_RANK}"; then
    echo "${NODE_RANK}"
    return
  fi
  if [[ -n "${MACHINE_RANK:-}" ]] && is_integer "${MACHINE_RANK}"; then
    echo "${MACHINE_RANK}"
    return
  fi
  echo "0"
}

detect_master_addr() {
  if [[ -n "${MASTER_ADDR_OVERRIDE}" ]]; then
    echo "${MASTER_ADDR_OVERRIDE}"
    return
  fi
  if [[ -n "${PET_MASTER_ADDR:-}" ]]; then
    echo "${PET_MASTER_ADDR}"
    return
  fi
  if [[ -n "${MASTER_ADDR:-}" ]]; then
    echo "${MASTER_ADDR}"
    return
  fi
  echo "127.0.0.1"
}

detect_master_port() {
  if [[ -n "${MASTER_PORT_OVERRIDE}" ]]; then
    echo "${MASTER_PORT_OVERRIDE}"
    return
  fi
  if [[ -n "${PET_MASTER_PORT:-}" ]] && is_integer "${PET_MASTER_PORT}"; then
    echo "${PET_MASTER_PORT}"
    return
  fi
  if [[ -n "${MASTER_PORT:-}" ]] && is_integer "${MASTER_PORT}"; then
    echo "${MASTER_PORT}"
    return
  fi
  echo "29500"
}

detect_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if [[ "${CUDA_VISIBLE_DEVICES}" == "all" ]]; then
      :
    elif [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
      echo "0"
      return
    else
      local csv="${CUDA_VISIBLE_DEVICES}"
      local count=1
      while [[ "${csv}" == *,* ]]; do
        csv="${csv#*,}"
        count=$((count + 1))
      done
      echo "${count}"
      return
    fi
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
    return
  fi
  echo "0"
}

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

NUM_MACHINES="$(detect_nnodes)"
MACHINE_RANK="$(detect_node_rank)"
MAIN_PROCESS_IP="$(detect_master_addr)"
MAIN_PROCESS_PORT="$(detect_master_port)"

if ! is_integer "${NUM_MACHINES}" || (( NUM_MACHINES <= 0 )); then
  echo "Error: invalid nnodes=${NUM_MACHINES}" >&2
  exit 1
fi

if ! is_integer "${MACHINE_RANK}" || (( MACHINE_RANK < 0 )) || (( MACHINE_RANK >= NUM_MACHINES )); then
  echo "Error: invalid node_rank=${MACHINE_RANK} for nnodes=${NUM_MACHINES}" >&2
  exit 1
fi

if [[ -z "${MAIN_PROCESS_IP}" ]]; then
  echo "Error: MASTER_ADDR is empty. Please set MASTER_ADDR or pass --master-addr." >&2
  exit 1
fi

if (( NUM_MACHINES > 1 )) && [[ "${MAIN_PROCESS_IP}" =~ ^(127\.0\.0\.1|localhost)$ ]]; then
  echo "Error: master_addr=${MAIN_PROCESS_IP} is invalid for multi-node training." >&2
  exit 1
fi

if ! is_integer "${MAIN_PROCESS_PORT}" || (( MAIN_PROCESS_PORT <= 0 )); then
  echo "Error: invalid master_port=${MAIN_PROCESS_PORT}" >&2
  exit 1
fi

VISIBLE_GPUS="$(detect_visible_gpus)"
if ! is_integer "${VISIBLE_GPUS}"; then
  VISIBLE_GPUS="0"
fi

if [[ -z "${NPROC_PER_NODE}" ]]; then
  if [[ -n "${PET_NPROC_PER_NODE:-}" ]] && is_integer "${PET_NPROC_PER_NODE}"; then
    NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
  elif [[ -n "${GPUS_PER_NODE:-}" ]] && is_integer "${GPUS_PER_NODE}"; then
    NPROC_PER_NODE="${GPUS_PER_NODE}"
  elif (( VISIBLE_GPUS > 0 )); then
    NPROC_PER_NODE="${VISIBLE_GPUS}"
  else
    echo "Error: failed to detect GPUs per node. Pass a leading integer, e.g. '8'." >&2
    exit 1
  fi
fi

if ! is_integer "${NPROC_PER_NODE}" || (( NPROC_PER_NODE <= 0 )); then
  echo "Error: nproc_per_node must be a positive integer, got ${NPROC_PER_NODE}" >&2
  exit 1
fi

if (( VISIBLE_GPUS > 0 && NPROC_PER_NODE > VISIBLE_GPUS )); then
  echo "Error: nproc_per_node=${NPROC_PER_NODE} exceeds visible GPUs=${VISIBLE_GPUS}" >&2
  exit 1
fi

TASK_BASENAME="train"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if (( i + 1 < ${#EXTRA_ARGS[@]} )); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
  esac
done

cd "${PROJECT_ROOT}"

if [[ "${SKIP_CONDA}" == "0" ]]; then
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
  else
    echo "Error: conda.sh not found at ${CONDA_SH}" >&2
    exit 1
  fi
fi

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-180}"
    RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"

    export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
    export RUN_ID_SYNC_PORT
    export RUN_ID_SYNC_TIMEOUT
    export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
    export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
    export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"

    RUN_ID="$(
      python - <<'PY'
import datetime
import os
from datetime import timedelta

import torch.distributed as dist

host = os.environ["RUN_ID_SYNC_HOST"]
port = int(os.environ["RUN_ID_SYNC_PORT"])
timeout_s = int(os.environ["RUN_ID_SYNC_TIMEOUT"])
machine_rank = int(os.environ["RUN_ID_SYNC_MACHINE_RANK"])
num_machines = int(os.environ["RUN_ID_SYNC_NUM_MACHINES"])
task_basename = os.environ.get("RUN_ID_SYNC_TASK_BASENAME", "train")

store = dist.TCPStore(
    host_name=host,
    port=port,
    world_size=num_machines,
    is_master=(machine_rank == 0),
    timeout=timedelta(seconds=timeout_s),
)
key = f"run_id::{task_basename}"
if machine_rank == 0:
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store.set(key, run_id)
run_id = store.get(key).decode("utf-8")
print(run_id)
PY
    )"

    echo "[run_id_sync] mode=tcpstore host=${RUN_ID_SYNC_HOST} port=${RUN_ID_SYNC_PORT} timeout_s=${RUN_ID_SYNC_TIMEOUT} run_id=${RUN_ID}"
  fi
fi

# Let Accelerator initialize its DeepSpeed plugin under torchrun.
export ACCELERATE_USE_DEEPSPEED="true"
export ACCELERATE_MIXED_PRECISION="no"
export ACCELERATE_CONFIG_DS_FIELDS="deepspeed_config_file,zero3_init_flag"
export ACCELERATE_DEEPSPEED_CONFIG_FILE="${ACCELERATE_DEEPSPEED_CONFIG_FILE:-scripts/ds_configs/ds_zero1_config.json}"
export ACCELERATE_DEEPSPEED_ZERO3_INIT="${ACCELERATE_DEEPSPEED_ZERO3_INIT:-false}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

WORLD_PROCESSES=$((NUM_MACHINES * NPROC_PER_NODE))

echo "[mobile-torchrun-zero1] project_root=${PROJECT_ROOT}"
echo "[mobile-torchrun-zero1] nproc_per_node=${NPROC_PER_NODE} nnodes=${NUM_MACHINES} node_rank=${MACHINE_RANK}"
echo "[mobile-torchrun-zero1] world_processes=${WORLD_PROCESSES} rdzv_endpoint=${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT}"
echo "[mobile-torchrun-zero1] task_basename=${TASK_BASENAME} run_id=${RUN_ID}"
echo "[mobile-torchrun-zero1] ds_config=${ACCELERATE_DEEPSPEED_CONFIG_FILE} zero3_init=${ACCELERATE_DEEPSPEED_ZERO3_INIT}"
echo "[mobile-torchrun-zero1] source_env PET_NNODES=${PET_NNODES:-unset} PET_NODE_RANK=${PET_NODE_RANK:-unset} PET_MASTER_ADDR=${PET_MASTER_ADDR:-unset} PET_MASTER_PORT=${PET_MASTER_PORT:-unset} PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE:-unset}"
echo "[mobile-torchrun-zero1] source_env WORLD_SIZE=${WORLD_SIZE:-unset} RANK=${RANK:-unset} MASTER_ADDR=${MASTER_ADDR:-unset} MASTER_PORT=${MASTER_PORT:-unset} GPUS_PER_NODE=${GPUS_PER_NODE:-unset}"

torchrun \
  --nnodes="${NUM_MACHINES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${MACHINE_RANK}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT}" \
  scripts/train.py \
  "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"
