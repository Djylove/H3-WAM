#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NPROC_PER_NODE=""
WAND_KEY_ARG=""
CONDA_SH="${CONDA_SH:-/mnt/cpfs/wxy/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-fastwam}"
SKIP_CONDA="${SKIP_CONDA:-0}"
EXTRA_ARGS=()

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_zero1_dlc_multinode.sh [nproc_per_node] [options] [hydra_overrides...]

Options:
  --wandb-key <key>    set WANDB_API_KEY for this run
  --conda-sh <path>    conda.sh path
  --conda-env <name>   conda env name
  --skip-conda         skip conda activation
  --help               show this help

Examples:
  bash scripts/train_zero1_dlc_multinode.sh 8 task=robotwin_hierarchical_3cam_384_1e-4
  bash scripts/train_zero1_dlc_multinode.sh --wandb-key <key> task=robotwin_hierarchical_3cam_384_1e-4
EOF
}

while (($#)); do
  case "$1" in
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

if [[ -z "${NPROC_PER_NODE}" ]]; then
  if [[ -n "${PET_NPROC_PER_NODE:-}" ]]; then
    NPROC_PER_NODE="${PET_NPROC_PER_NODE}"
  elif [[ -n "${KUBERNETES_CONTAINER_RESOURCE_GPU:-}" ]]; then
    NPROC_PER_NODE="${KUBERNETES_CONTAINER_RESOURCE_GPU}"
  elif [[ -n "${NPROC_PER_NODE:-}" ]]; then
    NPROC_PER_NODE="${NPROC_PER_NODE}"
  else
    echo "Error: failed to detect nproc_per_node from args or DLC env." >&2
    echo "Hint: pass an explicit leading integer like '8', or ensure PET_NPROC_PER_NODE is set." >&2
    exit 1
  fi
fi

detect_nnodes() {
  if [[ -n "${WORLD_SIZE:-}" ]] && is_integer "${WORLD_SIZE}"; then
    echo "${WORLD_SIZE}"
    return
  fi
  if [[ -n "${PET_NNODES:-}" ]] && is_integer "${PET_NNODES}"; then
    echo "${PET_NNODES}"
    return
  fi
  if [[ -n "${DLC_WORKER_NUM:-}" ]] && is_integer "${DLC_WORKER_NUM}"; then
    echo "${DLC_WORKER_NUM}"
    return
  fi
  if [[ -n "${WORKER_NUM:-}" ]] && is_integer "${WORKER_NUM}"; then
    echo "${WORKER_NUM}"
    return
  fi
  echo "1"
}

detect_node_rank() {
  if [[ -n "${RANK:-}" ]] && is_integer "${RANK}"; then
    echo "${RANK}"
    return
  fi
  if [[ -n "${PET_NODE_RANK:-}" ]] && is_integer "${PET_NODE_RANK}"; then
    if (( PET_NODE_RANK >= 1 )); then
      echo $((PET_NODE_RANK - 1))
    else
      echo "${PET_NODE_RANK}"
    fi
    return
  fi
  if [[ -n "${DLC_WORKER_INDEX:-}" ]] && is_integer "${DLC_WORKER_INDEX}"; then
    echo "${DLC_WORKER_INDEX}"
    return
  fi
  if [[ -n "${WORKER_INDEX:-}" ]] && is_integer "${WORKER_INDEX}"; then
    echo "${WORKER_INDEX}"
    return
  fi
  if [[ -n "${NODE_RANK:-}" ]] && is_integer "${NODE_RANK}"; then
    echo "${NODE_RANK}"
    return
  fi
  echo "0"
}

first_host_from_csv() {
  local csv="$1"
  local first="${csv%%,*}"
  first="${first// /}"
  if [[ "${first}" == *:* ]]; then
    first="${first%%:*}"
  fi
  echo "${first}"
}

detect_master_addr() {
  if [[ -n "${MASTER_ADDR:-}" ]]; then
    echo "${MASTER_ADDR}"
    return
  fi
  if [[ -n "${PET_MASTER_ADDR:-}" ]]; then
    echo "${PET_MASTER_ADDR}"
    return
  fi
  if [[ -n "${DLC_WORKER_HOSTS:-}" ]]; then
    first_host_from_csv "${DLC_WORKER_HOSTS}"
    return
  fi
  if [[ -n "${PAI_HOSTS:-}" ]]; then
    first_host_from_csv "${PAI_HOSTS}"
    return
  fi
  if [[ -n "${WORKER_HOSTS:-}" ]]; then
    first_host_from_csv "${WORKER_HOSTS}"
    return
  fi
  if [[ -n "${PAI_TASK_ROLE_worker_0_HOST:-}" ]]; then
    echo "${PAI_TASK_ROLE_worker_0_HOST}"
    return
  fi
  echo "127.0.0.1"
}

detect_master_port() {
  if [[ -n "${MASTER_PORT:-}" ]] && is_integer "${MASTER_PORT}"; then
    echo "${MASTER_PORT}"
    return
  fi
  if [[ -n "${PET_MASTER_PORT:-}" ]] && is_integer "${PET_MASTER_PORT}"; then
    echo "${PET_MASTER_PORT}"
    return
  fi
  echo "29500"
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

if ! is_integer "${NPROC_PER_NODE}" || (( NPROC_PER_NODE <= 0 )); then
  echo "Error: nproc_per_node must be a positive integer, got ${NPROC_PER_NODE}" >&2
  exit 1
fi

if ! is_integer "${NUM_MACHINES}" || (( NUM_MACHINES <= 0 )); then
  echo "Error: invalid num_machines=${NUM_MACHINES}" >&2
  exit 1
fi

if ! is_integer "${MACHINE_RANK}" || (( MACHINE_RANK < 0 )) || (( MACHINE_RANK >= NUM_MACHINES )); then
  echo "Error: invalid machine_rank=${MACHINE_RANK} for num_machines=${NUM_MACHINES}" >&2
  exit 1
fi

if ! is_integer "${MAIN_PROCESS_PORT}" || (( MAIN_PROCESS_PORT <= 0 )); then
  echo "Error: invalid main_process_port=${MAIN_PROCESS_PORT}" >&2
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

# Let Accelerator initialize its DeepSpeed plugin under torchrun without using
# `accelerate launch`, which is the part that falls back to localhost on DLC.
export ACCELERATE_USE_DEEPSPEED="true"
export ACCELERATE_MIXED_PRECISION="no"
export ACCELERATE_CONFIG_DS_FIELDS="deepspeed_config_file,zero3_init_flag"
export ACCELERATE_DEEPSPEED_CONFIG_FILE="${ACCELERATE_DEEPSPEED_CONFIG_FILE:-scripts/ds_configs/ds_zero1_config.json}"
export ACCELERATE_DEEPSPEED_ZERO3_INIT="${ACCELERATE_DEEPSPEED_ZERO3_INIT:-false}"

# Keep logging readable and match common multinode defaults.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

WORLD_PROCESSES=$((NUM_MACHINES * NPROC_PER_NODE))

echo "[dlc-torchrun-zero1] project_root=${PROJECT_ROOT}"
echo "[dlc-torchrun-zero1] nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK}"
echo "[dlc-torchrun-zero1] world_processes=${WORLD_PROCESSES} master=${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT}"
echo "[dlc-torchrun-zero1] task_basename=${TASK_BASENAME} run_id=${RUN_ID}"
echo "[dlc-torchrun-zero1] ds_config=${ACCELERATE_DEEPSPEED_CONFIG_FILE} zero3_init=${ACCELERATE_DEEPSPEED_ZERO3_INIT}"
echo "[dlc-torchrun-zero1] env WORLD_SIZE=${WORLD_SIZE:-unset} RANK=${RANK:-unset} PET_NNODES=${PET_NNODES:-unset} PET_NODE_RANK=${PET_NODE_RANK:-unset} PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE:-unset}"

python -m torch.distributed.run \
  --nnodes="${NUM_MACHINES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${MACHINE_RANK}" \
  --master_addr="${MAIN_PROCESS_IP}" \
  --master_port="${MAIN_PROCESS_PORT}" \
  scripts/train.py \
  "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"
