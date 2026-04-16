#!/usr/bin/env bash
set -euo pipefail

# Generic launcher for Alibaba Cloud DLC multi-node training with train_zero1.sh.
# DLC typically starts this script on every node and provides distributed env vars
# such as WORLD_SIZE / RANK / MASTER_ADDR / MASTER_PORT.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_SH_DEFAULT="/mnt/cpfs/wxy/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_DEFAULT="fastwam"

TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_zero1.sh"
CONDA_SH="${CONDA_SH_DEFAULT}"
CONDA_ENV="${CONDA_ENV_DEFAULT}"
SKIP_CONDA="0"
WAND_KEY_ARG=""

NPROC_PER_NODE=""
MASTER_PORT_OVERRIDE=""
NNODES_OVERRIDE=""
NODE_RANK_OVERRIDE=""
MASTER_ADDR_OVERRIDE=""
TASK_OVERRIDE=""

EXTRA_OVERRIDES=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/launch_train_zero1_dlc_multinode.sh [options] [-- <hydra overrides>]

Options:
  --task <override>         task override, e.g. task=robotwin_hierarchical_3cam_384_1e-4
  --nnodes <int>            force total node count
  --node-rank <int>         force current node rank (0-based)
  --master-addr <host>      force master address
  --master-port <int>       force master port
  --nproc-per-node <int>    GPUs per node; default is auto-detect
  --train-script <path>     override train script path
  --wandb-key <key>         set WANDB_API_KEY for this run
  --conda-sh <path>         conda.sh path
  --conda-env <name>        conda env name
  --skip-conda              skip conda activation
  --help                    show this help

Examples:
  bash scripts/launch_train_zero1_dlc_multinode.sh \
    --task task=robotwin_hierarchical_3cam_384_1e-4

  bash scripts/launch_train_zero1_dlc_multinode.sh \
    --task task=robotwin_hierarchical_3cam_384_1e-4 \
    -- learning_rate=1e-4 dataloader.batch_size=8
EOF
}

is_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

while (($#)); do
  case "$1" in
    --task)
      TASK_OVERRIDE="$2"
      shift 2
      ;;
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
    --nproc-per-node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --train-script)
      TRAIN_SCRIPT="$2"
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
    --)
      shift
      EXTRA_OVERRIDES=("$@")
      break
      ;;
    *)
      EXTRA_OVERRIDES+=("$1")
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

first_host_from_csv() {
  local csv="$1"
  local first="${csv%%,*}"
  first="${first// /}"
  if [[ "${first}" == *:* ]]; then
    first="${first%%:*}"
  fi
  echo "${first}"
}

detect_nnodes() {
  if [[ -n "${NNODES_OVERRIDE}" ]]; then
    echo "${NNODES_OVERRIDE}"
    return
  fi
  if [[ -n "${WORLD_SIZE:-}" ]] && is_int "${WORLD_SIZE}"; then
    echo "${WORLD_SIZE}"
    return
  fi
  if [[ -n "${PET_NNODES:-}" ]] && is_int "${PET_NNODES}"; then
    echo "${PET_NNODES}"
    return
  fi
  if [[ -n "${DLC_WORKER_NUM:-}" ]] && is_int "${DLC_WORKER_NUM}"; then
    echo "${DLC_WORKER_NUM}"
    return
  fi
  if [[ -n "${WORKER_NUM:-}" ]] && is_int "${WORKER_NUM}"; then
    echo "${WORKER_NUM}"
    return
  fi
  if [[ -n "${NNODES:-}" ]] && is_int "${NNODES}"; then
    echo "${NNODES}"
    return
  fi
  echo "1"
}

detect_node_rank() {
  if [[ -n "${NODE_RANK_OVERRIDE}" ]]; then
    echo "${NODE_RANK_OVERRIDE}"
    return
  fi
  if [[ -n "${RANK:-}" ]] && is_int "${RANK}"; then
    echo "${RANK}"
    return
  fi
  if [[ -n "${PET_NODE_RANK:-}" ]] && is_int "${PET_NODE_RANK}"; then
    if (( PET_NODE_RANK >= 1 )); then
      echo $((PET_NODE_RANK - 1))
    else
      echo "${PET_NODE_RANK}"
    fi
    return
  fi
  if [[ -n "${DLC_WORKER_INDEX:-}" ]] && is_int "${DLC_WORKER_INDEX}"; then
    echo "${DLC_WORKER_INDEX}"
    return
  fi
  if [[ -n "${WORKER_INDEX:-}" ]] && is_int "${WORKER_INDEX}"; then
    echo "${WORKER_INDEX}"
    return
  fi
  if [[ -n "${NODE_RANK:-}" ]] && is_int "${NODE_RANK}"; then
    echo "${NODE_RANK}"
    return
  fi
  if [[ -n "${MACHINE_RANK:-}" ]] && is_int "${MACHINE_RANK}"; then
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
  echo ""
}

detect_master_port() {
  if [[ -n "${MASTER_PORT_OVERRIDE}" ]]; then
    echo "${MASTER_PORT_OVERRIDE}"
    return
  fi
  if [[ -n "${MASTER_PORT:-}" ]] && is_int "${MASTER_PORT}"; then
    echo "${MASTER_PORT}"
    return
  fi
  if [[ -n "${PET_MASTER_PORT:-}" ]] && is_int "${PET_MASTER_PORT}"; then
    echo "${PET_MASTER_PORT}"
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

NNODES_DETECTED="$(detect_nnodes)"
NODE_RANK_DETECTED="$(detect_node_rank)"
MASTER_ADDR_DETECTED="$(detect_master_addr)"
MASTER_PORT_DETECTED="$(detect_master_port)"

if ! is_int "${NNODES_DETECTED}" || (( NNODES_DETECTED <= 0 )); then
  echo "Error: invalid nnodes=${NNODES_DETECTED}" >&2
  exit 1
fi

if ! is_int "${NODE_RANK_DETECTED}" || (( NODE_RANK_DETECTED < 0 )); then
  echo "Error: invalid node_rank=${NODE_RANK_DETECTED}" >&2
  exit 1
fi

if (( NODE_RANK_DETECTED >= NNODES_DETECTED )); then
  echo "Error: node_rank=${NODE_RANK_DETECTED} must be in [0, $((NNODES_DETECTED - 1))]" >&2
  exit 1
fi

if [[ -z "${MASTER_ADDR_DETECTED}" ]]; then
  echo "Error: failed to detect master address from DLC env vars." >&2
  exit 1
fi

if ! is_int "${MASTER_PORT_DETECTED}" || (( MASTER_PORT_DETECTED <= 0 )); then
  echo "Error: invalid master_port=${MASTER_PORT_DETECTED}" >&2
  exit 1
fi

if (( NNODES_DETECTED > 1 )) && [[ "${MASTER_ADDR_DETECTED}" =~ ^(127\.0\.0\.1|localhost)$ ]]; then
  echo "Error: master_addr=${MASTER_ADDR_DETECTED} is invalid for multi-node training." >&2
  exit 1
fi

VISIBLE_GPUS="$(detect_visible_gpus)"
if ! is_int "${VISIBLE_GPUS}"; then
  VISIBLE_GPUS="0"
fi
if (( VISIBLE_GPUS <= 0 )); then
  echo "Error: no visible GPUs detected on this node." >&2
  exit 1
fi

if [[ -z "${NPROC_PER_NODE}" ]]; then
  if [[ -n "${GPUS_PER_NODE:-}" ]] && is_int "${GPUS_PER_NODE}"; then
    NPROC_PER_NODE="${GPUS_PER_NODE}"
  else
    NPROC_PER_NODE="${VISIBLE_GPUS}"
  fi
fi

if ! is_int "${NPROC_PER_NODE}" || (( NPROC_PER_NODE <= 0 )); then
  echo "Error: invalid nproc_per_node=${NPROC_PER_NODE}" >&2
  exit 1
fi

if (( NPROC_PER_NODE > VISIBLE_GPUS )); then
  echo "Error: nproc_per_node=${NPROC_PER_NODE} exceeds visible GPUs=${VISIBLE_GPUS}" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "Error: train script not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ "${SKIP_CONDA}" == "0" ]]; then
  if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
  else
    echo "Warning: conda.sh not found at ${CONDA_SH}, continuing without conda activation." >&2
  fi
fi

export NNODES="${NNODES_DETECTED}"
export NODE_RANK="${NODE_RANK_DETECTED}"
export MASTER_ADDR="${MASTER_ADDR_DETECTED}"
export MASTER_PORT="${MASTER_PORT_DETECTED}"

LAUNCH_ARGS=("${NPROC_PER_NODE}")
if [[ -n "${TASK_OVERRIDE}" ]]; then
  LAUNCH_ARGS+=("${TASK_OVERRIDE}")
fi
LAUNCH_ARGS+=("${EXTRA_OVERRIDES[@]}")

echo "[dlc-zero1] project_root=${PROJECT_ROOT}"
echo "[dlc-zero1] train_script=${TRAIN_SCRIPT}"
echo "[dlc-zero1] nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE}"
echo "[dlc-zero1] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[dlc-zero1] visible_gpus=${VISIBLE_GPUS} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[dlc-zero1] task_override=${TASK_OVERRIDE:-unset}"
echo "[dlc-zero1] source_env WORLD_SIZE=${WORLD_SIZE:-unset} RANK=${RANK:-unset} MASTER_ADDR=${MASTER_ADDR:-unset} MASTER_PORT=${MASTER_PORT:-unset}"
echo "[dlc-zero1] source_env PET_NNODES=${PET_NNODES:-unset} PET_NODE_RANK=${PET_NODE_RANK:-unset} PET_MASTER_ADDR=${PET_MASTER_ADDR:-unset}"
echo "[dlc-zero1] source_env DLC_WORKER_NUM=${DLC_WORKER_NUM:-unset} DLC_WORKER_INDEX=${DLC_WORKER_INDEX:-unset} DLC_WORKER_HOSTS=${DLC_WORKER_HOSTS:-unset}"

bash "${TRAIN_SCRIPT}" "${LAUNCH_ARGS[@]}"
