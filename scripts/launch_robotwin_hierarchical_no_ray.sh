#!/usr/bin/env bash
set -euo pipefail

# No-Ray launcher for multi-node training in official image environments.
# Run this script once on EACH node with different --node-rank.

PROJECT_ROOT="/mnt/cpfs/wxy/FastWAM"
CONDA_SH="/mnt/cpfs/wxy/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="fastwam"

NNODES="4"
NODE_RANK=""
MASTER_ADDR=""
MASTER_PORT="29500"
NPROC_PER_NODE="8"
TRAIN_SCRIPT="scripts/train_zero1.sh"
TASK_OVERRIDE="task=robotwin_hierarchical_3cam_384_1e-4"

usage() {
  cat <<'EOF'
Usage (run on each node):
  bash scripts/launch_robotwin_hierarchical_no_ray.sh \
    --node-rank <0..N-1> \
    --master-addr <master_ip> \
    [--nnodes 4] [--nproc-per-node 8] [--master-port 29500] \
    [--wandb-key <key>] [--task task=robotwin_hierarchical_3cam_384_1e-4] [-- <extra hydra overrides>]

Examples:
  # Node 0 (master)
  bash scripts/launch_robotwin_hierarchical_no_ray.sh --node-rank 0 --master-addr 10.0.0.1 --wandb-key xxxx

  # Node 1/2/3
  bash scripts/launch_robotwin_hierarchical_no_ray.sh --node-rank 1 --master-addr 10.0.0.1 --wandb-key xxxx
  bash scripts/launch_robotwin_hierarchical_no_ray.sh --node-rank 2 --master-addr 10.0.0.1 --wandb-key xxxx
  bash scripts/launch_robotwin_hierarchical_no_ray.sh --node-rank 3 --master-addr 10.0.0.1 --wandb-key xxxx
EOF
}

WAND_KEY_ARG=""
EXTRA_OVERRIDES=()

while (($#)); do
  case "$1" in
    --nnodes)
      NNODES="$2"
      shift 2
      ;;
    --node-rank)
      NODE_RANK="$2"
      shift 2
      ;;
    --master-addr)
      MASTER_ADDR="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
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
    --task)
      TASK_OVERRIDE="$2"
      shift 2
      ;;
    --wandb-key)
      WAND_KEY_ARG="$2"
      shift 2
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
  echo "Error: WANDB_API_KEY is empty. Please set env or pass --wandb-key." >&2
  exit 1
fi

if [[ -z "${NODE_RANK}" || -z "${MASTER_ADDR}" ]]; then
  echo "Error: --node-rank and --master-addr are required." >&2
  usage
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Error: conda.sh not found at ${CONDA_SH}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export NNODES="${NNODES}"
export NODE_RANK="${NODE_RANK}"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"

echo "[launch-no-ray] nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE}"
echo "[launch-no-ray] master=${MASTER_ADDR}:${MASTER_PORT} train_script=${TRAIN_SCRIPT}"
echo "[launch-no-ray] task=${TASK_OVERRIDE}"

bash "${TRAIN_SCRIPT}" "${NPROC_PER_NODE}" "${TASK_OVERRIDE}" "${EXTRA_OVERRIDES[@]}"
