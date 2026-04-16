#!/usr/bin/env bash
set -euo pipefail

# One-click launcher for 4-node x 8-GPU robotwin hierarchical training.
# Usage examples:
#   bash scripts/launch_robotwin_hierarchical_multinode.sh
#   WANDB_API_KEY=xxxx bash scripts/launch_robotwin_hierarchical_multinode.sh
#   bash scripts/launch_robotwin_hierarchical_multinode.sh --wandb-key xxxx

PROJECT_ROOT="/mnt/cpfs/wxy/FastWAM"
RAY_ADDRESS="auto"
NUM_NODES="4"
NPROC_PER_NODE="8"
MASTER_PORT="29500"
TRAIN_SCRIPT="scripts/train_zero1.sh"
WORKDIR="/mnt/cpfs/wxy/FastWAM"
CONDA_SH="/mnt/cpfs/wxy/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="fastwam"
TASK_OVERRIDE="task=robotwin_hierarchical_3cam_384_1e-4"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/launch_robotwin_hierarchical_multinode.sh [--wandb-key <key>] [-- <extra hydra overrides>]

Options:
  --wandb-key <key>     Set WANDB_API_KEY for this run.
  --help                Show this help.

Examples:
  bash scripts/launch_robotwin_hierarchical_multinode.sh
  bash scripts/launch_robotwin_hierarchical_multinode.sh --wandb-key xxxxx
  bash scripts/launch_robotwin_hierarchical_multinode.sh -- task=robotwin_hierarchical_3cam_384_1e-4 eval_every=1000
EOF
}

WAND_KEY_ARG=""
EXTRA_OVERRIDES=()

while (($#)); do
  case "$1" in
    --wandb-key)
      if (($# < 2)); then
        echo "Error: --wandb-key requires a value." >&2
        exit 1
      fi
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

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Error: conda.sh not found at ${CONDA_SH}" >&2
  exit 1
fi

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "Error: project root not found at ${PROJECT_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

echo "[launch] image=official"
echo "[launch] ray_address=${RAY_ADDRESS} num_nodes=${NUM_NODES} nproc_per_node=${NPROC_PER_NODE}"
echo "[launch] train_script=${TRAIN_SCRIPT} task=${TASK_OVERRIDE}"
echo "[launch] conda_env=${CONDA_ENV}"

python scripts/train_ray_multinode.py \
  --address "${RAY_ADDRESS}" \
  --num-nodes "${NUM_NODES}" \
  --nproc-per-node "${NPROC_PER_NODE}" \
  --master-port "${MASTER_PORT}" \
  --train-script "${TRAIN_SCRIPT}" \
  --workdir "${WORKDIR}" \
  --conda-sh "${CONDA_SH}" \
  --conda-env "${CONDA_ENV}" \
  -- "${TASK_OVERRIDE}" "${EXTRA_OVERRIDES[@]}"
