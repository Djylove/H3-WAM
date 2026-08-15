#!/usr/bin/env bash
set -euo pipefail

: "${C31_RUN_INDEX:?set C31_RUN_INDEX to 0..3}"
if [[ ! "$C31_RUN_INDEX" =~ ^[0-3]$ ]]; then
  echo "C31_RUN_INDEX must be 0..3" >&2
  exit 2
fi

PROJECT_ROOT="${H3_WAM_PROJECT_ROOT:-/mnt/h3-wam/candidate-d0-rollout-96976ce/project}"
C30_ROOT="/mnt/h3-wam/eval/c30-action-conditioned-causal-dataset-v1"
C31_ROOT="/mnt/h3-wam/eval/c31-temporal-consequence-v1"
OUTPUT_ROOT="/mnt/h3-wam/outputs/c31-action-conditioned-consequence-v1"
PYTHON_BIN="/mnt/h3-wam/runtime/conda-py311/bin/python"
DOSSIER="$PROJECT_ROOT/experiments/dossiers/h3_c31_action_conditioned_consequence_v1.json"

cd "$PROJECT_ROOT"
status="$($PYTHON_BIN -c 'import json; print(json.load(open("/mnt/h3-wam/eval/c30-action-conditioned-causal-dataset-v1/COMPLETED"))["status"])')"
if [[ "$status" != "PASS_C30_ACTION_CONDITIONED_CAUSAL_DATASET" ]]; then
  echo "C30 data gate did not pass: $status" >&2
  exit 3
fi
for artifact in "$C31_ROOT/dataset.pt" "$C31_ROOT/h3_features.pt" "$DOSSIER"; do
  if [[ ! -s "$artifact" ]]; then
    echo "required C31 artifact absent: $artifact" >&2
    exit 4
  fi
done
$PYTHON_BIN skills/wam-evidence-gated-training/scripts/validate_dossier.py \
  "$DOSSIER" --target long

variants=(flattened flattened temporal temporal)
seeds=(42 314159 42 314159)
variant="${variants[$C31_RUN_INDEX]}"
seed="${seeds[$C31_RUN_INDEX]}"
run_name="${variant}_seed${seed}"
mkdir -p "$OUTPUT_ROOT/$run_name/checkpoints"

export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export CUDA_VISIBLE_DEVICES="${C31_GPU_INDEX:-$C31_RUN_INDEX}"
export PYTHONPATH=src:scripts/h3wam:.
exec "$PYTHON_BIN" scripts/h3wam/train_c31_action_conditioned_consequence.py \
  --dataset "$C31_ROOT/dataset.pt" \
  --features "$C31_ROOT/h3_features.pt" \
  --output "$OUTPUT_ROOT/$run_name/report.json" \
  --checkpoint-dir "$OUTPUT_ROOT/$run_name/checkpoints" \
  --model-variant "$variant" \
  --steps 10000 \
  --save-every 1000 \
  --batch-size 64 \
  --target-dim 256 \
  --hidden-dim 256 \
  --actions-per-latent 4 \
  --num-heads 8 \
  --learning-rate 3e-4 \
  --weight-decay 1e-2 \
  --seed "$seed" \
  --device cuda:0
