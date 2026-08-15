#!/usr/bin/env bash
set -euo pipefail

: "${C35_RUN_INDEX:?set C35_RUN_INDEX to 0..3}"
[[ "$C35_RUN_INDEX" =~ ^[0-3]$ ]] || { echo "C35_RUN_INDEX must be 0..3" >&2; exit 2; }

project="${H3_WAM_PROJECT_ROOT:-/mnt/h3-wam/candidate-d0-rollout-96976ce/project}"
c34_root="/mnt/h3-wam/eval/c34-combined-consequence-ranking-v1"
output_root="/mnt/h3-wam/outputs/c35-action-conditioned-consequence-v1"
python_bin="/mnt/h3-wam/runtime/conda-py311/bin/python"
dossier="$project/experiments/dossiers/h3_c35_action_conditioned_consequence_v1.json"

cd "$project"
for artifact in "$c34_root/dataset.pt" "$c34_root/h3_features.pt" "$dossier"; do
  [[ -s "$artifact" ]] || { echo "required C35 artifact absent: $artifact" >&2; exit 3; }
done
$python_bin skills/wam-evidence-gated-training/scripts/validate_dossier.py "$dossier" --target long

variants=(flattened flattened temporal temporal)
seeds=(42 314159 42 314159)
variant="${variants[$C35_RUN_INDEX]}"
seed="${seeds[$C35_RUN_INDEX]}"
run_name="${variant}_seed${seed}"
mkdir -p "$output_root/$run_name/checkpoints"

export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export CUDA_VISIBLE_DEVICES="${C35_GPU_INDEX:-0}"
export PYTHONPATH=src:scripts/h3wam:.
exec "$python_bin" scripts/h3wam/train_c31_action_conditioned_consequence.py \
  --dataset "$c34_root/dataset.pt" \
  --features "$c34_root/h3_features.pt" \
  --output "$output_root/$run_name/report.json" \
  --checkpoint-dir "$output_root/$run_name/checkpoints" \
  --model-variant "$variant" \
  --steps 10000 --save-every 1000 --batch-size 64 \
  --target-dim 256 --hidden-dim 256 \
  --actions-per-latent 4 --num-heads 8 \
  --learning-rate 3e-4 --weight-decay 1e-2 \
  --seed "$seed" --device cuda:0 \
  --experiment-id h3_c35_action_conditioned_consequence_v1
