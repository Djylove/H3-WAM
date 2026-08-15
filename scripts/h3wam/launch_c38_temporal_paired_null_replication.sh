#!/usr/bin/env bash
set -euo pipefail

: "${C38_RUN_INDEX:?set C38_RUN_INDEX to 0..3}"
[[ "$C38_RUN_INDEX" =~ ^[0-3]$ ]] || { echo "C38_RUN_INDEX must be 0..3" >&2; exit 2; }

project="${H3_WAM_PROJECT_ROOT:-/mnt/h3-wam/candidate-d0-rollout-96976ce/project}"
c34_root="/mnt/h3-wam/eval/c34-combined-consequence-ranking-v1"
output_root="/mnt/h3-wam/outputs/c38-temporal-paired-null-replication-v1"
python_bin="/mnt/h3-wam/runtime/conda-py311/bin/python"
dossier="$project/experiments/dossiers/h3_c38_temporal_paired_null_replication_v1.json"
cd "$project"
for artifact in "$c34_root/dataset.pt" "$c34_root/h3_features.pt" "$dossier"; do
  [[ -s "$artifact" ]] || { echo "required C38 artifact absent: $artifact" >&2; exit 3; }
done
$python_bin skills/wam-evidence-gated-training/scripts/validate_dossier.py "$dossier" --target long
seeds=(161803 271828 8675309 20260815)
seed="${seeds[$C38_RUN_INDEX]}"
run_name="temporal_seed${seed}"
mkdir -p "$output_root/$run_name/checkpoints"
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export CUDA_VISIBLE_DEVICES="${C38_GPU_INDEX:-0}"
export PYTHONPATH=src:scripts/h3wam:.
exec "$python_bin" scripts/h3wam/train_c31_action_conditioned_consequence.py \
  --dataset "$c34_root/dataset.pt" --features "$c34_root/h3_features.pt" \
  --output "$output_root/$run_name/report.json" \
  --checkpoint-dir "$output_root/$run_name/checkpoints" \
  --model-variant temporal --target-error-scaling raw \
  --condition-dropout-prob 0 --mechanism-gate paired_null \
  --steps 10000 --save-every 1000 --batch-size 64 \
  --target-dim 256 --hidden-dim 256 --actions-per-latent 4 --num-heads 8 \
  --learning-rate 3e-4 --weight-decay 1e-2 \
  --seed "$seed" --device cuda:0 \
  --experiment-id h3_c38_temporal_paired_null_replication_v1
