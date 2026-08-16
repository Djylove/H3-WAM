#!/usr/bin/env bash
set -euo pipefail

project=/mnt/h3-wam/candidate-d0-rollout-96976ce/project
python=/mnt/h3-wam/runtime/h3-int8-native/bin/python
torchrun=/mnt/h3-wam/runtime/h3-int8-native/bin/torchrun
output=/mnt/h3-wam/outputs/c57-lingbot-persistent-kv/long5000
cuda_lib=$($python -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["purelib"])/"nvidia"/"cu13"/"lib")')
export LD_LIBRARY_PATH=$cuda_lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export PYTHONPATH=$project/src:$project
mkdir -p "$output/checkpoints"
cd "$project"
exec "$torchrun" --standalone --nproc-per-node=8 \
  scripts/h3wam/train_c57_lingbot_persistent_kv.py \
  /mnt/h3-wam/data/c57-lingbot-replan8-v1/manifest_train.jsonl \
  --source-manifest /mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl \
  --cache-source-manifest /mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_all.jsonl \
  --cache-root /mnt/h3-wam/data/v7_dense_h3_cache \
  --kv-subdir h3_int8_dreamwam_kv_5x32_dense_v1 \
  --initialize-from /mnt/h3-wam/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt \
  --output "$output/report.json" \
  --checkpoint-dir "$output/checkpoints" \
  --steps 5000 --save-every 200 \
  --gradient-accumulation-steps 10 --num-workers 0 \
  --learning-rate 1e-5 --weight-decay 0.1 --warmup-steps 10
