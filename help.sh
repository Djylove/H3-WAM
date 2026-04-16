# export HF_ENDPOINT="https://hf-mirror.com"

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8

bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4

export WANDB_API_KEY=b8182b57eaa10a6c93943291158ee2f086aae4eb


cd /mnt/cpfs/wxy/FastWAM && \
bash scripts/train_zero1_dlc_multinode.sh \
  --wandb-key b8182b57eaa10a6c93943291158ee2f086aae4eb \
  task=robotwin_hierarchical_3cam_384_1e-4

# cd /mnt/cpfs/wxy/FastWAM && python scripts/train_ray_multinode.py \
#   --address auto \
#   --num-nodes 4 \
#   --nproc-per-node 8 \
#   --master-port 29500 \
#   --train-script scripts/train_zero1.sh \
#   --workdir /mnt/cpfs/wxy/FastWAM \
#   --conda-sh /mnt/cpfs/wxy/miniconda3/etc/profile.d/conda.sh \
#   --conda-env fastwam \
#   -- task=robotwin_hierarchical_3cam_384_1e-4