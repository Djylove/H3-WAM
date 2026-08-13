# cd /workspace/mnt/data/wxy/Hierarchical_WAM && \
# bash scripts/train_zero1_mobile_multinode.sh \
#   --wandb-key <your_key> \
#   task=robotwin_hierarchical_3cam_384_1e-4

cd /workspace/mnt/data/wxy/Hierarchical_WAM && \
bash scripts/train_zero1_mobile_multinode.sh \
  --wandb-key <your_key> \
  task=real_anygrasp_v2_hierarchical_1cam_384_1e-4
