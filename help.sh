# export HF_ENDPOINT="https://hf-mirror.com"

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8

export http_proxy=http://10.2.83.188:3128

bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4

bash scripts/train_zero1.sh 8 task=real_anygrasp_v2_hierarchical_1cam_384_1e-4
bash scripts/train_zero1.sh 8 task=robotwin_hierarchical_3cam_384_1e-4

torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=real_anygrasp_v2_hierarchical_1cam_384_1e-4

export WANDB_API_KEY=b8182b57eaa10a6c93943291158ee2f086aae4eb

export CUDA_VISIBLE_DEVICES=1,2

ssh-keygen -t ed25519 -C "your_email@example.com"

Host xingyu
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519

cd /mnt/cpfs/wxy/FastWAM && \
bash scripts/train_zero1_dlc_multinode.sh \
  --wandb-key b8182b57eaa10a6c93943291158ee2f086aae4eb \
  task=robotwin_hierarchical_3cam_384_1e-4

python scripts/test_hierarchical_generation.py \
  task=real_anygrasp_v2_hierarchical_1cam_384_1e-4 \
  test.checkpoint_path=runs/real_anygrasp_v2_hierarchical_1cam_384_1e-4/2026-06-18_01-04-19/checkpoints/weights/step_218394.pt \
  test.dataset_stats_path=runs/real_anygrasp_v2_hierarchical_1cam_384_1e-4/2026-06-18_01-04-19/dataset_stats.json 
  

python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_hierarchical_3cam_384_1e-4 \
  ckpt=runs/robotwin_hierarchical_3cam_384_1e-4/2026-05-19_16-43-59/checkpoints/weights/step_140898.pt \
  EVALUATION.task_name=blocks_ranking_rgb \
  EVALUATION.attention_viz_enabled=true \
  # 'EVALUATION.attention_viz_steps=[0,5,-1]' \
  # 'EVALUATION.attention_viz_layers=[-1]' \
  # EVALUATION.attention_viz_max_plans=1
cd /workspace/mnt/data/wxy/Hierarchical_WAM && \
bash scripts/train_zero1_mobile_multinode.sh \
  --wandb-key b8182b57eaa10a6c93943291158ee2f086aae4eb \
  task=robotwin_hierarchical_3cam_384_1e-4

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=1

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_hierarchical_3cam_384_1e-4 \
  ckpt=runs/robotwin_hierarchical_3cam_384_1e-4/2026-04-16_16-49-09/checkpoints/weights/step_187862.pt \
  EVALUATION.dataset_stats_path=runs/robotwin_hierarchical_3cam_384_1e-4/2026-04-16_16-49-09/dataset_stats.json \
  MULTIRUN.num_gpus=4

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_hierarchical_3cam_384_1e-4 \
  ckpt=runs/robotwin_hierarchical_3cam_384_1e-4/2026-04-16_16-49-09/checkpoints/weights/step_187862.pt \
  EVALUATION.dataset_stats_path=runs/robotwin_hierarchical_3cam_384_1e-4/2026-04-16_16-49-09/dataset_stats.json \
  MULTIRUN.num_gpus=8\
  EVALUATION.high_denoise_step=8\
  EVALUATION.low_denoise_step=6


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

