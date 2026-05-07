# python experiments/robotwin/run_robotwin_manager.py \
#   task=robotwin_hierarchical_3cam_384_1e-4 \
#   ckpt=runs/robotwin_hierarchical_3cam_384_1e-4/2026-04-16_16-49-09/checkpoints/weights/step_187862.pt \
#   EVALUATION.dataset_stats_path=runs/robotwin_hierarchical_3cam_384_1e-4/2026-04-16_16-49-09/dataset_stats.json \
#   MULTIRUN.num_gpus=8\
#   EVALUATION.high_denoise_step=0\
#   EVALUATION.low_denoise_step=null

# python experiments/robotwin/run_robotwin_manager.py \
#   task=robotwin_hierarchical_3cam_384_1e-4 \
#   ckpt=runs/robotwin_hierarchical_3cam_384_1e-4/2026-04-16_16-49-09/checkpoints/weights/step_187862.pt \
#   EVALUATION.dataset_stats_path=runs/robotwin_hierarchical_3cam_384_1e-4/2026-04-16_16-49-09/dataset_stats.json \
#   MULTIRUN.num_gpus=8\
#   EVALUATION.high_denoise_step=2\
#   EVALUATION.low_denoise_step=2

# python experiments/robotwin/run_robotwin_manager.py \
#   task=robotwin_hierarchical_3cam_384_1e-4 \
#   ckpt=runs/robotwin_hierarchical_3cam_384_1e-4/2026-04-27_20-00-56/checkpoints/weights/step_450000.pt \
#   EVALUATION.dataset_stats_path=runs/robotwin_hierarchical_3cam_384_1e-4/2026-04-27_20-00-56/dataset_stats.json \
#   MULTIRUN.num_gpus=8\
#   EVALUATION.high_denoise_step=2\
#   EVALUATION.low_denoise_step=2

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_hierarchical_3cam_384_1e-4 \
  ckpt=runs/robotwin_hierarchical_3cam_384_1e-4/2026-04-27_20-00-56/checkpoints/weights/step_450000.pt \
  EVALUATION.dataset_stats_path=runs/robotwin_hierarchical_3cam_384_1e-4/2026-04-27_20-00-56/dataset_stats.json \
  MULTIRUN.num_gpus=8\
  EVALUATION.high_denoise_step=0\
  EVALUATION.low_denoise_step=0

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_hierarchical_3cam_384_1e-4 \
  ckpt=runs/robotwin_hierarchical_3cam_384_1e-4/2026-04-27_20-00-56/checkpoints/weights/step_450000.pt \
  EVALUATION.dataset_stats_path=runs/robotwin_hierarchical_3cam_384_1e-4/2026-04-27_20-00-56/dataset_stats.json \
  MULTIRUN.num_gpus=8\
  EVALUATION.high_denoise_step=2\
  EVALUATION.low_denoise_step=0