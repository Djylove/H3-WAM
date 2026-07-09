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
  ckpt=runs/robotwin_hierarchical_3cam_384_1e-4/2026-05-19_16-43-59/checkpoints/weights/step_140898.pt \
  EVALUATION.dataset_stats_path=runs/robotwin_hierarchical_3cam_384_1e-4/2026-05-19_16-43-59/dataset_stats.json \
  MULTIRUN.num_gpus=8\

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_hierarchical_3cam_384_1e-4 \
  ckpt=runs/robotwin_hierarchical_3cam_384_1e-4/2026-06-09_19-41-53/checkpoints/weights/step_140898.pt \
  EVALUATION.dataset_stats_path=runs/robotwin_hierarchical_3cam_384_1e-4/2026-06-09_19-41-53/dataset_stats.json \
  MULTIRUN.num_gpus=8\
  EVALUATION.high_video_inference_steps=10\
  EVALUATION.high_denoise_step=5\
  EVALUATION.high_reuse_step=2\
  EVALUATION.low_denoise_step=10\
  EVALUATION.joint_denoise=false

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_hierarchical_3cam_384_1e-4 \
  ckpt=runs/robotwin_hierarchical_3cam_384_1e-4/2026-05-19_16-43-59/checkpoints/weights/step_140898.pt \
  EVALUATION.dataset_stats_path=runs/robotwin_hierarchical_3cam_384_1e-4/2026-05-19_16-43-59/dataset_stats.json \
  MULTIRUN.num_gpus=8\
  model.hierarchical_mask_low_predict=false\
  EVALUATION.high_video_inference_steps=10\
  EVALUATION.high_denoise_step=4\
  EVALUATION.high_reuse_step=null\
  EVALUATION.low_denoise_step=6\
  EVALUATION.low_reuse_step=2\
  EVALUATION.joint_denoise=true

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_hierarchical_3cam_384_1e-4 \
  ckpt=runs/robotwin_hierarchical_3cam_384_1e-4/2026-05-19_16-43-59/checkpoints/weights/step_140898.pt \
  EVALUATION.dataset_stats_path=runs/robotwin_hierarchical_3cam_384_1e-4/2026-05-19_16-43-59/dataset_stats.json \
  MULTIRUN.num_gpus=4\
  MULTIRUN.max_tasks_per_gpu=2\
  EVALUATION.high_video_inference_steps=20\
  EVALUATION.low_video_inference_steps=20\
  EVALUATION.high_denoise_step=0\
  EVALUATION.low_denoise_step=10

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_hierarchical_3cam_384_1e-4 \
  ckpt=runs/robotwin_hierarchical_3cam_384_1e-4/2026-05-19_16-43-59/checkpoints/weights/step_140898.pt \
  EVALUATION.dataset_stats_path=runs/robotwin_hierarchical_3cam_384_1e-4/2026-05-19_16-43-59/dataset_stats.json \
  MULTIRUN.num_gpus=4\
  MULTIRUN.max_tasks_per_gpu=2\
  EVALUATION.high_video_inference_steps=20\
  EVALUATION.low_video_inference_steps=20\
  EVALUATION.high_denoise_step=10\
  EVALUATION.low_denoise_step=0
