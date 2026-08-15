# H3-FACT F0/F1 future-proprio canary

本实验是 `novel_composition`，不是 FACT 官方复现。它只验证一个问题：在当前 H3
观测表征和 proprio 已知时，真实 32 步动作是否比打乱动作或零动作更能预测第 32 步后的
proprio。它不改 D0/C03，不导入动作生成器，也不训练 H3。

## 固定合同

- 上游来源：FACT `618a6c16868699b6d4138941de6a863589ac00dd`；只借鉴独立的
  action-to-consequence 信息流。
- 数据：v7 四套 LIBERO episode-disjoint train/validation manifest；从 manifest 的
  `dataset_root/episode/start` 定位 LeRobot parquet。
- 输入：缓存当前 state、冻结 H3 layer49 pooled 32-token feature、归一化 demo action32。
- 标签：同一 parquet 的 `observation.state[start + 32]`。
- 时序审计：parquet `state[start]` 必须与缓存 current state 的 `max_abs <= 1e-6`；拒绝 padded
  window 和任何 train/validation episode overlap。
- 三臂：`conditioned` 使用对应动作；`shuffled` 使用 batch 内无 self-map 的动作排列；
  `independent` 使用全零动作。三臂模型参数逐 bit 同初始化，数据、batch、优化器设置相同。
- 防泄漏：consequence `forward` 不接收 future target，并在模块内部 detach candidate action。
  因而 future loss 可更新 consequence action encoder，但不能更新上游 action generator。

## 单卡 s100 命令

```bash
cd /mnt/h3-wam/candidate-d0-rollout-96976ce/project

CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/mnt/h3-wam/runtime/h3-int8-native/lib/python3.11/site-packages/nvidia/cu13/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
/mnt/h3-wam/runtime/h3-int8-native/bin/python \
  scripts/h3wam/train_h3_fact_future_proprio.py \
  --train-manifest /mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl \
  --val-manifest /mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_val.jsonl \
  --source-manifest /mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_all.jsonl \
  --cache-root /mnt/h3-wam/data/v7_dense_h3_cache \
  --feature-subdir h3_int8_starwam_last32_dense_v1 \
  --steps 100 \
  --train-limit 1024 \
  --val-limit 256 \
  --batch-size 16 \
  --hidden-dim 256 \
  --seed 42 \
  --verify-h3-checkpoint-sha256 \
  --output /mnt/h3-wam/outputs/fact-lite-future-proprio-v1/f1_s100.json \
  --checkpoint /mnt/h3-wam/outputs/fact-lite-future-proprio-v1/f1_s100.pt
```

输出路径必须不存在；脚本拒绝覆盖。1024/256 行由固定 salt 对完整 split 做确定性 hash 选择，
不是 manifest 前缀。

## 放行判据

机械门全部满足：

1. train/validation episode 与 window overlap 均为 0，split 行与 source manifest 完全相同；
2. H3 cache 元数据和 checkpoint SHA256 匹配，缓存/原始 parquet 当前 state 对齐；
3. 三臂 loss/gradient finite，保存后 fresh restore `max_abs == 0`；
4. 单测证明 consequence loss 对模拟 action generator 的参数梯度为 `None`，但 consequence
   action encoder 梯度非零。

机制门同时满足：

1. `conditioned_true` validation normalized MSE 比 `independent` 至少低 1%；
2. `conditioned_true` 比 `shuffled_train_true` 至少低 1%；
3. 将 conditioned 模型的 validation action 无 self-map 打乱后，MSE 至少恶化 1%。

三项由报告中的 `PASS_MECHANISM_GATE` 统一判定。通过只允许按相同合同增加到 500 steps 复核，
再另立 future-H3 或 value 子实验；不通过则先检查动作/未来 state 映射和学习曲线。无论结果如何，
本实验都不能证明 failure-aware、best-of-N、动作策略提升或 LIBERO 成功率。
