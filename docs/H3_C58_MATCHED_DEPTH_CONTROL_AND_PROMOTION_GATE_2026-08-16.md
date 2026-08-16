# C58 matched D0-5 控制与晋级门槛

日期：2026-08-16

状态：代码与预注册完成；不占当前 GPU；等待 8×A800 空闲后先跑 10-step canary

## 为什么必须补这个控制

C58 从已训练的 D0-s14000 初始化后，把动作塔从 5 层扩成 30 层。如果只拿
C58 与原始 s14000 比较，任何改善都可能来自新增的 10k optimizer steps，不能
归因于 30 层结构。matched control 从同一个 D0 权重出发，丢弃 parent optimizer
和 scheduler，用和 C58 完全相同的新数据、顺序、flow noise、LR/schedule 和
10k step，只保留 5 层动作塔。

唯一变量：`action_layers = 5` 或 `30`。

## 冻结的成对训练合同

- Parent：D0-H32-s14000 SHA256 `36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3`。
- 两臂均只加载 parent model weights；optimizer/scheduler 均为 fresh。
- Manifest：200,779-row dense LIBERO train manifest。
- 共同 slice：offset `112000` 起，10 段 × 8000 rows，共 80,000 个不重复窗口。
- 8 rank、每卡 batch1、无 accumulation；每段 DistributedSampler 使用相同 seed42、epoch0、drop_last false。
- Flow seed：相同 completed step、rank、accumulation index 的无状态公式。
- AdamW：LR `1e-4`、WD `0.01`、betas `(0.9,0.95)`。
- Schedule：warmup 1000、cosine horizon 10000、min LR `1e-6`。
- Action：H32、shift5、7D、相同 minmax/pad-masked velocity MSE。
- Carrier：相同冻结 H3 repeat-layer49 K/V。
- 每 1000 step 保存；control 每个 milestone 都执行 strict restore。

`prepare_c58_matched_pair_contract.py` 在启动前重建 8-rank sampler 顺序，并冻结
每段 manifest order、rank order、rank-major 与 step-major SHA256。它还会检查
80k rows 与 parent 已消费 rows 零重叠。这样“同样本”不仅是相同 slice，而是
相同 rank、相同 optimizer step 的严格配对。

## GPU 放行

长训不能直接启动。先运行：

```bash
bash scripts/h3wam/launch_c58_matched_d0_control_probe.sh
```

10-step canary 必须同时通过：parent 输出 `max_abs=0`、5/5 block 梯度有限且
非零、loss 有限、fresh optimizer 声明、80 个训练样本、checkpoint 独立严格
恢复 `max_abs=0`。成功后生成 `CANARY_READY.json`，长训 launcher 才放行：

```bash
bash scripts/h3wam/launch_c58_matched_d0_control_long.sh
```

当前没有占用已有 GPU，也没有效果结论。

## 每 1000 step balanced-80 队列

空闲单张 A800 上运行：

```bash
bash scripts/h3wam/watch_c58_matched_balanced80.sh GPU_INDEX
```

队列等待 C58 与 matched control 的相同步数 checkpoint，固定 v7 balanced-80
样本 ID SHA256
`26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42`，
相同 seed、初始 action noise、shift5 与 10-step Euler sampler。每份报告必须
证明 checkpoint completed step、parent fresh optimizer 合同和两次独立 restore
`max_abs=0`。

## 预注册 offline gate

一个 milestone 只有同时满足以下条件才可进入 fresh LIBERO：

1. C58 normalized action MSE 相对 control 改善至少 1%；
2. physical action MSE 相对改善至少 1%；
3. gripper macro-F1 下降不超过 0.005；
4. language replacement sensitivity 至少保留 control 的 95%，且 mean-abs delta ≥ 0.01；
5. visual K/V shuffle sensitivity 至少保留 control 的 95%，且 normalized delta MSE ≥ `1e-4`。

在所有 eligible milestone 中选择 C58 physical MSE 最低者，同分选更早 step。
训练 loss 不参与选择；未跑完十个 milestone 时只输出 `WAIT`。

## 预注册闭环晋级门槛

Offline gate 只允许 fresh closed-loop，不是效果证明。最终使用 LIBERO
trials33..49、相同初态/环境 seed/policy noise、replan8、no ensemble，共每臂
680 episodes：

- 相对 matched control：至少 +3 percentage points、paired net wins ≥20、单侧 exact McNemar `p≤0.05`，且任一 suite 退化不超过 3 points；
- 相对 incumbent D0-s14000：overall 不低于 incumbent，任一 suite 退化不超过 3 points。

只有上述闭环门槛通过，才允许把收益归因于完整 30 层动作塔。
