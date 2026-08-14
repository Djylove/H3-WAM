# H3 R1 Candidate G：条件利用代码审计

日期：2026-08-14  
实验类别：`controlled_ablation`（Candidate G 本身是 `novel_composition`，不是任何上游的官方复现）  
训练许可：`NO_GO`（本轮只完成代码、损失反传和合同测试，未放行 GPU 训练）  
效果结论：`NOT_EVIDENCE_READY`

## 可证伪假设

在完全相同的 noisy action、flow target、timestep、语言和 proprio 下，加入“正确 H3 特征的加权 flow
loss 至少比跨样本错误 H3 特征低一个 margin”的单一约束，可以阻止 R1 在降低动作回归误差时把视觉
条件压成零；若 projector 梯度再次为零，或 paired visual/language sensitivity 仍随训练趋近零，则假设失败。

## 来源身份

固定来源由 `docs/UPSTREAM_SOURCES.lock.json`（SHA256
`afbd249b2eed3d58fff109e5e0cf21f9c4a6ec54e9c0bf8b5483b682fc12f96f`）约束：

| 项目 | commit | 工作树 | 代码级别 | 审计方式 |
|---|---|---|---|---|
| DreamWAM | `6e989facc0c452fd3488d75f60bc36411005558c` | clean | TRAINABLE | 固定本地文件 |
| FastWAM | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` | 828 个删除项，dirty | TRAINABLE（固定 commit） | 只读 `git show <commit>:<path>`，不信任/不恢复工作树 |
| FACT | `618a6c16868699b6d4138941de6a863589ac00dd` | clean | TRAINABLE | 固定本地文件 |
| MiniWorld | `e484206bbd4360ae56ed8abad51c83f2457ac092` | clean | TRAINABLE | 固定本地文件 |
| StarWAM | `cd76d96f273f81e228a05f40f9697fe2514e2356` | clean | TRAINABLE | 固定本地文件 |

论文只用于解释方法意图；下列判断全部来自上述固定 commit 的真实 config、forward、loss、optimizer
和 backward 代码。

## 上游可执行机制审计

### DreamWAM

- resolved joint config：action/video/flow/DINO/depth 权重依次为 `1/1/0.5/0.5→0.025/0.5→0.05`，
  local preview 开启、权重 `0.2`，训练 `batch=16, lr=1e-4, wd=0.01, steps=21700`：
  `third_party/DreamWAM/configs/dreamwam_joint.yaml:36-54,80-89`。
- `setting==joint` 明确令 action attend all video：
  `third_party/DreamWAM/dreamwam/model.py:403-407`。
- 同一个联合 forward 产生 action、video、flow、DINO、depth；主损失与各注入层 local-preview loss
  在 `model.py:777-881` 相加。`dreamwam/train.py:112-126` 对返回的 total 调
  `accelerator.backward` 并 optimizer step。因此这些辅助项确实进入 backward，不是日志指标。
- 推理时先填充 video K/V，再让 action 每个 denoise step 读取该 cache：
  `model.py:962-986`。这是推理路由，不是额外训练 loss。
- 未发现 paired wrong-condition、condition ranking 或 structured condition dropout。`gate_l1` 是正的
  稀疏惩罚，不能当成防坍塌项。

结论：DreamWAM 的证据最强处是“action-visible world tokens + 多种 structured future 目标 + 中间层
local preview 全部反传”；它是间接保持表征有用，不直接证明 action 必须区分正确/错误条件。

### FastWAM

- fixed blob 的 LIBERO joint override 为 `batch=16, lr=1e-4, epochs=10, grad_acc=1`，且把
  `mot_checkpoint_mixed_attn` override 为 false：
  `configs/task/libero_joint_2cam224_1e-4.yaml:8-26`。
- model config 为 30 层 video/action experts、shift `5`、action loss `1`；runtime 缺省 video loss 也是
  `1`：`configs/model/fastwam_joint.yaml:12-58`、`src/fastwam/runtime.py:156`。
- joint mask 明确 `action -> full video`：
  `src/fastwam/models/wan22/fastwam_joint.py:28-49`。
- text 分别进入 video/action pre-DiT，两个 expert 联合 forward；video 和 action weighted flow loss 在
  `src/fastwam/models/wan22/fastwam.py:479-563` 相加并返回，因此都会进入项目 trainer 的 backward。
- 未发现 wrong-condition、margin、condition dropout 或条件依赖门。

结论：这正是 R1 迁移的来源，但上游“允许 action 读取 video”不等于“训练后 action 必然读取 video”。
R1 s913 的反事实结果已经给出该逻辑缺口的实证反例。

### FACT

- resolved RobotWin 配方为 8 GPU、per-GPU batch 32、global batch 256、150k steps：
  `third_party/FACT/world_action_model/configs/robotwin.py:55-70`；loss 权重
  `visual=1, action=10, future_state=0.4, value=0.4`：`robotwin.py:112-120,314-319`。
- causal teacher mask 令 predicted action 只能看 state/ref/action，自身不能偷看 gt action；future
  state/value/image 可以看 clean gt action：
  `world_action_model/models/transformer_wa_casual.py:46-81`。
- trainer 同时构造 noisy predicted action 与 clean gt-action condition，并预测 future state/value：
  `world_action_model/trainer/wa_casual_trainer.py:686-718,790-814`。四项 loss 在
  `trainer.py:846-859` 产生，在 `trainer.py:470-499` 按 resolved weights 相加，最终由
  `fact_train/trainers/trainer.py:720-759` backward/step。
- failure rollout 只有在数据根存在 `meta/failure_rollouts.jsonl` 才加载：
  `fact_datasets/datasets/lerobot_dataset.py:527-535,641-647`；failure action imitation 被 mask，失败
  value 加 penalty：`wa_transforms_lerobot.py:497-514,536-553`。当前官方 resolved dataset 是否实际
  含该文件为 `UNKNOWN`。

结论：FACT 提供了可执行的 consequence/value teacher routing 和防动作泄漏 mask，但没有给 action
分支配 correct/wrong observation；不能直接解决 R1 的视觉条件坍塌。

### MiniWorld

- DROID launcher 是四阶段 curriculum：latent frames `6→16→32→64`，默认 8 ranks；前两阶段
  `100/50 epochs`，后两阶段各 `30k steps`：
  `third_party/MiniWorld/scripts/train_droid.sh:18-36,56-95`。
- CLI 的 `wm_cond_dropout_prob=0.1`：`miniworld/train.py:341-346`，并被传入模型：
  `miniworld/denoiser.py:132-143`。
- 模型按 sample 采 dropout mask，把 action condition 替换为 learned null action：
  `miniworld/miniworld.py:742-754,794-817`；train-time null 与 CFG unconditional branch 对齐：
  `miniworld/denoiser.py:169-173,213-225`。
- 训练使用 temporal-causal forward 和 masked video velocity loss：
  `denoiser.py:464-494`；`train.py:502-510` 对这个 loss backward/step。streaming inference 的 clean
  history K/V prefill 与条件/unconditional cache 位于 `denoiser.py:782-805,980-992`。

结论：这是唯一审计到的官方 structured condition dropout，但它训练的是 action-conditioned video
生成并依赖 CFG；它不直接约束 R1 action head 的 correct-vs-wrong 条件次序。单独照搬 0.1 dropout
甚至可能让已有 action prior 更容易工作，因此本轮不实现它。

### StarWAM 与当前 R1

- 官方 feature-conditioned recipe 取 observation 的 last-layer 32 tokens、包含 text、训练 backbone，
  但 `video_loss=0, action_loss=1`：
  `third_party/StarWAM/examples/libero/configs/recipes/starwam_libero_feature_conditioned_wan22_5b.yaml:74-105`。
- 官方 action context 是 `[text, projected video features]`，只有 weighted flow action loss：
  `third_party/StarWAM/starwam/wam/feature_conditioned_action_model.py:108-141,270-295`；trainer 在
  `starwam/training/trainer.py:613-627` 对该 loss backward。
- 当前 H3 R1 是 `[text, proprio, projected cached H3]`：
  `src/fastwam/models/h3wam/starwam_feature_action.py:142-223`，同样只有 action flow 主目标。

这解释了为什么 R1 容易找到更便宜的 noisy-action/proprio prior：没有 world loss 更新 frozen H3，也没有
loss 要求 action 输出识别“当前样本的视觉”而非“任意视觉”。

## 与 s913 条件坍塌逐项对照

证据源：`experiments/evidence/h3_r1_a2_f2_conditioning_collapse_v1.json`。

| 现象 | s100 | s913/step914 | 判定 |
|---|---:|---:|---|
| physical MSE | 0.343551 | 0.124351 | generic regression 变好 |
| visual-shuffle action delta MSE | 0.236696 | 4.096e-7 | 只保留 s100 的 0.000173% |
| language mean-abs delta | 0.433882 | 0.019489 | 下降 95.51% |
| feature-projector grad | 正值路径可训练 | step914 精确 0 | fail closed |

Candidate F 在 s850/step851 复现同一模式，说明 clean reconstruction 只是已有 velocity error 的重加权，
没有补上条件绑定。

## 唯一实现：Candidate G paired visual margin

父基线：R1 pinned StarWAM weighted masked flow，Candidate F 关闭。  
唯一变量：额外一次 wrong-H3 forward，并加入
`relu(L_flow(correct H3) - L_flow(wrong H3) + margin)`。  
不变量：动作/噪声/timestep/target、语言、proprio、mask、scheduler、ActionDiT、H3 cache、数据 split、
优化器和严格梯度门。

实现位置：

- CLI 默认关闭：`scripts/h3wam/train_h3_int8_starwam_action.py:125-140`。
- local batch >1 时 cyclic cross-sample；实际 per-rank batch=1 时从下一 DDP rank 取 detached H3：
  `train_h3_int8_starwam_action.py:442-464`。
- hinge、wrong forward、同 flow noise/target、加入 total 并 backward：
  `train_h3_int8_starwam_action.py:467-561`。
- checkpoint 写入 Candidate G 身份与所有 fixed inputs：
  `train_h3_int8_starwam_action.py:726-744`；Candidate F/G 同时开启会在 `:665-669` fail closed，
  防止把两个变量静默混成一次实验。
- 原来的 expert/projector/proprio 必须 finite 且严格 `>0` 的硬门完全保留：
  `train_h3_int8_starwam_action.py:1063-1068`。

这是项目内 novel composition，不声称来自某篇论文或某个官方 repo。它由两类 code-backed 事实驱动：
上游普遍把条件路由到 action，但没有 correct/wrong 约束；本项目 paired visual shuffle 已直接测到 action
对视觉不变。

## 测试与门

执行：

```bash
.venv/bin/pytest -q \
  tests/test_h3_int8_starwam_action_trainer.py \
  tests/test_h3_starwam_feature_action.py
```

结果：`31 passed`（trainer 单文件 `26 passed`，另含 5 个 policy tests）。覆盖：

- 默认关闭不改变原合同；
- local cyclic negative 与 per-rank batch=1 的 DDP next-rank negative；
- negative H3 detached；
- hinge 正确方向与双分支梯度；
- 两次 forward 的 noisy action bit-equivalent；
- Candidate G 才增加 weighted loss；
- Candidate F/G 不能静默叠加；
- checkpoint contract；
- 零梯度硬门没有被放宽。

## 差异矩阵、预算与下一步

| 字段 | 父 R1 | Candidate G | 状态 |
|---|---|---|---|
| H3 | frozen cached INT8 features | 相同 | EXACT |
| 正样本 flow | weighted masked flow | 相同 | EXACT |
| 负样本 | 无 | 只替换 H3，其他输入固定 | INTENTIONAL_DEVIATION |
| 新参数 | 0 | 0 | EXACT |
| forward 成本 | 1× ActionDiT | 2× ActionDiT | INTENTIONAL_DEVIATION |
| projector grad gate | finite 且 >0 | finite 且 >0 | EXACT |
| language binding | 无直接损失 | 仍无直接损失 | UNKNOWN / 未解决 |
| DDP negative sample ID 去重审计 | 无 | tensor route 已测，真实 IDs 尚未落盘校验 | UNKNOWN |

当前只允许 `PROBE_ONLY`：不得保存候选 checkpoint，不得消耗正式数据预算。正式 canary 的 resolved
global batch、样本数、effective epochs 和墙钟尚未注册，均为 `UNKNOWN`；计算式为
`global_batch=world_size×per_device_batch×grad_accum`，
`effective_epochs=steps×global_batch/unique_train_windows`，墙钟必须先测 Candidate G 双 forward 的
真实 sec/step。

进入 `GO_CANARY` 前还需：真实 8-rank 单步 all-gather sample ID 无 self-map 证据、同一 batch 上
`L_correct/L_wrong/margin active rate/projector grad` 原始 JSON，以及 fresh restore。任何 projector grad=0
立即 `NO_GO`，不允许改成 epsilon/warning。即使机械门通过，效果仍为 `NOT_EVIDENCE_READY`；后续必须
在固定 balanced-80 的 early/late checkpoints 同时报 physical、gripper、visual shuffle、language
replacement，再决定是否进入 rollout。

尚未放行，因此没有真实训练启动命令；本轮也没有启动任何 GPU 训练。
