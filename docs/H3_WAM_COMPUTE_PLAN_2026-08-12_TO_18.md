# H3-WAM 8 月 12–18 日算力使用计划

更新时间：2026-08-12（Asia/Shanghai）

## 唯一目标

在 8 月 17 日晚前得到一个**跨任务、可复现的 H3-WAM 闭环正例**，并判断增益来自哪里；
8 月 18 日只做最终评测、checkpoint/配置/manifest 固化和结果汇总，不再启动无法完成的长训练。

闭环成功率是主指标。action val MSE、训练 loss、物体位移和语言反事实只用于决定是否继续，
不能单独证明模型有效。

## 当前基线与已经排除的方向

| 实验 | 离线结果 | 闭环结果 | 决策 |
|---|---:|---:|---|
| M13 dense frozen-H3，step200 | val40 `0.140886` | Goal 4 tasks `0/4` | 继续到 step800 门槛，不能仅因 MSE 下降晋级 |
| M13 dense frozen-H3，step400 | val40 `0.122312` | Goal 4 tasks `0/4` | 可作为后续适配父 checkpoint |
| M11 frame-indexed，step200 | val40 `0.151389` | task3 trials0–9 `0/10` | 只给 step400 最后一次门槛 |
| M14 tail-2 H3 canary，40 steps | val40 `0.119368` | 正在跑 task3 trials0–2 | 离线比父模型改善约 `2.4%`，等待闭环后决定扩训 |
| M13 step400 controller sweep | — | replan 1/2/5、scale0.5 均 `0/1` | 不再消耗大算力调 replan/scale |

现有证据说明：数据密度不足曾经是问题，但把 dense 数据继续堆给冻结 H3 仍没有产生成功；
当前更可能的瓶颈是语言/目标绑定、接触阶段被平均损失稀释，以及离线分布到闭环状态的偏移。

## 四条并行线路

### P0-A：语言与目标绑定（最快可证伪）

1. 对 M13/M14 做同图像、同状态、正确/错误指令的动作反事实测试；当前冻结 H3 特征的
   correct/wrong cosine 约 `0.994`，说明任务条件过弱。
2. 从最佳 dense checkpoint 训练 100-step language-ranking canary；更新 DoT/KV fusion 和
   H3 尾部 2 层，不做 task-specific loss。
3. 通过门槛：val 不比父模型差 5%，correct/wrong 动作差异显著扩大，并且固定多任务
   rollout 出现正确目标接触或至少 `1/10` success。

失败则不扩到 full-H3；成功再做 300-step 版本并比较 tail-2/tail-10。

### P0-B：接触/阶段均衡动作学习

1. 用全部 40 tasks 构造 task-balanced、gripper-transition/high-motion/late-phase 加权 sampler；
   不针对单一任务挑数据。
2. 在物理动作空间增加前 8–10 个实际执行动作的 loss，并提高 gripper/contact 维度权重；
   同时保留原 flow-matching loss，避免只拟合抓取开关。
3. 报告分阶段 val：approach、pre-contact、contact、post-contact，而不是只报总 MSE。

通过门槛同样是多任务闭环正例；总 MSE 小幅变差但 contact 指标和闭环改善可以晋级。

### P1-C：Dense DreamWAM structured future

1. 对完整 dense manifest 生成 RAFT motion + H3-VAE cache；用两台机器 16 GPU 分片，预计
   在一个夜间窗口内完成，产物采用原子写入并支持续跑。
2. 先跑 paper-I/O 配方的 100-step motion canary：新通道 `0.1×` 初始化、独立 I/O LR，
   H3 tail-2；只有 motion loss 真下降且 action/闭环不退化才扩训。
3. motion 有收益后才增加 DreamWAM 的 depth/DINO training-only supervision；SAM3D 线继续
   暂停，避免同时改变过多变量。

这条线的意义不是增加推理负担，而是用结构化未来监督改变 H3 表征；部署仍只需 RGB、语言、
proprioception。

### P1-D：多任务 recovery distillation / DAgger

若 8 月 14 日晚前三条线仍没有闭环正例，立即启用这个保底线：

1. 在至少 8 个代表任务上收集 H3 policy 的失败状态；
2. 用已验证的 FastWAM/专家策略为这些状态提供 recovery action；
3. 将 recovery windows 与原 40-task dense 数据混合训练，不使用任务专属 switch；
4. 测试未参与 recovery 采集的 held-out tasks，区分记忆与泛化。

这是解决 offline-to-online 分布偏移的路线，不把 teacher success 冒充 H3 success。最终模型必须
单独闭环执行。

## 暂不占主算力的方向

- **RoboTTT/长上下文**：适合解决长 rollout 中的在线适应；当前模型第一目标选择就错，先不做
  完整 TTT。P0 出现正例后可做 2–4 帧 history + previous-action 小 canary。
- **MiniWorld**：可做 action-conditioned candidate scorer 或世界模型，但不是现成动作策略；
  只有已有可用 policy candidate 后才值得训练/接入。
- **全量 H3 解冻**：M14 tail-2 先回答“表征能否被推动”。没有语义/闭环信号时把 50 层全部
  解冻只会放大成本和遗忘风险。
- **完整 2,000-rollout benchmark**：候选未在固定 canary 获得正例前不运行。

## 节点排期

| 日期 | 32611 | 30907 | 32409 | 30234 |
|---|---|---|---|---|
| 8/12 | M11/M13 独立评测；之后 P0-B 数据审计 | M13 dense 长线主训练 | M14 闭环；之后 P0-A | M11 frame-indexed 长线主训练 |
| 8/13 | P0-B 100–200 step canary | M13 保留到预定终点 | P0-A 100-step canary | M11 保留到预定终点 |
| 8/14 | dense motion cache shard-0 / 独立评测 | M13 完成后转 cache 或晋级线 | P0-A 晋级实验 | M11 完成后转 cache shard-1 |
| 8/15 | motion 100-step canary | recovery rollout shard-0 | 最佳 P0 300-step | recovery rollout shard-1 |
| 8/16 | 候选多任务评测 | 最佳线扩训 | held-out recovery 训练 | history 小 canary（仅在已有正例时） |
| 8/17 | 四套评测 task shard-0 | task shard-1 | task shard-2 | task shard-3 |
| 8/18 | \multicolumn{4}{c}{冻结最佳模型、汇总成功率、保存环境/配置/数据指纹；不再开长任务} |

排期是动态的：32611、32409 优先快速消费 checkpoint 并做机制 canary；30907 的 M13 dense
和 30234 的 M11 frame-indexed 是本阶段两条长线基准，**即使中间 checkpoint 闭环为 0 也保留
到预定终点**，用于回答训练长度是否构成瓶颈。长线只有遇到 NaN、持续 OOM、数据损坏或
checkpoint 无法恢复才停止。单一新机制最多占两台机器，保证至少两种机制同时被验证。

## 统一晋级和停机规则

每条线依次经过四个门槛：

1. **工程门槛**：finite、无 OOM、checkpoint 可恢复；
2. **离线门槛**：相对父 checkpoint val 退化不超过 5%，并报告 phase/suite breakdown；
3. **语义门槛**：正确/错误指令产生可测动作差异，正确目标 predicate 有进展；
4. **闭环门槛**：固定 canary 至少 `1/10`，随后跨至少 2 个 suite、2 个 task 复现。

上述早停规则仅适用于新开的 canary/超参分支，不适用于已经在跑的 M13、M11 长线基准。
短线首个 checkpoint 同时没有离线、语义和闭环改善就停止该超参分支。只保留 parent、当前
最优和最新可恢复 checkpoint；训练日志、manifest、评测 JSON 永久保留。8 月 17 日前没有正例时，
不宣称 H3-WAM 已成立，而是交付失败定位与 recovery-distillation 结果。

## 最终交付

- 最佳/次佳 checkpoint 与精确启动命令；
- 数据 manifest、normalization stats、代码 commit 和依赖快照；
- 固定 seeds 的 action val、语言反事实、predicate progression、闭环 success JSON；
- 5090 部署所需的冻结/量化路径单独记录，但不占用本周 A800 的主要训练窗口。
