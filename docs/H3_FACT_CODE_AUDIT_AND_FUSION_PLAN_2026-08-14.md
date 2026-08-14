# FACT 对 H3-WAM 的代码审计与融合计划

日期：2026-08-14

## 结论

FACT 值得进入候选池，而且它补的是当前 H3-WAM 最缺的一块：不只模仿成功动作，还学习“执行
某段动作后会发生什么、任务是否更接近成功”。但不能直接把 FACT 的 Wan2.2 全量训练命令搬到
当前项目。第一阶段应保留冻结 INT8 H3 和现有 ActionDiT，把 FACT 的 causal act-then-imagine、
failure-aware loss 与 best-of-N value ranking 拆成可验证模块；只有它们各自有效后才融合成统一
action-consequence expert。

## 官方代码身份

- 仓库：`https://github.com/Bariona/FACT.git`
- 固定 commit：`618a6c16868699b6d4138941de6a863589ac00dd`
- commit 时间：2026-08-11；当前仓库只有一个公开 commit，必须按新项目对待，不能仅凭 README
  将其结果当成熟基线。
- 本地只读副本：`third_party/FACT`
- 许可证：Apache-2.0。

## FACT 实际实现了什么

训练序列是：

`[state | reference image | noisy predicted action | clean GT action | future state | value | future image]`

其 causal mask 保证 predicted-action token 只能看 state、reference 与自身，不能看 clean action、
future state、value 或 future image；未来分支可以看 clean action，但不能反向把未来标签泄漏给动作。
推理分成两段：先仅生成动作，再把生成的 clean action 作为条件预测 future state/value/video。

失败 episode 的 action imitation loss 被置零，避免模仿失败动作；future state、future image 和加入
失败惩罚的 value 仍参与训练。推理可以采样 N 个动作候选，再选择预测 cost/value 最低者。官方配置
还使用 state/action/future-state/value/video 的联合 flow loss、robot-token FFN adapter 和不同的
action/base learning rate。

关键代码：

- causal mask：`third_party/FACT/world_action_model/models/transformer_wa_casual.py:48-80`
- teacher-forcing token timestep：`third_party/FACT/world_action_model/pipeline/utils.py:138-177`
- joint target/loss 与 failure mask：
  `third_party/FACT/world_action_model/trainer/wa_casual_trainer.py:682-866`
- 两阶段推理：`third_party/FACT/world_action_model/pipeline/wa_pipeline.py:545-830`
- failure metadata：`third_party/FACT/fact_datasets/datasets/lerobot_dataset.py:527-535,641-648`
- value target/action mask：
  `third_party/FACT/world_action_model/transformers/wa_transforms_lerobot.py:496-515`
- best-of-N：`third_party/FACT/scripts/inference_server.py:297-430`

## 与当前项目的差异

| 维度 | FACT 官方代码 | 当前 H3-WAM R1 | 判断 |
| --- | --- | --- | --- |
| 世界主干 | Wan2.2 可训练统一 Transformer | 冻结 INT8 H3 cache | 不能直接恢复官方权重/训练合同 |
| 动作生成 | 同一 causal transformer 内的 flow action tokens | 独立 30 层 ActionDiT | 数学兼容，carrier 不同 |
| 未来监督 | RGB latent + future state + value | 当前只有 action | FACT 是重要补充 |
| 失败数据 | `failure_rollouts.jsonl`，失败动作不模仿 | 当前 LIBERO demo 基本是成功轨迹 | 核心 failure-aware 收益需要新数据 |
| 推理 | action 后再推 future/value，支持 best-of-N | 单次 action flow | 可作为独立第二阶段增加 |
| 官方预算 | 150k steps，batch 32/GPU，RoboTwin 48×14D | LIBERO 32×7D、当前仅 100-step canary | 不可用短 canary 冒充复现 |

## H3-FACT 融合路线

### F0：数据和防泄漏合同

从 episode-disjoint LIBERO dense windows 生成：当前 observation H3 tokens、32-step action、chunk
末端 future proprio、未来窗口 H3 tokens、任务进度。训练/验证仍按 episode 隔离。实现自动测试：
action 分支对 future target 的梯度必须为零；打乱 clean action 应恶化 consequence prediction。

### F1：FACT-lite consequence expert

保持当前 R1 ActionDiT 完全不变。新增独立 consequence expert，输入
`current H3 tokens + text + proprio + clean action chunk`，输出 future proprio、future H3 tokens 和
progress cost。用独立模块先保证不可能发生 future-to-action 泄漏。这是对 FACT 的
`INTENTIONAL_DEVIATION`：预测 H3 future representation 而非 RGB latent，但更符合冻结 H3、关注动作
生成的当前阶段。

### F2：failure-aware 数据回灌

把闭环失败保存为 canonical episode，而不只保存 mp4：必须含逐步 observation、实际执行 action、
任务 predicate、failure onset 和终止原因。失败 episode 不训练 action imitation，只训练 future H3、
future state 与 cost。现有 R0 八个失败视频只够做诊断，不足以训练该分支。

### F3：best-of-N 动作选择

从相同 ActionDiT checkpoint 以固定 seeds 采样 N 个 action chunks，用 F1/F2 cost head 排序。先比较
N=1 与 N=4 的离线 ranking accuracy、动作多样性与额外延迟，再做相同初始状态的闭环 A/B。只有
cost 能稳定排序成功/失败候选时才允许增加 N。

### F4：统一 causal action-consequence expert

只有 F1-F3 各自有正证据后，才把 action、clean-action condition、future H3/state/value 合入同一个
带 FACT mask 的 transformer。这一步再与 DreamWAM carrier 胜者融合；不在第一轮同时改变 causal
mask、carrier、H3 解冻和损失权重。

## “蛊王”淘汰门槛

1. 当前 R1 s1/s50/s100 balanced held-out 决定基础 ActionDiT 是否学到泛化信号。
2. visual shuffle 必须证明动作依赖 H3；否则先修 carrier，不启动 FACT 大训练。
3. F1 必须在 episode-disjoint future H3/state 上优于 action-independent baseline，并通过防泄漏测试。
4. F2 cost/value 必须在未见 episode 上把失败排在成功之后；没有足够失败数据时不得宣称
   failure-aware 成功。
5. F3 N=4 必须在固定闭环初始状态下提高成功/目标接触，且推理时延可接受。
6. 最终融合只从通过单变量门槛的候选中选择，不比较训练 loss 选“蛊王”。

当前许可：`GO_CODE_AND_DATA_CONTRACT / NO_GO_LONG_TRAIN`。
