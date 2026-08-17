# C60 失败因果诊断与单变量修复预注册

日期：2026-08-17。状态：`DIAGNOSIS_COMPLETE / NO_20K / PREREGISTER_C63_STAGE2_RANKING`。
本文只使用已经完成的 C60 s1k/s5k/s10k per-sample 结果、C60 held-out failure split 和固定的官方
FACT 源码；没有启动续训，也没有把离线诊断包装成闭环证据。

## 结论先行

C60 不是单纯“训练步数不足”。它已经学到更好的多数连续动作维度，但少数接触/对象选择状态出现大幅
退化，使 s10k 的物理动作 MSE 反而略差于 s5k；从 s1k 到 s10k 的均值退化则主要由 gripper 维度
贡献。继续把同一目标训到 20k，既不能修复这个长尾，也不能把 H3 世界预测能力直接变成动作选择能力。

更根本的合同错位是：官方 FACT 推理先生成动作，再用候选动作条件化未来状态、价值和未来图像，并在
`best_of_N > 1` 时按预测 value 的 `argmin` 选择动作；当前 C60 部署强制 `consequence_best_of_n=1`，
并把完整 C60 模型包装成 action-only adapter。于是 H3/未来/value 分支只在训练时通过共享梯度间接影响
动作塔，部署时完全没有参加动作选择。

因此下一条单变量候选不是再补一个 loss，也不是继续 20k，而是 **保持 C60 s10k 权重、动作生成器、
归一化、32-step horizon、shift=5 和 10-step solver 全部不变，只启用 C60 自己已经训练的 Stage-2
`forward_consequence`，用预测 value 排序候选动作**。先做同状态成功/失败动作对的离线可证伪诊断；
只有通过后才允许收集四 suite 的 confirmatory 对，再决定是否做 N=1 vs N=4 闭环。

## 固定证据身份

- 官方 FACT 固定提交：`618a6c16868699b6d4138941de6a863589ac00dd`。2026-08-17 查询远端
  `main=9427ea451e806220742148049ef0576e43ef7382`；与固定提交的执行代码无差异，差异仅为 README
  live-demo 三行。
- C60 s1k/s5k/s10k checkpoint SHA256 分别为
  `b44c71ca87ba80f3646f27313f1b13f3910b6cb3a2f5f11428891834e70dd856`、
  `fc5984d5df2ee89023051c5494e14567fdfbd5c97bfdd36e0cdec32aa30f8fcb`、
  `d6659c6b387f062a99f670a1d902b56df71a6bf1472aa4e46e56c9213ba75a36`。
- 三份固定 balanced80 报告 SHA256 分别为
  `e6e1111cc273ded8bdb95b8f708825418a4cd1284e08be1854051bbd5e7a8258`、
  `a1a0f0f00dadb753b70937de8cfc2106d6be9f59bfe80da2c3379575f4c8f3e6`、
  `45dabec4dcc7a563b3b4c1208bc84857b8025ddfb342f21e130548abdfcee5ea`。
- C60 failure dataset SHA256
  `1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4`；observations JSONL
  SHA256 `b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55`。
- 680 对闭环结论保持不变：C60 `313/680`、C58 `295/680`，`+2.647pp`，paired `63:45`，
  one-sided McNemar `p=0.050716`；正式状态仍是
  `FAIL_C60_FACT_EXPANDED_PAIRED / KEEP_C58_PARENT / NOT_EVIDENCE_READY`。

## s1k/s5k/s10k 的失败形状

所有比较都使用同一 80 个 episode-disjoint validation windows、相同 action noise、solver、归一化和
sample identity。

### s10k 相对 s1k

- physical MSE 平均变化 `+3.835e-5`；51 个样本改善、29 个退化，median `-9.196e-4`。
- 退化不是整体漂移，而是重尾：positive degradation mass 的 top-5 占 `65.31%`，top-10 占
  `84.59%`；最大单样本退化 `+0.038421`。
- 最大退化集中于长程、多对象和接触末端：alphabet soup + tomato sauce 双对象入篮、butter 入篮、
  black bowl 空间关系、双 mug/双 plate、cream cheese + butter 双对象入篮。
- 连续动作大多改善；但 gripper physical MSE 从 `0.054659` 升到 `0.061830`（`+13.12%`），
  normalized MSE 从 `0.222202` 升到 `0.249278`（`+12.19%`），gripper macro-F1 从
  `0.941795` 降到 `0.933197`。s1k→s10k 的总体均值退化由 gripper 主导。

### s10k 相对 s5k

- physical MSE 平均变化 `+3.018e-5`；42 个样本改善、38 个退化，median `-1.287e-4`。
- top-5/top-10 占 positive degradation mass 的 `58.62%/73.25%`；最大退化为 spatial 的
  “black bowl next to plate”样本，`+0.021700`。
- spatial suite 平均退化 `+0.001113`，而 LIBERO-10/object/goal 均值分别为
  `-0.000016/-0.000238/-0.000738`。此阶段 gripper 反而改善 `-3.60%`，主要退化转移到
  x/y/z 和 rz。这说明 s10k 不是全面变坏，而是接触与空间状态的少数大错压过多数小改进。

该形状与“再训练更多步即可解决”不一致：s10k 比 s5k 赢的样本更多、median 更好，但均值被少数
严重错误反转。长训门要求 s10k physical MSE 不差于 s5k，实际 `0.0252567073 > 0.0252262835`，
所以没有 20k 许可。

## 与官方 FACT 的逐层合同对齐

| 层 | 官方 FACT `618a6c1` | 当前 C60 | 状态 | 影响 |
|---|---|---|---|---|
| 数据 | LeRobot dense frames、episode mixture、48-step action/future | expert/success/observational failure/causal failure 固定 4/2/1/1 rank mixture；32-step | `INTENTIONAL_DEVIATION` | causal failure effective epoch `5.184`，expert 仅 `0.199`，失败接触状态可压过广覆盖 expert 状态 |
| action target | 选定绝对维转 state-relative delta，z-score，48-step | FastWAM 7D min-max，32-step | `INTENTIONAL_DEVIATION` | 机器人合同不同可以合理偏离，但不能声称 target exact port |
| value/failure | 明确 failure onset；active 后加 penalty；failure episode 屏蔽 action loss | 有 exact intervention onset、failure action mask、value penalty | `EQUIVALENT` | C60 的 causal 标签机制不是首要缺陷 |
| loss 权重 | visual/action/state/value=`1/10/.4/.4` | future-representation/action/state/value=`1/10/.4/.4` | `INTENTIONAL_DEVIATION` | 权重 exact，visual target 被 H3 future representation 代理，不是 RGB flow 的 exact reproduction |
| backbone | Wan 全 transformer 可训练，VAE 冻结 | INT8 H3 冻结，30 层 ActionDiT/FACT heads 可训练 | `INTENTIONAL_DEVIATION` | 符合当前“保留 H3 基模、训练动作专家”决策，不应以全量 Wan 训练步数类比 |
| teacher forcing | clean GT action 条件化未来/value/image；action 不能读未来 | 同 timestep，clean action 条件 consequence；因果 mask | `EQUIVALENT` | 额外机械测试中 joint action 与 `forward_action` 最大差 `1.19e-7`，无明显 train/infer action 前向裂缝 |
| 推理 | Stage-1 action；Stage-2 action-conditioned future/value；BoN 按 value `argmin` | 强制 N=1；`_H3FACTActionOnlyAdapter`；不调用 consequence/value | **`MISMATCH`** | 世界预测能力没有直接影响执行动作，是当前最高优先级缺口 |

官方证据位置：`third_party/FACT/world_action_model/configs/robotwin.py:63`、`:84`、`:99`、`:112`、
`:183`、`:207`、`:269`；数据 target 位于
`third_party/FACT/world_action_model/transformers/wa_transforms_lerobot.py:427`；两阶段推理位于
`third_party/FACT/world_action_model/pipeline/wa_pipeline.py:550`；BoN/value argmin 位于
`third_party/FACT/scripts/inference_server.py:297`、`:428`。本地 action-only 限制位于
`scripts/h3wam/serve_rollout_policy.py:2466`、`:2502`、`:2581`；完整 consequence API 已存在于
`src/fastwam/models/h3wam/fact_layerwise_tower.py:414`。

## C63：唯一允许的单变量修复候选

假设：在完全相同的当前 state/observation/text 下，C60 s10k 的 Stage-2 value 对一个成功父轨迹的
未来 32-step executed-action chunk 给出的 cost，应低于同一恢复状态下失败 intervention 的
32-step executed-action chunk。如果连这个最容易的 held-out 两选一排序都做不到，C60 的 consequence
head 就不能用于 BoN，继续训练或闭环尝试都没有依据。

固定 32 对来自 C60 validation split 的 32 个 branch onset，覆盖 11 个成功父轨迹；其中
`libero_spatial=30`、`libero_object=2`。这个 split parent-source-disjoint 于训练，但 suite 极不平衡，
所以它只能是**离线诊断门**，不能作为四 suite 泛化证据。

### Mechanical gates

1. 严格校验上面的 checkpoint、dataset、observations 和成功父 trajectory SHA；32/32 对必须完整。
2. 每对的 branch onset 与成功父轨迹在 `sim_state`、`previous_action` 和 simulator step 上逐字段完全
   相等；不允许只按 task/trial 猜对齐。每对两个候选统一使用 branch onset 的 RGB/proprio/text 作为
   模型输入，所以进入 Stage-2 的当前观测逐字节相同。
3. successful/failed action 都从真实 executed actions 重建为 32 step，再经过同一 gripper conversion、
   min-max normalization 和 padding round-trip；两候选必须确实不同。
4. Stage-2 输入只含当前 observation/proprio/text、候选 action 和固定初始噪声；future state、outcome、
   success label 不得作为模型输入。
5. 每对两候选复用完全相同的 future-state/value/representation 初始噪声、shift=5、10-step solver；
   交换评测顺序后 score 差异在容差内不变。
6. 严格恢复完整 C60 s10k；action generator 参数和输出字节不修改。候选 action 改变时 value score
   必须产生有限且非零的变化，否则判定 head 未使用 action。

预注册后、任何模型前向前的 mechanical audit 发现：C54 虽然 32/32 branch onset 与成功父轨迹的
`sim_state`、`previous_action` 完全一致，但 state restore 后重新观测的 RGB 和 EEF 数值不是逐字节
一致（EEF 最大差约 `6.3e-5..8.3e-4`）。因此上面的 gate 2 在读取任何 value score 前由“父/分支
RGB/proprio 也相等”修正为“动作来源 simulator state 精确相等，且两候选统一复用 branch 当前模型
输入”。32 对身份和全部统计门均未改变；评测结果中必须报告该 erratum，不能把它隐去。

第二项 pre-score implementation erratum：首轮 v1 在 H3 inference tensor boundary 被安全门拒绝，
没有产生 score；v2 在第一对完成模型计算后、写出任何 shard/score 文件前，被 value shape gate 拒绝。
审计确认官方 FACT `wa_pipeline.py:317-318` 明确初始化 future state `[B,1,D]`、value `[B,1,1]`，
Stage-2 在 `:737-738` 按 token 数构造 timestep，并在 `:820-823` 保持原 shape 更新。本地 C60
`_vector_tokens`/decoder 同样只有一个 value token。v2 错误地用 `[B,1]` sample 更新 `[B,1,1]`
prediction，广播成 `[B,B,1]`。修复固定所有 Stage-2 track 为三维 token tensor，并在每一步严格断言
`future_state=[B,1,8]`、`value=[B,1,1]`、`future_representation=[B,1,7168]`；最终直接读取唯一的
`value[:,0,0]`，禁止 mean 或末 token 选择。32 对、噪声、solver 和统计门仍未改变。

### 冻结 offline gate

- primary：成功 action 被选中至少 `22/32`；one-sided exact binomial `p<=0.05`（22/32 的
  `p=0.0250512`）。
- margin：`score(failure)-score(success)` 的 median 必须 `>0`，所有 score 必须 finite。
- spatial safety：至少 `20/30` 选择成功 action（one-sided exact `p=0.0493686`）。
- object sanity：2 对中至少 1 对选择成功；明确标为 underpowered，不据此声称 object 泛化。
- 任一 mechanical gate 或统计门失败：`NO_GO_C60_STAGE2_RANKING / KEEP_C58`，不得启动 20k、BoN
  闭环或事后更换阈值。
- 全部通过：只给出 `GO_COLLECT_CROSS_SUITE_C63_CONFIRMATORY_PAIRS`。先冻结四 suite 均衡同状态对，
  再做 confirmatory ranking；仍不得直接宣布模型晋级。

## 后续放行路径

1. 先实现并运行上述 C63 离线诊断；这是前向评测，不是训练。
2. 若通过，收集四 suite 均衡且与 C60/C63 调参集 disjoint 的同状态动作对，重新冻结 primary/suite
   gates；若不通过，停止 Stage-2/BoN 路线并把失败归因到 value head，而不是继续堆训练步数。
3. 只有 cross-suite ranking 通过，才创建独立 dossier 做同一 C60 checkpoint 的 N=1 vs N=4
   matched closed-loop canary；唯一变量为“是否按内置 Stage-2 value 选候选”。
4. 只有 matched canary 通过预注册成功率和 suite-safety 门，才考虑训练一个专门改善 value ranking
   的后继模型；C60 本身仍不回溯改名为成功。
