# H3-WAM 四条完整端口并行实验

日期：2026-08-16

## 决策

C55 已经证明“在 D0 后面追加少量 future/state/value head”能显著改善离线世界预测，
但不能转化成闭环动作收益：680 组同状态三臂评测中，joint 为
`231/680=33.97%`，action-only 为 `234/680=34.41%`，D0 为
`270/680=39.71%`。因此停止继续给 D0 打辅助 head，改为四条官方训练结构的完整 H3
骨干端口。

四条线共享相同的动作归一化、LIBERO 任务集合、D0 基线、评测协议和数据身份；在完成
默认关闭一致性、梯度、恢复、单批过拟合和真实显存门禁前，不读取闭环收益。通过机械门禁后
立即独占一个 8×A800 节点长训，互不串行等待。

## 固定上游

| 支线 | 官方代码 commit | 必须保留的核心机制 |
|---|---|---|
| C56 FACT | `618a6c16868699b6d4138941de6a863589ac00dd`；远端 `9427ea4` 仅 README 变化 | 单一 causal backbone 中 `[P,A,G,V,I]` token 顺序、teacher-forced clean action、失败动作 mask、future/value flow、两阶段推理 |
| C57 LingBot-VA | `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb` | 观测与实际执行动作共同进入跨 replan 持久 rolling KV；训练与 rollout 使用相同 KV 生命周期 |
| C58 FastWAM | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` | 完整 30 层 ActionDiT、视频骨干插值初始化、逐层视觉/动作联合注意力、flow action chunk |
| C59–C61 failure data | FACT 论文算法 1/2 与上述官方 loader | 失败轨迹不学动作、仍学真实失败未来与 base value；只有显式 onset 后才加失败 penalty |

DreamWAM `6e989facc0c452fd3488d75f60bc36411005558c`、MiniWorld
`e484206bbd4360ae56ed8abad51c83f2457ac092` 和 StarWAM
`cd76d96f273f81e228a05f40f9697fe2514e2356` 继续作为初始化、结构化未来和 carrier
实现参考，但本轮不再把它们压缩成额外浅层 head。

## 服务器分配

| 节点 | 地址 | 当前独占任务 |
|---|---|---|
| n0 | `117.50.181.177:32611` | C58b online frozen-H3 / full-30 ActionDiT 长训 |
| n1 | `117.50.181.177:30907` | C57 LingBot 长训；利用显存余量并置 C56 train-only scale gate |
| n2 | `117.50.181.177:32409` | C58 repeated-layer49 对照臂续训与固定评测队列 |
| n3 | `117.50.181.177:30234` | C61 四候选 causal failure rollout 扩容 |

共享工作区固定为 `/mnt/h3-wam`，项目为
`/mnt/h3-wam/candidate-d0-rollout-96976ce/project`。不得使用 `/root` 保存项目或权重。

## Failure data 的修正

FACT 代码只公开 `failure_rollouts.jsonl` 的消费合同，没有公开 failure-onset 自动标注器。
论文算法 2 也明确写成“when available”。因此：

- 终局失败、超时、关节位移或碰撞仅能进入 review queue，不能自动变成 onset；
- 整个失败 episode 的 `action_loss_mask=0`；
- 未标注 onset 的失败仍训练 observed future 和 base temporal value，但不加 penalty；
- 显式干预后失败的分叉轨迹，从干预边界开始训练 failure-active value；
- 官方代码使用 remaining-time `+ penalty`，论文公式使用 progress `- penalty`；两者分别保存，
  禁止静默混用。

已冻结的数据资产：

| 资产 | 结果 |
|---|---|
| C59 outcome-only overlay | 560 episodes、362 failures、21559 samples；0 条伪造 onset；`COMPLETED.json` 位于 `/mnt/h3-wam/eval/c59-fact-failure-active-overlay-v1` |
| C60 state-aligned causal failure | 83 failed branches、3115 samples；train 51 ep/17 parent sources，validation 32 ep/11 parent sources；dataset SHA256 `1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4` |
| C61 rollout expansion | C48 train-only 的 141 个成功父轨迹，d3/d5 两个状态，每状态四个首动作 seed，共 1128 jobs；同组 continuation seed 完全一致 |

C60 的 split 以成功父 episode 为单位，d3/d5 和所有 action arms 不得跨 train/validation。
第一版按 group split 的产物已移到 `.invalid-group-split`，禁止训练读取。

## 每线晋级门槛

1. 上游 commit、实际执行源码 hash、数据与 checkpoint SHA 全部冻结。
2. 新机制默认关闭时与 D0/FastWAM 父实现 bit-exact。
3. 新增 token/层/持久 state 有非零有限梯度，未来 token 不得泄漏到 Stage-1 action。
4. checkpoint 与 runtime state 恢复后同输入输出 bit-exact。
5. 真实 H3/A800 前向、反向、峰值显存和吞吐通过，不以 toy tensor 代替。
6. 先做固定 episode-disjoint 离线机制门；只选择预注册 milestone，不看闭环结果挑点。
7. 闭环统一与 D0 同初始状态、同 noise、同 replan8 配对；最终以 simulator predicate 成功率
   决策，future loss 不能替代成功率。

四线完成后先做单线淘汰。只有闭环胜者才进入两两融合；融合仍采用单变量比赛，不把四种机制
一次性堆在一起。

## 在线训练与缓存边界

C58b 已在同一80样本上证明在线冻结INT8 H3与逐层磁盘K/V在30层逐tensor bit-exact，并证明在线
H3、30层ActionDiT反向和AdamW可在单张A800约42.21GiB reserved内完成。正式C56b/C58b因此使用
`online_frozen_int8_per_rank_v1`，不再构建新的全量K/V缓存；机械parity缓存完成审计后可删除。

已有缓存只允许被已经启动且合同冻结的C57/C61读取，不再扩建。原始观测windows、manifest、split、
normalization stats、checkpoint、评测轨迹和事故证据不属于可删缓存，必须保留以便严格恢复和复现。
