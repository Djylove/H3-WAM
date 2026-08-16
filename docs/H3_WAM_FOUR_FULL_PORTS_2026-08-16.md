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
| n2 | `117.50.181.177:32409` | C57 固定评测队列；C58b 最终 READY 后自动切换 C56 主臂 10k |
| n3 | `117.50.181.177:30234` | C61 四候选 causal failure rollout；数据终审后自动切换 C56+C61 匹配臂 10k |

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

## 2026-08-16 在线执行状态

以下状态是训练中的证据快照，不代表最终效果结论：

- 已删除约 182 GiB 不再进入正式合同的 C56a structured5、StarWAM/DreamWAM NO_GO、
  C58b parity 与机械探针缓存；共享盘仍约有 24 TiB 可用。C57 已冻结的持久 KV 资产与正式
  raw windows 保留，不再生成新的 H3/KV 缓存。
- C58b 的 8-GPU 10k 在线长训已通过 10-step DDP、30/30 层梯度、无磁盘 KV、严格恢复门；
  `s1000..s4000` 已原子落盘，训练进入 s5000。最终 watcher PID `952239` 等待
  `online-long10000/READY.json`，随后运行 online-H3/no-disk-KV balanced80，并以同
  trial33、environment seed42、policy seed330042 配对 C58b 与固定 SHA 的 D0，各跑完整
  40-task LIBERO micro-benchmark，报告 exact McNemar。
- C56b 完整 FACT 端口的 mixed8 loss/scale gate 与 10-step optimizer canary 已通过：四类 rank
  配比固定为 `4 expert / 2 success / 1 observational failure / 1 causal failure`，30 层均有
  非零梯度，失败样本动作损失严格为零，future-to-action leak 为零，restore max-abs 为零。
  n2 watcher PID `387447` 只接受 C58b s10000 最终 checkpoint 的路径、大小和 SHA，随后启动
  C60 主臂 10k。
- C61 共 1128 个 exact-state 反事实分叉正在 n3 采集。finalizer PID `861397` 只在 1128 份
  results 与 trajectory 和单节点完成 marker 全部齐全后生成正式数据；matched watcher PID
  `906377` 再校验 `COMPLETED.json`、dataset/observations SHA，并以与 C56 主臂完全一致的
  父模型、seed、LR、loss、target norm、10k steps 启动。唯一变量是 C60/C61 causal pool。
- C56 两臂各自 s10000 后必须通过最后 1000 步有限 loss、30/30 梯度、零 future leak、
  checkpoint contract 和 bit-exact restore 才能发布 READY。随后统一 balanced80，并在相同
  trial33/seed 下输出 C61-vs-C60、C60-vs-C58b、C61-vs-C58b 三组配对闭环效应；不能仅凭
  训练 loss 或 C61-vs-C60 单一比较宣布 FACT 有效。
- C57 5k 训练固定每 200 步保存并在 80 条、每 suite 20 条的 episode-disjoint heldout 上评测；
  相对 D0 的曲线为 s200 `-75.58%`、s400 `-36.35%`、s600 `-24.82%`、s800
  `-17.64%`、s1000 `-13.77%`，样本胜率从 `6.25%` 升至 `35%`，但仍是 NO_GO。
  s1000 首次评测暴露了 CUDA/cuBLAS 动态库顺序错误；空闲 A800 最小 BF16 Linear 可复现，
  固定使用 h3-int8 runtime 自带 cu13 后探针和完整 80 条评测均通过。n2 queue PID `388059`
  与终点 watcher PID `387552` 已加入真实 C56 进程识别、失败隔离和 fail-closed 重试。

所有中间趋势只用于诊断，不用于挑 checkpoint。C57 只允许预注册 s5000、C58b/C56/C61 只允许
预注册 s10000 进入最终闭环；闭环报告未生成前，四条线的效果状态均为
`NOT_EVIDENCE_READY`。
