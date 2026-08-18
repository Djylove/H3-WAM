# H3-WAM 完整实验复盘与下一阶段路线

日期：2026-08-18（Asia/Shanghai）

项目：MiniMax-H3 World Action Model on LIBERO

性质：阶段终局复盘；覆盖本地 5090 探索、A800 多线训练、完整闭环评测和停机归档

## 0. 阅读方式

本文回答六个问题：

1. 最初想验证什么；
2. H3 在整个 WAM 中到底承担什么角色；
3. 训练路线为什么从全量微调转向冻结 H3 + 动作专家；
4. 哪些实验真正成功，哪些只是离线信号或负结果；
5. 目前模型能力、训练预算和证据还差什么；
6. 下一次获得算力后，应按什么顺序恢复。

如果只看结论，阅读第 1、7、10、12 节。如果要复现实验，继续查看第 4、5、9、11 节。逐实验原始记录仍以
[实验总账](H3_WAM_EXPERIMENT_LEDGER.md)、[候选注册表](H3_WAM_CANDIDATE_REGISTRY_2026-08-14.md)
和 `experiments/dossiers/` 为准；本文不替代原始 dossier，而是把它们串成一条完整研发逻辑。

## 1. 最终结论

本轮已经证明：

> 冻结 MiniMax-H3 INT8，将当前双目观测、文本和多层 H3 K/V 交给有足够容量的独立动作专家，能够形成统一的
> LIBERO 四套件策略，并在 680 对相同初态闭环中稳定超过历史父模型。

本轮没有证明：

- 全量或 BF16 微调 H3 能提高机器人动作成功率；
- 视频/未来预测指标提高会自动带来动作提高；
- 当前 FACT consequence auxiliary objective 能增益动作；
- MiniWorld/LingBot context 已经形成可融合的上下文冠军；
- C69 已经按冠军晋级协议正式替代 C58b；
- 当前模型已达到 FastWAM 官方数据曝光和训练预算。

当前证据分层如下：

| 身份 | 模型 | 证据 |
|---|---|---|
| 正式晋级 carrier | C58b FastWAM full30 + H3 layer-wise | `295/680`，相对 D0 `+3.676pp`，通过全部预注册门 |
| 最高点估计、待直接确认 | C69 matched action-only | `338/680`；尚未与 C58b 在同一授权中做直接配对晋级 |
| consequence 归因处理臂 | C67 FACT joint | `324/680`，没有胜过同预算 C69 |
| 高分但未晋级 | C60 FACT | `313/680`，相对 C58b 的增益、净胜和显著性均略低于门槛 |
| 下一代架构种子 | C71 Light-WAM state fusion | 视觉依赖和 normalized MSE 强，physical/语言门失败，未做 rollout |
| 历史父模型 | D0 H32 | `270/680`，构成 C58b 的正式父基线 |

因此当前正式 fusion lineage 只有：

```text
MiniMax-H3 INT8 frozen
        |
D0 H32 five-block action parent            270/680
        |
C58b FastWAM full30 layer-wise carrier     295/680  ← promoted
```

C69应当写在谱系旁边，而不是直接覆盖谱系：

```text
C58b ── continue 20k matched action-only ── C69   338/680
                                                └─ top promotion candidate
```

## 2. 项目最初的问题

项目最初希望把 FastWAM/DreamWAM 中使用的 Wan2.2 视频世界模型替换成 MiniMax-H3，并回答：

> H3 的首帧条件视频生成和世界预测能力，能否转成机器人动作生成与闭环成功率？

早期存在三种可能路线：

1. 直接微调 H3，让同一个大模型同时学习视频和动作；
2. 把动作 token 塞入 H3 的生成路径；
3. 将 H3 作为视觉/世界特征骨干，另训练动作专家。

5090 实验首先证明 H3 INT8 可以在单卡运行，也证明首帧视觉特征确实能因果影响动作；云端 A800 的作用不是
再次证明“能跑”，而是扩大数据、动作专家容量、训练预算和完整 LIBERO 评测。

## 3. 为什么放弃全量微调主线

早期 BF16、LoRA、tail 解冻和共享 video-action 训练反复出现同一形状：

- video/world 或普通 held-out regression 指标持续改善；
- 视觉生成表征越来越强；
- 动作的 gripper、接触、语言响应或闭环成功率反而下降；
- 晚期 checkpoint 有时出现 visual conditioning 响应接近消失，动作头仍能降低平均 MSE。

这说明视觉预测目标和机器人控制目标没有天然同向：重建背景、纹理和一般运动可以改善平均视频指标，却不一定保留
目标物体选择、接触瞬间、夹爪切换和空间关系。与此同时，H3 表征不断漂移，动作头需要持续追逐新的特征分布。

因此主线改为：

```text
H3 INT8：冻结、只负责当前观测世界特征
Action expert：BF16、承担动作生成和主要优化预算
World/consequence/context：作为独立赛道，必须单变量晋级后才能融合
```

这不是说 H3 不应再适配，而是把适配推迟到动作专家已经成熟之后，并限制为残差 LoRA/adapter，而不是重新全量解冻。

## 4. 最终系统结构与合同

### 4.1 输入与输出

- 任务：LIBERO-10、Goal、Object、Spatial，共 40 个任务；
- 图像：双相机当前观测/首帧；
- 文本：完整任务语言；
- proprio：8D；
- action：7D absolute action，horizon 32；
- 推理：shift-5 flow，10-step solver，replan 8，episode 最多 400 步；
- 成功：只读取 LIBERO 环境的真实 `check_success`，物体位移和离线 MSE不能替代。

### 4.2 H3 的实际角色

H3 在最终主线中不是直接输出动作，也不在部署时先生成完整 RGB 视频。它接收当前观测和文本，产生多层 K/V 世界特征；
ActionDiT 将动作 query 与这些 K/V 做融合，再通过 flow solver 生成 32 步动作块。

这使首帧生成能力有用：H3 为单帧观测提供包含外观、语义和潜在运动的世界表征，但动作推理不需要等待完整视频生成。

### 4.3 数据合同

最终 dense 训练合同包含 `200,779` 个 expert train windows、`22,150` 个 validation windows；训练/验证按
episode 隔离。一个训练 window 覆盖 33 个原始时刻和 32 个连续动作，不再使用每 episode 仅抽 5 帧的早期稀疏方案。

数据教训是本项目的关键资产之一：episode 数量大不代表状态覆盖充分。动作训练必须覆盖接近、抓取、接触、搬运、释放和
成功前后的逐帧状态，不能以少量均匀快照冒充 dense sampling。

### 4.4 评测合同

候选先跑固定 episode-disjoint balanced80，报告：

- normalized/physical action MSE；
- gripper macro-F1；
- visual feature shuffle；
- language replacement；
- strict checkpoint restore。

通过离线门后才允许闭环。正式晋级使用 680 对：四个 suite × 10 tasks × trials 33..49；每个 episode 使用独立
simulator 和 policy process，并逐对核验初态、checkpoint、source freeze、seed 和结果 SHA。

晋级门固定为：

- 总成功率至少提升 3pp；
- paired net wins 至少 20；
- 单侧 exact McNemar `p <= 0.05`；
- 任一 suite 不退化超过 3pp。

## 5. 研发过程

### 阶段 A：本地 5090 与 standalone H3

本地首先完成 H3 INT8 首帧特征链路和小动作头。历史三个单任务 head 曾有较高成功率，zero-feature 消融为零，证明 H3
当前观测表征不是装饰性输入。迁移到 standalone native H3 后，固定 30 episode 回归得到 `22/30`，未通过预注册
`26/30`，说明链路可用但旧单任务头不能直接作为统一策略。

该阶段同时发现旧 Comfy capture 的多层特征存在 storage alias：五个槽位最终都是 layer49 副本。因此早期结果只证明最后层
H3 特征有价值，不证明真正的多层融合。

### 阶段 B：统一动作专家与 conditioning collapse

StarWAM/早期 shared-H3 ActionDiT 路线证明 30 层动作模型能训练，普通 MSE随 steps下降；但晚期视觉打乱和语言替换响应下降，
gripper退化，构成 conditioning collapse。结论不是“30层无效”，而是当前接口、缩放、目标或训练策略让动作头学会数据平均，
没有持续利用条件信息。

这个阶段产出仍然重要：完整 ActionDiT、flow scheduler、strict restore、episode-disjoint evaluator、特征缩放和动作归一化都被后续复用。

### 阶段 C：D0 dense carrier 与 horizon

DreamWAM-style D0 使用冻结 H3 K/V 和五层动作专家。稀疏数据闭环失败后，项目把数据密度作为唯一变量，建立 dense stride-1
窗口和 H32 动作合同。D0-H32-s14000最终成为完整 benchmark 的历史父模型，成功率 `270/680`。

D0 证明冻结 H3 路线能形成统一策略，但五层动作专家和重复 layer49 carrier 仍可能限制 H3 多层世界知识的利用。

### 阶段 D：C58b FastWAM full30 layer-wise carrier

C58b直接使用 FastWAM 固定源码中的完整30层 ActionDiT，并把 H3 的50层按单调深度映射到30个动作 block：

```text
H3 layers 0 ... 49  →  ActionDiT blocks 0 ... 29
```

相对 D0 的主变量是动作塔容量和逐层 carrier。H3继续冻结，数据、动作目标和评测合同保持不变。C58b训练10k steps、80k
samples，约 `0.398448` expert epoch。

最终结果：

- C58b：`295/680 = 43.382%`；
- D0：`270/680 = 39.706%`；
- 绝对提升：`+3.676pp`；
- discordant：C58b赢87、D0赢62，净胜25；
- 单侧 exact McNemar：`p=0.02446`；
- suite safety：PASS。

这是本项目第一条完整通过 source、mechanical、paired 和 rollout gate 的 H3-WAM 主线。

### 阶段 E：consequence、ranking 与 FACT

项目没有直接跳到 C60，而是先验证动作条件后果是否可学：

- E49/C38：action-conditioned future-H3 机制成立；
- C44：离线 binary ranking 通过；
- C51/C52：dense value 和 fresh counterfactual ranking 通过；
- C45/C46、C53/C54：对应在线 best-of-N/ranker 未产生稳定闭环增益。

这形成了一个重要负结论：**后果和值可以被预测，不等于它们能在部署时选出更好的动作。** behavior-policy 数据、触发状态、
候选分布和训练/部署合同不一致都会让离线 ranker 失效。

C55随后把 future/state/value 辅助目标放回共享动作块，离线一度改善，但680闭环为 joint `231`、action-only `234`、D0 `270`，
正式否证浅层联合辅助。

C60使用 C58b 30层 carrier 和更完整 FACT causal tower，得到：

- C60：`313/680 = 46.029%`；
- C58b：`295/680 = 43.382%`；
- `+2.647pp`、净胜18、单侧 `p=0.0507164`。

C60点估计更高，但三个正向门都略低于阈值，因此保持 `KEEP_C58_PARENT`。同时审计发现官方 FACT 部署使用 Stage-2
action-conditioned consequence/value 做 best-of-N，而本地 C60 rollout仍是 N=1 action-only adapter；这意味着 consequence分支
在训练时共享梯度，却没有在部署时直接参与选择。

为了分清“更多动作训练”和“consequence目标”，项目完成了同预算20k归因：

| 指标 | C67 FACT joint | C69 action-only |
|---|---:|---:|
| 成功 | 324/680 | 338/680 |
| 成功率 | 47.647% | 49.706% |
| 配对独胜 | 23 | 37 |
| C67-C69 | -2.059pp | — |
| 双侧 exact p | 0.09246 | — |

因此只能得出：当前 consequence auxiliary objective 没有产生可检测的增量价值。两臂都使用 H3，不能把该结果解释为“H3世界特征无效”。

C69虽然有当前最高点估计，但本轮预注册问题是 C67 对 C69 的机制归因，不是 C69 对 C58b 的冠军晋级；C58b没有在这一授权中
与 C69重新逐对运行。因此 C69是 `TOP_PROMOTION_CANDIDATE`，不是已经完成统计晋级的冠军。

### 阶段 F：context / persistent KV

项目先后尝试 executed-action history、LingBot predicted-cache rollback、真实 observation/action persistent KV、MiniWorld sink/FIFO、
framewise context 和 temporal RoPE。

- C14：窄任务有短暂收益，但新 trials未复现并伤害Object；
- C57：完整5000 steps、约1.99 epochs，fixed80反而比D0差1.166%；
- C62：correct/shuffle context不可分，context-on比off差3.378%；
- C64：证明 temporal key RoPE 能改变K，但只完成机械门；
- C66：历史确实进入模型，但 clean context相对context-off明显退化。

所以 context赛道没有冠军。可复用的是 persistent lifecycle、executed-action feedback、rollback、valid mask、framewise bridge 和 RoPE
实现，不是任何已训练权重。

### 阶段 G：C70 sampler 与 C71 Light-WAM

C70只调整expert/success/failure采样覆盖，并保持C67结构和20k预算。它通过机械门但terminal balanced80的paired action win gate失败，
没有进入rollout。它排除了“只调整采样覆盖即可修复 FACT”的简单假设。

C71直接移植 Light-WAM 三层state fusion的最小机制：H3层14/27/41、learned-query pooling和浅动作trunk。9918步终点相对C58b：

- normalized MSE改善4.70%；
- gripper macro-F1改善0.49%；
- visual shuffle delta为0.193922，视觉依赖强；
- physical MSE恶化11.49%；
- language relative-L2只有0.064134，而C58b为0.894055。

C71证明轻量动作专家能有效吸收H3视觉表示，但没有守住物理动作和语言条件，因此停止在rollout之前。它是下一轮架构种子，
不能与C58b融合后直接宣称更强。

## 6. 主要实验结果总表

| 模型 | 主要变量 | 预算 | 离线/机制 | 闭环 | 最终身份 |
|---|---|---:|---|---:|---|
| D0-H32 | dense数据、五层动作头 | s14000 | 合同通过 | 270/680 | historical parent |
| C58b | full30 + layer-wise H3 | 10k / 80k samples | balanced80通过 | 295/680 | promoted carrier |
| C60 | C58b + FACT joint | 10k | 最高早期信号 | 313/680 | failed promotion |
| C67 | FACT joint长预算 | 20k / 160k | conditioning-safe | 324/680 | attribution treatment |
| C69 | matched action-only | 20k / 160k | attribution gate通过 | 338/680 | top candidate/control |
| C70 | sampler coverage | 20k / 160k | terminal gate失败 | 未跑 | eliminated ablation |
| C71 | Light-WAM shallow fusion | 9918 / 79344 | 视觉强、物理/语言失败 | 未跑 | architecture seed |
| C57 | LingBot persistent KV | 5k / 400k samples | fixed80低于D0 | 未跑 | context no-go |
| C62 | MiniWorld rolling context | 100 / 800 samples | clean≈shuffle且差于off | 未跑 | context no-go |

注意：不同线路的 `samples` 来自不同stream和global batch，不能只按steps横向排序。闭环结果也只能在身份、初态和授权协议成对时做统计归因。

## 7. 当前最合理的模型判断

### 7.1 H3能否作为WAM基础模型

可以。C58b的680对结果已经证明H3 frozen carrier具有真实闭环价值，而不是只改善离线视频指标。更准确的表述是：

> H3是可用的世界/视觉基础模型；机器人能力上限目前主要受动作专家、动作数据曝光和部署合同限制。

### 7.2 C69是否可能已经强于C58b

很可能，但尚未完成严格晋级证明。表面差值为 `338-295=43` 次成功，即 `+6.324pp`，足以成为最高优先验证对象。缺失的是同一
authorization下的 C69/C58b 680对初态、suite safety、net wins和McNemar报告。

### 7.3 更多训练是否可能继续提高

证据支持这一假设：C58b只有约0.398个expert epoch，C69继续20k后，共享动作塔累计expert exposure约0.797 epoch，仍不到一轮。
但这不能写成单调定律；C60和若干早期线已经出现平均指标改善、少数接触状态严重退化。

当前C69混合采样每step含4个expert rank。以200,779个expert windows计算，从C58b父模型起达到累计expert exposure所需的C69增量步数约为：

| 累计expert epochs | C69增量steps | 用途 |
|---:|---:|---|
| 1 | 30,195 | 首个完整数据曝光门 |
| 2 | 80,390 | 中程长训主比较 |
| 5 | 230,974 | 充分训练研究点 |
| 10 | 481,948 | 接近官方epoch量级的远期预算 |

success/failure streams有各自重复次数，不能被这个expert epoch数字掩盖。下一轮应报告逐stream exposure，而不是只写总epoch。

### 7.4 能否让视觉与动作同步提升

当前冻结H3方案实现的是“保留视觉能力、提升动作利用”，不是更新H3视觉能力。真正双升需要在成熟动作专家上增加零初始化残差adapter：

```text
f_final = f_frozen_H3 + alpha * delta_f_adapter
```

H3主体保持冻结，adapter学习动作相关视觉增量；动作专家使用更高LR，adapter使用低10～100倍的LR，并用原始H3 feature anchor限制漂移。
必须同时通过world/visual、language、physical action、gripper和闭环门，不能用单一视频指标放行。

## 8. 失败原因矩阵

| 失败表现 | 主要原因 | 已有证据 | 下一次不能怎么做 |
|---|---|---|---|
| 视频更好、动作更差 | 目标梯度冲突、H3表征漂移 | BF16/full-H3与shared路线 | 不直接恢复全量H3微调 |
| MSE下降但条件消失 | 动作头学数据平均、conditioning collapse | R1/Candidate F晚期 | 不用latest或更多steps绕过反事实门 |
| consequence可预测但动作不提高 | 离线目标不约束生成；候选分布错位 | C38/C44/C51正，C45/C46/C53/C54负 | 不再训练外挂线性ranker |
| FACT joint不如action-only | 当前future/state/value辅助梯度无增量 | C67 vs C69 | 不原样延长C67 |
| context正确但比off差 | 时序平均、RoPE/cadence错位、历史扰乱当前动作 | C57/C62/C66 | 不把多个context修复一次性堆入 |
| visual direct-action强但physical差 | 输出标定、语言约束不足 | C71 | 不因视觉指标强直接rollout |
| 稀疏数据闭环无接触 | 接触阶段state coverage不足 | D0早期与数据审计 | 不恢复5帧/episode抽样 |
| 大缓存拖慢研发 | 数TB K/V、预计算长、引入离线边界 | C58b cache canary | 继续online frozen H3，不生成全量K/V cache |

## 9. 方法、源码与上游身份

| 方法 | 固定上游 | 本地核心实现 |
|---|---|---|
| D0 DreamWAM carrier | DreamWAM `6e989fac...` | `dreamwam_kv_carrier.py`、`train_h3_int8_dreamwam_kv_carrier.py` |
| C58b full30 | FastWAM `45d8e145...` | `fastwam_full_tower.py`、`c58_online_training.py` |
| FACT joint/action-only | FACT `618a6c16...` | `fact_layerwise_tower.py`、`fact_online_data.py`、`train_c56b_fact_online.py` |
| MiniWorld context | MiniWorld `e484206b...` | `c62_miniworld_context.py`、`c64_miniworld_framewise_context.py` |
| LingBot persistent KV | LingBot pin见dossier | `lingbot_persistent_kv.py`、`c66_lingbot_fastwam_persistent.py` |
| Light-WAM fusion | Light-WAM `b2785f66...` | `lightwam_state_fusion.py`、`train_c71_lightwam_online.py` |
| StarWAM早期动作头 | StarWAM `cd76d96f...` | 对应R1训练与评测脚本，保留为负结果和合同资产 |

所有H3替换均属于 `backbone_port` 或 `novel_composition`，不是上游官方模型的完全复现。详细逐字段差异见
[阶段代码与预算审查](H3_WAM_PHASE_REVIEW_2026-08-17.md) 和 [上游锁文件](UPSTREAM_SOURCES.lock.json)。

## 10. 下一阶段实验计划

### P0：先确认C69是否为新冠军

不训练新模型，直接执行 C69-s20000 vs C58b-s10000 的680对完全相同初态闭环：

- 固定source snapshot、H3、动作codec、normalization、solver和replan；
- 每对重新运行两个arm；
- 使用C58b既有3pp/净20/p值/suite safety门；
- 不用历史C58结果与新C69结果做非配对拼接。

若通过，fusion parent更新为C69；若不通过，保留C58b并把C69定义为长预算诊断。

### P1：action-only长预算学习曲线

父模型固定为C58b；保持C69 action-only合同，训练到30,195、80,390，并在资源允许时到230,974增量steps。建议：

- 每5k或10k保存一次完整checkpoint；
- 每个阶段保存训练曲线和strict restore，但不保存全部1k大权重；
- balanced80异步评测固定milestone；
- 30k、80k和最终点才进入闭环，禁止事后挑最佳点；
- 统计expert、success、observational failure、causal failure四流各自exposure。

这条线回答“动作专家是否仍受训练不足限制”，优先级高于新辅助loss。

### P2：视觉—动作双升的残差适配

只有P1得到成熟action-only父模型后启动：

1. A：frozen H3 + action expert继续训练；
2. B：A + zero-init H3 residual adapter，只接受action loss；
3. C：B + action-related world target，其他合同不变。

建议目标优先级：动作flow > object motion/contact/future proprio > feature anchor > 一般RGB重建。首轮不要同时加入context、ranking和sampler变化。

### P3：C71视觉路径修复

保留三层state fusion和learned-query pooling，只允许一个主变量：

- language-preservation loss；或
- physical-space action calibration；或
- 从C58/C69蒸馏动作输出。

先恢复physical和language paired gate，再考虑rollout。不要只延长C71步数。

### P4：context与FACT Stage-2

- context：从C64拆分 cadence-only、learned-null、temporal RoPE三个父子实验；
- FACT：不再恢复C67 auxiliary长训。先只读验证现有Stage-2是否能在同状态候选中排序，再做N=1 vs N=4闭环；
- 两条线都必须先产生独立赛道胜者，才能和action champion融合。

## 11. 恢复训练的最短操作顺序

1. checkout Git `main` 并记录commit；
2. 验证 H3 INT8 SHA256 `e889202c...d03c47a`；
3. 验证目标checkpoint size/SHA和strict restore `max_abs=0`；
4. 检查dense manifest、train stats和selected80 hash；
5. 运行一个真实LIBERO process，确认gripper、normalization、horizon32/replan8；
6. 先做P0 C69/C58b直接晋级，不改权重；
7. 再启动P1 action-only长预算；
8. P1形成稳定父模型后，才启动P2/P3；
9. 任何训练都同时保存resolved command、source freeze、data hash、checkpoint contract和评测JSON。

归档资产和checkpoint SHA见
[停机归档manifest](../experiments/archive/h3_wam_shutdown_manifest_2026-08-18.json)。本机核心结果包为
`artifacts/core_results_json_2026-08-18.tar.zst`，完整执行源码包为
`artifacts/execution_source_0cc9d9e.tar.zst`。

## 12. 项目当前状态定义

截至本轮结束：

```text
H3 feasibility                 PASS
Native INT8 H3 without ComfyUI PASS
Dense unified LIBERO policy    PASS
C58b carrier promotion         PASS / EVIDENCE_READY
C69 direct champion promotion  PENDING
Full-H3 fine-tuning            STOP AS MAINLINE
FACT consequence increment     NOT DETECTED
Context track champion         NONE
Light-WAM track champion       NONE
Full FastWAM-level exposure    NOT YET REACHED
```

最重要的研发判断是：

> H3不是当前主要瓶颈。现阶段应把算力优先投入到充分训练和严格确认动作专家；在动作父模型成熟后，再以可回退的残差adapter探索视觉—动作双升。

这使下一阶段不需要重新清零，也不需要继续围绕全量微调反复试错。我们已经有明确父模型、最高候选、失败边界、评测门和恢复路径。

## 13. 关联文档

- [云资源停机归档与研发交接](H3_WAM_SHUTDOWN_HANDOFF_2026-08-18.md)
- [实验资产总账](H3_WAM_EXPERIMENT_LEDGER.md)
- [候选模型注册表](H3_WAM_CANDIDATE_REGISTRY_2026-08-14.md)
- [代码、来源与训练预算审查](H3_WAM_PHASE_REVIEW_2026-08-17.md)
- [context / consequence历史总审计](H3_CONTEXT_CONSEQUENCE_HISTORY_2026-08-17.md)
- [C67/C69最终归因报告](C67_C69_PAIRED_ATTRIBUTION_RESULT_2026-08-18.md)
- [C60失败因果诊断](H3_C60_FAILURE_CAUSAL_DIAGNOSIS_2026-08-17.md)
- [checkpoint与归档manifest](../experiments/archive/h3_wam_shutdown_manifest_2026-08-18.json)
