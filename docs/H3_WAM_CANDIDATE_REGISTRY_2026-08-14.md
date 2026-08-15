# H3-WAM 开源候选注册表与炼蛊谱系

更新时间：2026-08-15（Asia/Shanghai）

## 目标与边界

可证伪总假设：把多个官方开源 WAM 的独立强机制适配到同一个 MiniMax-H3、dense LIBERO 数据与
闭环协议后，至少一个候选会在保持视觉/语言条件依赖的同时产生固定 LIBERO 成功；不同赛道的胜者逐级
融合后应相对直接父模型获得可归因的额外成功率。

当前 `D0 sparse s963` 只是 **incumbent（当前擂主）**，不是 representation/carrier 冠军，更不是最终
H3-WAM。所有 H3 替换均标为 `backbone_port` 或 `novel_composition`，不得称为官方复现。

## 统一实验合同

- backbone：MiniMax-H3 INT8，checkpoint SHA256
  `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a`；除单独登记的局部解冻候选外冻结；
- 数据：LIBERO 四套、episode-disjoint v7 dense；train `200,779` windows，val `22,150` windows；
- train manifest SHA256：`b0d611c21059fa7da6fb08162b03efadd59aff68354bb101be41d3ae20d98eb1`；
- 动作：7D、horizon 32、训练 split 独立统计、gripper 最后一维；
- 初始预算：global batch 8，先 `7,704` unique samples / `963` steps（dense epoch `0.03837`）；
- 规模预算：胜者才进入 `200,776` samples / `25,097` steps（dense epoch `0.99999`）；
- 离线：同一 balanced-80、same-noise solver、physical error、gripper macro-F1、language replacement、
  无 self-map visual shuffle；
- 在线：固定 task/trial、wait/max/replan、action clip、gripper decode 和 success predicate；只有 predicate
  success 计成功，物体位移不代替成功率。

## 官方来源注册表

| 来源 | 固定本地 revision | 本地状态 | 开源级别 | 主要可迁移机制 | 当前边界 |
|---|---|---:|---|---|---|
| FastWAM | `45d8e1458921` | dirty 828；只读固定 commit | TRAINABLE | dense 33-step span、ActionDiT、shifted flow、联合 video/action 训练、LIBERO evaluator | H3 完整联合端口未完成 |
| StarWAM | `cd76d96f273f` | clean | TRAINABLE | feature-conditioned、MoT、Shared-DiT 三家族、30层 ActionDiT | R1 有稀疏端口；缺统一 v7 dense 重测 |
| DreamWAM | `6e989facc0c4` | clean | TRAINABLE | layer-wise video K/V、ActionDiT、RGB/motion/depth/DINO structured future | D/D0 carrier 已落地；完整 structured future 未晋级 |
| ImageWAM | `5d4a341ed20a` | clean | TRAINABLE | source-conditioned target image/edit feature + ActionDiT | H3 首帧 I2V/target-image adapter 未完成 |
| FACT | `618a6c168686` | clean | TRAINABLE | causal act-then-imagine、future state/value、failure-aware、best-of-N | 已有6048条因果rollout transition；缺failure onset与反事实结果，暂只准value诊断 |
| MiniWorld | `e484206bbd43` | clean | TRAINABLE | block-causal diffusion forcing、长度课程、rolling KV | 是 world model，不是可直接 rollout 的动作策略 |
| LingBot-VA | `7c6ffa9bfc4b` | clean | TRAINABLE | shared video/action blocks、persistent KV、executed-action history | H3 shared 端口已测；完整 history 合同未对齐 |
| DiT4DiT | `66a6f3a12e2c` | dirty 579；只读固定 commit | TRAINABLE | 中间层 visual feature + 独立 ActionDiT，video loss 更新 backbone | 本地早期探索不等价于官方完整端口 |
| Motus | `f771216802b8` | clean | TRAINABLE | video/action/text 三专家 MoT、多阶段预训练/SFT | 依赖 Stage-2 初始化；H3 端口未开始 |
| Cosmos Policy | `18a2accadf4e` | dirty 23；只读固定 commit | TRAINABLE | proprio/action/future-state/value latent slots、success/failure mixture | 2B 配方不能直接缩放到 H3 33B |

远端发现但尚未固定源码的 MemoryWAM、Kairos、WLA、ABot-PhysWorld 与 minWM 只进入 discovery queue。
其中仅发布说明/推理代码或仅 world-model 训练而没有动作策略闭环者，不占用正式训练预算；先完成
`SOURCE_GATE` 并固定 revision，再决定进入上表。

## 实验候选池

| ID | 赛道 | H3 候选 | 官方代码来源 | 直接父模型 | 唯一变量 | 当前阶段 | 训练许可 |
|---|---|---|---|---|---|---|---|
| C00 | incumbent | D0 sparse s963 | DreamWAM + local H3 adapter | 无 | 历史稀疏基线 | offline pass / rollout `0/2` | NOT_EVIDENCE_READY |
| C01 | carrier | D0 dense | DreamWAM | C00 | 5 windows/episode → dense starts | cache running | GO_CANARY |
| C02 | carrier/spatial | D dense five-layer K/V | DreamWAM | D0-H32 matched training lineage | layer49 repeat → aligned layers 9/19/29/39/49 | s14000 Spatial5/20、Object3/8，均输父模型 | NO_GO；FAIL_PAIRED_GATE |
| C03 | carrier/action | StarWAM dense full30 s100 anchor | StarWAM | historical v8 StarWAM s100 | v8 frame-indexed → common v7 valid-window split/cache | dossier pass；dual H3 cache running；historical s850/s913 condition collapse | GO_CANARY s100 only；NO_GO_LONG |
| C04 | action/co-training | FastWAM-H3 | FastWAM | fixed FastWAM Wan baseline + C03 interface baseline | Wan video expert → H3 while preserving official action path | source pass；adapter incomplete | PROBE_ONLY |
| C05 | carrier | ImageWAM-H3 target image | ImageWAM | carrier winner | video K/V → H3 target-image/edit representation | source pass；adapter incomplete | PROBE_ONLY |
| C06 | carrier/action | DiT4DiT-H3 | DiT4DiT | carrier winner | selected intermediate H3 features + official ActionDiT | source pass；dirty vendor audit pending | PROBE_ONLY |
| C07 | temporal/context | LingBot-H3 executed history16 | LingBot-VA | shared-H3 parent | explicit executed-action feedback；尚非 persistent observation KV | s2500 offline champion，但 fixed rollout `0/1` 且无接触 | NO_GO current port；persistent-KV 需新 dossier |
| C08 | temporal/context | MiniWorld-H3 rolling memory | MiniWorld | C07 or carrier/action winner | rolling KV + diffusion-forcing curriculum | policy bridge absent | PROBE_ONLY |
| C09 | consequence/ranking | FACT-lite H3 | FACT | carrier/action/context winner | causal future state/value head；action path cannot see future | future-proprio/future-H3机制门通过；160条rollout已抽出6048 transitions，但failure onset未知 | GO_CANARY value-only dossier；NO_GO failure imitation/ranking |
| C10 | consequence/ranking | FACT best-of-N | FACT | C09 | rank sampled action chunks by learned progress/value | waits C09 calibration | NO_GO parent gate |
| C11 | structured future | DreamWAM motion/depth/DINO | DreamWAM | carrier/action winner | add exactly one structured future target per child | source pass；target caches pending | PROBE_ONLY |
| C12 | action/co-training | Motus-H3 three-expert | Motus | best joint-training baseline | three-expert MoT | Stage-2/H3 mapping unknown | PROBE_ONLY |
| C13 | consequence/state | Cosmos-H3 latent slots | Cosmos Policy | consequence winner parent | future state/value latent slots | scale/data mismatch open | PROBE_ONLY |
| C14 | temporal/progress | D0-H32 executed-action history16 adapter | LingBot-VA persistent video/action KV + local D0 parent | D0-H32-s14000/replan8 | add zero-init 16-action progress adapter；freeze parent | s14500 trials0/1长程+1，但trials2/3无新增；s17000过训练 | 机制证据保留；NO_GO_FUSION |
| C15 | carrier/spatial | D0 dual-view grid K/V | local H3 adapter + DreamWAM carrier | D0-H32-s14000/replan8 | 32-token一维池化 → 左右相机各4×4网格池化 | Spatial7/20与父持平；Object2/8低于父5/8 | NO_GO；FAIL_PAIRED_GATE |
| C16 | temporal/value | held-out progress value diagnostic | FACT | D0-H32-s14000 rollout trajectories | 只对成功轨迹按官方future-state time-to-go造target；失败轨迹右删失 | 1001 transitions；trial3 val238；LIBERO-10 val正样本0 | GO_DIAGNOSTIC；NO_GO_BEST_OF_N |
| C17 | temporal/value | expert progress value diagnostic | FACT + v7 dense expert windows | frozen H3 carrier | task+absolute-step+H3预测offset32 time-to-go；不改动作 | feature PASS，但shadow AUROC0.1875；absolute-step形成时间捷径 | NO_GO_POLICY_INTEGRATION；保留诊断证据 |
| C18 | temporal/value | time-blind expert progress diagnostic | C17冻结特征与split | C17 | 删除absolute-step，仅task+H3预测time-to-go | feature MAE下降53.8%，但shadow AUROC0.5469<0.65 | NO_GO_POLICY_INTEGRATION；需failure/action outcome |
| C19 | deployment/data | old-trajectory exact restore | LIBERO `regenerate_obs_from_state` | D0保存的qpos/qvel/time | 要求恢复观测等于原轨迹观测 | state exact但图像/proprio不等；旧state缺observable动态 | NO_GO_ORIGINAL_OBSERVATION_CLAIM |
| C20 | deployment/data | paired canonical branch restore | C19 + fixed per-reset seed | C19 | 两个独立env从同state执行同8步动作 | 四suite×3状态图像逐像素一致，数值差≤1e-10 | GO_COUNTERFACTUAL_COLLECTION_CANARY |
| C21 | consequence/data | same-state stochastic-continuation canary | D0-H32-s14000/replan8/no-ensemble + C20 | C20 | 固定规范state/环境/模型，改变整条policy noise schedule；4 suite×4分支 | 16条中11成功；Object同状态3/4成功；四组动作均不同 | GO_ENTROPY_CALIBRATION_ONLY；NOT_EVIDENCE_READY |
| C22 | consequence/data | multisuite stochastic-continuation entropy sweep | C21 | C21 | 8源episode×距成功1/3/5 replans×4 noise schedule | 96条71成功；7/24 mixed覆盖四suite | GO_CAUSAL_FIRST_ACTION_CANARY；NOT_EVIDENCE_READY |
| C23 | consequence/action | first-action-only causal branch | C22高熵state + D0父策略 | C22 | 只改变首replan noise；同组后续noise逐值固定 | 32条18成功；Spatial d5同状态2/4；全部seed/动作审计通过 | GO_EPISODE_DISJOINT_CAUSAL_DATASET_CANARY；NOT_EVIDENCE_READY |
| C24 | consequence/action | first-action execution horizon sweep | C23同8 states/32 seeds | C23 horizon8 | 仅首chunk执行16或32步；后续仍replan8 | h16:2 mixed/2 suites；h32:3 mixed/2 suites | GO_CAUSAL_DATASET_H32；NOT_EVIDENCE_READY |
| C25 | consequence/data | episode-disjoint h32 causal dataset canary | C24 h32 | C24 | 14源episode/32 state组/128分支；源episode隔离split | 9 mixed：train6/val3，覆盖10/Object/Spatial | GO_FROZEN_H3_ACTION_CRITIC_CANARY；NOT_EVIDENCE_READY |
| C26 | consequence/ranking | frozen-parent causal action critic | FACT consequence + D0 live H3 + C25 | C25 | action-only、H3×action、FACT consequence三臂；21 train pairs/9 untouched val pairs | train均21/21；val为0/9、4/9、0/9，全部失败；H3 top1 2/3、p=0.6875 | NO_GO_BEST_OF_N；小样本跨episode反转 |
| C27 | consequence/data | fresh expanded causal action outcomes | C25 execution contract | C26失败诊断 | 排除全部C22/C25源episode；39新源episode/78 state组/312分支；先冻结split | 312/312完成；198成功；train 13 mixed/42 pairs，fresh val 4 mixed/12 pairs，三suite均覆盖 | PASS_DATA_GATE；仅放行C28一次确认 |
| C28 | consequence/ranking | fresh frozen-H3 action critic confirmation | C26 train-LOO selected config | C27 passed data | C25全部转train+C27 train；C27 val保持新鲜；10 full-pair steps；action-only与H3×action | train H3 65/72；fresh val H3 6/12/top1 2/4/p=.586，action-only 7/12/top1 2/4 | NO_GO_BEST_OF_N；静态H3×action不跨episode泛化 |

`PROBE_ONLY` 只允许代码审计、adapter 单测、真实 forward/backward 和不保留权重的一步探针；不能生成
候选 checkpoint 或宣称效果。

## 第一轮赛程与资源顺序

1. 完成 C01/C02 共用的 v7 五层 K/V cache 全量审计；缓存未完成前不启动正式训练。
   同一次 H3 forward 还会写 C03 所需的 layer49 pooled feature；双输出与两条独立路径均已逐 bit 对齐，
   证据见 `experiments/evidence/h3_int8_dual_cache_parity_v1.json`。
2. C01 与 C02 使用同初始化、同 dense 样本和同 s963 预算配对；立即做 balanced-80 和固定 2-task
   rollout。两者只是 carrier 赛道选拔。
3. 同期只用 CPU/小 GPU 完成 C03/C04/C05 的 SOURCE/MECHANICAL dossier，不等待 C01/C02 结果才读代码。
4. C03 只补一个严格 v7 `s1/s50/s100` 跨家族锚点：旧 v8 与 v7 train ID 重合 `90.07%`，旧线又在
   `s850/s913` 复现条件坍塌，因此禁止无新机制地重跑 `s963`。C03 s100 只有保持视觉/语言依赖才进入
   固定 rollout；C04 完成 adapter 后另立 dossier。未比较前 C01 不称冠军。
5. 只有第一个固定闭环正例出现后，才将 C07/C08 上下文胜者和 C09 consequence 胜者逐级融合；若基础
   动作策略首步就选错目标，不能用 memory/TTT 掩盖基础策略失败。
6. 一个节点保留评测或等价 GPU 配额；长训 checkpoint 每 500–1000 steps 保存并异步消费。

当前 C01/C02 的完整规模预算为每臂 `25,097` steps、约 `8.5 h / 8×A800`；只有 s963 paired gate
胜者继续完整 dense epoch。C03 之后的吞吐、显存和墙钟为 `UNKNOWN`，必须先用真实 H3 probe 测量。

## 2026-08-15 长预算更新

- dense D/D0 已在相同 7,704 样本上完成 paired gate；aligned five-layer D 未击败 repeat-layer49 D0，
  所以 D0 是当前 carrier/action 擂主，不是最终赛道冠军。
- 为排除训练不足，D0-H32 从严格可恢复的 s963 续训到 s20000；同时以 H8 作为唯一 horizon 变量从
  同 seed 新训到 s20000。每臂 global batch8、160,000 样本、`0.796896` effective epoch，每1000步
  保存并跑同一个 balanced-80 反事实 evaluator。
- H8 来源于5090已验证的短 action chunk 控制经验，但这里使用统一四套 LIBERO dense 数据和 D0 flow
  expert，因此属于新的 controlled ablation，不继承单任务成功率结论。
- 训练许可 `GO_LONG`；在固定400-step闭环出现目标接触/成功前，效果状态仍为
  `NOT_EVIDENCE_READY`。后续融合谱系固定为 `D0 carrier -> horizon winner -> FACT consequence`。
- 首个同 step 诊断点：H8-s1000 normalized/physical MSE 为 `0.246269/0.129493`，优于
  H32-s1000 的 `0.340939/0.157451`，且 H8 的 visual-shuffle penalty 为 `0.039905`；但
  H32-s2000 已继续降到 `0.213848/0.106330`。H8-s2000 进一步达到 `0.153454/0.081639`，
  gripper accuracy `0.867188`（H32 `0.745703`），同 step 仍领先。这是离线学习曲线，不是
  闭环晋级结论。
- 缓存与相同输入的在线 H3 K/V 已有 bitwise parity（K/V `max_abs=0`）；缓存的真实边界是冻结
  H3、32-token 压缩以及 demo→rollout 分布偏移。H8 复用 H32 packed-layout cache，闭环前必须固定
  `feature_audio_horizon=32` 或补做 H8/H32 K/V parity。
- 中段曲线修正了早期判断：到 matched s9000，H32 normalized/physical MSE 为
  `0.083179/0.037419`，H8 为 `0.088563/0.039321`；H32 的 language/visual counterfactual
  响应也更强。两臂均未坍塌，继续跑满预注册 s20000，不按早期 s2000 提前选胜者。
- 共享 H3 adapter s2000 的 task3/trial0 完整400步闭环为 `0/1`：目标碗和顶层抽屉均无位移，反而
  误触 stove button `0.221442`，分类为 `FAIL_WRONG_OBJECT_NO_TARGET_CONTACT`。32409 已转为并发消费
  D0-H32 s11000（replan32）与 D0-H8 s9000（replan8）的同任务闭环。
- 首批闭环已取得真实正例：统一 D0-H32 s11000 在 `libero_object task0/trial0` 第137步成功，在
  `libero_goal task5/trial0` 第206步成功；同一轮另3项失败，总计 `2/5`。因此“冻结INT8 H3加统一
  dense动作专家能完成至少两个跨suite任务”的存在性结论为 `EVIDENCE_READY`；完整benchmark和
  H32优于H8仍是 `NOT_EVIDENCE_READY`，因为首轮闭环使用 s11000 对 s9000，尚非同step归因。
- task3各4 trials 均 `0/4`：H8四次都移动目标碗、平均最大位移 `0.5148`，但0次拉开顶层抽屉；
  H32四次都接触碗，2次明显拉动顶层抽屉，平均抽屉/碗位移 `0.0689/0.1811`。当前解释是H8偏短期
  目标接触，H32更可能学习先后顺序；必须由 matched-step/final paired rollout 确认。
- H32-s11000 正例复核后，goal task5 为 `1/3`、object task0 为 `1/4`，新增 object task1 为 `0/1`；
  正例可复现身份但当前成功率不稳。32409 已启动严格 matched-s12000 的 H8/H32 配对：goal5 与
  object0 各 trial0/1，共8 episodes，action horizon/replan 是唯一变量。
- matched-s12000 trials0/1：H32 在 goal5 为 `2/2`、Object0 为 `0/2`；H8 在 goal5 为 `0/2`、
  Object0 为 `1/2`（第149步成功）。离线H8 action/gripper略优，但H32语言/视觉反事实响应明显更强；
  闭环呈任务类型分工而非单边碾压。已继续 trials2/3，将每个任务/模型扩为4 episodes。
- matched-s12000 trials0–3 完整结果：H32 goal5/object0 分别 `2/4、2/4`，H8 为 `0/4、2/4`；
  聚合 H32 `4/8` 对 H8 `2/8`。因此 H32 的“训练H32+部署replan32”bundle 在该两任务配对screen
  达到 `EVIDENCE_READY` 并成为 horizon 赛道 rollout-gate 暂定胜者；完整LIBERO优越性仍为
  `NOT_EVIDENCE_READY`，且本实验没有拆开训练horizon与执行replan两个耦合因素。
- 已启动 matched-s12000 跨任务 screen（object1、spatial0 两个trial、goal3，共8 episodes）。同时在32409
  排队最终 s20000 H32/H8 各8 episodes：仍用 goal5/object0 trials0–3 与 s12000 同输入复测。s20000
  只有在反事实条件响应不塌缩且成功数至少达到 s12000 H32 的 `4/8` 时才替换当前 rollout incumbent；
  否则保留 s12000，不能因为“步数更大”自动晋级。
- 跨任务 screen 已完成，H32/H8 都是 `0/4`。这否定了把两任务 `4/8` 外推为泛化成功率：H32 在
  goal3 trial4 同时移动了顶层抽屉 `0.1557` 和目标碗 `0.6229` 却未完成放置；H8 在 object1
  trial0 主要移动错误的 tomato sauce `0.4946`，目标 cream cheese 仅 `0.0370`。当前闭环瓶颈
  更具体地落在多阶段完成与对象选择。32409 已插入 H32-s14000 的相同8 episodes复测，之后自动接
  s20000 配对评测。
- H32-s14000 同输入复测仍为 `4/8`（goal5 `2/4`、object0 `2/4`）；相对 s12000 低约17%的离线
  physical MSE 没有提高聚合成功率，且 object0 成功seed发生变化。现已固定 H32-s14000/action
  horizon32，仅把执行 replan32 改为 replan8 再跑同8 episodes；这是拆解训练horizon与部署replan
  耦合的单变量实验，假设是更频繁重规划能把成功率提高到 `>4/8`。
- 上述单变量实验已完成：H32-s14000/replan8 达到 `6/8`，其中 goal5 `2/4`、object0 `4/4`；同一
  checkpoint/replan32 是 `4/8`、object0 `2/4`。因此长32步开环执行是已证实的部署瓶颈，当前
  rollout incumbent 更新为 H32-s14000/replan8。最终 s20000 队列同步改为两臂都用 replan8，只有
  在同8 episodes 达到至少 `6/8` 且条件响应不塌缩时才替换 incumbent。
- H32-s20000/replan8 最终门为 `4/8`：object0 `4/4`，但 goal5 从 s14000 的 `2/4` 降为 `0/4`；
  离线条件响应仍健康。它没有达到预注册的 `>=6/8` 晋级线，明确拒绝并保留 H32-s14000/replan8。
  这表明继续优化偏向短 pick-place，却损失持续 push，训练步数与闭环能力不是单调关系。H8-s20000
  完成后，自动队列会跳过已有H32结果，只补H8的8 episodes。
- s20000 最终配对完成：统一replan8后，H32为 `4/8`、H8为 `2/8`；H8的 goal5/object0 分别
  `0/4、2/4`，与s12000聚合结果相同，继续训练没有提升。两者均未达到 H32-s14000/replan8 的
  `6/8`，所以最终checkpoint都不晋级。已启动 H32-s14000/replan4 的同8 episodes单变量消融；
  因推理调用约翻倍，只有达到 `>=7/8` 才替换replan8。
- replan4 仅 `3/8`（goal5 `0/4`、object0 `3/4`），拒绝晋级。同checkpoint曲线为 replan32
  `4/8`、replan8 `6/8`、replan4 `3/8`，说明太短会打断动作时序意图。下一项固定replan8，启用
  仓库中已实现并有单测的 FastWAM-style temporal action ensemble；仅平均重叠chunk，不增加重规划
  次数，必须在同8 episodes超过 `6/8` 才晋级。
- temporal ensemble 为 `5/8`（goal5 `2/4`、object0 `3/4`），也拒绝晋级。首次运行在仿真环境
  因重依赖导入缺少safetensors，分类为基础设施失败；已把ActionEnsembler拆为轻量模块，云端直导入和
  14项部署单测通过后重跑，空输出完整隔离保留。当前冠军固定为 H32-s14000/replan8/no-ensemble。
- 已开始冠军的全任务广度screen：LIBERO Goal/Object/Spatial/10各10任务、每任务固定trial0，共40
  episodes（5波×8卡、最多16000仿真步）。该结果只作为每任务一次的coverage screen，不能冒充官方
  多trial benchmark；其目的是真正暴露跨任务泛化瓶颈，并决定下一轮该改数据、目标还是动作执行。
- 全40任务trial0已完成 `14/40=35%`：Goal `1/10`、Object `8/10`、Spatial `4/10`、LIBERO-10
  `1/10`。成功覆盖14个不同任务，不是针对单任务过拟合；泛化集中在单阶段对象迁移，长程组合仍弱。
  该数字只代表每任务一次的coverage，不冒充官方benchmark。已启动相同40任务的trial1复验，检查
  35%是否依赖单一初始状态。
- trial1 为 `15/40=37.5%`；两轮合计 `29/80=36.25%`，Goal/Object/Spatial/LIBERO-10 分别
  `4/20、16/20、7/20、2/20`。8个任务两次均成功，21/40任务至少成功一次，说明trial0不是单一
  初始状态偶然。已排队trial2和trial3再加80 episodes，总覆盖将达到160 episodes；仍不冒充官方
  50-trial benchmark。

## 蛊王融合谱系

```text
carrier/action 赛道冠军
  └─ + temporal/context 冠军
       └─ + consequence/ranking 冠军
            └─ + structured-future 冠军（若独立通过）
                 └─ unified LIBERO benchmark + multi-seed + deployment audit
```

每个子节点仅新增一个已经独立晋级的机制，并与父节点在同数据、预算、seed、solver 和 rollout 合同下
配对。融合失败时淘汰组合而不是篡改父模型；只有最终节点通过统一 LIBERO benchmark、多 seed、消融、
显存和延迟门，才称为“蛊王”。

## 2026-08-15 组合/空间/对象选择专项

可证伪假设：`D0-H32-s14000/replan8` 的剩余失败主要不是基础物体类别识别，而是视觉空间载体压缩与
跨 replan 进度丢失；若该判断正确，C02 应在固定 Spatial 子集提高成功数，C14 应在固定 Goal/10
组合子集增加成功且 Object 不下降超过 1 个 episode。任一候选只改善离线 MSE、没有增加 simulator
predicate success，均判假并停止。

### 当前闭环归因

固定父模型、replan8、trial0–3 共160 episodes：Goal `10/40`、Object `28/40`、Spatial `17/40`、
LIBERO-10 `2/40`，总计 `57/160=35.625%`。joint-delta 只作为失败诊断，不冒充成功：Object 的12次
失败中8次已经搬动正确目标、没有明显错对象；Spatial 的23次失败中14次搬动正确 bowl、1次有明显
错 bowl 证据；LIBERO-10 有19次部分必需目标进展与12次错对象/错阶段证据。原始审计在
`/mnt/h3-wam/eval/audits/d0-h32-s14000-replan8-trials0123-failure-audit-v1.json`，160个结果文件的
hash manifest SHA256 为 `892367c7d1b5bca07987c91fcf94c7f8ee385c75c5e0ad5840442c4aa23019a2`。

### 差异矩阵

| 字段 | 父模型 | C02 五层空间载体 | C14 历史进度适配器 | 状态 |
|---|---|---|---|---|
| H3 | INT8、冻结、同 SHA | 相同 | 相同 | ALIGNED |
| 动作专家 | DreamWAM 5-block ActionDiT | 相同初始化/训练合同 | 父模型全部冻结 | ALIGNED |
| 视觉 K/V | layer49×5、32 token | layer9/19/29/39/49、各32 token | 相同父模型 | INTENTIONAL |
| 历史 | rollout 虽发送16步但服务端忽略 | 仍忽略 | 16步真实已执行动作，左补零+valid mask | INTENTIONAL |
| 数据/顺序 | v7 uniform，前112000样本到s14000 | 同 manifest、seed、sample offset | 从s14000后不重叠窗口训练新增adapter | ALIGNED / ADAPTER-ONLY |
| 部署 | replan8、无ensemble | 通过门后同协议 | 通过门后同协议 | ALIGNED |

未决差异：DreamWAM 官方没有 MiniMax-H3 端口，二者仍是 `backbone_port`；C14 把 LingBot-VA 官方
持久 video/action KV 简化为零初始化的 action-history progress adapter，属于 `novel_composition`，
不是官方复现。OptimusVLA 当前 GitHub 只有 README/assets、没有可训练实现，因此只作为结论补充并从
训练候选排除；HELM 当前没有核验到官方训练仓库，也不占用 GPU 预算。

### 预算、门槛与真实命令

- C02：global batch8，从s963续至s14000，总112000样本，`0.557827` effective epoch；每1000步保存、
  restore、balanced80。最后一次s3000→s14000恢复段实测墙钟69分04秒（此前s963→s3000不含在内）。真实命令：
  `nohup bash scripts/h3wam/launch_dense_d0_horizon_long.sh d_h32_resume > /mnt/h3-wam/eval/d-h32-s14000-five-layer.queue.log 2>&1 &`。
- C14机械门已通过：global batch8、1 step、8 samples，父模型 probe `max_abs=0`；五个 ActionDiT block
  与 proprio gradient 全为0，history gradient `0.103162`、更新 `1.0073e-05`，原子checkpoint重新加载
  `max_abs=0`。3k adapter canary 已完成：global batch8、24000样本、`0.119534` effective epoch、每500步
  保存，最终checkpoint `d0_history16_s17000.pt`；真实命令为
  `nohup bash scripts/h3wam/launch_dense_d0_history16_adapter.sh > /mnt/h3-wam/eval/d0-history16-adapter-s3000-v1.queue.log 2>&1 &`。
  完整3k adapter段墙钟14分46秒。
- C02 promotion：同父模型的固定 Spatial tasks `0–9`、trials0–1 必须超过父模型 `7/20`，同时固定
  Object 子集不得下降超过1个成功；否则拒绝。
- C14 promotion：固定 Goal task3 与 LIBERO-10 tasks0/3/7/9、每项trials0–1必须超过父模型；另跑
  Object tasks0/1/5/9回归门。出现父模型条件依赖退化、Object下降>1或没有新增长程成功即停止。

许可状态：C02 `FAIL_PAIRED_GATE / NO_GO`；C14-s14500只保留
`MECHANISM_EVIDENCE / NO_GO_FUSION`，C14-s17000 `FAIL_PAIRED_GATE`；C15
`GO_CANARY / NOT_EFFECT_EVIDENCE`。整体 incumbent 仍是
`D0-H32-s14000 -> replan8 -> no ensemble`，尚无可融合的时间或空间赛道冠军。

### 专项首轮结果（2026-08-15）

- C14 的 s14500/s15000/s16000/s16500/s17000 已在固定episode-disjoint val40、相同噪声和10步Euler上
  做零历史父对照、正确历史、错配历史评测。最终s17000 physical MSE为 `0.029188`，略优于零历史父
  对照 `0.029282`；错配历史为 `0.029642`，正确历史输出相对错配历史 mean-absolute delta为
  `0.024256`。语言/视觉响应分别为 `0.175044/0.110229`，没有条件坍塌。s14500离线最强
  (`0.028316`)，曲线非单调，因此s17000不是自动最优。
- C14 最终点的固定 LIBERO-10 tasks0/3/7/9、trials0/1 为 `2/8`，父模型同输入为 `1/8`；新增收益
  集中在task3，候选两次均成功（255/359步），父模型只有trial1成功。Goal3仍为 `0/2`。这只证明一个
  小的长程增益；Object0/1/5/9回归从父模型 `5/8` 降到 `2/8`，超过最多下降1次的预注册限制，故
  s17000正式标为 `FAIL_PAIRED_GATE / NOT_EVIDENCE_READY`。失败审计显示Object失败为4次无目标接触、
  2次搬动目标但未完成，并非明显选错物体；说明浅历史分支主要扰乱了接触/控制，而非改善目标绑定。
- 因历史学习曲线非单调且s14500是预注册保存点中离线最优者，使用完全相同的18个长程/Goal/Object
  episodes做了固定checkpoint-selection复核。s14500长程组合为 `2/10`（父模型 `1/10`），Object为
  `5/8`（父模型 `5/8`），通过“长程增加且Object不少于4/8”的窄门；Object剩余3次失败均为无目标接触，
  没有明显错对象证据。因此s14500只通过了首批checkpoint-selection窄门，不能据此称时间赛道winner。
  独立trials2/3确认中，长程为 `0/10`，与父模型 `0/10` 相同；Object为 `4/8`，父模型 `5/8`。
  合并trials0–3只是长程 `2/20 vs 1/20`、Object `9/16 vs 10/16`，新增成功集中在task3的前两个trial，
  没有跨trial稳定复现。因此撤销“时间赛道winner”称号，只保留历史确实影响输出的机制证据。
- C02 s6000 的 balanced80 相对同step D0：normalized MSE `0.097605 vs 0.110490`、physical MSE
  `0.047004 vs 0.050610`、gripper macro-F1 `0.907355 vs 0.880319`；language sensitivity较低
  (`0.162029 vs 0.194681`)，visual-shuffle action MAE略高 (`0.144402 vs 0.139896`)。因此五层载体
  保持视觉依赖且中段离线更好，但仍必须由final s14000的固定Spatial闭环决定是否有效。
- C02 s6000 与s8000的早期Spatial tasks0–7/trial0闭环均为 `0/8`；固定父擂主同任务为 `3/8`。s6000失败分解是
  4次无目标接触、2次移动正确目标但未完成、2次明显错对象证据。由于s6000并非与父擂主同step的最终
  promotion点，连续两个负点仍不提前中止预注册s14000长线，但已强烈反驳“对齐五层K/V自然改善空间关系”；
  禁止把s6000/s8000离线改善解释为空间能力改善。
- C02最终s14000 balanced80虽有 normalized/physical MSE `0.057962/0.025511`、gripper macro-F1
  `0.942977`、语言/视觉响应 `0.206101/0.134579`，但固定Spatial仅 `5/20`（父 `7/20`），Object回归
  `3/8`（父 `5/8`，且低于4/8下限）。112000样本后仍双门失败，故训练不足解释被排除，C02停止。
- 对象选择不另起大模型：父模型Object全量已达 `28/40`，多数失败是正确目标已移动但未完成；C14与C02
  都可能破坏Object。因此Object在下一轮作为硬回归门，主要优化转向长程结果判断与保留空间拓扑。
- 新C15针对一个已核验的本地偏差：真实H3首帧条件K/V是双相机 `7×14=98` token，当前实现用
  `adaptive_avg_pool1d`压到32，DreamWAM官方并无此压缩。C15保持32 token和相同存储，只对左右相机
  分别做4×4二维池化。64个真实INT8样本、8卡机械审计全部finite、每层独立，峰值约20.1GiB；
  artifact为 `/mnt/h3-wam/eval/c15-grid-cache-mechanical-v1/COMPLETED`。下一步从固定D0-s14000恢复
  模型/optimizer/scheduler/RNG，在完全相同的112000–120000样本上训练1000步，与现成D0-s15000配对。
  真实一步适配机械门也已通过：global batch8、8 samples、完成到s14001，五个block梯度均finite且非零，
  head update `3.05176e-05`，保存后严格恢复的probe `max_abs=0.0`；checkpoint SHA256为
  `4fbbb0db2c5990074645f74ba3a6e77489e025f8a64b2f82a7a087c18ab39512`，artifact在
  `/mnt/h3-wam/outputs/c15-grid-adaptation-mechanical-v1/COMPLETED`。这仍只算链路证据，不算效果改善。
  真实命令：`nohup bash scripts/h3wam/launch_c15_grid_cache_stage8k.sh ... &`，缓存READY后自动执行
  `scripts/h3wam/launch_c15_grid_adaptation_s1000.sh`；在离线和Spatial闭环胜出前不称改善。
  两条评测接力也已预置：30907的 `launch_c15_balanced80_gate.sh` 等验证缓存后做同噪声离线配对；
  32409的 `launch_c15_spatial_object_gate.sh` 等checkpoint后立即跑Spatial 20 episodes和Object 8 episodes。
  promotion保持Spatial严格超过父模型 `7/20` 且Object不少于 `4/8`，避免用空间提升掩盖对象选择退化。
  8k正式缓存已通过全量审计（missing/invalid均0）；1000步适配完成到s15000，global batch8、8000样本、
  `0.039845` epoch，训练/严格恢复墙钟 `135.78/40.78s`，restore probe `max_abs=0.0`。checkpoint SHA256
  `ee6ad4df1cf1b0ca8229fc67c06ad9a554ad5f3406515ecaa7e9f85b83cd27e4`。Spatial闭环已自动启动；
  训练与恢复PASS仍不是效果证据，等待20+8固定闭环门。
  最终闭环为Spatial `7/20`（父 `7/20`，未严格提高）、Object `2/8`（父 `5/8`且低于4/8下限），因此
  `FAIL_PAIRED_GATE / NO_GO_FUSION`。30907的剩余验证缓存和balanced80等待队列已停止，保留已有缓存与
  checkpoint作为负结果证据，不再为失败候选消耗8卡。artifact：
  `/mnt/h3-wam/eval/c15-grid-closed-loop-gate-v1/COMPLETED`。
- FACT官方代码固定在 `618a6c168686`：其failure-aware训练依赖明确的failure activation，best-of-N依赖
  校准的time-to-go/value排序。现有160条父模型rollout可组成6208个replan states和6048条因果transition，
  train trials0/1/2与val trial3任务重叠但episode不重叠；然而没有failure onset与counterfactual action
  outcome。因此只放行value-only feature/target dossier，不放行failure imitation或best-of-N闭环。
  审计：`/mnt/h3-wam/eval/audits/d0-h32-s14000-replan8-fact-transition-audit-v1.json`。
- C16继续逐行对齐FACT官方transform（commit `618a6c168686`，文件SHA256
  `ed76964b005420e752d15d140156962d6c18abd40e58f9140313857d5ebd7110`），冻结出1001条可监督成功
  transition：train763、trial3 val238，来自43/14个成功episode；103个失败episode因没有failure onset全部按右删失，
  不伪造惩罚target。val只有Goal91/Object88/Spatial59条，LIBERO-10为0，因此目前只放行“冻结H3特征能否预测
  held-out进度”的诊断，不放行failure training或best-of-N。artifact：
  `/mnt/h3-wam/eval/fact-value-target-dossier-v1/report.json`；manifest SHA256
  `386b4e574c13635007e04a1b19f0867434eacd24f32984e0f3b02df30619c587`。
- 为补上C16没有LIBERO-10正样本的问题，C17从现有v7专家窗口冻结出progress targets：future index为
  `min(start+32,length-1)`，raw value为 `(length-future-1)/(length-1)`。train200779窗/1542 episodes，
  val22150窗/170 episodes，episode overlap0；其中LIBERO-10为82760/9104窗。offset32是按本项目H32做的
  明示本地改动，FACT官方RobotWin为48，不能冒充复现。当前只放行冻结H3 progress head canary；未证明
  action-conditioned held-out ranking前仍禁止best-of-N。artifact：
  `/mnt/h3-wam/eval/expert-progress-targets-v1/report.json`。
  冻结特征probe已在32611启动：每suite确定性分层抽train1000/val500，合计4000/2000；只压缩layer49
  K/V为512维mean/std，用两路低优先级CPU读取，避免抢C15的GPU与共享I/O。promotion要求相对
  task+absolute-step ridge的总MAE至少改善5%，且四个suite均不得退化超过5%。真实命令：
  `nohup bash scripts/h3wam/launch_c17_progress_probe.sh > /mnt/h3-wam/eval/c17-frozen-h3-progress-probe-v1.queue.log 2>&1 &`。
  probe已完成并通过：held-out总MAE `0.088255→0.062385`（ratio `0.70687`），R²
  `0.81456→0.89832`；四suite均改善，LIBERO-10 MAE `0.067854→0.057894`。这说明冻结H3 K/V包含
  超出task+absolute-step的阶段进度信息，C17晋级shadow progress trace；但它尚未看到可验证的备选动作
  outcome，因此仍不授权best-of-N或声称动作生成改善。artifact：
  `/mnt/h3-wam/eval/c17-frozen-h3-progress-probe-v1/COMPLETED`。
  同一拟合已导出为17KB严格契约权重 `probe.pt`（40 contexts、553维标准化输入、554个含bias权重），
  restore复算与原预测逐元素一致；下一步只做闭环shadow trace，不直接改动作。
- C17 shadow在固定擂主的16条闭环上完成（四suite各2成功+2失败、同checkpoint/seed/init/replan8）：
  16/16首动作块逐值一致、16/16 outcome一致，证明17KB头只读且没有改变动作；但最终remaining-progress
  AUROC仅`0.1875`，成功/失败终值中位数分别`0.05746/0.0`。失败episode跑到step400时也被
  `absolute_step/400`压成0，形成明确的时间捷径，故`FAIL_PROGRESS_SHADOW_GATE / NOT_EVIDENCE_READY`。
  artifact：`/mnt/h3-wam/eval/c17-progress-shadow-v1/report.json`，16 episodes墙钟344秒。
- C18以C17为父，只删除absolute-step并复用完全相同的4000/2000特征、split、ridge和target。离线
  task-only→task+H3 MAE为`0.21545→0.09952`（ratio`0.46190`），R²为`0.00418→0.74192`，
  四suite全部改善；但相同16条shadow的AUROC仅提升到`0.546875`，仍低于预注册`0.65`。
  成功/失败终值中位数为`0.15174/0.15648`，虽方向正确但分离不足；首动作与outcome仍16/16一致。
  因而停止仅靠成功专家time-to-go标签的调参，不授权best-of-N。下一次critic实验必须先取得可审计的
  failure onset或同状态备选动作outcome。artifacts：
  `/mnt/h3-wam/eval/c18-timeblind-progress-probe-v1/COMPLETED`、
  `/mnt/h3-wam/eval/c18-timeblind-progress-shadow-v1/report.json`。
- C19检查旧轨迹能否恢复到原始观测：四suite各取失败episode的首/中/末，共12个state。LIBERO
  `get_sim_state()`保存的time/qpos/qvel可max-abs0写回，但恢复后的图像与proprio不等于旧缓存，故不能
  把旧trajectory state称为“原轨迹精确快照”。artifact：
  `/mnt/h3-wam/eval/c19-libero-state-restore-v1/COMPLETED`。
- C20改问counterfactual真正需要的可证伪问题：两个独立env从同一规范化恢复state开始，执行同一8步
  chunk，是否得到相同起点/终点。首跑遗漏LIBERO `seed()`使用process-global RNG，Goal/Spatial的
  reset布局不同，保留为harness失败；v2在每次reset前固定seed42后，四suite×3状态的双相机起点/终点
  均逐像素一致，proprio/state均在`1e-10`容差内，steps/success predicate一致。因此只放行小规模
  paired counterfactual collection canary；仍未得到alternative-action outcomes，不放行critic训练。
  artifact：`/mnt/h3-wam/eval/c20-libero-branch-repeatability-v2/COMPLETED`。
- C21固定`D0-H32-s14000/replan8/no-ensemble`、环境seed42与四个规范branch state，唯一改变每次
  rollout的policy diffusion-noise seed。四suite各4分支、共16条：Goal `4/4`、Object `3/4`、
  Spatial `0/4`、LIBERO-10 `4/4`；Object形成一组同状态混合成败标签。四组最小首动作块pairwise
  RMS分别为`0.22723/0.14624/0.27201/0.09239`，均高于`1e-6`，通过预注册的
  `PASS_COUNTERFACTUAL_OUTCOME_CANARY`。预算为2 waves×8 A800、最多16×400环境步，实测墙钟
  `280s`。事后代码审计确认`policy-noise-seed-base`也改变所有后续replan seed，因此该结果只放行
  高结果熵state校准，不能把成败归因于首动作，也不放行critic数据集；原artifact中的
  `GO_DATASET_EXPANSION`许可被此审计收窄为`GO_ENTROPY_CALIBRATION_ONLY`。尚未训练或验证
  critic/best-of-N，效果状态仍为`NOT_EVIDENCE_READY`。artifact：
  `/mnt/h3-wam/eval/c21-counterfactual-outcome-canary-v1/COMPLETED`，SHA256
  `c258cb829c45e504e03e5a183008d2820a1a32152bb8ff70723ba1acf8895f8c`。
- C22按预注册的8个成功源episode、距成功`1/3/5` replans和4条完整noise schedule完成96分支：
  `71/96`成功、`7/24` mixed group，且Goal/Object/Spatial/LIBERO-10四suite均有mixed group；四个
  shard各24条的墙钟为`457/453/458/460s`。所有组首动作均不同，通过
  `PASS_COUNTERFACTUAL_ENTROPY_SWEEP`。30234首波因节点本地LIBERO site缺失而0结果退出，固定tar
  恢复后仅重跑该shard；原traceback和incident JSON均保留。该门只选择高熵state，不是首动作因果
  标签。artifact：`/mnt/h3-wam/eval/c22-counterfactual-entropy-sweep-v1/COMPLETED`，SHA256
  `05f18c76e9c460e06f8e4290f7a3332d2cf784ee35022dcb1973f63644cc3978`。
- C23由C22 Bernoulli entropy确定性选8个state（每suite先取1组，再全局排序），每组复用4个首动作
  seed，但固定共同的后续seed schedule。机械smoke中首动作逐值复现C22，而结果由成功变失败，直接
  证明C22 outcome受后续随机性混杂。正式32分支中`18/32`成功；Spatial task0/trial3/d5为`2/4`
  mixed，其余组为0/4或4/4。32/32首动作与C22逐值一致，32/32 continuation schedule合法，8/8组
  动作不同，判定`PASS_FIRST_ACTION_CAUSAL_CANARY`。这首次放行episode-disjoint因果数据集canary，
  仍不放行critic/best-of-N效果声明。artifact：
  `/mnt/h3-wam/eval/c23-first-action-causal-canary-v1/COMPLETED`，SHA256
  `11dfef6ce6523ac58f8cb8aae7166e81173b2e3e7724b95332b9ab0348b4143f`。
- C24复用C23相同8 state/32 seed/continuation schedule，唯一改变首chunk执行长度；后续均保持
  replan8。h16为`17/32`成功、2 mixed（Object/Spatial），未过门；h32为`16/32`成功、3 mixed
  （LIBERO-10两组、Spatial一组），通过预注册的`>=3 mixed / >=2 suites`。两候选32/32首动作均
  bit-exact于C23，seed/action机械门全过，选择h32放行episode-disjoint数据canary；这不代表部署
  replan32已优于replan8。artifact：`/mnt/h3-wam/eval/c24-first-action-horizon-sweep-v1/COMPLETED`，
  SHA256 `a85d7d0dd03c355906cfa5f8277b8abf3e1d2675b5291153a1fae03e8b53f54e`。
- C25固定C24 h32合同，在14个源episode上采集32 state组/128分支；train22组、val10组均按源episode
  隔离。最终`70/128`成功、9 mixed，其中train6、val3，覆盖LIBERO-10/Object/Spatial；Goal 8组均
  同质。源split、128条seed schedule及32组动作多样性全部通过，判定
  `PASS_EPISODE_DISJOINT_CAUSAL_DATASET_CANARY`。该结果只放行冻结H3/动作父策略的小critic canary，
  必须先过held-out within-state ranking，尚不放行best-of-N。artifact：
  `/mnt/h3-wam/eval/c25-episode-disjoint-causal-dataset-v1/COMPLETED`，SHA256
  `641ce0fb53853c555da2ad69cbe1b2d6451faec470c362c6db1ad7aef3fa4165`。
- 评测基础设施发现：30234上 `h3-int8-native` 的PyTorch2.10/CUDA13在A800执行最小BF16 Linear会报
  `CUBLAS_STATUS_INVALID_VALUE`；同节点共享的PyTorch2.8/CUDA12.8可执行。history离线评测改用后者，
  该失败归类为infra，不计作policy trial；闭环仍用已验证的INT8 H3运行时节点。
