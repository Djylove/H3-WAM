# H3-WAM 开源候选注册表与炼蛊谱系

更新时间：2026-08-14（Asia/Shanghai）

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
| FACT | `618a6c168686` | clean | TRAINABLE | causal act-then-imagine、future state/value、failure-aware、best-of-N | 缺 canonical failure rollout 数据 |
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
| C02 | carrier | D dense five-layer K/V | DreamWAM | C01 fresh-init twin | layer49 repeat → aligned layers 9/19/29/39/49 | cache running | GO_CANARY after full audit |
| C03 | carrier/action | StarWAM dense full30 s100 anchor | StarWAM | historical v8 StarWAM s100 | v8 frame-indexed → common v7 valid-window split/cache | dossier pass；dual H3 cache running；historical s850/s913 condition collapse | GO_CANARY s100 only；NO_GO_LONG |
| C04 | action/co-training | FastWAM-H3 | FastWAM | fixed FastWAM Wan baseline + C03 interface baseline | Wan video expert → H3 while preserving official action path | source pass；adapter incomplete | PROBE_ONLY |
| C05 | carrier | ImageWAM-H3 target image | ImageWAM | carrier winner | video K/V → H3 target-image/edit representation | source pass；adapter incomplete | PROBE_ONLY |
| C06 | carrier/action | DiT4DiT-H3 | DiT4DiT | carrier winner | selected intermediate H3 features + official ActionDiT | source pass；dirty vendor audit pending | PROBE_ONLY |
| C07 | temporal/context | LingBot-H3 executed history16 | LingBot-VA | shared-H3 parent | explicit executed-action feedback；尚非 persistent observation KV | s2500 offline champion，但 fixed rollout `0/1` 且无接触 | NO_GO current port；persistent-KV 需新 dossier |
| C08 | temporal/context | MiniWorld-H3 rolling memory | MiniWorld | C07 or carrier/action winner | rolling KV + diffusion-forcing curriculum | policy bridge absent | PROBE_ONLY |
| C09 | consequence/ranking | FACT-lite H3 | FACT | carrier/action/context winner | causal future state/value head；action path cannot see future | future-proprio 与 future-H3 均通过 s100/s500 机制门；failure data absent | consequence mechanism champion；NO_GO value/ranking |
| C10 | consequence/ranking | FACT best-of-N | FACT | C09 | rank sampled action chunks by learned progress/value | waits C09 calibration | NO_GO parent gate |
| C11 | structured future | DreamWAM motion/depth/DINO | DreamWAM | carrier/action winner | add exactly one structured future target per child | source pass；target caches pending | PROBE_ONLY |
| C12 | action/co-training | Motus-H3 three-expert | Motus | best joint-training baseline | three-expert MoT | Stage-2/H3 mapping unknown | PROBE_ONLY |
| C13 | consequence/state | Cosmos-H3 latent slots | Cosmos Policy | consequence winner parent | future state/value latent slots | scale/data mismatch open | PROBE_ONLY |

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
