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
| C07 | temporal/context | LingBot-H3 persistent history | LingBot-VA | carrier/action winner | persistent observation KV + executed-action feedback | partial H3 port；contract mismatch open | PROBE_ONLY |
| C08 | temporal/context | MiniWorld-H3 rolling memory | MiniWorld | C07 or carrier/action winner | rolling KV + diffusion-forcing curriculum | policy bridge absent | PROBE_ONLY |
| C09 | consequence/ranking | FACT-lite H3 | FACT | carrier/action/context winner | causal future state/value head；action path cannot see future | future-proprio s100/s500 mechanism pass；future-H3 s100 armed；failure data absent | GO_CANARY future-H3；NO_GO value/ranking |
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
