# Candidate D：DreamWAM K/V carrier 到冻结 INT8 H3 的最小端口

日期：2026-08-14
状态：独立 cache/trainer 已接通；全量 8,560-window cache 审计通过；一次授权 repair retry 的
单步机械 probe 通过；默认关闭；未保存 checkpoint、未启动长训、无效果结论。

## 1. 结论

当前 H3-StarWAM 的 `last32` 路径不等价于 DreamWAM。它只把 H3 第 49 层的一份 hidden feature 投影成 context，并由所有 ActionDiT block 重复做 cross-attention。DreamWAM 的核心 carrier 是：每个 action block 使用该层独有的 video K/V，与该 action block 自己产生的 action K/V 拼接，再执行 self-attention；之后才做 text cross-attention 和 FFN。

Candidate D 按这一语义做最小端口：一次冻结 INT8 H3 前向，在 H3 层 `(9, 19, 29, 39, 49)` 直接截取各层 RoPE 后 K 和原始 V；五个官方 DreamWAM ActionDiT block 一一消费五份不同 K/V。它不重复注入 `last32`，也不对 H3 做第二次 QKV 投影。

这个原型证明接口在数学形状和实现路径上可行，不证明任务成功率。是否优于现有 StarWAM，必须通过下文的成对 canary 决定。

## 2. 可复现源码证据

| 项目 | 固定 commit | 本地来源 |
|---|---|---|
| DreamWAM | `6e989facc0c452fd3488d75f60bc36411005558c` | `third_party/DreamWAM` |
| StarWAM | `cd76d96f273f81e228a05f40f9697fe2514e2356` | `third_party/StarWAM` |

DreamWAM byte pin：

| 文件 | SHA-256 |
|---|---|
| `dreamwam/layers.py` | `3cd38ad24eff05e748d9353af3f39200e93b16b6d07d22f153ccef0f36becd96` |
| `dreamwam/experts.py` | `9ba51dbb15b8df8e4ff01c5a08acf443a950c422544e4497d13e0e2658bd489c` |
| `dreamwam/mot.py` | `5467d135287a6e77074cb653fc3d72218490fcfa40ac486b61d5cc5975ab6c01` |

关键代码链：

1. DreamWAM `JointMoT._attention_input` 先为每层生成并规范化 Q/K，施加 RoPE，同时生成 V：`third_party/DreamWAM/dreamwam/mot.py:51`。
2. `prefill_video_cache` 在每个 video block 保存该层的 K/V，而不是保存最终 hidden state：`mot.py:144`。
3. action Q 对 `[video_K | action_K]`、`[video_V | action_V]` 做注意力：`mot.py:194`。
4. 结果再经过 self-attention 输出/残差、text cross-attention 与 FFN：`mot.py:93`。
5. 当前 H3-StarWAM 明确只允许一层 feature，并把它投影后追加到 context：`src/fastwam/models/h3wam/starwam_feature_action.py:142`。

## 3. 差异矩阵

| 维度 | 官方 DreamWAM | 当前 H3-StarWAM | Candidate D |
|---|---|---|---|
| 视觉来源 | 可训练/可缓存 VideoDiT | 冻结 H3 第 49 层 hidden `last32` | 冻结 INT8 H3 的真实逐层 K/V |
| 层身份 | 每个 action block 对应独有 video block K/V | 所有 action blocks 重复同一 context | H3 `(9,19,29,39,49)` 分别对应 5 个 action blocks |
| 融合位置 | action self-attention 内联合 K/V | ActionDiT text/context cross-attention | action self-attention 内联合 K/V |
| action 自身 K/V | 有，并与 video K/V 拼接 | 由 StarWAM ActionDiT 内部处理，视觉不在 self K/V 中 | 有，并与 H3 video K/V 拼接 |
| text/proprio | 独立 cross-attention context | 与视觉 feature 混成同一 context | text + proprio 独立 cross-attention context |
| 视觉更新 | 完整 JointMoT 可逐层更新 video tokens | 无 | H3 前向内部逐层更新；carrier 只读缓存 |
| 反传到视觉基模 | 完整训练时可以 | 冻结 H3 时不可以 | 不可以；本候选明确冻结 H3 |
| 层数 | 官方配置的完整 VideoDiT/ActionDiT 深度 | 当前 ActionDiT 深度，但重复单层 context | 稀疏 5 层最小端口，不宣称完整等价 |
| attention width | VideoDiT 与 ActionDiT 相同后直接拼接 | hidden 5376 先投影到 context 5120 | H3 `56*128=7168`，ActionDiT 同宽后直接拼接 |
| 默认状态 | 项目主路径 | 当前研究路径 | `enabled=False`，零参数、不可误调用 |

## 4. 等价边界

已对齐的 DreamWAM 语义：

- action Q 在每层直接读取该层 video K/V 和 action K/V；
- 使用 DreamWAM 官方 `ActionDiT`、`JointMoT._attention_input` 与 `_post_attention`；
- self-attention 后仍保留 text cross-attention、FFN、时间调制和 action RoPE；
- H3 K/V 是同一次正常 attention QKV 计算的产物，不做额外近似投影。

尚未对齐、不得混称“复现 DreamWAM”的部分：

- 只选 H3 50 层中的 5 层；官方路径是逐层对应；
- H3 K/V 来自 H3 的 packed multimodal sequence，官方 K/V 来自独立 VideoDiT；
- 冻结/预计算 K/V 使 action loss 无法更新 H3；
- 32-token canary 是存储友好的稀疏 token 版本，98-token 才更接近现有完整 observation token 范围；
- 没有实现 DreamWAM video/action 双向的完整 JointMoT 更新，本候选对应官方的 video-cache action-forward 路径。

## 5. 实现与安全开关

- `src/fastwam/models/h3wam/dreamwam_kv_carrier.py`
  - `H3DreamWAMKVCarrierPolicy(enabled=False)` 默认不创建任何训练参数，并拒绝 forward；
  - 开启后从 byte-verified pinned DreamWAM 源加载官方模块；
  - cache layer key 必须与 `carrier_layers` 精确一致；
  - 对完全重复/别名 K/V storage fail-fast，防止把一份 `last32` 冒充多层 cache；
  - 默认生产形状：5 action blocks、hidden 1024、56 heads、head dim 128。
- `src/fastwam/models/h3wam/int8_backbone.py`
  - `capture_kv_layers=()` 默认关闭，原输出路径不变；
  - 开启后，在每个所选 H3 block 的同一次 QKV 计算中返回 RoPE 后 K 与原始 V；
  - `kv_capture_indices` 可固定 token 子集；不传时只截取该 packed sequence 的 video indices。

独立 opt-in 路径现已接入：

- `scripts/h3wam/precompute_h3_int8_features.py` 只有显式选择
  `h3_dreamwam_kv_v1` 时才写五层 K/V，旧 `last32` 默认路径不变；
- `scripts/h3wam/train_h3_int8_dreamwam_kv_carrier.py` 严格校验 cache、manifest、checkpoint
  contract，并继承 parent v2 的 FP32 continuous timestep 与 rank-distinct flow RNG；
- trainer 的 BF16 mixed-precision forward 边界保留 FP32 source timestep，不修改 pinned
  DreamWAM 源码，也不把 timestep 降成 BF16；
- `scripts/h3wam/audit_h3_dreamwam_kv_cache.py` 独立逐文件校验 raw bytes、身份集合、metadata、
  shape/dtype/finite 和 storage alias。

## 6. 存储、参数与计算预算

BF16 K+V 单样本精确缓存量：

| carrier | bytes/sample | MiB/sample | 21,700 samples（GiB） |
|---|---:|---:|---:|
| 5 layers × 32 tokens | 4,587,520 | 4.375 | 92.71 |
| 5 layers × 98 tokens | 14,049,280 | 13.398 | 283.93 |

公式：`layers * tokens * 56 heads * 128 head_dim * 2(K,V) * 2 bytes`。这还不包含 metadata、文件系统 block 和数据索引。

默认 5-block action carrier 的估算参数：

- 每 block 约 67,195,904 参数；五层约 335,979,520；
- 加 action/text/time embedding、head 和 proprio encoder 后约 349,944,839 参数；
- 纯 BF16 权重约 0.652 GiB，训练优化器与 activation 另计。

运行时每次 observation/replan 只需一次冻结 H3 前向；每个 flow denoise step 运行五个 action blocks。预计算训练 cache 不需要重复运行 H3，但必须在 manifest 固定 H3 checkpoint hash、层号、token indices、timestep/condition、dtype、shape 和采样策略。

## 7. 唯一变量 paired canary（仍未执行）

### 7.1 不变项

两组必须使用完全相同的：

- H3 INT8 checkpoint 与 SHA、VAE/condition 路径；
- frozen H3，不加载 H3 LoRA，不局部解冻；
- 数据 manifest、episode-level split、样本顺序、seed `42`；
- observation/action horizon、action normalization、flow scheduler/shift；
- batch/effective batch、optimizer、LR、warmup、gradient clip、训练步数；
- checkpoint/eval steps 和固定 offline validation examples；
- text context、proprio 输入及 mask；
- 训练前 restore smoke、第一批 input/output/gradient finite 检查。

### 7.2 唯一变化项

配置 `action_carrier`：

- `starwam_last32`：当前冻结 H3 第 49 层 32-token hidden → context cross-attention；
- `dreamwam_h3_kv_5x32`：H3 层 `(9,19,29,39,49)` 的 32-token K/V → 5-block joint K/V carrier。

这是一个明确的“carrier bundle”分类变量；它同时包含 carrier 结构及其必需表示，不能把结果误归因于单独的“多层数”或“缓存 K/V”。若 D 胜出，再做 `(49,)`、`(29,49)`、五层以及 32/98 tokens 的消融。

### 7.3 建议顺序与放行门

1. 同一 64-sample debug manifest：restore + forward/backward + 20-step overfit，仅验证 plumbing；不报告能力结论。
2. 同一分层 1,024-sample canary manifest：两组各 1,000 steps，steps `0/250/500/750/1000` 保存；不提前挑最好结果。
3. 用相同 checkpoint step 比较固定 validation action flow MSE、每 action 维 MAE、normalized prediction 越界率；同时记录吞吐、峰值显存和参数量。
4. Candidate D 必须在至少两个连续 checkpoint 上改善主 action metric，且没有明显扩大越界率，才允许进入长训。
5. 通过 offline gate 后，才在同一 LIBERO task/seed/replan/action-chunk 协议做小规模闭环；未通过时不靠增加 rollout trial 掩盖问题。

首个 canary 建议用 32 个确定性 video token，并把 token index 列表写进 manifest。98-token 是 D 胜出后的 fidelity 升级，而不是首轮同时变化的变量。

## 8. 已完成验证

隔离 CPU runtime 执行：

```text
PYTHONPATH=src python -m pytest -q \
  tests/test_h3_dreamwam_kv_carrier.py \
  tests/test_h3wam_int8_backbone.py \
  tests/test_h3_int8_online.py \
  tests/test_h3_int8_precompute.py \
  tests/test_h3_starwam_feature_action.py
```

原模型端口回归结果：`24 passed in 6.99s`。接入 trainer 后，本地 carrier/trainer 定向回归为
`12 passed`；30907 上对精确部署的 trainer test SHA 执行为 `6 passed in 7.068s`。

覆盖项包括：默认关闭且零参数、pinned 官方源码加载、distinct-layer K/V 改变输出、反向梯度、重复 cache 拒绝、缺层拒绝、缓存预算、INT8 attention 真实 K/V 捕获，以及 FP32 flow timestep 经 BF16 autocast 前向仍保持源精度。该结果只证明模型端口与旧默认链路兼容，不代表训练或闭环效果。

## 9. 30907 cache 审计与机械 probe

固定输入：

- source manifest：8,560 windows，SHA-256
  `d343a360753bd01821fd87ed4a85ca9240ecf4794f8cf0c457921bac2dd3f0e3`；
- train/validation：7,710/850 windows，1,542/170 episodes，episode overlap `0`；
- INT8 H3 checkpoint SHA-256：
  `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a`；
- cache：`h3_int8_dreamwam_kv_5x32_canary_v1`，8 shards 各 1,070 files。

独立全量 audit 结果：8,560/8,560 完成；missing/extra/tmp/duplicate/tensor-metadata error
全部为 `0`；总大小 `39,314,799,920` bytes；aggregate cache SHA-256
`4d4e7aee64d7167f7fba6982ee182d4fb927ac74c2a3f0698fe15e2b8de80461`；audit
用时 `201.575s`，exit `0`。

首次单步 probe 在 report 生成前暴露纯机械 dtype 冲突：FP32 continuous timestep 产生的
sinusoidal embedding 进入 BF16 Linear。修复只在 Candidate D trainer forward 边界启用标准 BF16
autocast，未改 pinned DreamWAM，也未降低 source timestep 精度。经明确授权只做一次 repair retry：

- `1 GPU × 1 sample × 1 optimizer step`，`7.576s`；
- loss `1.484375`，prediction std `1.374959`；
- 五个 ActionDiT block gradient norms 均有限且非零：
  `[73.1104, 68.5613, 73.5351, 68.2277, 78.6729]`；
- proprio gradient norm `0.280970`；head update max abs `1.19209e-7`；
- peak allocated/reserved `3.285/3.439 GiB`；exit `0`；
- `saved_checkpoint=null`，没有 `.partial` 或任何 checkpoint 文件。

机械 report SHA-256：
`7a5d4ada789cbf7ed468d0c39d6be584cb44780eb48e1d79f6fb2aee18d8cc69`。完整可机读证据见
`experiments/evidence/h3_int8_dreamwam_kv_candidate_d_mechanical_v1.json`。

当前裁决严格限定为 `MECHANICS_PASS_ONLY`：cache/Data 与单步 trainability 通过；没有 parent
paired baseline、连续 checkpoint、restore、offline mechanism signal 或 closed-loop canary，因此
`effectiveness=NOT_EVIDENCE_READY`，也未获长训授权。
