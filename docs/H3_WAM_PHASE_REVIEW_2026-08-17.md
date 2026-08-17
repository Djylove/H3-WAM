# H3-WAM 阶段性代码、来源与训练预算审查（2026-08-17）

## 0. 晚间执行更新

- C67 已完成 20k；固定 s10→s20 的 normalized/physical MSE 分别恶化 `1.526%/1.729%`，visual
  response 只保留 `81.13%`，正式 `NO_C67_PAIRED_680_ROLLOUT`。因此 C68 同配方 30k 不再放行。
- C69 同预算 action-only 归因线继续执行，已跨过 s5000；仍只允许最终 C69-s20 对 C67-s20 解释
  auxiliary objective 的净作用，中间 preview 不用于选点。
- C70 只改 sampler 为平均 `6/1/0.5/0.5`。非保留单步 probe 与真实 10 步 canary 均通过：30/30
  shared gradients、future leak 0、所有 loss 有限、12GB checkpoint strict restore max-abs 0；机械状态
  `GO_C70_LONG / NOT_EVIDENCE_READY`。20k 长训必须继续每1k原子保存和严格恢复，最终只比较预注册
  C70-s20 与 C67-s20；离线门失败则不做 LIBERO。

## 1. 本轮唯一问题与结论

**可证伪问题**：在保持 C58 parent、冻结 online INT8 H3、C67 的 FACT 结构、数据和评测合同不变时，
当前 20k 预算是否已经足以排除“训练曝光不足”，以及下一轮应优先增加 steps、改变 sampler，还是改变
action/future 接口？

结论是：**当前证据不足以排除训练曝光不足，但也不支持把原 4/2/1/1 配比直接机械拉到 100k/150k。**

- C58 parent 已用 80,000 个不重复专家窗口训练 10k steps，相当于 `0.398448` expert epoch；C67 自身到
  s20k 又看到 80,000 个专家样本。因此共享 action/FACT 30 层到 C67-s20 的累计 expert exposure 是
  `0.796896` epoch，而不是只看 C67 dossier 得到的 `0.398448`。
- C67 新增的 future-state/value/future-representation 编解码器从零初始化，只拥有 C67 自身的
  `0.398448` expert epoch。对这些新模块，“不足一轮”判断成立。
- 同一个 s20k 合同却会把 success rollout 看 `16.214` 遍、observational failure 看 `1.544` 遍、causal
  failure 看 `10.368` 遍。把该配比拉到 100k 会变成 `81.070/7.722/51.840` 遍；这同时改变过拟合风险，
  不能只叫“补足 epoch”。
- C67 s1k..s4k 的 optimizer curve 显示 future-state/value 在学习，但 action loss 与 7,168 维
  future-H3 representation loss 尚未出现持续下降。训练 loss 不能替代 held-out/rollout，但这个形状说明
  “只加同配比 steps”不是唯一合理假设。

因此本轮不改正在运行的 C67。先让它完成预注册的同轨迹 s10k/s20k 比较；下一轮长训必须把
`budget-only`、`sampler coverage` 和 `FACT auxiliary attribution` 分成三个候选，不得一次混改。

## 2. 当前真实执行合同

| 字段 | 当前 C67 执行事实 | 身份 |
|---|---|---|
| 分类 | C58 FastWAM carrier + FACT causal tracks + H3 backbone port | `novel_composition/backbone_port`，不是官方复现 |
| H3 | MiniMax-H3 INT8，50 层 online forward，每 rank 一份；`requires_grad=false` | intentional deviation |
| carrier | H3 50 层映射到 30 个 layer-wise K/V prefix | local tested port |
| action/consequence trunk | 一个 30 层 ActionDiT 同时处理 `[A,G,V,I]`；A 看不到 clean action/future targets | FACT-style causal port |
| trainable | 30 层 ActionDiT、action/proprio、future-state/value/future-representation encoders/decoders | H3 本体不更新 |
| action | 32×7，shifted flow，shift=5，action weight=10 | FastWAM-compatible action objective |
| future | future H3 K/V 池化为 56×128=`7168` 维标准化向量；另有 state8/value1 | local replacement，不等价 RGB/video loss |
| sampler | 每 step 固定 rank：4 expert + 2 success + 1 observational failure + 1 causal failure | local fixed mixture |
| optimizer | AdamW；base/action LR `2e-5/2e-4`；warmup500；20k cosine；GB8 | local budget ablation |
| budget | C67 20k/160k samples；每 1k 保存 12.202 GB full-state 并 strict restore | running |
| decisive eval | 固定 balanced80 learning curve；仅 s10/s20 可进入 paired 680 LIBERO | pre-registered |

执行代码证据：

- 固定 rank mixture：`scripts/h3wam/train_c56b_fact_online.py:52-55`；
- episode→frame sampling：`scripts/h3wam/train_c56b_fact_online.py:120-170`；
- frozen H3 与 30-layer port：`scripts/h3wam/train_c56b_fact_online.py:336-390`；
- loss weights 和 global masked mean：`scripts/h3wam/train_c56b_fact_online.py:275-304`；
- future target 为 online future-H3 K/V 表示：`scripts/h3wam/train_c56b_fact_online.py:232-272`；
- 共享 causal tower：`src/fastwam/models/h3wam/fact_layerwise_tower.py:55-412`。

## 3. C67 的逐流 exposure，而不是一个汇总 epoch

训练池为 expert `200,779`、success `2,467`、observational failure `12,950`、causal failure `1,929` 个
train windows。以下 exposure 仅计算 C67 增量；`累计 expert` 另加 C58 parent 的 `0.398448`。

| C67 steps | samples | expert | success | observational | causal | 共享塔累计 expert |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 8,000 | 0.020 | 0.811 | 0.077 | 0.518 | 0.418 |
| 4,000 | 32,000 | 0.080 | 3.243 | 0.309 | 2.074 | 0.478 |
| 10,000 | 80,000 | 0.199 | 8.107 | 0.772 | 5.184 | 0.598 |
| 20,000 | 160,000 | 0.398 | 16.214 | 1.544 | 10.368 | 0.797 |
| 30,000 | 240,000 | 0.598 | 24.321 | 2.317 | 15.552 | 0.996 |
| 50,195 | 401,560 | 1.000 | 40.693 | 3.876 | 26.021 | 1.398 |
| 100,000 | 800,000 | 1.992 | 81.070 | 7.722 | 51.840 | 2.391 |
| 150,000 | 1,200,000 | 2.988 | 121.605 | 11.583 | 77.760 | 3.386 |

两个容易误判的数字：

1. C67 自身达到 1 expert epoch 需要 `ceil(200779/4)=50,195` steps；
2. 共享动作塔连同 C58 parent 达到累计 1 expert epoch，只需 C67 约 `30,195` steps。

以 s1k..s4k 实测 optimizer `1.21–1.23 sec/step` 计算，20k 纯训练约 6.8h；但每 1k 的 12.2 GB
checkpoint+restore 使里程碑间隔约 25.2min，端到端约 8.4h。相同 cadence 下，30k 约 12.6h、50k 约
21h。存储分别约 366 GB、622 GB；共享盘仍有约 24 TB 空闲，但到期墙钟比容量更紧。

## 4. 当前曲线不能证明效果，也不能证明已饱和

来自 n4 原始 `train_s1000..s4000.json` 的每 1k mean：

| segment | total | action | future H3 rep | future state | value |
|---|---:|---:|---:|---:|---:|
| 1–1k | 3.2436 | 0.07049 | 1.93180 | 0.83766 | 0.67958 |
| 1k–2k | 2.8158 | 0.07012 | 1.94229 | 0.16469 | 0.26593 |
| 2k–3k | 2.7455 | 0.06897 | 1.92575 | 0.10992 | 0.21520 |
| 3k–4k | 2.7391 | 0.06953 | 1.92924 | 0.08962 | 0.19682 |

解释边界：state/value heads 正在拟合；action 与 future-H3 rep 暂时近似平台。由于四流交替、flow timestep
随机且这是 train loss，这张表只支持“继续获取预注册 held-out 曲线”，不支持 early stop、选 s3k，或宣称
20k 必然无效。当前每个里程碑均 strict restore，future→action leak 为 0。

## 5. 与官方执行代码的再次对齐

| 来源与 revision | 官方/执行预算 | 可迁移机制 | 当前 H3-WAM 差异 | 判定 |
|---|---|---|---|---|
| FastWAM `45d8e145...` | LIBERO `num_epochs=10`、per-device batch16、8 GPU；完整 loader epoch | 33-frame span、stride1 dense windows、9 video/32 action、30-layer ActionDiT | 当前 dense 合同基本继承窗口，但 H3 frozen K/V 替代可训练 video DiT；累计 expert 尚不足1轮 | `INTENTIONAL_DEVIATION` |
| FACT local `618a6c1...`；upstream `9427ea4...` 只多 README live-demo | `max_steps=150000`、8×32 GB256；expert/failure episode mixture；future RGB/state/value | act-then-imagine、clean-action consequence、failure imitation mask | 当前保留 causal mask，但 GB8、H3 frozen、future-H3 vector 替代 future video；固定4/2/1/1不是官方 episode-count mixture | `INTENTIONAL_DEVIATION` |
| DreamWAM `6e989fac...` | 21,700 steps、batch16；33/4→9-frame+32-action | RGB+motion joint denoise；DINO/depth structured future；逐层 shared attention | 当前没有 RGB/motion/depth/DINO，只预测单个池化 future-H3 target | `MISMATCH`，不能称 DreamWAM port |
| MiniWorld `e484206b...` | 6/16/32/64 latent-frame curriculum；100 epochs、50 epochs、30k、30k | action-conditioned streaming WM、Muon、长度课程、rolling KV | 当前没有逐阶段时域课程，C62只移植过 rolling context 且失败 | `MISMATCH` |
| Light-WAM `b2785f66...` | launcher 默认 150k steps、4×16 GB64、25 epochs fallback | frozen compact backbone + all-layer LoRA；8/16/24 adapters；learned-query pooling；1-block action trunk；前8步动作加权 | 当前是 frozen INT8 H3 + 30-block action trunk，无 backbone LoRA/learned-query pooling | `TRAINABLE SOURCE / HIGH PRIORITY` |
| WLA `155ac94e...` | LIBERO all action/image-action 均 100k steps、per-device batch32 | image-action/action-only官方配对；history8；world expert通过meta-query影响action | AR/RynnBrain/Sana 与 H3 接口不同；可借鉴严格同预算 world-on/off 配对 | `TRAINABLE SOURCE / MECHANISM REFERENCE` |
| GAM `18f5cf09...` | LIBERO 150k steps、GB24；冻结前13个GFM blocks；12-layer future predictor | 3D geometry prior、causal future latent、action history、future horizon mixture | 当前 future target 是H3 K/V均值，不含显式几何；架构替换量大 | `TRAINABLE SOURCE / LATER TRACK` |
| Faster-WAM/DoT arXiv:2608.02365 | code未确认；paper-only | all-layer K/V + RoPE realignment + single-layer action head | 当前已有all-layer映射但仍用30-layer action tower | `PAPER_ONLY / use Light-WAM code first` |
| SelfWAM arXiv:2608.00725 | code未确认；paper-only | clean-action-conditioned future + robot self-mask | clean-action路径已有；self-mask可由LIBERO simulator segmentation提供，不需要启动SAM3D | `PAPER_ONLY / later controlled ablation` |
| RepWAM `ad32f521...` | 当前仓库主要为方法/开放计划，未确认完整trainable recipe | semantic visual-action tokenizer、latent action pretraining | 需要额外 tokenizer/pretraining stage，不能当下一轮快速修复 | `PARTIAL / research reserve` |

### 5.1 对“别人到底训练多久”的直接回答

可执行新代码的共同特征不是某个统一 epoch 数，而是**百万级以上 sample exposure + 明确的阶段/配对**：

- 当前 C67-s20：160k 增量样本；连同 C58 parent 也只有 240k 历史训练样本；
- FastWAM：完整训练集 10 epochs；
- DreamWAM：21.7k steps，batch16（分布式后实际 global sample 数更大）；
- WLA：100k steps，per-device batch32；
- Light-WAM：150k steps，默认4卡×16，约9.6M samples；
- GAM：150k×GB24，约3.6M samples；
- FACT：150k×GB256，约38.4M samples。

这些数字不能直接按模型大小横比，但足以说明：C67 的 20k×GB8 属于有意义的诊断长训，不是来源级
充分训练。尤其当 action tower 有约十亿级参数且新 consequence heads 从零开始时，不能用当前预算给
“H3/FACT上限”下结论。

## 6. 最新论文带来的、但由代码约束的启发

### 6.1 优先减少动作专家深度，而不是继续堆辅助 head

Faster-WAM 提出 single-layer action head；Light-WAM 已提供可执行的 learned-query multi-layer state
fusion 和一层 trunk。我们的 C58 已证明 H3 all-layer carrier 有效，但仍让 30 层动作塔重复变换这些信息。
因此下一条新架构线应从 Light-WAM 固定 commit 直接移植 pooling/trunk，而不是凭论文重写 DoT。

### 6.2 future target 应更“动作相关”

DreamWAM、GAM、SelfWAM、RepWAM 从不同角度指向同一问题：RGB/高维生成表示包含大量与动作无关的变化，
应使用 motion/geometry/semantic/self-motion 或 representation-aligned target。C67 的 7,168 维 pooled H3 K/V
loss 到 s4k 几乎不变，是把这一机制列为优先假设的本地证据；但在 C67 完成前不得换 target。

### 6.3 world objective 必须有 action-only 同预算父对照

WLA 公开了 LIBERO all 的 `action` 与 `image_action` 两套同为100k的配置。我们历史 C60 只与已经结束的
C58 parent 比，不是“继续训练同样 steps 但关闭 FACT auxiliary”的严格同预算对照。下一轮必须补这个
paired arm，否则即使长训成功，也无法区分更多 action optimization 与 consequence objective 的贡献。

## 7. 修订后的三条可归因实验线

### P0：完成 C67，不改当前合同

- 目标：同一20k cosine trajectory 的 s10k vs s20k；回答 C67 增量 expert `0.199→0.398`、共享塔累计
  expert `0.598→0.797` 时，held-out conditioning/action 和 paired LIBERO 是否改善。
- 当前：s4k checkpoint 已生成；四个 strict restores 通过。仍是 `NOT_EVIDENCE_READY`。
- 禁止：根据 s1k..s4k train loss 改 LR、sampler、目标或提前选择 checkpoint。

### P1-A：C68 one-cumulative-expert-epoch budget arm

- 父模型：同一 C58 s10k；唯一变量相对 C67 是 fresh 30k scheduler/budget。
- 合同：保持4/2/1/1、所有数据hash、loss、seed/evaluator不变；30k trajectory 内冻结 s20/s30 比较。
- 意义：C68-s30 让共享塔累计 expert exposure 约 `0.996`，第一次真正跨过“约一轮专家数据”。
- 预算：240k增量samples；实测端到端约12.6h；30个1k checkpoint约366GB。
- 放行：C67 final preview 必须表明 conditioning 未坍塌；若 s10→s20 已系统恶化，不用“加epoch”绕过。

### P1-B：C69 matched action-only attribution arm

- 父模型、30层架构、样本顺序、4/2/1/1 masks、20k schedule、seed全部与C67相同；唯一关闭
  future-representation/state/value losses及其新heads更新。
- 比较：C69-s20 vs C67-s20，held-out action/visual/language/gripper + fixed paired rollout。
- 意义：分清 C67 任何变化来自“继续训练 action tower”还是 FACT consequence supervision。
- 注意：failure ranks 在 action-only arm 不产生 imitation；实现必须证明 DDP masked mean 与 C67 action
  分量逐步等价，不能临时改成8个expert ranks。

### P2-A：C70 sampler-coverage arm

- 保持20k steps/GB8/20k schedule；rank周期改为平均 `6 expert + 1 success + 0.5 observational + 0.5 causal`。
- 这样 C70 自身 expert `0.598`、共享塔累计约 `0.996`；success约8.1、observational约0.77、causal约5.18。
- 与 C67-s20 是同算力的 sampler 单变量，回答“更多真实状态覆盖”是否优于反复小 failure pool。
- 该 arm 必须独立 dossier；不能与 C68 结果拼成同一个候选。
- 当前：probe 与 10-step checkpoint/restore canary 已通过，长训 dossier 通过 `--target long`；效果仍未知。

### P2-B：Light-WAM shallow state-fusion H3 port

- 直接固定 `b2785f66...` 的 adapter/state-fusion/action expert 代码路径；先保持 H3 frozen，不宣称官方复现。
- 用现有30层H3 K/V作为多层来源，learned-query pooling + 1-block trunk；与 C58 30-layer tower做相同
  expert samples/steps的 paired comparison。
- INT8 H3 上不假装实现官方 all-layer LoRA；QLoRA/局部解冻是另一个变量，后置。

## 8. 决策门

1. **现在**：C58仍是唯一 champion；C67已失败，C68不启动，C69与C70保持独立单变量长线。
2. **C69/C70 final 后**：C69回答 auxiliary attribution，C70回答 sampler coverage；二者都必须先过固定
   balanced80，再决定是否进入 fresh paired LIBERO，不能用中间点或跨实验拼接胜者。
3. **闭环**：任何 line 只有通过固定 held-out gate 才进 fresh LIBERO；train loss和future loss均不能晋级。
4. **来源边界**：Faster-WAM/SelfWAM 在作者训练代码未确认前只能提出假设；实际实现优先 Light-WAM、
   DreamWAM、GAM、WLA 已公开执行代码。

## 9. 本轮来源身份

| 项目 | URL | revision/状态 |
|---|---|---|
| FastWAM | https://github.com/yuantianyuan01/FastWAM | local=head `45d8e145...` |
| FACT | https://github.com/Bariona/FACT | code pin `618a6c1...`; head `9427ea4...` 仅 README +3 lines |
| DreamWAM | https://github.com/hustvl/DreamWAM | local=head `6e989fac...` |
| MiniWorld | https://github.com/zhao-yian/MiniWorld | local=head `e484206b...` |
| StarWAM | https://github.com/shaohua-pan/StarWAM | local=head `cd76d96f...` |
| Light-WAM | https://github.com/L1ziang/Light-WAM | reviewed temp clone `b2785f66...` |
| WLA | https://github.com/SJTU-DENG-Lab/WLA | reviewed temp clone `155ac94e...` |
| GAM | https://github.com/cvlab-kaist/Geometric-Action-Model | reviewed temp clone `18f5cf09...` |
| ImageWAM | https://github.com/yuyangalin/ImageWAM | local=head `5d4a341e...` |
| RepWAM | https://github.com/wdrink/RepWAM | head `ad32f521...`; training completeness not confirmed |

本轮没有修改任何正在执行的训练、数据、checkpoint 或 evaluator，也没有把临时 review clones 写入项目。
