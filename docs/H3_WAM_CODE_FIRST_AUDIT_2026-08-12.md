# H3-WAM 开源代码优先审计

更新时间：2026-08-12（Asia/Shanghai）

## 结论

下一条主线不再把“冻结 H3 + 小动作头 + 少量 LoRA”继续放大。最稳妥的可执行方向是把 H3 当作
`backbone_port`，移植一个已有完整训练和评测代码的 video-action 方法，仅替换基础视频模型。
当前首选是 LingBot-VA/FastWAM 风格的可训练双流路径；DiT4DiT 作为更受控、成本更低的并行替代。
M11/M13 继续完成，身份明确标为 legacy baseline。

2026-08-12 已完成第一轮最小反向门控消融：真实 H3 的梯度、FSDP、checkpoint restore 均通过，
但配对 val40 action loss 仅改善 `0.0142%`，因此该 canary 的结果是 `NO_GO_LONG`。这不是
LingBot-VA 完整架构的否定结论；被否定的是“在现有 one-way 模型尾部增加少量 gate 即可获得
策略收益”的简化假设。

## 代码事实对齐

| 项目 | 开源等级 | 实际代码中的关键设计 | 对 H3-WAM 的用途 |
|---|---|---|---|
| FastWAM | TRAINABLE | 33帧、stride4得到9帧；32动作；30层动作专家；video/action 联合训练；global batch128、10 epochs | 数据/动作接口和闭环父基线 |
| DreamWAM | TRAINABLE | RGB + flow + DINO + depth 结构化未来监督；约21,700 steps；动作专家30层 | 结构化未来监督，不应由60-step探索替代 |
| MiniWorld | TRAINABLE world model | action-conditioned streaming world model；DROID 上由6/16/32/64 latent frames递进 | 可作候选预测器，不是现成动作策略 |
| LingBot-VA | TRAINABLE | Wan2.2 双流 video/action flow matching；全30层；FSDP；5000 steps；streaming KV 与执行动作/观测反馈 | 最接近可直接移植的 H3 双流主线 |
| DiT4DiT | TRAINABLE | Cosmos特征在中层 hook 后 `detach`；独立16层 ActionDiT；video loss单独更新 backbone | 受控验证“世界学习 + 独立动作策略” |
| Cosmos-Policy | TRAINABLE | latent action/proprio/future-state/value slots；成功与失败 rollout；40k steps、global batch1920 | 目标架构与数据课程参考，非当前直接复刻 |
| Motus | TRAINABLE | 30层1:1 video/action MoT；VGM→latent-action pretrain→robot SFT 三阶段 | 后续大规模预训练路线 |
| ImageWAM/BadWAM/Motubrain | PARTIAL/对比 | 提供方法与实现差异参考 | 不单独作为 GO_LONG 预算来源 |

准确 URL 与 commit 见 `UPSTREAM_SOURCES.lock.json`。这里的 `TRAINABLE` 表示本地审计确认至少有
dataloader、forward/loss、optimizer/launcher、checkpoint 和 evaluator 的可执行链路，不代表与 H3
天然兼容。

## 已发现的论文—代码差异

- FastWAM 的实际 scheduler 使用 shifted-uniform 变换；不能仅凭论文描述改成 logit-normal。
- DiT4DiT 的 action feature hook 有 `detach`，动作 loss 不更新视频 backbone；其 backbone 依靠
  future-video loss 更新。“联合 forward”不能误写成端到端动作梯度。
- launcher 对默认 config 有大量覆盖，训练预算必须读取最终启动参数，不以默认 YAML 为准。

## 新主线的唯一可证伪假设

在相同 LIBERO dense 数据、动作归一化和闭环协议下，把 H3 接入经过官方代码验证的全层
video-action 双流训练，会比冻结 H3 head-only 父基线产生更强的语言目标区分和至少一个闭环成功。

## 实施门槛

### A. LingBot-VA/FastWAM → H3 主线

保持不变：两相机输入、动作 horizon/normalization、数据 split、评测任务和随机种子。唯一主变量是
Wan video backbone 替换为 H3，并实现对应 token shape、RoPE、cross-stream attention 和 checkpoint
转换。

在占用长线算力前必须通过：

1. H3 checkpoint restore 与首帧条件 forward；
2. video loss 对 H3 参数、action loss 对 action/fusion 参数的 finite 非零梯度测试；
3. 保存恢复同输入一致；
4. 一个完整 dense window 的帧/动作 indices 与官方接口等价；
5. 100-step canary 的固定 closed-loop 评测。

100-step 只决定是否继续，不证明效果。只有相对 M13 父基线 val 不退化超过5%、语言反事实差异
扩大且出现 `>=1/10` 固定闭环成功，才进入长训练。

### A1. 已完成的最小反向门控 canary

- action 只能写入 future-video rows，observation rows 受测试保护；
- 2-step 真实 H3 smoke 的反向 gate gradient norm 为 `46.4479`，保存/恢复链路通过；
- 100-step A/B 严格共享初始化、数据、seed、loss 和 FSDP 布局；
- B 相对 A 的 val40 action loss 仅改善 `0.0142%`，video loss 退化 `0.0043%`；
- 结论：不晋级 rollout、不扩 step、不增加卡，保留为工程基件。

### A2. 下一轮完整 block-causal 双流整改

下一轮不只是放大 gate，而是逐项移植官方代码中的因果信息结构：

1. 把 observation-video、future-video、noisy-action、executed-action/observation feedback
   声明为显式 stream，而不是把 action 仅作为 H3 尾层附加 K/V；
2. 按时间 chunk 构造 block-causal mask：动作只能读取当前及过去观测，future-video 可以读取
   对应动作，禁止 future observation 泄漏；
3. 训练时同时计算 video/action flow matching，逐层记录两个 loss 到 H3、video expert、action
   expert 和 fusion 的梯度覆盖；
4. 推理端增加与训练一致的 observation/action KV 更新，并用单元测试验证 train/inference mask
   等价；
5. 先跑参数量受控的 2-step restore/gradient smoke，再跑固定 100-step A/B；只有达到预注册
   held-out、语言反事实和 `>=1/10` 闭环门槛才申请长训卡。

在 A2 的 mask、stream contract 和 gradient coverage 测试完成前，空闲 A800 用于 smoke 和评测，
不启动未经证据门禁的长任务。

#### A2 phase 1 实施状态

已按官方 `wan_va/modules/model.py::_get_mask_mod` 实现可广播的 dense SDPA 参考 mask，并实现
H3 video Q/K/V 与 ActionDiT action Q/K/V 的四流联合注意力核心。当前已验证：

- noisy action chunk 可读取同 chunk clean video，但不能读取未来 video；
- noisy future-video chunk 只可读取更早的 clean action，不能读取同 chunk action target；
- noisy stream 同 slot 自注意、clean stream 因果注意及 window 限制与上游布尔条件一致；
- video objective 对 action expert、action objective 对 H3 均产生有限非零直接梯度；
- H3 当前 cache 的 12 个 latent frames 与 32 个动作不使用 Wan 的固定 reshape，而按共同
  动作时间轴映射成 8 个 chunk；每个 latent frame 的全部 spatial tokens 共享 chunk id；
- 21 项模块测试通过。

这一阶段仍不是可训练整模：下一步必须接入 H3 的 packed video rows、每 token timestep/RoPE、
clean/noisy stream embeddings 与 FSDP checkpoint，完成真实 H3 smoke 后才能升为 `GO_CANARY`。

### B. DiT4DiT → H3 受控线

保留官方 detached feature 边界，用真实 future-video loss 更新 H3，用独立 ActionDiT 学动作。这个
实验回答“动作梯度是否必须穿透 H3”，不与 A 同时改 sampler/loss/controller。若 video loss 没有
到达 H3，实验直接 NO_GO。

## 当前不放行的路线

- 继续扩大 head-only/LoRA 步数：离线下降但多轮闭环为零，没有新机制变量；
- 直接全量解冻 H3：在架构、梯度、checkpoint restore 尚未逐项对齐前风险不可控；
- 仅凭 DoT 论文重写：缺少同等级官方完整代码时只能做小 canary；
- SAM3D：按当前决策暂停；
- 完整 benchmark：候选未先获得固定 canary 正例前不投入。

因此当前门禁结论为：旧 M11/M13 **继续作为基线**；H3 双流移植在完成上述工程/梯度测试前为
`NO_GO_LONG`，测试完成后才可升为 `GO_CANARY`。
