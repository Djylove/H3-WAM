# H3-WAM 开源代码优先审计

更新时间：2026-08-12（Asia/Shanghai）

## 结论

下一条主线不再把“冻结 H3 + 小动作头 + 少量 LoRA”继续放大。最稳妥的可执行方向是把 H3 当作
`backbone_port`，移植一个已有完整训练和评测代码的 video-action 方法，仅替换基础视频模型。
当前首选是 LingBot-VA/FastWAM 风格的可训练双流路径；DiT4DiT 作为更受控、成本更低的并行替代。
M11/M13 继续完成，身份明确标为 legacy baseline。

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
