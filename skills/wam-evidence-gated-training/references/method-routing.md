# WAM 方法路由与证据边界

本文件用于选择该查哪套实现，不替代每次重新浏览论文和官方仓库。

| 项目 | 一手来源 | 可支持的结论 | 不能直接支持的结论 | 审计重点 |
|---|---|---|---|---|
| Fast-WAM | arXiv:2603.16666；`yuantianyuan01/FastWAM` | video co-training 可在推理时不生成 future；官方 LIBERO 数据/动作/评测口径；动作目标是 shifted flow | 冻结任意视频 backbone、只训浅 head 也会成功；官方含 clean-action regression 或 policy teacher roll-in | 33-step span、stride4 得 9 视频帧/32 actions、10 epochs、30-layer ActionDiT、动作归一化与 rollout；从源码区分官方 flow 与本地 novel composition |
| StarWAM | `shaohua-pan/StarWAM` | Wan feature-conditioned 30-layer ActionDiT、last-layer pooled visual context、shifted flow、可训练视觉主干 | 冻结 INT8 H3 last-layer cache 等价于官方 joint backbone training | launcher resolved config、feature timestep、context 顺序、backbone optimizer group、10-step Euler、pre-denorm clamp |
| Faster-WAM / DoT | arXiv:2608.02365；截至审计时未确认作者官方代码 | 单层 action head 可从视频 backbone 所有层 K/V 获得表示；需要 RoPE realignment | 任意自写 KV fusion 都等价；论文重实现可作为训练主线；冻结 backbone 足够 | 没有作者代码时标 `paper-only backbone_port`；可做有界诊断长训，但不能冒充官方复现；测试 all-layer K/V、mask、RoPE、gradient，并以机制与闭环结果单独决定 `EVIDENCE_READY` |
| Faster-WAM future conditioning | arXiv:2608.04404 | SparseMoT + Interval KV-Fusion 在推理保留 future 表征 | 与 DoT 同一项目；head-only 等于 future conditioning | 区分两篇同名 Faster-WAM；确认 future 是怎样计算/复用和推理成本 |
| DreamWAM | arXiv:2608.04996；`hustvl/DreamWAM` | RGB+motion 联合 latent denoise、geometry/semantics supervision；action Q 在每层读取该层 video K/V | 只加 RAFT loss 60 steps即复现；把同一个 last-layer context 重复注入每层等价于 DreamWAM carrier | 官方 config 21,700 steps、LR/warmup、逐层 K/V 身份与 alias、flow init、DINO/depth routing、joint vs uncond |
| MiniWorld | arXiv:2608.01127；`zhao-yian/MiniWorld` | action-conditioned、block-causal、diffusion-forcing streaming video WM；多阶段长度 curriculum | 它是可直接 rollout 的机器人 action policy；可直接替换 H3 head | DROID action keys/camera、6→16→32→64 latent-frame stages、Muon、rolling KV cache；若接入 WAM 要另建 policy bridge |
| FACT | `Bariona/FACT` | causal act-then-imagine；pred action 不看未来，clean action 条件 future state/value/video；失败样本屏蔽 imitation；best-of-N value ranking | 冻结 H3 action head 天然具备 consequence/value；没有失败 canonical 数据也能声称 failure-aware | teacher-forcing mask、两阶段推理、failure onset/action mask、future/value target、BoN 排序；新仓库/单 commit 时提高审计强度 |
| ImageWAM | arXiv:2606.19531；`yuyangalin/ImageWAM` | text-guided target edit/cache 可能比密集视频更聚焦任务相关变化 | H3 I2V 天然等价 image-editing prior | source-conditioned target frame、KV cache 接 action expert、冻结理解模块、single-step inference；适合作为目标/语言绑定独立路线 |
| LingBot-VA | `Robbyant/lingbot-va`；LeRobot `policies/lingbot_va` | Wan2.2 双流视频/动作 flow matching、block-causal streaming、真实观测与执行动作回灌 KV cache | 只训浅动作头能获得同样长上下文能力 | launcher 覆盖、预计算 latent 的 frame ids、30D action mask、两套 scheduler、完整 transformer 反传、LIBERO client |
| DiT4DiT | `Mondo-Robotics/DiT4DiT` | future-video loss 更新视频 backbone；独立 16 层 ActionDiT 从指定中间层取特征 | action loss 直接更新视频 backbone；默认 YAML 就是实际运行配置 | hook 中 `detach`、launcher 将 logit-normal 攓为 uniform、8-action/9-frame窗口、80k steps、64 GPU evaluator |
| Motus | `thu-ml/Motus` | 三专家 MoT、视频和动作联合 flow matching；Stage-2 latent-action 预训练后 Stage-3 embodiment SFT | 直接从 H3 原始权重做少量 LIBERO SFT等价于 Motus | Stage-2 checkpoint、30层 1:1 joint attention、动作/视频频率、全参数训练、RoboTwin evaluator |
| Cosmos Policy | `NVlabs/cosmos-policy` | 将 proprio/action/future state/value 编码进视频 latent slots，LIBERO 四 suite 完整训练/评测 | 其 2B 配方可原样缩放到 32B H3 | demo+成功/失败 rollout mixture、latent mask、chunk16、global batch1920、40k step、action L1 和 device-sensitive eval |
| RoboTTT | NVIDIA project/paper | 长上下文 rollout 的 test-time adaptation | 能修复第一步就选错目标的 policy | 仅在静态/短程闭环已有正例后评估，严格区分在线适应与训练增益 |

## 代码优先选择规则

1. 优先选择包含 dataloader、真实训练 loop、官方 config、checkpoint restore 和闭环 evaluator 的仓库。
2. 只有模型定义或伪代码的仓库不能作为训练配方来源。
3. README 参数与代码默认值冲突时，以 launcher 展开后的 resolved config 和执行代码为准。
4. 有官方 checkpoint 时先在官方 evaluator 上复现至少一个已报告基线，再移植 H3。
5. H3 替换实验必须保留同仓库、同数据、同 evaluator 的原 backbone 对照；否则无法归因。

## 项目替换规则

将 Wan2.2 换成 MiniMax-H3 时，逐项重新证明：

- latent/video token layout 和条件首帧语义；
- text encoder/token 与 action head 接口；
- backbone layer 数、hidden/head dimensions 和 K/V 投影；
- 3D RoPE 到 action 1D RoPE 的对齐；
- first-frame causal mask 与 future leakage；
- flow timestep/velocity target；
- VAE 时空压缩和 future frame indices；
- 哪些 H3 参数被 video/action/aux loss 实际更新。

只要其中一项没有直接测试，就标 `UNKNOWN`，最多允许工程 smoke，不允许长训。
