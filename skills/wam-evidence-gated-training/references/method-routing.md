# WAM 方法路由与证据边界

本文件用于选择应审计哪套实现，不替代每次重新检查官方仓库。

| 项目 | 代码支持的用途 | 不能直接推出 | 首要审计项 |
|---|---|---|---|
| FastWAM | 完整 video/action 联合训练、LIBERO 数据动作评测基线 | 冻结任意视频 backbone + 浅 head 也有效 | 33帧/stride4/9视频帧、32动作、30层 ActionDiT、10 epochs |
| Faster-WAM/DoT | paper-only all-layer K/V + RoPE realignment 思路 | 自写 KV fusion 等价或可 GO_LONG | 官方代码身份、K/V、mask、RoPE、梯度 |
| DreamWAM | RGB+motion+DINO+depth structured future | 60-step RAFT-only 可复现/证伪 | 21,700 steps、初始化、loss schedule、layer routing |
| MiniWorld | action-conditioned streaming world model | 是可直接 rollout 的 action policy | 6→16→32→64 curriculum、Muon、rolling KV |
| LingBot-VA | 双流 video/action flow、streaming observation/action feedback | 浅动作头有同样上下文能力 | launcher、latent frame ids、30D mask、完整反传 |
| DiT4DiT | future-video loss + 独立 ActionDiT | action loss 更新视频 backbone | hook `detach`、launcher overrides、80k steps |
| Motus | VGM→latent-action pretrain→robot SFT 三阶段 MoT | 少量 LIBERO SFT 等价 | Stage-2 checkpoint、1:1 joint attention、全参训练 |
| Cosmos-Policy | latent action/state/value slots 与 rollout data mixture | 2B 配方可直接缩放到32B | chunk16、global batch1920、40k steps、eval |
| ImageWAM | target-edit/cache 的语言目标绑定路线 | H3 I2V 天然等价 | source conditioning、KV cache、冻结边界 |

## H3 backbone port 必证项目

- latent/video token layout 与首帧条件语义；
- text/action 接口、hidden/head dimensions；
- 3D video RoPE 与 action RoPE；
- causal mask 与 future leakage；
- flow timestep/velocity target；
- VAE 压缩和 frame indices；
- video/action/aux loss 分别更新哪些参数。

任一项缺少直接测试就标 `UNKNOWN`，最多允许工程 smoke。
