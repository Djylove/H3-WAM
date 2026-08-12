# MiniWorld 对 H3-DreamWAM 的可用性评估

更新时间：2026-08-10

## 结论

MiniWorld 值得借鉴，但不能原样替换 H3-DreamWAM。它解决的是
`历史观测 + 已知动作 -> 未来视频`，当前项目解决的是
`当前观测 + 指令 + proprio -> 未来动作`。MiniWorld 没有动作生成头、语言条件、proprio
输入或 LIBERO 闭环成功率，因此其公开视频结果不能证明策略能力。

最有价值的不是把 H3 换成 MiniWorld，而是吸收三项训练原则：

1. 用块因果注意力和 Diffusion Forcing 缩小训练与流式推理的差异；
2. 用短到长的时域 curriculum 学局部接触，再学长上下文；
3. 用零初始化的 AdaLN-LoRA 残差逐层引入动作条件，避免新动作分支破坏已有视频尺度。

这三项恰好对应当前失败：joint300 的视频 MSE 继续改善，但动作 MSE 和闭环没有改善；
当前 joint100 checkpoint 又没有训练任何 ActionDiT block。下一轮应冻结已晋级 H3 世界
分支，先让 ActionDiT 尾部以稳定的门控和短到长 curriculum 真正学起来。MiniWorld 独立
world-model canary 放在第二优先级，不抢占动作闭环主线。

## 上游审计

- 仓库：`third_party/MiniWorld`
- 固定提交：`e484206bbd4360ae56ed8abad51c83f2457ac092`
- 仓库创建于 2026-07-30，目前只有 2 个 commit；源码可通过 Python compile 检查。
- 当前没有 `LICENSE` 文件。只作为研究参考，不直接复制代码到可发布实现。
- 开源内容包含 DROID/RealEstate10K loader、训练、流式采样、吞吐测试，以及 0.5B/1B
  的两类 checkpoint；不包含单元测试、LIBERO adapter 或策略评测代码。
- Video DiT 从零训练，但 Wan2.2 VAE 是冻结的预训练模型，所以“from scratch”不代表
  所有视觉模块都从零训练。

## MiniWorld 真正做了什么

```text
首帧/历史 RGB + 未来动作
          │
          ▼
Wan2.2 VAE latent
          │
          ▼
块因果 Video DiT + Rectified Flow
  ├─ chunk 内双向注意力
  ├─ chunk 间因果注意力
  ├─ 动作经 AdaLN-LoRA 调制
  └─ 非递减 chunk 噪声（CoPP / Diffusion Forcing）
          │
          ▼
未来 RGB 流式 rollout
```

DROID 默认把 Cartesian position 与 gripper action 取出，经 q01/q99 归一化；Wan VAE
每 4 个原始动作对应一个未来 latent frame。训练随机混合“仅首帧 clean”和“首 chunk
clean”，并按 6、16、32、64 latent frames 四阶段继续训练。默认 1B 模型采用 8 卡 DDP、
BF16、Muon：前两阶段 100/50 epoch，后两阶段各 30k step。

推理使用滚动 KV cache、resident sink 和异步 chunk 去噪。论文报告 1B DROID 在关闭 CFG
的吞吐测试中，由全窗口 3.31 FPS 提升到 7.29 FPS，首 chunk 74.0 秒降到 4.86 秒。但默认
质量采样仍是 100 个去噪步，这不是可直接用于机器人高频控制的实时策略。

## 与当前方案的差异

| 维度 | H3-DreamWAM | MiniWorld | 影响 |
| --- | --- | --- | --- |
| 主任务 | 观测到动作 | 动作到视频 | 不能直接替换 ActionDiT |
| 视觉初始化 | 33.1B H3 视频基模 | 0.5B/1B/3B Video DiT 从零训练 | MiniWorld 更轻，但丢失 H3 先验 |
| VAE | H3 VAE，H3 特有时序布局 | Wan2.2 VAE，4 个动作/latent | 动作对齐不能机械复用 |
| 指令 | H3 文本 packed rows | 无语言输入 | 单独使用无法区分 LIBERO 任务目标 |
| proprio | 已共享给 H3 与 ActionDiT | 无 | 不适合直接做闭环策略 |
| 动作 | 独立 ActionDiT 生成 32-step chunk | 已知动作只作为视频条件 | MiniWorld 本身不是 WAM policy |
| 长上下文 | 当前固定窗口联合采样 | 块因果训练 + rolling KV | 值得移植训练思想 |
| 训练稳定性 | Action 主体梯度尺度曾失稳 | 动作调制低秩残差零初始化 | 值得用于 ActionDiT 尾层解冻 |
| 评测 | LIBERO action MSE + 闭环成功 | 50 个 held-out 视频的视觉/几何指标 | MiniWorld 结果不能替代成功率 |

## 对当前失败的解释价值

MiniWorld 强化的是动作对视觉后果的因果建模，而当前 H3-DreamWAM 的 Video Expert 不读
动作，Action Expert 单向读取 Video K/V。joint300 出现“video 更好、action 更差”并不
意外：继续优化世界重建不保证动作头获得可执行控制能力。

MiniWorld 同时说明，不应继续只训随机输出头或单纯加大 Flow loss。它的动作条件在每层
通过共享调制和零初始化低秩残差进入，并且整个 Video DiT 从头到尾参与优化。映射到我们
这里，至少要训练 ActionDiT 的真实 block；为了避免此前全层解冻的梯度爆炸，应通过
零初始化 gate、尾层逐级解冻和独立 clipping 控制尺度。

## 推荐实验顺序

### M4-A：先修复动作主线

1. 固定 `multisuite_uniform_joint100.pt` 的 H3 I/O、H3 tail 和 Flow 分支，不再追 video
   MSE。
2. ActionDiT 采用 H8 -> H16 -> H32 curriculum；每阶段只在 episode-disjoint、40-task
   uniform 数据上训练，不做任务加权。
3. 先开最后 2 个 Action block，再到 4 个；Video K/V 残差增加可学习 gate，初值为 0，
   Action I/O 与新 block 分组学习率、分组裁剪。
4. 保存 100/300/600/1000-step checkpoint；每个点先跑 40-task sampler，再跑固定三个
   LIBERO-goal 闭环。首个出现目标物体/关节有效位移的 checkpoint 才晋级。

这一步能直接检验 MiniWorld 的“稳定引入动作条件 + 短到长训练”是否解决现有 0/3，且
不引入第二个大模型，归因最清楚。

### M4-B：MiniWorld 独立 causal-world canary

在不改主策略的前提下，用发布的 `MiniWorld_0_5b_droid.pt` 初始化，增加 LIBERO
LeRobot adapter：

- 保留 DROID 与四套 LIBERO 混合训练，使用 dataset-specific action normalizer；
- 首轮只用 agent-view，避免双视角拼接与因果模型同时变化；
- 对 LIBERO 精确重建“每个 latent 对应哪些控制步”，不能沿用 Wan 的固定 4-action
  规则去假设 H3 时序；
- 只更新 action encoder、AdaLN-LoRA 与最后若干 block，先做 2k-step canary；
- 评估真实动作 rollout 视频，并用打乱/反向/零动作做 counterfactual action-following
  测试。只有真实动作明显优于反事实动作，才说明它学到了控制因果而非外观续写。

M4-B 的产物先作为世界模型/候选动作评估器，不直接宣称为策略。

### M5：通过 canary 后再融合

优先考虑两个方向：

1. 将 MiniWorld clean-history 的因果视觉 token 作为第二路 K/V 交给 ActionDiT，H3 继续
   提供语言和通用视觉先验；
2. 由 ActionDiT 生成少量候选 action chunks，MiniWorld 预测各候选后果，再用目标/价值
   头重排。

方向 1 更适合训练期表征增强，方向 2 更容易做闭环因果验证但推理慢。两者都应在
MiniWorld canary 证明 action-following 后再做。

## 暂不做的事情

- 不把 MiniWorld-1B 直接当 ActionDiT；它没有动作输出。
- 不立即丢弃 H3；MiniWorld 没有语言条件，且论文目标不是策略成功率。
- 不直接把 MiniWorld 的 block-causal mask 塞进 H3 packed attention；H3 的文本、视频、
  audio/proprio row layout 和 Wan latent 时间轴不同。
- 不先跑完整 60k-step 长视频训练；当前最主要的证据缺口是动作闭环，不是 253 帧视频。
- 不以论文相对视觉指标作为“超过 FastWAM”的证据；最终仍以 LIBERO benchmark 成功率
  和固定 seed 配对结果为准。

## 资源判断

- 8x80GB 足够复现 0.5B/1B 的 DDP 配方，也足够并行推进 H3-DreamWAM 的 Action tail；
  第一轮只建议 0.5B canary，避免在方向未验证前投入 1B 的四阶段完整训练。
- 5090 适合 MiniWorld 0.5B 单卡推理/小 batch 局部微调，但默认 100-step 采样延迟不适合
  在线高频控制。最终部署需要减少采样步、蒸馏或只离线使用世界模型。
- 上游 checkpoint 大小约为：0.5B DROID 2.23GB，1B DROID 3.85GB。先下载 0.5B，不下载
  全部四个 checkpoint，避免无谓占用存储。

## 决策门槛

MiniWorld 路线晋级需要同时满足：

1. counterfactual action-following 明显成立；
2. 作为辅助特征或候选重排后，固定 LIBERO 闭环至少从当前 0/3 提升到 1/3；
3. 改善来自多个任务而非单任务专调；
4. 增益能在可接受的推理预算内保留。

否则保留它的 curriculum、零初始化动作调制和滚动缓存思想，不把 MiniWorld 模型本体纳入
最终 H3-WAM。
