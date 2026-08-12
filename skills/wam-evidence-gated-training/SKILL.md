---
name: wam-evidence-gated-training
description: 审计并放行 World Action Model 研究实验：强制对齐论文、作者官方仓库、固定 commit、本地实现、数据采样、优化预算、动作接口和闭环评测，再生成 GO_CANARY、GO_LONG 或 NO_GO 结论。
---

# WAM evidence-gated training

把开源实现转成可审计的实验合同。没有证据 dossier，不启动新的 GPU 训练；已经明确要求保留的
长线实验继续运行，但标记为 legacy baseline。论文只作为方法意图、消融和代码缺失细节的辅助证据，
不能代替可执行代码。

## 1. 固定问题

只写一个可证伪问题，并将实验标为 `reproduction`、`backbone_port`、
`controlled_ablation` 或 `novel_composition`。模型替换不得称为官方复现。给每个实验指定一个
不变父基线和唯一主变量。

## 2. 代码优先建立来源身份

证据优先级为：作者官方执行代码、resolved config/launcher/checkpoint、固定 commit 的本地镜像与
真实日志、论文正文/附录、第三方资料。先读 dataloader、forward/loss、optimizer、launcher 和
evaluator，再读 README。

记录 URL、commit、branch、dirty diff、tag/checkpoint，并将实现分为：

- `TRAINABLE`：包含真实数据、loss、optimizer/launcher、restore 和 evaluator；
- `PARTIAL`：只缺一到两项，只可作为模块参考；
- `INFERENCE_ONLY`：不作为训练预算依据。

必须读取 launcher 对 config 的最终覆盖；检查 `detach`、`no_grad`、`requires_grad` 和 optimizer
param groups。论文与官方代码冲突时，以代码作为复现实验基准，把论文版本作为独立消融。

具体方法边界见 [references/method-routing.md](references/method-routing.md)。

## 3. 逐字段对齐

对官方代码、resolved config、发布 checkpoint、本地启动命令和论文逐项核对：

- 架构：backbone、token、mask、融合层、RoPE、动作头、trainable/frozen 参数；
- 数据：suite/task/episode、camera、窗口、frame/action stride、horizon、padding、split、normalization；
- objective：flow/timestep、video/action/aux loss 及权重；
- 优化：global batch、samples、epochs、optimizer、分组 LR、warmup、scheduler、steps；
- 评测：环境版本、max/replan、denoise、seed、action clip、gripper、trials 和 success predicate。

状态只能是 `EXACT`、`EQUIVALENT`、`INTENTIONAL_DEVIATION`、`MISMATCH`、`UNKNOWN`。
等价和有意偏离必须给测试、父基线、假设和独立消融；所有字段都要有证据定位。

## 4. 数据与动作合同

长训练前输出 episode/task/suite 数、原始帧数、有效 window 数、任务分布、完整 window indices、
episode-disjoint split、manifest hash、动作/状态每维单位与统计、delta/absolute 与 gripper round-trip，
并用专家 demo replay 验证接口。任一高风险项未知，不进入长训练。

## 5. 架构与梯度

最小测试必须证明 shape/mask/RoPE、action loss 和 video loss 的真实梯度路径、冻结边界、finite
gradient 以及 checkpoint 恢复一致性。冻结 backbone 且无 world loss 的实验应命名为
`action-only-on-frozen-features`。DoT 移植还必须证明 all-layer K/V 与 RoPE realignment。

## 6. 分级放行

从 [references/dossier-template.json](references/dossier-template.json) 复制 dossier，运行：

```bash
python skills/wam-evidence-gated-training/scripts/validate_dossier.py DOSSIER.json --target canary
python skills/wam-evidence-gated-training/scripts/validate_dossier.py DOSSIER.json --target long
```

- `GO_CANARY`：合同和梯度门全部通过，只允许 1–100 step smoke/canary；
- `GO_LONG`：另需 TRAINABLE 官方实现、restore、可信父基线、机制指标和固定闭环 canary；
- `NO_GO`：存在 UNKNOWN/MISMATCH、多变量无消融或父基线/restore/闭环未通过。

离线 MSE 下降不是机制指标。语言路线看指令反事实，motion 路线看真实 motion loss 和梯度，闭环
路线看 success predicate。

## 7. 昂贵算力调度

保留至少一个长线基线，其余节点并行验证不同机制。评测及时消费 checkpoint；训练预算同时写
样本数和 effective epochs；到期前一天停止新长任务并固化 checkpoint、commit、config、manifest
与 eval JSON；永久记录 NO_GO。

每次输出必须包含：可证伪假设、来源身份、差异矩阵、global batch/样本/epoch/墙钟、父基线、唯一
变量、晋级/停止门槛、GO/NO-GO 和真实启动命令（或明确未放行）。
