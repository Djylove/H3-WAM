# Candidate D DreamWAM K/V carrier：balanced-80 同协议 adapter

日期：2026-08-14
状态：本地代码与合成测试完成；30907 已产生并严格恢复一个仅 1-step 的 schema-v2
Candidate D 机械 checkpoint；尚未运行 balanced-80、未启动效果训练。

## 结论

Candidate D 可以在不修改现有 StarWAM evaluator 行为的前提下，接入相同的 balanced-80 **选择规则、任务配额、扰动、solver 和指标实现**。实现位于 `scripts/h3wam/evaluate_h3_dreamwam_kv_carrier.py`：它把一条样本的 5 层 H3 K/V 打包成一个视觉 carrier，通过独立 wrapper 解包给 DreamWAM action policy。

但当前 Candidate D v4 与 R1/G v8 来自不同冻结 manifest：v4 的 80 个 IDs hash 为 `b507e1ff6031f01c88cd6181aaeb4cba33b76e2c67737a986bf764c76be87519`，v8 为 `75d888fbb4298bef3517b623c00861ac6fe036495dee3bf4f0c68b5c097c5f54`。因此当前 adapter 能做 Candidate D 自身机制筛选，**不能把其结果称为与 R1/G 的严格 paired 父对照**。

30907 的外部只读审计已确认 v4 selector 得到 80 条、40 个任务、K/V missing=0，选中 cache 总字节 `367,427,920`。这些值是外部审计证据，不是本地 adapter 在本任务中重算的结果。

这只解决“可比评测接线”，不构成 Candidate D 有效性证据。

## 锁死的同协议项

- source/train/val manifest 身份、episode-disjoint 检查：复用 StarWAM evaluator。
- 选择规则/配额：固定 salt `h3-int8-starwam-balanced-val-v1`，LIBERO-40 每任务 2 条，共 80 条。
- IDs gate：默认硬锁已审计 Candidate D v4 hash `b507e1ff...87519`；`--expected-selected-ids-sha256` 只用于未来显式迁移至另一个已审计 manifest，实际选择结果不匹配即失败。
- visual shuffle：固定 salt `h3-r1-visual-shuffle-v1`；右移置换、无 self-map。
- visual shuffle 原子单位：一条 source 样本的完整 5 层 K/V bundle；不在样本内打乱层或 K/V。
- seed：42。
- batch size：1。历史噪声按 `seed + 1_000_003 * batch_index` 生成，改 batch size 会改变逐样本噪声，因此 adapter 拒绝其他值。
- 推理：固定 StarWAM `FlowMatchScheduler(shift=5)`，10 步 Euler。
- baseline、language replacement、visual shuffle：同一初始 action noise。
- normalization：训练/normalized 指标保留 `starwam_minmax_clip5` 域；physical 指标复用官方先 clamp `[-1,1]` 再 min-max 反归一化。
- preclip：normalized 指标在 physical clamp 之前计算；输出报告分别命名两个域，不能混读。
- gripper：最后一个 action 维度，normalized sign 阈值为 0。
- language replacement：复用同一 context-ID replacement 和 padding/mask 路径。
- checkpoint：schema、source hash、train hash、stats hash、DreamWAM 固定 commit/源码 SHA、K/V schema/层/形状、shift 和 model spec 均严格匹配；模型用两个独立实例 `strict=True` 恢复并要求同噪声预测逐元素一致。
- H3 权重身份：Candidate D checkpoint contract 必须包含固定字段 `h3_checkpoint_sha256=e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a`，同时 `verify_h3_checkpoint_sha256=true`；adapter 拒绝缺失、其他值或仅声明但未校验的 checkpoint，并在 report 的 `checkpoint` 与 `protocol_identity` 中显式记录。每条 cache 仍核验其 `checkpoint` path 与 contract 完全相同。
- cache aggregate hash：由独立全量 cache audit 计算和绑定，不由只读取 balanced-80 选中项的 evaluator 冒充全量审计。可通过 `--cache-audit-aggregate-sha256` 将外部 audit 的 lowercase SHA256 写进 report；adapter 只验证格式并标注 `external_audit_declaration_not_recomputed_by_selected-cache-evaluator`。

## 与 StarWAM balanced-80 的差异矩阵

| 项目 | StarWAM R1 | Candidate D adapter | 分类 |
|---|---|---|---|
| 选择算法/任务配额、seed、batch、shift、steps | 40×2、42、1、5、10 | 完全复用且硬拒绝漂移 | EXACT |
| 当前 selected IDs | v8 `75d888...5f54` | v4 `b507e1...87519` | MISMATCH |
| action/state normalization、physical clamp、gripper | evaluator 原实现 | 直接调用相同函数/accumulator | EXACT |
| language/visual 扰动与噪声 | 同噪声 | 直接调用相同 sampler | EXACT |
| 视觉 carrier | layer49 last32 `[1,32,5376]` | 5 层 K/V `[5,2,32,56,128]` | INTENTIONAL_DEVIATION |
| visual shuffle 单位 | 单个 last32 feature tensor | 完整 5 层 K/V bundle | EQUIVALENT_INTERVENTION |
| action head | StarWAM 30-block ActionDiT | DreamWAM 5-block carrier ActionDiT | INTENTIONAL_DEVIATION |
| 载入代码 | StarWAM evaluator 原生 | 独立 wrapper；不改原文件 | INTENTIONAL_DEVIATION |
| 闭环 success | evaluator 不提供 | evaluator 不提供 | UNKNOWN |

## 待解项与证据门

1. 当前只有 checkpoint SHA256 `daa58403d5501efc003c1f5d1c297e308ba3e51267eca86e7c1df7c908224d39`
   的单样本、单步机械 checkpoint；fresh restore `max_abs=0`。它可用于 evaluator 机械验证，不能作为
   “完成训练”的模型或效果证据，也不能与 StarWAM 数字作方法归因。
2. 当前 trainer 正式训练必须带 `--verify-h3-checkpoint-sha256`；旧的、不含字段或 verification flag 为 false 的 Candidate D checkpoint 会被 adapter 有意拒绝，不能静默补值。
3. 30907 外部审计已确认真实 80 个选中 ID 的 K/V cache `missing=0`；运行 evaluator 时仍须逐条通过
   BF16、shape、finite、timestep=1、无 alias 和 source provenance 检查，不能只引用文件存在性。
4. aggregate cache SHA 只能由对应 cache root 的外部全量 audit artifact 绑定；若未向 evaluator 提供，report 中为 `null`，不代表 cache 未通过或已通过全量审计。
5. 当前 v4/v8 IDs 不一致，不能做严格 paired attribution。必须二选一：把 Candidate D 数据/cache 迁移到 v8 并锁 `75d888...5f54`，或在 v4 的 `b507e1...87519` IDs 上重跑直接父模型。
6. 离线 normalized error 改善仍不等价于 action 可执行，更不等价于 LIBERO success。只有同一环境、任务、seed、trial 数和 action adapter 的闭环 rollout 能晋级。
7. 如果 Candidate D 离线预测变强但 gripper 或闭环退化，应先检查 carrier intervention、动作维度指标和 rollout，不把继续堆训练步数作为默认修复。

放行顺序：保存 Candidate D checkpoint → 固定 v4 balanced-80 adapter 做自身筛选 → 迁移 Candidate D 到 v8 matched IDs，或在 v4 同 IDs 重跑直接父模型 → 只有 paired IDs 一致后才作方法归因 → 仅将非塌缩且动作/gripper 指标有改善的 checkpoint 送入相同 LIBERO 闭环。

## 本地验证

```text
.venv/bin/python -m unittest -v tests.test_h3_dreamwam_kv_evaluator
5 tests passed
```

覆盖：协议硬约束、checkpoint 合同拒错、canonical normalization/data path、完整 K/V bundle shuffle，以及 wrapper 解包后的 layer-specific/no-alias 语义。
