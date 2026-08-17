# C63 FACT Stage-2 同状态动作排序结果

日期：2026-08-17。正式状态：
`FAIL_C63_STAGE2_WITHIN_STATE_DIAGNOSTIC / NO_GO_C60_STAGE2_RANKING_KEEP_C58`。

## 固定身份

- C60 s10000 checkpoint SHA256：
  `d6659c6b387f062a99f670a1d902b56df71a6bf1472aa4e46e56c9213ba75a36`。
- 只读前向源码 snapshot：
  `/mnt/h3-wam/runtime-snapshots/c63-fact-stage2-bbf799a-lean-ro`，对应代码 commit `bbf799a`。
- 只读结果 root：
  `/mnt/h3-wam/outputs/c63-fact-stage2-within-state-bbf799a-v3`。
- 固定 32-pair manifest SHA256：
  `c3ead728d362d53ad57f02508cb66febfa128dad79e95911d0da89cd40d8c25a`。
- 最终 `RESULTS.json` SHA256：
  `108608743603fca6631fb636f30f2c5e9b95f9912cbbfb215dfbdd8397638deb`。
- 最终聚合器 commit：`4777cc9`；聚合器只修复 failed shard 也应形成结构化 FAIL，未重新运行
  模型前向、未改变任何统计门。

## 结果

- 成功父动作被 C60 Stage-2 value 选中 `27/32`，one-sided exact binomial
  `p=5.6537101e-05`。
- `failure_score-success_score` median `+0.126953125`、mean `+0.322784424`；正值表示失败
  action 的预测 cost 更高。
- spatial 为 `25/30`，object 为 `2/2`；但 object 只有两对，不能据此声称 object 泛化。
- 32/32 score finite；32/32 候选顺序交换复测严格相同，最大差 `0.0`。
- 31/32 对产生非零 action-conditioned value 差异；唯一 tie 为 pair18，
  `libero_spatial/task4/trial32`，成功与失败 raw score 都是 `0.25`（normalized 都是 `-0.75`）。
- 另外四个未选成功动作的 spatial pair，其失败减成功 margin 分别为
  `-0.015625/-0.0234375/-0.015625/-0.009765625`。

独立脚本直接重读八份 shard 后得到相同的 `27 win / 1 tie / 4 loss`、p 值、median、mean 和 suite
计数。八份 shard 中只有含 tie 的 shard2 正确标为 `FAIL_C63_STAGE2_SHARD_MECHANICS`；聚合器没有
丢弃这个失败，而是把它纳入最终门判定。

## 为什么统计信号很强仍正式 FAIL

预注册 mechanical gate 要求 32/32 对在候选 action 改变时 value score 都 finite 且 nonzero；
pair18 在 BF16 最终 value 上精确打平。因此：

- primary `>=22/32`：PASS；
- exact-binomial `p<=0.05`：PASS；
- median margin `>0`：PASS；
- spatial `>=20/30`：PASS；
- object sanity `>=1/2`：PASS；
- **all action-conditioned value deltas nonzero：FAIL**。

不能在看到 `27/32` 后把 all-nonzero 改成 `31/32`，也不能把 tie 当半胜或丢掉。因此 frozen
permission 必须是 `NO_GO_C60_STAGE2_RANKING_KEEP_C58`；不启动 C60 20k，不直接做 BoN 闭环，
不改变 C60 680 对失败结论。

## 研究含义

结果仍提供了一条重要但受限的诊断证据：C60 内置 Stage-2 value 并非完全忽略 action；在固定的
同状态两选一问题上，它大多数时候能把成功父动作排在失败 intervention 前面。这支持“此前 C60
部署没有直接消费世界预测/value，是关键 deployment gap”的因果解释，但**不等于**它能从模型自己
采样的多个相近动作中稳定选优，更不等于 LIBERO 闭环提升。

下一步如果重开该路线，必须创建新的、事前冻结的实验，而不是修改 C63：使用四 suite 均衡、与本次
32 对 disjoint 的同状态候选，明确 tie 处理、数值精度和候选多样性门，再判断能否进入 N=1 vs N=4
闭环。当前 C63 自身已经终止，C58 继续作为正式 parent。

