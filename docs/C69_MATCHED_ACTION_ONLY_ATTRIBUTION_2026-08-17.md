# C69 matched action-only attribution

## 结论边界

C69不是新的“动作补丁”，而是C67必须具备的严格同预算对照。它回答一个单一问题：C67后续若改善，
究竟来自额外20k动作优化，还是来自FACT future-representation/state/value监督。完成训练前，C69仅有
机械训练许可；C58仍是唯一闭环晋级的carrier champion。

## 固定与唯一变量

- 相同：C58 s10000父节点、完整30层H3-to-Action shared tower、P/A/G/V/I joint causal forward、
  在线冻结INT8 H3、所有数据hash、seed20260816、4 expert + 2 success + 1 observational failure +
  1 causal failure、失败action mask、AdamW base/action LR `2e-5/2e-4`、warmup500、cosine20k。
- 唯一变量：C67损失`[10,1,0.4,0.4]`变为C69的`[10,0,0,0]`；六个后果专属
  encoder/decoder冻结并从optimizer排除。
- 两个failure ranks继续执行相同样本和forward，但action mask为0。六个有动作监督的ranks按`8/6`
  缩放后经过DDP rank mean，逐步严格等于C67 action component；不得改成8个expert ranks。

## 已完成的机械证据

- 本地相关测试：23项PASS，包括全局masked action梯度等价、auxiliary-only冻结与long finalizer fail-close。
- 真实8×A800十步：10/10 finite，30/30共享block全局梯度为正，future target action leak为0。
- strict restore prediction max-abs为0。
- canary checkpoint：
  `/mnt/h3-wam/outputs/c69-matched-action-only-v1/canary10-v1/c69_action_only_s10.pt`，SHA256
  `af29173c780691f3f1a6f8d7efef1a49e24d349c2c938796a66d08e5865d4b07`。
- `GO_LONG.json` SHA256：`5d448cbf94bc9820a66d148e294f133b469fa5bb66913b389b5456376b4c89a5`。

## 正式运行

- source commit：`a60b056567cecfacc880606816881766657f934a`
- source tree：`5514af79eee87170672050c3cf5e9c7e47386798`
- read-only snapshot：`/mnt/h3-wam/code-snapshots/h3-wam-a60b056-c69-long-v1`
- SOURCE_FREEZE SHA256：`a9197001d9b545ba7542dccc864c104ac2ee99a6defbd4c50bb9acab9ef66d68`
- release：`/mnt/h3-wam/releases/c69-matched-action-only-20k-a60b056.json`
- output：`/mnt/h3-wam/outputs/c69-matched-action-only-v1/online-long20000-v1`
- 预算：8×A800、20000 optimizer steps、global batch8、160000 samples、每1000步checkpoint+strict restore。

## 后续判定

1. C67与C69都必须先完成全部20个checkpoint、restore和固定balanced80诊断。
2. 唯一主要比较是C67-s20000 joint对C69-s20000 action-only，不按train loss换里程碑。
3. 只有conditioning安全门通过才进入paired LIBERO；闭环门固定为`+3pp`、净胜至少20、one-sided exact
   McNemar `p<=0.05`、任一suite不低于`-3pp`。
4. C67胜出才说明FACT后果监督在当前H3合同下有增量价值；C69持平或胜出则停止该world objective，
   保留动作优先路线。两者都不能仅凭离线MSE替代C58。
