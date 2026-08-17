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

## 异步里程碑评测

- 只读评测实现已固定在commit `55b622fec64cf3fac7ddd9a41524f4ed72490865`，快照为
  `/mnt/h3-wam/code-snapshots/h3-wam-55b622f-c69-preview-v1`，SOURCE_FREEZE SHA256为
  `9e6cae8f01af159b9214428815ca4e226272044f2c739594916b9a26fb24ca78`。它不包含trainer调用，
  只在checkpoint、train report和strict restore三者齐备后读取；20个点全部完成前不聚合、不选点。
- C69-s1000 checkpoint SHA256为
  `7dd5ddcd6d755fff5ce24d266b52868c66994344e9e1afafd5c675079ce9922f`，机械审计和固定balanced80
  conditioning四门全部PASS。normalized/physical MSE为`0.058800/0.025693`，gripper macro-F1
  `0.939408`，prediction std `0.477610`，language delta `0.224368`，visual-shuffle MSE
  `0.034714`。同一步C67为`0.058345/0.025330/0.940178`；这只是第一点描述，不能据此判定FACT增益、
  提前停止C69或选择任一checkpoint。
- 评测队列在30907上运行，后续每个1000步点自动使用同一80 IDs、seed42、10-step solver和在线冻结
  INT8 H3；报告权限保持`PREVIEW_ONLY_PENDING_TRAINING_COMPLETE_REBIND`。

截至C67-s17000的异步preview，17/17 conditioning gates均通过，但曲线并未支持“训练越久越好”：
s10000的normalized/physical/macro-F1为`0.059716/0.025007/0.934451`，s16000为
`0.060807/0.025483/0.931187`。这些preview按预注册只用于缩短最终等待，不得提前停训或挑s4k；它们
说明C68 30k不能因“卡空闲”自动放行，必须等待固定s20000终点和C69归因结果。

最终零重评rebind与固定C67-s20000/C69-s20000归因合同见
`docs/C67_C69_FIXED_S20_ATTRIBUTION_GATE_2026-08-17.md`。该链路只允许预注册的objective、loss和六类
auxiliary freeze差异；离线输出不得选点或声明winner，双方20/20 conditioning安全后才可放行独立paired
LIBERO归因评测。
