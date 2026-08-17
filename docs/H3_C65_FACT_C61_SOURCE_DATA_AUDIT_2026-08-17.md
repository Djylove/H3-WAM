# C65 FACT 四 suite 同状态排序：C61 来源与数据审计

日期：2026-08-17。结论：`FAIL_C65_C61_FOUR_SUITE_DATA_GATE_NO_SCORE`。

C65 是全新候选，不修改 C63。C63 的 `27胜/1平/4负`、单侧精确二项
`p=5.65e-05` 是很强但未放行的诊断信号；它仍因事前冻结的“32 对全部非零”门在一个 BF16
exact tie 上失败。C65 要回答的是更严格的问题：在四 suite 均衡、来源独立、接近实际部署候选分布的
同状态成功/失败动作上，这个 Stage-2 value 是否稳定排序正确。

## 只读审计结果

审计器逐条读取 C61 冻结的 1128 个结果和轨迹，没有模型前向、训练或 BoN。结构身份全部通过：

- 1128/1128 的 branch 起点与父轨迹在 `sim_state`、`previous_action`、`step` 上 byte-exact；
- 每个状态恰有四个不同的首个 `32×7` 动作候选；
- 四候选仅改变首个 diffusion seed，组内 continuation noise 相同；
- 741 个分支成功、387 个分支失败；
- 全部分支由同一 D0-H32-s14000 checkpoint 生成，SHA256 为
  `36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3`；
- 与 C63 的 parent trajectory、branch trajectory、sim state、candidate action 以及
  suite/task/trial/step 重叠均为 0。

严格可用 pair 必须来自同一恢复状态的候选集合，并且候选采用完全相同的执行合同。库存如下：

| suite | 四候选状态 | mixed 状态 | 独立 mixed 父 source | success×failure 组合数 |
|---|---:|---:|---:|---:|
| Spatial | 98 | 26 | 21 | 86 |
| Object | 126 | 18 | 15 | 62 |
| Goal | 40 | 4 | 2 | 13 |
| LIBERO-10 | 18 | 0 | 0 | 0 |

因此四 suite 的严格均衡容量为 **0**，不是 48，也不是 161。48 是 mixed state 总数，161 是相关的
组合 pair 数；它们不能绕过 LIBERO-10 的 0，也不能把同一 state/source 的相关候选当成独立二项试验。

完整机器可读证据为
`experiments/evidence/h3_c65_c61_same_state_source_data_audit_v1.json`，SHA256
`87ab2752b84192b0be6234eec1511b647631b6d0849e3940504cdbbbdf008e8c`。

## 为什么不用“成功父动作 vs 失败候选”凑数

C61 成功父轨迹采用 replan8；干预分支的首动作却固定执行 32 步，之后才回到 replan8。直接把二者当成
同合同 pair，会把动作差异与执行 horizon 差异混在一起。即使接受这个不对称，出现过失败的独立父
source 也只有 Spatial 47、Object 16、Goal 9、LIBERO-10 2，仍不具备四 suite 确认性检验的能力。

另一个 gap 是候选分布：C61 来自 D0，而要部署的是 C60/C58 lineage。C61 可以说明 D0 附近的动作
敏感性，但不是当前 selector 真正会看到的候选分布。

## 冻结的补采合同

补采必须从 C60 s10000（SHA256 `d6659c6b...75a36`）的 fresh 成功轨迹建立不可变 source pool，排除
C61/C63 的 trajectory/state/action 身份。每个恢复状态事前冻结八个 C60 diffusion seed；首段和后续均
采用真实部署的 replan8，组内 continuation seed schedule 相同。全部分支完成前不读取中间 outcome。

一个 source 最多进入一对：若八候选同时包含成功和失败，固定取最小成功 candidate id 与最小失败
candidate id；否则该 source 只计入 mixed coverage 分母，不进入排序。数据门是每 suite 至少 20 个独立
mixed source，并固定为每 suite 20 对，共 80 对。任一 suite 不足即发布缺口并停止，不跑 score。

两个空闲节点按 suite 分片，写入全新的 `/mnt/h3-wam/eval/c65-*` 根；不得读取或写入 n0/n3 正在执行的
full50 输出根。

## 数据门通过后的事前 score 门

- 固定 C60 s10000，不训练；使用官方 FACT 唯一 `value[:,0,0]`、`raw=normalized+1`、argmin；
- BF16 部署精度不变；两个 finite raw scalar exact equal 定义为 tie/abstention，不设 epsilon、不以
  FP32 补救、不随机破平；
- overall 非 tie coverage 至少 `76/80=95%`，每 suite 至少 `19/20=95%`；
- 非 tie 中成功动作整体 preference 至少 65%，且 one-sided exact binomial `p<=0.05`；
- 每 suite 非 tie preference 至少 60%；80 对（tie margin=0）的 failure-minus-success 中位数严格大于 0；
- 相同当前 RGB/proprio/text、相同 Stage-2 初始噪声、反转候选顺序 max-abs=0、全部 score finite；
  future、terminal 与 outcome label 不得进入模型输入。

通过只允许建立另一个事前注册的 N=1 vs N=4 闭环实验，不能直接部署 BoN；失败则继续保留 C58。
