# C56 FACT / C61 终点评测预注册

日期：2026-08-16。本文在 C56/C60 与 C56/C61 两臂完成 s10000 前冻结，训练中间结果不得修改门槛。

## 两个独立问题

1. `C60 main vs C58b parent`：完整 FACT 训练是否比相同 H3 父模型的 FastWAM 动作塔更有效。
2. `C61 matched vs C60 main`：只扩大 exact-state causal-failure 数据是否带来额外闭环收益。

禁止在同一批 40 条结果上先选 C60/C61 较高者、再把它与 C58b 比较并宣称胜出。两个问题分别报告，
避免事后选择偏差。

## 机械放行

- 两臂只能使用预注册 s10000；最后 1000 步 loss 有限、30/30 层梯度有限非零、future-to-action leak 为 0；
- checkpoint 文件身份、父 checkpoint 与 causal 数据 SHA 全部匹配，独立进程恢复 `max_abs=0`；
- 同一 balanced-80、同一噪声和 solver；两臂都通过 prediction std、gripper、语言替换和视觉 shuffle 门；
- 任一门失败都停止 LIBERO，不以训练 loss 或较早 checkpoint 替代。

## trial33 固定筛选

四个 LIBERO suite、每 suite 10 task、每 task 仅 trial33，共 40 个 paired episode/arm。执行合同与
C58b 正式对照逐项一致：`wait_steps=30`、`replan=8`、`horizon=32`、10 次 flow evaluation、无 ensemble、
normalized pre-clamp，episode/noise seed 为 `42 + task_id*100000 + trial*1000`。聚合器必须逐 episode
核验实际 replan seed 序列。

trial33 仅决定是否扩大评测，不承担统计显著的最终晋级结论：

- FACT port 放大条件：C60 overall successes 高于 C58b，且 C60 paired wins 多于 C58b wins；
- causal-data 放大条件：C61 overall successes 高于 C60，且 C61 paired wins 多于 C60 wins；
- 任一 suite 少于直接对照 3 个及以上成功（30 percentage points）视为安全失败；
- 未达到正向条件记为 `NO_GO_EXPANSION`；即使 trial33 的 exact McNemar `p<=0.05`，仍只记
  `GO_EXPANDED_PAIRED_EVAL`，不能写成 benchmark 泛化或擂主晋级。

## 完整晋级

通过 trial33 筛选的比较固定扩至 trials33..49，即四 suite × 10 task × 17 trials = 680 paired episodes/arm。
终局效果门与 C58b 扩展评测一致：

- overall success rate 至少 `+3 percentage points`；
- paired net wins 至少 `20`；
- 单侧 exact McNemar `p<=0.05`；
- 任一 suite 相对直接对照退化不超过 `3 percentage points`。

C61 若要成为 FACT 最终臂，必须同时通过 `C61 vs C60` 和 `C61 vs C58b`；C60 只需通过
`C60 vs C58b`。完整 680 对之前，报告状态只能是 micro-benchmark evidence。

## 资源与墙钟

- s10000 后在 n3 单卡依次完成两臂 balanced-80，预计 8–12 分钟；
- 机械门通过后，n3 八卡并行运行两臂 × 四 suite 的 trial33，预计 45–75 分钟；
- n2 训练结束后只作故障回退，不并行拆分同一 40 条合同；n0 保留 C58b 正式对照与扩展任务；
- trial33 终点评测总墙钟预计 1–1.5 小时。训练 READY 前 watcher 只能等待，不得创建效果结果。
