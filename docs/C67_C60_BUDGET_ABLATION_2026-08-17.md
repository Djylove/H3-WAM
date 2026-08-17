# C67：C60 训练预算受控实验

日期：2026-08-17。状态：`GO_LONG_BUT_MANUAL_RELEASE_REQUIRED / NOT_EVIDENCE_READY`。本文件只冻结实验，
不创建 release marker，也不启动 GPU。

## 要回答的问题

C60 在 10k 后没有晋级，究竟是“FACT/H3 方向无效”，还是“80000 个样本、aggregate 0.366761 epoch
不够”？历史 C60 自身不能直接续跑：其 10k cosine scheduler 已到 LR=0，改变 horizon 再加载会违反
checkpoint optimization contract。C67 因此从固定 C58 parent 重新起一条 horizon=20000 的轨迹，并在同一
轨迹冻结 s10000 control 与 s20000 treatment。

这不是两个不同配方：两端的源码、初始化、数据、采样顺序、seed、loss、optimizer、scheduler、动作合同和
评测均相同；s20 唯一多出的内容是 step10001..20000 及对应的80000个确定性样本。历史 C60-s10 仅是外部
anchor，不能替代 C67-s10，也不能与 C67-s20 构成“只差步数”的配对。

## 固定身份与预算

- 父模型：C58B-s10000 SHA256
  `2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541`；两端都从它开始。
- 外部锚点：C60-s10000 SHA256
  `d6659c6b387f062a99f670a1d902b56df71a6bf1472aa4e46e56c9213ba75a36`；只作 sanity check。
- 官方源码：FACT commit `618a6c16868699b6d4138941de6a863589ac00dd`；FastWAM commit
  `45d8e1458921d83f8ad6cf9ce993d371208dabd0`。
- 模型/数据/损失：冻结 INT8 H3 online、30层共享 FACT tower、4 expert + 2 success + 1 observational
  failure + 1 causal failure、loss `10:1:0.4:0.4`，完全复用 C60。
- 优化：8×A800、microbatch1、global batch8、AdamW base/action LR `2e-5/2e-4`、warmup500、cosine
  horizon20000、seed20260816。s10000 的 cosine factor 为 `0.5201329701`，base/action LR 约
  `1.04026594e-5/1.04026594e-4`；s20000 才到0。
- s10预算：80000 samples；aggregate `0.366761` epoch；expert/success/observational/causal 暴露
  `0.199224/8.107013/0.772201/5.184033`。
- s20预算：160000 samples；aggregate `0.733522` epoch；四流暴露翻倍为
  `0.398448/16.214026/1.544402/10.368066`。注意 episode-then-frame 采样使小 failure stream 多轮复用，
  不能把 aggregate epoch 当成各流都“没见完一轮”。
- 每1000步保存12.2GB full state并独立 strict restore，共20个 checkpoint；预计约244.04GB
  （227.28GiB），launch 前至少保留300GiB。训练稳态线性估计8.82h，另加 checkpoint I/O；s1k后重估 ETA。

## checkpoint 与完整性门

固定保存 s1k..s20k，绝不基于中间 loss/offline/rollout 早停或改选 endpoint。每一段必须满足：

1. 1000个连续 absolute step，所有 loss 有限，30/30 shared block gradient 有限且正，future-to-action
   leakage为0，frozen H3无梯度；
2. s2k起的每段必须从前一里程碑 full-state checkpoint 加载，`restore_at_load_max_abs=0`；
3. 每个落盘 checkpoint 另启8 rank restore-only，prediction `max_abs=0`；model、optimizer、scheduler、
   contract与step共同保存；
4. 20个 checkpoint 合同必须逐字段相同，尤其是 C58 parent、H3、全部 data SHA、seed和
   `scheduler_horizon=20000`；s10与s20 SHA独立发布；
5. 任何缺段、覆盖已有 output、source/release hash漂移或可用空间小于300GiB都 fail closed。

## offline 晋级门

在同一 episode-disjoint balanced80（40 task×2）、相同 sample/noise/seed42/shift5/10-step solver 上评测
20个 milestone，形成曲线但不据此改 endpoint。s20 进入闭环必须同时满足：

- 20/20 strict restore 和 conditioning gate 通过；
- s18–s20 的 mean normalized MSE 和 physical MSE 均至少优于 s10–s12 mean 1%；
- s20 相比 s10 的 normalized/physical MSE 均至少下降1%，且两种逐样本 error 的 s20 win rate 均
  `>=55%`（ties不计胜）；
- s20 gripper macro-F1 不低于 s10 超过0.005；language replacement delta 与无 self-map visual shuffle
  response均保留 s10 的至少90%；prediction std 有限且 conditioning不坍塌；
- C67-s10 同样完整报告，以便发现新20k scheduler在中点是否已偏离历史 C60。该 bridge 只限制外推：
  若 s10 与历史 C60 差异明显，仍可回答 C67内部 budget effect，但不得宣称旧 C60“只差步数”。

任一项失败，状态固定为 `NO_OFFLINE_EVIDENCE_FOR_MORE_STEPS`，不做闭环。

## paired LIBERO 门

offline 全过后，仅比较预注册的 C67-s20 treatment 与 C67-s10 control。沿用完整四 suite×10 tasks×
trials33..49=680 对、每 episode 新 simulator+policy process、full initial-state exact、wait30/max400、
replan8/horizon32/eval10及同一 task seed。禁止用历史 C60 rollout 替代 C67-s10 新执行结果。

“训练步数不足得到支持”必须同时满足：

- s20-s10 overall success `>=+3pp`；discordant net wins `>=20`；one-sided exact McNemar `p<=0.05`；
- 任一 suite 不比 s10 低超过3pp；680对完整且全部 initial-state/process gate通过；
- s20 总成功率不低于历史 C60 的 `313/680`，防止只赢一个被新 schedule 拉低的内部 control。

若 offline 过而 closed loop 未过，结论为
`MORE_STEPS_IMPROVE_OFFLINE_NOT_ACTION_EXECUTION`；若二者全过，结论只能写
`EVIDENCE_READY_BUDGET_ABLATION_ONLY`。由于 trials33..49 已被历史实验使用，它不是新的 benchmark
champion证据，不改变 `KEEP_C58_PARENT`，也不允许把旧 C60 680对失败事后改写为成功。若希望晋级谱系，
必须另取未使用初态或外部 benchmark 做一次独立确认。

## 停止规则

- s10和s20以外 checkpoint只画预注册曲线，不做闭环选择；
- 训练 loss下降不是效果证据；预测能力上升但动作成功率未过门即判未解决；
- 若 s20未过，结论只覆盖“当前 C58初始化、C60数据/目标和20k cosine合同”，不证明任何更长训练永远无效；
- 本仓库不自动生成 `C67_RELEASE_FILE`。只有独立审查把 dossier、trainer、launcher、C58 READY、C58
  checkpoint和 output root 的 hash/路径写入手工 release JSON 后，launcher才可能通过。
