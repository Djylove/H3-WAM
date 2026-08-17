# C66-k1 严格单变量机制 Canary

## 决策

只运行一次从 C58 fresh parent 初始化的 100-step、8×A800 bounded canary。它回答的问题仅是：C66 全历史训练的退化是否主要来自一次提交七个历史 chunk，而不是 C66 block-internal persistent K/V 机制本身。

即使所有离线门通过，结果也只能标记 `MECHANISM_SIGNAL_ONLY_NO_LONG_OR_ROLLOUT`，不能自动启动 long training 或 LIBERO rollout。

## 证据起点

- C66 full-history s100 report SHA256 `55abab8d6f4e71d52941f84eea7725a4f83615a7a56c890446ba50439fc88c34`，结论 `FAIL_C66_PAIRED_CANARY / NO_GO_C66_LONG_TRAINING`。clean MSE 0.105681，shuffle 0.119431，off 0.079910；clean 能识别正确历史，但相对 off 退化 32.25%。
- context-length diagnostic SHA256 `50a726dd6bc69fa185c9c9bf17cac9ed138d9d8ef6a229b886d44af76c241237`，只读诊断选出的 trained best window 是 k1；它明确没有训练或 rollout 权限。
- C58 fresh parent SHA256 `2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541`；INT8 H3 SHA256 `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a`。

## 单变量矩阵

| 项目 | C66 full-history | C66-k1 | 是否变化 |
|---|---|---|---|
| 初始化 | C58 fresh | C58 fresh | 否 |
| 数据 | train800 / heldout64，episode-disjoint | 同一文件、同一 SHA | 否 |
| seed | 66017 | 66017 | 否 |
| world / steps | 8 ranks / 100 | 8 ranks / 100 | 否 |
| optimizer | AdamW, lr 1e-5, betas 0.9/0.95, wd 0.01, warmup 10 | 完全复用 | 否 |
| 模型/H3/目标 | C66 full30 + frozen INT8 H3 + flow matching | 完全复用 | 否 |
| source row | 7 chunks / 15 obs / 56 actions | 同一完整 source row | 否 |
| 实际 committed context | 7 chunks / 15 obs / 56 actions | 最近 1 个完整 chunk / 2 obs / 8 actions | **是，唯一变量** |

实现仍先走 C66 原始 full materialization，再裁剪 committed history，因此 H3 编码、数据访问和 target 生成没有另加变量。k1 observation 已按绝对 frame 13/14 reindex；state 从 `frame_st_id=13, action_st_id=48, next_update_id=12` 开始，commit 后保持 rollout 绝对坐标 15/56/14。

## Fixed heldout arms

- `clean-k1`：最近完整 chunk 的 observation、proprio、executed action。
- `history-action-shuffle-k1`：observation/proprio 不变，把紧邻前一个完整 chunk 的 8 个 executed actions 放入最近 chunk；它是确定性、同 episode、非 self-map 的控制。
- `context-off`：不启用 persistent context。

不能对单元素列表做 rotate，因为那会成为恒等映射，导致 shuffle 门失效。

## 门与边界

复用或收紧 C66 原门：800 个唯一 train 样本、64 heldout、episode-disjoint、30/30 block gradient、H3 无梯度、runtime restore exact、shuffle prediction effect、clean 至少优于 shuffle 1%、clean 相对 off 退化不超过 5%。另加 k1 完整 chunk、绝对坐标、distinct donor 和 no-long/no-rollout 门。

失败时停止；通过时只证明 k1 有 bounded offline mechanism signal。后续是否值得独立设计更大训练预算，必须另行审查，不继承本次权限。

## 执行路径

- 训练入口：`scripts/h3wam/train_c66_k1_bounded_mechanism_canary.py`
- 不可变源码/数据门：`scripts/h3wam/audit_c66_k1_single_variable.py`
- 8 卡启动器：`scripts/h3wam/launch_c66_k1_bounded_canary_8gpu.sh`
- 输出：`/mnt/h3-wam/outputs/c66-k1-bounded-mechanism/s100-fresh-v1`

启动器要求项目位于 `/mnt/h3-wam/code-snapshots/`、无 symlink、所有文件只读；审计固定 C66 trainer/model/freezer/diagnostic 源码和 plan/train/heldout/parent/H3/C66/diagnostic 的 SHA。输出目录已存在时拒绝覆盖。

## 正式结果与最终决策

正式 report SHA256 `70975e1b9de6612f6bdb65ff8d0bbeb9fdff3530b82e6b22cc4a7c781aba908a`，checkpoint SHA256 `861e95d891ca9128c2cb3bcc514243104fe70fb05c01fc9c0076d384a9201eeb`。固定 heldout64 的结果为：

- clean-k1 MSE `0.0821909424`；shuffle-k1 `0.0828969684`；context-off `0.0780162402`；
- clean 相对 shuffle 改善 `0.851691%`，未达到预注册 `>=1%`；
- clean 相对 context-off 退化 `5.351068%`，超过预注册 `<=5%`；
- 正式结论 `FAIL_C66_K1_BOUNDED_MECHANISM / NO_GO_C66_K1_LONG_OR_ROLLOUT / NOT_LIBERO_EVIDENCE`。

因此 k1 明显缩小了 full-history 的 `+32.249%` context harm，但仍没有通过两个 efficacy 门。不能把接近阈值解释成通过，不能追加 steps、改阈值或发起 rollout。

## `runtime_restore_exact` 假阴性诊断

正式 trainer 的精度作用域不一致：`train_c66_lingbot_c58_persistent_canary.py:476-483` 在 CUDA BF16 autocast 中产生 clean/shuffle/off，而 `:492-503` 离开 autocast 后才对 restored state 重算 prediction；因此 `:512` 比较的并非同一数值执行合同。

只读 v2 诊断从正式 checkpoint 回放相同 heldout64/noise，optimizer steps 和新 checkpoint 都为 0。RESULTS SHA256 `f2bc3344ec7dded536605d5bb935f4fdbfc821296e284b56df14c21ccc416019`：

- 64/64 snapshot 的所有 K/V、metadata bit-exact，最大差异 `0`；
- 64/64 k1 absolute coordinates 均为 frame/action/update `15/56/14`；
- 同为 autocast 内时 original-state 对 restored-state 最大差异 `0`，同为 autocast 外时也为 `0`；
- 同一 state 在各自精度作用域重复 forward 最大差异 `0`；
- 只有 formal 的“clean 在内、restore 在外”跨作用域比较非零：64/64 非零，全局最大 `0.0234375`（heldout index 44）。

分类收口为 `EVALUATION_AUTOCAST_SCOPE_MISMATCH_NOT_SERIALIZATION_OR_K1_PREFIX`：不是模型机制非确定性，不是 state 序列化缺陷，也不是 k1 absolute-prefix bug，而是 restore gate 的精度上下文假阴性。正式 artifact 和原 gate verdict 保持不改；即使排除该假阴性，两个独立 efficacy 门仍失败，所以最终 NO_GO 不变。
