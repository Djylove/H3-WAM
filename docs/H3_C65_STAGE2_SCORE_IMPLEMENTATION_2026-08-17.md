# C65 fresh four-suite Stage-2 score implementation

日期：2026-08-17。当前许可：`GO_CANARY` for conditional read-only score；效果状态：
`NOT_EVIDENCE_READY`。

## 可证伪问题

在每个 LIBERO suite 恰好20个、共80个 fresh/source-independent/same-state C60 成功-失败动作对上，冻结
C60 Stage-2 value 是否能以至少95%非平局覆盖率、非平局中至少65%总体成功动作偏好、每 suite 至少60%
偏好、单侧精确二项 `p<=0.05` 和严格正的全80对 median margin 稳定排序？

## 代码与数据边界

- `finalize_c65_c60_pair_collection.py` 必须先完整审计3072个事前冻结 branch；任一 suite 少于20个独立
  mixed source 时只写 `NO_SCORE_DATA_COVERAGE_GAP`，不产生 `PAIRS.json`。
- `evaluate_c65_fact_stage2_pairs.py` 只读取 pair 中的 current trajectory 第0行 RGB/proprio、任务文本和
  candidate action。future observation、terminal state、outcome 和 success label 均不进入模型。
- scorer复用C63已执行的C60 restore、online INT8 H3 current K/V、十步 shift5 Stage-2 solver、共享初始
  noise和候选反序复测；使用唯一 `value[:,0,0]`，`raw=normalized+1`，低值胜。
- 精确BF16相等是tie/abstention，不做epsilon、FP32补救或随机破平。tie可存在，但总体最多4个、每 suite
  最多1个；其统计处理只发生在聚合器，不能让单个shard提前解释效果。

## 执行与结果门

- 8 GPU，各10对；optimizer steps=0、training samples=0，预计不超过0.5小时。
- 80/80身份、action hash、finite和candidate-order invariance必须通过。
- 总体non-tie `>=76/80`，每suite `>=19/20`；总体conditional preference `>=65%` 且精确二项
  `p<=0.05`；每suite conditional preference `>=60%`；全80对
  `median(failure_score-success_score)>0`。
- PASS只给`GO_SEPARATE_PREREGISTERED_N1_VS_N4_CLOSED_LOOP_ONLY`，不把C60晋级，也不改变C58擂主。

入口为 `launch_c65_fact_stage2_score_8gpu.sh`；`watch_and_launch_c65_fact_stage2_score.sh` 只在完整
`DATA_GATE.json` 明确给出 `GO_SCORE_C65` 且节点GPU空闲后接力。源码必须来自新的只读git snapshot，
不能使用共享live worktree。
