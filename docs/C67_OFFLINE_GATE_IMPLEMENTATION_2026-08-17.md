# C67 离线效果门实现

日期：2026-08-17。该实现只负责 C67 的预注册离线门，不启动训练或 LIBERO rollout，也不读取任何
rollout 结果。训练必须先完整生成 `TRAINING_COMPLETE.json` 和 s1k..s20k 全部 checkpoint；不允许用
中途离线结果改变训练、停止规则或 endpoint。

## 固定执行链

- 单点评测：`scripts/h3wam/evaluate_c67_fact_milestone_balanced80.py`；只接受 C67 horizon20000
  checkpoint、C67 milestone restore audit、完整的 training-complete marker 和固定 C60 数据 SHA。
- 八卡队列：`scripts/h3wam/launch_c67_fact_milestone_balanced80_queue.sh`；固定评测 s1k..s20k 共20点，
  每点使用同一 balanced80、seed42、shift5、10-step solver、H3、normalization 和 noise。
- 固定聚合：`scripts/h3wam/aggregate_c67_fact_milestone_balanced80.py`；读取20份报告后一次性产生
  `${C67_OFFLINE_ROOT}/RESULTS.json`，禁止从曲线中另选 checkpoint。

正式调用方式：

```bash
PROJECT_ROOT=/mnt/h3-wam/code-snapshots/REVIEWED_C67_READONLY \
C67_TRAIN_ROOT=/mnt/h3-wam/outputs/c67-c60-budget-ablation-v1/online-long20000-v1 \
C67_OFFLINE_ROOT=/mnt/h3-wam/outputs/c67-c60-budget-ablation-v1/balanced80-s1k-s20k-v1 \
bash scripts/h3wam/launch_c67_fact_milestone_balanced80_queue.sh
```

## 聚合门与输出合同

聚合器严格执行原预注册门：20/20 training restore、fresh restore、conditioning 和 finite metrics；
s18–s20 的 normalized/physical MSE 均值至少比 s10–s12 低1%；s20 的两种 MSE 均至少比 s10
低1%；s20 对 s10 的80个逐样本误差中，两种指标的胜率均至少55%（tie留在80个分母内且不算胜）；
gripper macro-F1 最多下降0.005；language/visual response 各至少保留90%。

全部通过时输出：

- `format=h3wam-c67-budget-balanced80-result-v1`
- `status=PASS_C67_BUDGET_BALANCED80_GATE`
- `permission=GO_C67_PAIRED_680_ROLLOUT`

任何一门失败时固定输出 `FAIL_C67_BUDGET_BALANCED80_GATE / NO_C67_PAIRED_680_ROLLOUT`。结果显式
发布 matched control s10000 与 treatment s20000 的 checkpoint/restore SHA，供后续 rollout 授权器严格
对接。即使全部通过，也只表示可运行预注册680对闭环，不构成动作成功率证据。
