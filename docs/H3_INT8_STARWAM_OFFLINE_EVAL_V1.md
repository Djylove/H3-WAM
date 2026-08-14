# H3 INT8 + StarWAM ActionDiT R1 离线筛选器

日期：2026-08-14
状态：本地 CPU 合成数据验证完成；尚未对真实 checkpoint 出具指标；不是 LIBERO 闭环证据。

## 可证伪问题

在固定的 episode-disjoint 验证集上，schema2 checkpoint 是否能从当前观测对应的 H3 layer49 `last32` 缓存、语言和 proprio，以 StarWAM 官方 shift=5、10 步 Euler flow 推理产生非塌缩且更接近专家 chunk 的 causal action？

该工具只负责离线机制筛选。即使离线指标改善，也不能据此宣称 LIBERO 成功率改善。

## 来源身份

- 上游：`third_party/StarWAM`
- 固定 commit：`cd76d96f273f81e228a05f40f9697fe2514e2356`
- 主要审阅文件：
  - `starwam/wam/feature_conditioned_action_model.py`
  - `starwam/eval/policy.py`
  - `starwam/modules/scheduler.py`
  - `starwam/training/metrics.py`
  - `examples/libero/configs/recipes/starwam_libero_feature_conditioned_wan22_5b.yaml`
- evaluator 会把 commit，以及 ActionDiT、WanBlock、policy、feature-conditioned model、scheduler、官方 metrics 文件的 SHA256 写入结果 JSON。

## 使用方式

```bash
.venv/bin/python scripts/h3wam/evaluate_h3_int8_starwam_action.py \
  /path/to/checkpoint_step_1000.pt \
  --source-manifest /path/to/source.jsonl \
  --train-manifest /path/to/train.jsonl \
  --val-manifest /path/to/val.jsonl \
  --cache-root /path/to/cache \
  --output /path/to/eval_step_1000.json \
  --device cpu \
  --batch-size 1 \
  --samples-per-task 10 \
  --language-sensitivity
```

真实 checkpoint 筛选时，保持同一组 source/train/val manifest 和 seed，逐 checkpoint 输出独立 JSON。默认 `batch-size=1` 最接近上游逐样本评测行为。

`--samples-per-task N` 是严格的 LIBERO-40 balanced-val 模式：完整冻结 val manifest 必须恰好覆盖 40 个任务，每个任务至少 N 个 window；筛选器以固定 salt `h3-int8-starwam-balanced-val-v1` 和 sample ID 的 SHA256 排序，每任务取前 N 个。选样不依赖 manifest 行顺序，并且与 `--limit`、`--sample-offset` 互斥。报告的 `data.selection` 记录 salt、N、每任务计数和 selected ID hash；source/train/完整 val manifest 的 path、hash 和样本数保持不变。

`--limit` 只适合非 balanced 的快速小样本诊断；跨 checkpoint 完整比较必须使用同一固定样本集合。

## 严格输入合约

筛选器拒绝静默兼容，任一字段不一致即失败：

- checkpoint 顶层必须精确为 schema2：`schema_version`、`completed_steps`、`model`、`optimizer`、`lr_scheduler`、`contract`、`probe_prediction`、`probe_sample_ids`；
- contract 必须匹配固定 StarWAM commit/source hash、layer49、32 tokens、缓存策略、source/train manifest hash 和样本数、stats hash、min-max clip5 normalization、shift=5、模型结构、action horizon 和 feature subdir；
- source/train/val manifest 必须非空、ID 唯一，train/val 每行必须与 source 中对应行完全一致；train/val 同时要求 window-disjoint 和 episode-disjoint；
- balanced-val 必须在完整 val manifest 上验证 40 任务覆盖和每任务样本数，然后才进行确定性选样；不能与位置切片混用；
- 每个缓存必须是完成的 H3 INT8 layer49 `last32`：`[1,32,h3_feature_dim]`、finite、timestep=1.0，并匹配 context、horizon、backbone、quantization 和 source 样本数；
- context 必须是 text-only；action/state/stats 维度必须精确匹配 checkpoint；
- 模型必须在新实例中以 `strict=True` 恢复。筛选器随后删除该实例，再创建第二个新实例，用相同输入和噪声做 10 步推理，要求输出逐元素完全一致。

输出原子写入，报告必含 checkpoint/source/split/stats/selected-samples hash 与实际样本数。

## 指标

只统计非 padding 动作：

- action MSE/MAE，以及逐维 MSE/MAE；
- chunk ADE：每个有效 timestep 的动作向量 L2 误差均值；
- endpoint：每个样本最后一个有效 timestep 的动作向量 L2 误差均值；
- prediction mean/std（逐维和全局），用于发现动作塌缩；
- gripper sign accuracy、precision、recall、F1、macro-F1 和混淆计数；
- 可选语言替换敏感性：保持视觉、proprio 和初始高斯噪声不变，仅换成另一任务的文本 context，报告预测差异；
- 同时报告模型 normalization 域指标，以及按官方 `[-1,1]` clamp 后 min-max 反归一化的物理域指标。

语言差异大只证明模型对文本敏感，不证明它理解了正确语义；仍需配对指令机制评测和闭环验证。

## 与固定上游实现的差异矩阵

| 项目 | 级别 | 固定上游 | 当前筛选器 | 说明 |
|---|---|---|---|---|
| Action head | EXACT | StarWAM ActionDiT/WanBlock | 从固定 third-party commit 导入并校验 SHA | 未重写网络结构 |
| Flow scheduler | EXACT | `FlowMatchScheduler(shift=5)` | 同一上游类、同一 shift | 公式一致性有单测 |
| 推理积分 | EXACT | 高斯初始化，10 步，`sample += velocity * delta` | 同一调度与 Euler 更新 | 默认 batch=1 对齐逐样本评测 |
| 动作反归一化 | EXACT | 预测先 clamp 到 `[-1,1]` 后 min-max 反归一化 | 物理域指标使用同一行为 | normalization 域另行保留诊断值 |
| 条件顺序 | EXACT | text + proprio + visual feature | 本地 adapter 保持相同条件接口 | feature 来源见下一行 |
| 视觉特征 | INTENTIONAL_DEVIATION | 在线 Wan2.2 feature-conditioned model | 冻结的 H3 INT8 layer49 `last32` 缓存 | 实验类别是 backbone port，不是官方复现 |
| 训练/评测 split | INTENTIONAL_DEVIATION | 官方 recipe 的 `val_split: 0.0`，官方 batch evaluator 未强制 episode 隔离 | 强制 source/train/val identity 与 episode-disjoint | 防止 window 泄漏；属于本项目评测合同 |
| balanced-val selection | INTENTIONAL_DEVIATION | 无固定 40-task balanced selector | 固定 salt+sample ID hash，每任务严格 N 条 | 完整 val hash 不变；缺任务/缺样本直接失败 |
| checkpoint restore | INTENTIONAL_DEVIATION | 官方 policy loader 使用 non-strict restore | schema2 精确字段、contract 校验、`strict=True`、双 fresh restore 一致性 | 用于拒绝错误 checkpoint，不改变模型计算 |
| 官方离线指标 | EQUIVALENT | `evaluate_batch` 主要给 action MSE，逐样本调用 `infer_action` | 保留同域 MSE，并增加 MAE/ADE/endpoint/std/gripper | MSE 和 scheduler 有合成测试；新增指标不声称上游官方 |
| 语言替换评测 | INTENTIONAL_DEVIATION | 无 | 同噪声替换文本的敏感性诊断 | 只作为机制信号 |
| batch RNG | INTENTIONAL_DEVIATION | 上游逐样本推理 | 默认 batch=1；允许显式增大 batch | 跨 checkpoint 比较必须固定 batch、seed 和样本 hash |
| 闭环 success | UNKNOWN | 由 simulator rollout 给出 | 本工具不执行闭环 | 离线通过后仍须固定 LIBERO rollout |

## 验证记录

```text
.venv/bin/python -m unittest -v tests.test_h3_int8_starwam_action_evaluator
11 tests passed

.venv/bin/python -m unittest -v \
  tests.test_h3_int8_starwam_action_trainer \
  tests.test_h3_starwam_feature_action \
  tests.test_h3_episode_disjoint_manifests \
  tests.test_h3_int8_starwam_action_evaluator
26 tests passed
```

覆盖：shift5/10-step schedule、padding-aware 指标、gripper F1、schema2 双 fresh restore、严格 state load、episode 泄漏拒绝、manifest/cache/hash 合约、语言替换敏感性，以及 balanced-val 的确定性、40 任务覆盖、参数互斥和不足 N 拒绝。另已通过 `py_compile`、CLI `--help` 和 `git diff --check`。

## 证据门

- 训练许可：本文件不放行新长训；这里只新增只读离线评测能力。
- 当前效果结论：`NOT_EVIDENCE_READY`。
- 晋级条件：对固定 val manifest 的多个真实 checkpoint 运行本筛选器，排除动作塌缩并选出预注册候选；随后由同一 LIBERO 环境、任务、seed、trial 数和 action adapter 的闭环评测决定 success。
- 停止条件：contract/restore/split 任一失败，或多个 checkpoint 在相同样本上的动作指标持续恶化/塌缩时，不把更多训练步数当作默认修复。
