# H3-WAM 实验资产账本

更新时间：2026-08-12（Asia/Shanghai）

本文档保存 H3-WAM 已完成和正在运行的关键尝试，避免云资源结束后只剩模型文件而无法解释。
历史 `M*` 名称曾被复用，因此这里增加稳定的 `E*` 编号。所有成功率均指真实 LIBERO
闭环 success predicate；物体移动、离线 MSE 或训练 loss 不能替代成功率。

## 当前结论

- 已证明 H3 表征可以训练出显著下降的动作离线损失，但尚未证明通用 H3-WAM 闭环策略成立。
- dense 数据修正显著改善离线误差，M13 step400 达到 `0.122312`，但固定四任务仍为 `0/4`。
- 冻结 H3 的 head-only 路线语言区分度很弱，correct/wrong instruction 动作余弦约 `0.994`。
- controller replan/action-scale 扫描没有修复失败，主要矛盾已经从“部署参数”转到目标绑定、
  接触阶段学习和 offline-to-online 分布偏移。
- DreamWAM 的极短 motion canary 只是未对齐的探索，不能据此否定其官方完整训练方法。

## 稳定实验记录

| ID | 历史名称/路线 | 数据与训练变量 | 最好离线证据 | 固定闭环证据 | 结论 |
|---|---|---|---:|---:|---|
| E00 | H3 BF16 本地可训练性 | H3 tail，500 steps | val `1.16094 → 0.94047` | 非统一泛化协议 | 只证明 H3 局部微调链路可用 |
| E01 | DreamWAM M1 task3 | 单任务、tail-4、time-conditioned | held-out 改善 | `0 success` | 单任务拟合不晋级 |
| E02 | H3-DoT v2 head-only | sparse multi-task，150 steps | val40 `0.2104` | task0 `0/1` | 冻结表征不足 |
| E03 | full50 小试 | 解冻更多层，10 steps | val约 `0.2094` | `0/1` | 预算过小且无闭环证据 |
| E04 | sparse full training | 40 tasks sparse windows | step300 val850 `0.25234` | Goal `0/20`，cross-suite `0/32` | 数据抽样方案失败 |
| E05 | DreamWAM motion | RAFT motion，修正初始化后60 steps | flow约 `1.7995` | `0 success` | 仅可作重新对齐后的 canary 参考 |
| E06 | dense sampling correction | 40 tasks、逐帧 windows | canary `0.240706 → 0.212337` | `0/4` | dense sampling 有效但不充分 |
| E07 | M13 dense long | 200,779 train windows，global batch128 | step200 `0.140886`；step400 `0.122312` | 各自固定任务 `0/4` | 保留为长线基线 |
| E08 | M11 full frame-indexed long | 277,713 windows，global batch128 | step200 `0.151389` | task3 `0/10` | 保留至预定终点 |
| E09 | M14 tail-2 | M13 step400 父模型，40 steps | `0.119368`，比父模型约好2.4% | task3 `0/3` | 小幅离线增益不足以晋级 |
| E10 | controller sweep | replan 1/2/5、action scale0.5 | 不适用 | 每项 `0/1` | 停止调部署超参 |

## 2026-08-12 活跃长线

| 线路 | 节点 | 训练规模 | 最后观测进度 | 保留原因 |
|---|---|---|---:|---|
| M13 dense | `117.50.181.177:30907` | 1569 steps / 1 epoch | step783 | 回答更长训练是否能突破闭环零成功 |
| M11 frame-indexed | `117.50.181.177:30234` | 2170 steps / 1 epoch | step454 | 与 dense-uniform 采样形成长期对照 |

进度是快照而不是实时状态。恢复研究时必须先从日志和 checkpoint manifest 重新确认。

## 云端资产位置

共享根目录为 `/mnt/h3-wam`，主要资产包括：

- 模型：`/mnt/h3-wam/models/MiniMax-H3`、`RAFT`、`fastwam_release`；
- 数据/缓存：`downloads`、`libero_fastwam_extracted`、`v2_full_cache`、
  `v7_dense_*`、`v8_frameindexed_*`；
- 输出：`outputs/h3dotwam*` 与对应 rollout/eval JSON；
- 当前 checkpoint：M13 至少有 step200/400/600，M11 至少有 step200/400。

这些大文件不进入 Git。Git 保存配置、manifest、评测 JSON、锁定 commit 和恢复说明；选出的最终
checkpoint 应另存对象存储，并记录 hash。任何删除前先保留 parent、best、latest 三类 checkpoint。

## 数据采样教训

最早每个 episode 仅抽约 5 个 window，远少于官方逐帧窗口。修正后：

- v7 dense：200,779 train windows、1542 episodes、40 tasks；
- v8 full frame-indexed：277,713 windows；
- 每个 window 必须保存完整 frame/action indices、padding 和 stride；
- split 必须 episode-disjoint，并锁定 manifest hash。

“多训练几步”只有在样本数、global batch 和 effective epochs 可计算时才有意义。

## 下一次恢复顺序

1. 读取本账本和 `UPSTREAM_SOURCES.lock.json`，恢复固定上游 commit。
2. 核验 M11/M13 的最新 checkpoint、日志、resolved config 与数据 manifest hash。
3. 先完成固定闭环评测，不以离线 MSE 自动晋级。
4. 新主线按 `H3_WAM_CODE_FIRST_AUDIT_2026-08-12.md`，优先移植完整可训练代码路径。
5. 每个新实验只改一个变量，并在启动前记录父基线、预算、晋级门槛和停止条件。
