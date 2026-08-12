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
- H3/ActionDiT 的 action→future-video 反向梯度链路已在真实 33B H3、8×A800 上跑通；但
  严格配对的 100-step 消融仅带来 `0.0142%` held-out action-loss 改善，低于实验噪声量级，
  因此停止 gate-only 长训，转向 LingBot-VA 的完整 block-causal 双流接口。

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
| E07 | M13 dense long | 200,779 train windows，global batch128 | step200 `0.140886`；step400 `0.122312`；step800 `0.114559` | 各 checkpoint 固定四任务均 `0/4` | 保留为长线基线；离线继续改善但闭环未突破 |
| E08 | M11 full frame-indexed long | 277,713 windows，global batch128 | step200 `0.151389` | task3 `0/10` | 保留至预定终点 |
| E09 | M14 tail-2 | M13 step400 父模型，40 steps | `0.119368`，比父模型约好2.4% | task3 `0/3` | 小幅离线增益不足以晋级 |
| E10 | controller sweep | replan 1/2/5、action scale0.5 | 不适用 | 每项 `0/1` | 停止调部署超参 |
| E11 | H3 bidirectional engineering smoke | 真实 H3 33B、8×A800、tail-2、2 steps | loss `35.8310 → 31.4696`；反向 gate grad norm `46.4479` | 不适用 | 工程/梯度链路通过，不构成效果证据 |
| E12 | LingBot-inspired gate-only A/B | 同初始化/seed/800 dense windows；A 输出头，B 额外112个反向 gate scalars | train action：A `26.697083`，B `26.692795`；val40：A `24.207277`，B `24.203841` | 按预注册规则不晋级 rollout | held-out 仅改善 `0.0142%`，`NO_GO_LONG`；停止 gate-only 放大 |
| E13 | LingBot four-stream real-layer smoke | H3真实末层 + action expert；noisy/clean video/action；2 steps | velocity-head loss `9.8255 → 8.0418`；双专家梯度非零；reserved `10.37 GiB` | 不适用 | 单层工程门通过；整模 packed/FSDP 前仍 `NO_GO_LONG` |
| E14 | four-stream full50 FSDP smoke | 8×A800、50层、尾2层可训练、1 step | loss `26.0150`；H3/action grad `70.23/10131.99`；reserved `21.16 GiB` | 不适用 | 全层/FSDP门通过；等待真实 dense window smoke |
| E15 | 独立 ActionDiT 真实 dense 更新 | 8×A800、1274 video tokens、32 actions、tail-2、LR `1e-6`、无 warmup | step1 `33.3022`；step2 `7505.1816`；action grad `5.87e5 → 3.81e7` | 不适用 | `NO_GO`；checkpoint 只作诊断，不得续训 |
| E16 | shared-H3 four-stream 工程门 | 官方共享 block 结构；真实 H3 单层2步 + full50真实dense 10步；tail-2；10步warmup | 单层 action `2.3449 → 2.3432`；full50 action `1.07875 → 1.07266`；H3/action grad稳定 | 不适用 | 数值、FSDP、save/restore 均通过；只晋级 multi-window canary，不构成泛化证据 |
| E17 | shared-H3 multi-window canary | 100 steps、global batch8、800 train windows、随机 H3/LingBot timestep、tail-2 | val40 action `1.256609 → 1.250917`（改善 `0.453%`）；video `0.169534 → 0.165862`（改善 `2.166%`） | 尚未放行 | 首个 episode-disjoint 双目标正向信号；晋级扩展 canary，不是 `GO_LONG` |
| E18 | shared-H3 s100→s200 扩展门 | 续训100 steps、下一批800 windows、其余协议固定 | val40 action `1.245850`（较初始化改善 `0.856%`）；video `0.160928`（改善 `5.076%`） | 尚未放行 | 通过预注册 `0.75%` offline 门；仍是 teacher-forced，不是部署证据 |
| E19 | adapter-only 配对消融 | 与 E17 同 seed/首800 windows/100 steps；冻结全部 H3 block 与 video output | val40 action `1.252830`（改善 `0.301%`）；video `0.169371`（改善 `0.096%`） | 不适用 | tail-2 比 adapter-only 仅多改善 `0.152` 个百分点，未达预注册 `0.25`；不能声称尾层更新是主要贡献 |
| E20 | shared-H3 无泄漏逐 chunk sample40 | 8 chunks；每 chunk video 4步→提交→action 4步→提交；未来 clean key 完全屏蔽 | 生成 video MSE `0.695563 → 0.627430`（改善 `9.795%`）；action `1.258840 → 1.255442`（改善 `0.270%`） | 尚未放行 | 双模态生成信号为正，放行最小闭环工程 canary；动作增益小，仍非 `GO_LONG` |

## 2026-08-13 LingBot 核心结构纠偏

重新逐行核对作者仓库固定 commit
`7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb` 后确认：官方 LingBot-VA 先将
`[noisy video, clean video, noisy action, clean action]` 四路 token 拼接，再让全部 token
通过同一组 `WanTransformerBlock`。动作侧新增的是 `action_embedder`、独立
`condition_embedder_action` 与 `action_proj_out`，不存在一套独立的 30 层 ActionDiT body。

因此 E13–E15 的“独立 ActionDiT + joint attention”只保留为工程探索，不再称为官方结构对齐。
E15 同时存在固定 `sigma=0.5`、无 10-step warmup、动作未使用官方 `action_snr_shift=0.05`
等优化协议偏差，所以它不能否定 LingBot 方法本身；它只否定当前实现直接扩成长训。新主线的
唯一结构变量改为：将动作投影到 H3 hidden width，并让四路 token 共享 H3 的 50 个 blocks；
H3 没有第四个原生 modality tag，首个 canary 明确复用未参与当前任务的 audio tag `2` 作为动作
AdaLN 通道，这是 `INTENTIONAL_DEVIATION`，必须单独记录和消融。

共享 H3 实现随后通过 34 项相关测试、真实 H3 单层 2-step 和 8×A800 full-50 真实 dense
10-step 门。后者在 10-step linear warmup 下 action loss 从 `1.078752` 降到 `1.072662`，video
loss `0.182335 → 0.182341`，H3/action gradient norm 始终约 `0.21–0.25/8.14–9.49`；峰值
`42.01 GiB reserved`。独立进程加载保存的 stage 后得到 action loss `1.072317`，证明 checkpoint
restore 成功。该实验重复同一窗口，故只回答数值稳定性；下一门必须轮换 train windows，并在
episode-disjoint val40 上与未训练 shared-H3 初始化做同噪声对照。

E17 已完成上述对照。训练每个 rank/step 轮换不同 window，累计 800 samples；video 使用 H3 原生
shift `12.0`，action 使用 LingBot 配方的 shift `0.05`，val40 则按 sample 固定噪声。held-out
action/video 均改善，但 action 幅度仍只有 `0.453%`，故只放行到累计 3200 samples 的扩展 canary。
同时用空闲节点跑相同数据、噪声、LR 的 adapter-only 对照，隔离 H3 tail-2 更新是否必要；在两者
val40 结果出来前不做闭环、不增加服务器。

E18/E19 将扩展预算收紧为各 100 steps 后完成。shared-H3 累计 1600 个 train samples 后，固定
val40 action/video 相对初始化分别改善 `0.856%/5.076%`，说明第一轮很小的正向信号可重复扩大；
但同预算 adapter-only 的 action 也改善 `0.301%`，而 E17 tail-2 只比它多 `0.152` 个百分点，未过
`0.25` 个百分点的机制门。更重要的是，这些 val loss 使用官方训练式 clean future stream，属于
teacher-forced 指标；动作预测能读取同 chunk 的 clean video。下一步不得直接称闭环或 `GO_LONG`，
必须先实现官方式 chunk-causal 推理：当前 video chunk 从噪声生成并固化后，当前 action chunk 才
能读取该生成结果，任何时候都不得把数据集的未来 video/action 作为 clean stream 输入。

E20 已补齐这一缺口。采样器显式维护 clean stream validity mask，未观察/未生成 token 不只是置零，
而是从 attention key 中完全屏蔽；34 项采样、mask、梯度相关测试通过。相同 seed 的 val40 配对中，
s200 对生成 video/action MSE 均优于未训练初始化，说明 teacher-forced 改善没有在自由生成时完全消失。
不过 action 改善仅 `0.270%`，因此只允许接通服务端和跑最小 LIBERO 闭环工程 canary，不允许据此
增加训练步数、宣称泛化或启动全量微调。

## 2026-08-12 活跃长线

| 线路 | 节点 | 训练规模 | 最后观测进度 | 保留原因 |
|---|---|---|---:|---|
| M13 dense | `117.50.181.177:30907` | 1569 steps / 1 epoch | step920 | 回答更长训练是否能突破闭环零成功 |
| M11 frame-indexed | `117.50.181.177:30234` | 2170 steps / 1 epoch | step590 | 与 dense-uniform 采样形成长期对照 |

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
2026-08-12 检查共享盘为 `49T` 总量、`22T` 已用、`28T` 可用（44%）；当前无需因空间删除
checkpoint ladder。

## E12 严格 A/B 结果

A、B 都从同一份 H3→ActionDiT 初始化开始，使用 seed `2026`、相同 800 个 v7 dense
window、100 optimizer steps、global batch `8`、相同 loss/学习率和 `tail_sharded` FSDP 布局。
B 的唯一变量是在最后两层加入 112 个 action→future-video gate scalars。

| 指标 | A：output-only | B：output + bidirectional tail-2 | B 相对变化 |
|---|---:|---:|---:|
| train mean action loss | 26.697083 | 26.692795 | 改善 0.0161% |
| val40 mean action loss | 24.207277 | 24.203841 | 改善 0.0142% |
| val40 mean video loss | 0.363572 | 0.363588 | 退化 0.0043% |

两个 arm 的首步 total/video/action loss 完全相同，证明数据顺序和初始化可比；B 的 gate 从零
更新到 `max_abs=0.009644`，证明机制确实被优化。A 最初用 `head` FSDP 布局反向时因单卡
峰值约 `77.9 GiB` OOM，随后改为与 B 完全相同的 `tail_sharded + frozen body/shared state`
布局后重跑；该修正消除了内存布局这一混杂变量。最终两边峰值均约
`41.45/58.41 GiB allocated/reserved`。

按预注册规则，B 没有获得有意义的 held-out 优势，因此不做选择性闭环试验，也不通过增加
steps 或 gate 数量来追逐噪声。E12 的正面价值是排除了“只补一个尾部反向残差就足够”这一
假设，并验证 H3 双向流的保存、恢复、梯度和多卡链路可复用。

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
4. 新主线按 `H3_WAM_CODE_FIRST_AUDIT_2026-08-12.md`，实现完整 chunk-level block-causal
   video/action 双流；不得把 E12 gate-only 版本直接扩成长训。
5. 每个新实验只改一个变量，并在启动前记录父基线、预算、晋级门槛和停止条件。
