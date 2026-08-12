# H3-WAM 实验资产账本

更新时间：2026-08-13（Asia/Shanghai）

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
- shared-H3 已证明生成 video 可学（E22 严格无泄漏 video 改善 `12.696%`），但 action
  仅改善 `0.384%`。E24–E27 四个作者代码驱动的小型 canary 均未过动作门；主瓶颈是
  video/action 时序合约与动作建模，而不是 GPU 数量、采样步数或简单扩大训练步数。

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
| E21 | shared-H3 首次闭环工程门 | s200、goal task3/trial0、sample4、replan32、max80 | 3次 replan；动作有限、零饱和；平均推理 `71.90s` | `0/1`；物体最大位移约 `1e-16` | 服务/FSDP/VAE/仿真链路通过，但未产生有效物体交互；`NO_GO_LONG` |
| E22 | shared-H3 s300 bounded continuation | s200再训100 steps、下一批800个不重复 window；累计2400 windows | teacher-forced val40 action/video `1.240105/0.158307`；无泄漏 sample40 action/video `1.254009/0.607254` | 按门控不重跑 | 生成 video 较初始化改善 `12.696%`，action 仅改善 `0.384%`，未过 `0.5%` 动作门；停止纯续训 |
| E23 | 官方 20/50 denoise-step 消融 | 固定 s200、固定 val8/seed；4/4 改为 video20/action50 | action `1.207244 → 1.202721`（改善 `0.375%`）；video `0.585749 → 0.671369`（退化 `14.617%`） | 不重跑 | 未过预注册 `1%` 动作门且耗时 `665s`；高采样步数不是当前主瓶颈 |
| E24 | LingBot quantile normalization s100 | 与 E17 同初始化/seed/800 windows；仅将 min/max `[-1,1]` 改为 q01/q99 `[-1.5,1.5]` | val40 action `1.304397 → 1.295573`（改善 `0.676%`）；video 改善 `2.165%` | 按门控不重跑 | 比 E17 action 多改善 `0.224` 个百分点，略低于预注册 `0.25`；方向有效但单独不足以晋级 |
| E25 | per-chunk action timestep s100 | E24 合约；每4个动作独立采样 timestep，其余固定 | val40 action `1.304397 → 1.295933`（改善 `0.649%`）；video 改善 `2.195%` | 按门控不重跑 | 低于 E24 的 `0.676%` 且未过 `0.926%` 门；噪声粒度单独不是瓶颈 |
| E26 | LingBot noisy clean-video s100 | E24 合约；训练时以0.5概率给 clean-video stream 加高噪声 | clean val40 action `1.297608`（较初始化改善 `0.520%`）；masked action `1.299689` | 按门控不重跑 | masked action 与 E24 `1.299678` 持平且略差；未缩小训推偏移，停止该线 |
| E27 | detached generated-video conditioning s100 | E24 合约；action 训练改为消费模型首次 forward 的 detached one-step `x0` | clean action `1.295345`（较初始化改善 `0.694%`）；masked action `1.299328` | 按门控不重跑 | masked action 较 E24 仅改善 `0.027%`，远低于预注册 `0.5%`；不晋级 causal sample/闭环 |

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

adapter-only 的相同无泄漏 sample40 为 video/action `0.702341/1.258167`，基本等于初始化，故
tail-2 更新对生成式 world 分支的贡献已得到比 teacher-forced MSE 更清晰的支持。E21 随后首次
打通 shared-H3 闭环服务：模型 READY 后在真实 LIBERO 环境完成 3 次 replan 和 80 steps，动作均
finite，归一化动作无饱和，平均环境动作幅度 `0.405`；但 task3 `0/1`，所有物体 joint 位移都在
`1e-16` 量级。非 KV-cache 的完整历史重算耗时 `71.90s/replan`。因此 E21 只通过工程链路门，
没有通过策略效果门。下一对照固定 checkpoint，改用官方 LIBERO 配置的 video/action denoise
steps `20/50`，判断 sample4 是否是动作质量瓶颈；同时仅保留再 100 steps 的 s300 bounded canary。

E22/E23 已完成并同时否定“再堆训练步数”和“只堆采样步数”两个捷径。s300 的 teacher-forced
action 继续下降到 `1.240105`，但严格无泄漏 action 只从初始化的 `1.258840` 降到 `1.254009`
（`0.384%`）；同期生成 video 已改善 `12.696%`。这说明当前优化主要被 world/video 分支吸收，
并没有等比例转化为可执行动作。固定 s200 的官方 `20/50` 采样在 val8 上只改善 action
`0.375%`，同时 video 退化 `14.617%`，推理耗时达到 `665s`，故不得进入闭环。后续只允许从作者
开源代码中选择一个动作侧不一致项做 canary；在动作无泄漏门通过前，不再增加 steps 或服务器。

E24 对作者代码中的 action normalization 做了严格单变量复现。四套 LIBERO 原始数据共
`1712 episodes / 277713 frames`，从原始帧（而不是高度重叠的 window）计算 q01/q99；六个连续
控制维与作者发布统计接近，本地 gripper 的 `[0,1]` 支持在归一化后与作者 `[-1,1]` 支持等价。
quantile 将三个小幅旋转维的标准差放大约 `1.31–2.12×`，但 val40 动作改善只从 E17 的
`0.453%` 提升到 `0.676%`，比基线多 `0.224` 个百分点，未达到预注册 `0.25`。因此保留 quantile
作为下一次完整代码对齐实验的默认动作合约，但不为 E24 单独做无泄漏采样或闭环。进一步核对
发现更关键的训练偏差：LingBot 为每个 latent/action frame 独立采样 timestep，而当前 H3 端为
整段 action 使用单一 timestep；下一 canary 必须先补齐 per-chunk action diffusion forcing。

E25 已完成这一可控修正：36 项相关测试和真实 2-step 反向 smoke 均通过，100-step 训练峰值
与 E24 相同，证明多 timestep 的 AdaLN 与 final norm 路径实现正确。但 val40 action 改善为
`0.649%`，不仅未达预注册 `0.926%`，还略低于 E24 的 `0.676%`，因此不做无泄漏采样或闭环。
逐行代码对齐还发现 LingBot 训练的 `noisy_cond_prob=0.5`：一半 batch 会给 clean video stream
加入高噪声，而 action clean stream 保持干净。这比 action timestep 粒度更直接作用于
teacher-forcing/exposure-bias；下一 canary 应固定 E24，其唯一变量为 noisy clean-video condition。

E26 复现了该官方机制，并在强制 corruption 的真实 2-step smoke 后按概率 `0.5` 训练 100 steps。
clean val40 action/video 为 `1.297608/0.167225`，action 相对 quantile 初始化改善 `0.520%`，保住了
最低 clean 门，但弱于 E24。更关键的 masked-clean-future 配对结果为：初始化
`action/video=1.304024/0.421788`，E24 `1.299678/0.420131`，E26 `1.299689/0.420463`；E26 对
E24 没有动作优势，video 也略差。因此 noisy condition 没有在当前预算下缩小暴露偏移，不晋级
causal sampling 或闭环。E24 的 quantile normalization 保留为当前动作合约；下一结构性候选应
训练 action 直接消费 detached generated-video latent（或等价 scheduled sampling），而不是继续
调 noise granularity、condition corruption、denoise steps 或纯训练步数。

E27 已完成这一 scheduled-sampling canary。实现用解析单测锁定 H3 clean-time velocity
约定下的 `x0 = xt + sigma * v`，37 项测试和真实 2-step 反向 smoke 通过，峰值
`43.03 GiB`。100-step clean val40 的 action/video 为 `1.295345/0.166247`，action 较初始化
改善 `0.694%`；masked-clean-future 为 `1.299328/0.420343`。与 E24 的 masked action
`1.299678` 相比仅改善 `0.027%`，masked video 退化 `0.050%`。虽然方向略正且满足
clean 保底门，但未达预注册 `0.5%` 的机制门，因此不做 causal sample 和闭环。

E24–E27 已经单变量排除了 quantile 幅度、action timestep 粒度、clean-video 噪声和
one-step scheduled sampling 这四个“小修补”作为主瓶颈。下一步不再盲目扩步数，而是先
完成 video latent frame 与 32-step action horizon 的时序对齐审计：当前 12 个 future-video latent
与 8 个 action chunk 采用比例映射，这是相对 LingBot 官方 frame-stride/latent-frame 关系的
明确偏离。只有证据包锁定对齐合约且 smoke 通过后，才启动新 canary。

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
3. 固定 E24 作为当前动作合约；E25–E27 不再扩步数，也不重复 causal sample/闭环。
4. 逐行对齐 LingBot 的 `latent_frame_num × frame_stride` 与本地 12 future-video latents / 32
   actions / 8 chunks，生成可复现的 frame-action index 对照表和新 evidence dossier。
5. 只在时序偏离被证明且修正的 2-step smoke 通过后，启动单变量 s100 canary；预注册
   动作门后再决定是否进入无泄漏 sampling 和 LIBERO 闭环。
