# H3-WAM 实验资产账本

更新时间：2026-08-17（Asia/Shanghai）

本文档保存 H3-WAM 已完成和正在运行的关键尝试，避免云资源结束后只剩模型文件而无法解释。
历史 `M*` 名称曾被复用，因此这里增加稳定的 `E*` 编号。所有成功率均指真实 LIBERO
闭环 success predicate；物体移动、离线 MSE 或训练 loss 不能替代成功率。

## 当前结论

- R1 baseline v2 与 Candidate F v2 的晚期 checkpoint 虽继续降低 physical MSE/ADE，却在相同
  balanced-80 反事实评测中几乎失去视觉与语言响应，并分别于 step914/851 触发零视觉梯度；
  两条线均为 `FAIL_CONDITIONING_COLLAPSE`，不得靠增加 steps 或放宽机制门晋级。
- Candidate G paired visual margin 在 s50 通过梯度、跨 rank 无 self-map 和 restore 机械门，但
  严格同 step 父对照中 physical MSE、gripper macro-F1、语言与相对视觉敏感度均无净增益；
  判定 `FAIL_PAIRED_GATE`，不扩 s100/长训/rollout。
- Candidate D 的五层 DreamWAM K/V cache 已完成 8,560/8,560 全量审计，且唯一授权的修复后
  单步 probe 通过有限非零梯度与参数更新门。后续 D/D0 严格配对近一轮实验全部完成：D 在
  s10–s500 的 physical MSE/ADE 领先，但 s963 被重复 layer49 的 D0 全面反超。两臂从 s250
  起均形成持续的目标对齐 visual-shuffle 惩罚，证明冻结 H3 条件加独立动作专家路线具备离线
  机制信号；淘汰“五层对齐必然更优”的假设，选择 D0 s963 进入最小闭环 canary。
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
  仅改善 `0.384%`。E30 对齐官方 `1e-5` LR 后，无泄漏 video/action 改善升至
  `14.020%/0.900%`，但固定闭环仍 `0/1` 且无物体接触。E32 再对齐官方 AdamW
  `weight_decay=0.1` 后，raw/causal action 反而退化 `3.005%/2.836%`；主瓶颈不是 GPU
  数量、采样步数、简单扩大训练步数或继续照搬优化超参，而是动作专用建模与闭环目标绑定。

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
| E08 | M11 full frame-indexed long | 277,713 windows，global batch128 | step200 `0.151389`；step1600 `0.108532`（改善28.31%） | step200 task3 `0/10`；step1600 task3 `0/10` | 保留至预定终点；已从无接触进步到明显物体位移，但仍未成功 |
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
| E28 | H3 physical-time alignment s100 | E24 合约；将 video chunk mask 和 action RoPE 对齐 H3 非均匀 VAE 物理时间 | val40 video/action `0.146827/1.298060`；改善 `1.867%/0.067%` | 按门控不重跑 | action 远低于预注册 `0.926%`；不晋级 no-leak sample40/闭环 |
| E29 | 官方 flow loss weighting s100 | E24 合约；训练时逐 video row/action token 使用 LingBot/DreamWAM scheduler `training_weight` | val40 video/action `0.165724/1.297436`；改善 `2.530%/0.534%` | 按门控不重跑 | action 未过 `0.926%` 门且比 E24 弱 `0.143` 个百分点；不晋级 sample/闭环 |
| E30 | 官方 LR `1e-5` s100 | E29 合约；只将 LR 从 `1e-6` 对齐 LingBot LIBERO 配置 `1e-5` | raw val40 video/action 改善 `18.112%/3.067%`；无泄漏 sample40 改善 `14.020%/0.900%` | task3 trial0 `0/1`；最大物体位移 `8.08e-17` | 当前最强因果离线信号，但闭环不通过；保留优化配方，先修动作历史协议再扩训 |
| E31 | 初始动作锚的冷启动近似 s100 | E30 合约；每次窗口/重规划前置4步零动作 | raw改善 video/action `16.967%/1.322%`；无泄漏改善 `13.195%/0.653%` | 撤销上游对齐声明 | pinned LingBot 只在 `frame_st_id==0` 固定首帧并持续维护 KV；本地冷启动近似不等价，旧 checkpoint 已 fail closed |
| E32 | 官方 AdamW weight decay `0.1` s100 | E30 合约；只将本地默认 `0.01` 对齐官方 `0.1` | raw video/action 改善 `17.705%/-3.005%`；无泄漏改善 `13.680%/-2.836%` | 按门控不重跑 | 两个动作指标均反向；`NO_GO_LONG`，本地短程局部解冻保留 `0.01` |
| E33 | E30 execution cadence | checkpoint/horizon32 固定；replan 从32改为4/8/16 | 不适用 | 三臂均 `0/1`、最大物体位移 `≤1.49e-16` | saturation 升至 `12.29–12.68%`；开环长度不是简单主因，停止部署扫描 |
| E34 | R1 baseline v2 A2 | 同一 balanced-80；s100→last-safe s913 | physical MSE `0.343551→0.124351`；visual-shuffle MSE delta `0.236696→4.10e-7`；language mean-abs delta `0.433882→0.019489` | 未放行 | gripper F1 退化；step914 visual grad=0；`FAIL_CONDITIONING_COLLAPSE` |
| E35 | Candidate F v2 F2 | clean regression weight1；同一 balanced-80；s100→last-safe s850 | physical MSE `0.352986→0.202512`；visual-shuffle MSE delta `0.209107→9.85e-7`；language mean-abs delta `0.415041→0.032469` | 未放行 | gripper macro-F1 `0.550772→0.316026`；step851 visual grad=0；`FAIL_CONDITIONING_COLLAPSE` |
| E36 | Candidate G paired visual margin | 8-rank wrong-H3 hinge；s1→s50；同 step baseline s50 | G/parent physical MSE `0.412425/0.405095`；visual relative scale `1.063807/1.064937` | 未放行 | 机械门通过，但 gripper macro-F1 低3.04%、语言略低；`FAIL_PAIRED_GATE`，不扩训 |
| E37 | Candidate D DreamWAM 5-layer K/V carrier | 8,560-window cache 全量 audit；修复后 1 GPU × 1 sample × 1 step；无 checkpoint | loss `1.484375`；5 block/proprio gradients 全部有限非零；head update `1.19e-7` | 未放行 | cache/Data 与 trainability 机械门通过；`MECHANICS_PASS_ONLY / NOT_EVIDENCE_READY`，禁止据此长训或声称效果 |
| E38 | Candidate D s1 save/restore + v4 balanced-80 adapter | schema-v2 1-step checkpoint；fresh restore；v4 40×2 evaluator | restore max-abs `0`；physical MSE `0.492577`；gripper macro-F1 `0.363093`；language relative-L2 `0.471810`；visual delta MSE `0.177422` | 未放行 | `MECHANICAL_GATE PASS / EVALUATOR_MECHANICS PASS`；v4/v8 IDs mismatch 且仅一步，`NOT_EFFECTIVENESS / NOT_STRICT_PAIRED_IDS` |
| E39 | Candidate D/D0 v4 严格配对学习曲线 | 同初始化/样本顺序/8-rank；D 五层对齐，D0 重复 layer49；s10/s50/s250/s500/s963 | s963 D/D0 physical MSE `0.142015/0.110084`，ADE `0.945540/0.804093`，越界率 `6.13%/4.47%`，gripper macro-F1 `0.780779/0.784651` | 待最小 canary | 每臂累计7704唯一 IDs、restore max-abs `0`；两臂 visual shuffle 均显著恶化动作/夹爪；`REJECT_ALIGNED_5LAYER_AS_WINNER / PROMOTE_D0_S963_TO_MINIMAL_CLOSED_LOOP_CANARY` |
| E40 | Candidate D0 s963 在线 K/V 与闭环 canary | INT8 H3 layer49 在线 K/V；goal0 80步、spatial0 120步；10-step flow | 训推 K/V bitwise exact；动作 saturation `0`；峰值 `30.61 GiB` | `0/2` | goal0 错推 bowl/plate、drawer delta0；spatial0 无接触；`FAIL_CLOSED_LOOP`，不跑 benchmark |
| E41 | Candidate D0 v7 dense 数据线 | 架构/目标/优化不变；5 windows/episode → 200,779 dense train windows；32卡 cache | cache 生成中；v7 balanced-80 IDs `26b032...9c42` | 待评测 | 唯一变量是状态/接触密度；先等预算 s963，再到 s25097 近一轮；禁止在 dense 结果前再改架构 |
| E42 | H3 INT8 双输出 dense cache | 同一次 H3 forward 写 DreamWAM 五层 K/V 与 StarWAM layer49 pooled hidden；单样本与两条独立路径比对 | feature/KV 均 bitwise equal、max-abs `0`；32 workers；切换时保留30,005份K/V；初始补写约`9.16 samples/s` | 不适用 | `MECHANICAL_PASS`；避免为 C03 再完整跑一遍22万样本，但不构成任何策略效果证据 |
| E43 | dense carrier 三线流水线 | 32卡双 cache → 双全量 audit → D/D0 各8卡 s963；C03 8卡 s100；第四节点逐 checkpoint balanced-80 | 已 armed；快照 K/V `30,002`、StarWAM `14,623`、错误0；D/D0/C03 与三条评测 watcher 均等待 audit READY | 待评测 | v7/v8 train ID 重合 `90.07%`，故 C03 锁死 s100、禁止重复 s963；D/D0 严格唯一变量配对，任何线均未获得长训或效果声明许可 |
| E44 | dense 双 cache 全量审计与三线解锁 | 64个单线程内容分片；逐文件 hash/反序列化/元数据/finite 检查；精确目录 ID 集合；固定 manifest/H3 hash | K/V `222,929` 条、`953.56 GiB`；StarWAM `222,929` 条、`71.98 GiB`；缺失/额外/临时/元数据错误均为0；D/D0/C03 共24卡已启动 | 等待首个 checkpoint 与 balanced-80 | 3个无生产者旧 `.tmp` 已可恢复隔离，完整 final 均存在；过度订阅与慢目录扫描归类为基础设施问题；审计 PASS 只放行训练，不构成效果证据 |
| E45 | FACT-lite H3 future-proprio F1 s100 | 同 H3/state 输入下，正确 horizon32 动作 vs batch derangement vs zero-action；episode-disjoint 1024/256 | val MSE `0.016670` vs shuffled-train `0.108835` vs independent `0.113512`；正确模型 shuffle 后 `0.235171`；restore diff0 | 不适用 | 三个预注册机制门均通过；`PASS_MECHANISM_GATE / EVIDENCE_READY`，只放行同合同 s500，不声称策略或成功率 |
| E46 | repaired LingBot adapter-only s1000 闭环 | Goal task3/trial0；80步；horizon/replan32；4-step flow；quantile clip1.5 | action saturation `9.375%`；EEF 三轴移动范围 `0.0586/0.0566/0.0408m`；物体最大位移 `1.46e-16` | `0/1` | 服务与恢复成功但无物体接触；`FAIL_CLOSED_LOOP`，停止 s1000 benchmark 晋级；保留已批准的独立 5000-step 诊断曲线 |
| E47 | repaired LingBot adapter-only fresh s5000 | 从确定性初值连续5000步；global batch8；每500步独立 val40/sample40 | s500 causal video/action `0.635340/1.149157`；s1000 `0.645313/1.074492`，action 较 step0/s500/旧s1000 改善`17.79%/6.50%/7.92%`，video 较s500回退`1.57%`但仍优于step0`7.22%` | s500与s1000 Goal task3/trial0 均`0/1`且无接触；s1000 EEF范围仅`0.0440/0.0180/0.0450m`，物体最大位移`1.46e-16` | `PASS_OFFLINE_GATE / FAIL_CLOSED_LOOP / STOP_BENCHMARK`；保留预注册长曲线作诊断，但主优化转向接触阶段覆盖/目标，不能再把平均MSE下降当充分条件 |
| E48 | FACT-lite future-proprio F1 s500 | E45 完全相同三臂和1024/256 split；仅100→500步 | val MSE `0.012862` vs shuffled-train `0.068679` vs independent `0.073846`；正确模型 shuffle 后 `0.212931`；restore diff0 | 不适用 | 机制门重复通过；停止扩大同一 target，放行独立 future-H3 s100；value/ranking 继续受 failure data gate 阻塞 |
| E49 | FACT-lite future-H3 F1H s100/s500 | current/start+32 H3 layer49 mean 经固定256D投影；三臂同初始化、1024/256 episode-disjoint | s100 增益约`3.10%-4.08%`；s500 MSE `180.116` vs shuffled-train `194.065` vs independent `193.492`，正确模型 shuffle 后`212.738`，增益`6.91%-18.11%`；restore diff0 | 不适用 | s100/s500 重复通过，成为 consequence 机制冠军；停止孤立 target 扩步，value/ranking 继续受 canonical failure data gate 阻塞 |
| E50 | executed-history16 s2500 闭环 | history 曲线最佳点 causal action/video `0.376101/0.255801`；Goal task3/trial0；replan16 | saturation `2.60%`；EEF 三轴范围`0.122/0.0968/0.229m`；物体最大位移`7.90e-17` | `0/1` | 离线最优点仍无物体接触；`FAIL_CLOSED_LOOP / STOP_HISTORY_BENCHMARK`，上下文线不能凭离线 MSE 晋级 |

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

E28 已将这一偏离实现为可选的物理时间合约：根据 H3 释放实现的非均匀
`5/3 × (1,4,4,4,4)` rotary/VAE 时钟和缓存的 33→39 帧最近邻采样，同时修正 video
chunk mask 与 action RoPE。33 项云端测试、真实 2-step 反向和 checkpoint restore 全部
通过；s100 在 8×A800 上用时 `483s`，峰值 `42.03 GiB reserved`。因新时间合约会
改变未训练前向，配对评估另算同构初始化 `video/action=0.149620/1.298935`；s100 为
`0.146827/1.298060`，对应 video 改善 `1.867%`，action 仅改善 `0.067%`。远低于
预注册 `0.926%` action 门，所以不做 no-leak sampling/闭环，也不扩训。该实验
再次确认当前修正主要转化为 world/video 改善，action 学习仍是独立瓶颈。

E29 补齐了 LingBot-VA 与 DreamWAM 训练代码共同使用、而本地 E17–E28 漏掉的
timestep-dependent flow loss weighting。解析实现由调度器等价单测锁定，并分别按 video row
和 action token 归约；35 项云端测试、真实 2-step 反向、stage 重载均通过。s100 用时约
`479s`，峰值 `42.02 GiB reserved`。固定 val40 的 video/action 为
`0.165724/1.297436`，相对同构初始化改善 `2.530%/0.534%`；动作改善反而比未加权 E24 少
`0.143` 个百分点，也未达到预注册 `0.926%`。因此不做 causal sample/闭环，不靠增加步数
追逐该失败点。权重实现保留为上游对齐选项，但它不是当前动作瓶颈的解法。

E30 将唯一优化变量改为 LingBot-VA LIBERO 官方 `1e-5` learning rate，其余保持 E29
不变。s100 固定 val40 的 video/action 相对同构初始化改善 `18.112%/3.067%`，严格
无泄漏 sample40 改善 `14.020%/0.900%`，是当前 shared-H3 最强因果离线信号。闭环前又
发现服务端错误沿用旧 min/max 解码 quantile checkpoint；现已让 checkpoint 自带的归一化
合约成为唯一事实源，并按官方训练 clipping 将 quantile 范围保留为 `[-1.5,1.5]`，云端
25 项相关测试通过。修复后的 task3/trial0 仍为 `0/1`，最大物体位移 `8.08e-17`，所以 E30
明确为 `NO_GO_LONG`：官方优化协议值得保留，但当前动作历史/执行接口还不能扩成长训。

逐行核对作者数据集、server 和 LIBERO client 后定位到下一处成套偏差：作者在原始动作域
前置一帧（LIBERO 为4步）全零动作作为历史，采样时固定该帧为 clean condition，首次执行时
跳过它；本地此前既没有训练锚，也直接执行 action0。E31 已把三处作为一个不可拆分的
冷启动训练—推理近似实现，25 项测试通过；真实 H3 两步 smoke 的 H3/action 梯度为
`0.327/10.633`、`0.236/15.739`，独立进程 restore val8 成功。最终 s100 raw video/action
改善 `16.967%/1.322%`，但严格无泄漏只改善 `13.195%/0.653%`，低于 E30 的
`14.020%/0.900%`。2026-08-13 复核上游 server/client 后确认该实现遗漏了 `frame_st_id` 与持久
observation/action KV 生命周期，不能称为上游合同；训练器现拒绝该 flag，服务端拒绝旧 E31 stage。

官方 LIBERO 配置还使用 AdamW `weight_decay=0.1`，而 E30 实际为 PyTorch 默认 `0.01`。
E32 以此为唯一变量完成 100 steps：训练耗时 `461.4s`、峰值保留显存 `42.02 GiB`。
固定 raw val40 的 video/action 改善为 `17.705%/-3.005%`，严格无泄漏 sample40 为
`13.680%/-2.836%`。视频学习保留、动作学习却在两个独立协议上同时反向，未过预注册
E30 `0.900%` 动作门，因此不做闭环、不续训。官方长程全参数配方的正则强度不能直接移植到
当前 100-step H3 tail-2 局部解冻预算。

E33 不改模型，只将 E30 的执行周期从32缩短至4/8/16。三个相同 task3/trial0 均完成80步，
全部 `0/1` 且没有物体接触；动作饱和比例为 `12.68%/12.29%/12.60%`，反而高于 replan32
的 `10.59%`。因此不能把 E30 闭环失败简单归因于32步开环执行，停止 cadence 扫描。

## 2026-08-13 长线实时状态

| 线路 | 节点 | 训练规模 | 最后观测进度 | 保留原因 |
|---|---|---|---:|---|
| E32 official weight decay | `117.50.181.177:30907` | 100-step bounded canary | 已完成；causal action 退化 `2.836%`，未过门 | 分支停止，30907/32409 已释放 |
| M11 frame-indexed | `117.50.181.177:30234` | 2170 steps / 1 epoch | 已越过 step1800；最新已保存 step1800 | 仍在 8×A800 上约 100% 运行；step1600 val40 `0.108532`、task3 `0/10`，终点后核验 final |

上述状态于 2026-08-13（Asia/Shanghai）用进程、GPU 和日志三重核对：E32 训练及配对
raw/sample40 已完成，30907/32409/32611 已释放；30234 继续 M11。恢复研究时仍必须
先从日志和 checkpoint manifest 重新确认。

## 云端资产位置

共享根目录为 `/mnt/h3-wam`，主要资产包括：

- 模型：`/mnt/h3-wam/models/MiniMax-H3`、`RAFT`、`fastwam_release`；
- 数据/缓存：`downloads`、`libero_fastwam_extracted`、`v2_full_cache`、
  `v7_dense_*`、`v8_frameindexed_*`；
- 输出：`outputs/h3dotwam*` 与对应 rollout/eval JSON；
- 当前 checkpoint：M13 有 final step1569 及 step200–1400 ladder；M11 已有 step200–1600 ladder，
  并继续训练至 step2170。shared-H3 E24–E30 的 s100 stage 也均保留在
  `/mnt/h3-wam/outputs/h3-lingbot-shared/`。

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
2. 等 M11 到 step2170 后核验 final checkpoint、日志、resolved config 与数据 manifest hash；
   M13 已完成，不再重跑。
3. E30 固定为当前 shared-H3 优化基线；E31/E32 均未超过其 `0.900%` causal action 门，已停止。
4. 下一证据审计转向动作专用建模：比较 LingBot/MiniWorld/DiT4DiT 的 action
   carrier 宽度、state 注入位置和 chunk 目标，不再从 video condition 侧做小修补。
5. 新实验必须由作者开源代码中的明确差异驱动，且先交付配对基线、单变量门和
   2-step/restore smoke；未过 offline action 门不调度 LIBERO 闭环。

## 2026-08-13 训练预算扩展

用户明确要求优先检验“训练量不足”。在不修改数据、动作合约和核心目标的前提下，启动三条
可区分结论的规模线，并保留一台独立评测节点：

| 节点 | 实验 | 预算 | 里程碑/门禁 |
|---|---|---:|---|
| 30907 | shared-H3 tail-2 从累计 step500 续至2500 | global batch8，新增16000窗口，累计0.0996 epoch | step1000/1500/2000/2500；严格 causal sample40 和固定闭环决定是否继续 |
| 32611 | shared-H3 tail-4 等预算对照 | global batch8，20000窗口，0.0996 epoch | step500/1000/1500/2000/2500；唯一变量为可训练H3尾层2→4 |
| 30234 | M11 frame-indexed 第二完整epoch | global batch128，2170 steps，277760样本 | 每200步checkpoint；终点复用 val40 + Goal task3十次闭环 |
| 32409 | shared-H3 checkpoint消费者 | 不训练 | 自动等待并依次评测tail-2/tail-4里程碑，防止只看终点 |

启动后三条训练均进入真实GPU计算且首步 finite。tail-2 的 step500 配对结果为：teacher-forced
video/action MSE `0.117352/1.167875`，但无泄漏 causal video/action MSE
`0.419902/1.313663`；即视频和 teacher-forced action 随训练改善，而 causal action 相对
step100 的 `1.295219` 暂时回退。这是扩大预算时必须持续监测的过拟合/训推偏移信号，不能仅凭
训练 loss 宣称成功。

## 2026-08-13 tail-2 累计 step10000 扩展

tail-2 在累计 step2500 首次得到明确的无泄漏因果动作改善：video/action MSE
`0.279897/1.231785`，相对未训练初始化分别改善约 `59.8%/5.75%`。用户据此明确要求将相同配方
一步扩到累计 step10000。该实验从 s2500 权重继续 7500 optimizer steps，global batch8，新增
60000 windows（`0.298835` effective epoch），累计80000 windows（`0.398447` epoch）。唯一变量
仍是训练预算；优化器 moments 不包含在 stage 中，因此续训会重新初始化 AdamW 状态。

检查点按**累计步数**保存 s3000、s4000、...、s9000，终点保存 s10000；32409 在旧的
tail-2/tail-4 ladder 完成后自动消费全部里程碑，运行 val40 与无泄漏 causal sample40。因为 s2500
尚无闭环 success，这条线是用户授权保留的 legacy scale probe，当前仍为 `NO_GO_LONG`；任何里程碑
都必须通过固定 LIBERO success predicate 才能晋级。

## 2026-08-13 动作训练噪声支持三臂消融

在 tail-2 长训继续运行的同时，上游代码审计发现当前 shared-H3 配方直接继承了 LingBot
LIBERO 的 `action_snr_shift=0.05`。该分布平均噪声 sigma 约 `0.113`，仅约 `4.76%` 的训练
样本满足 `sigma>0.5`，而因果动作生成从纯噪声 `sigma=1` 开始。作为对照，LingBot 的
demo/Robotwin/Franka 配置使用 shift `1.0`，FastWAM 与 DreamWAM 使用 shift `5.0`；后两者
对应的 `sigma>0.5` 覆盖约为 `50%/83.33%`。

为避免把训练支持与采样器同时改变，新增 `action_train_shift` 和 `action_infer_shift` 两个独立
合约，并保持所有主评测 `action_infer_shift=0.05`。E30 已提供 shift `0.05` 的配对父基线；
在 `32611/32409` 并行启动 shift `5/1` 两条 s100 canary。两条实验都使用相同 H3 初始化、
seed `2026`、前800个训练 window、global batch8、LR `1e-5`、WD `0.01`、warmup10、tail-2、
quantile 动作归一化和 weighted flow objective，唯一训练变量是 scheduler shift。

主门要求因果 action MSE 不高于 `1.282267`（比 E30 `1.295219` 再改善1%）、因果 video MSE
不高于 `0.627947`，raw action MSE 低于 `1.304397`，raw video MSE 不高于 `0.146192`。
三臂先统一 video/action 4步以严格复用 E30；随后固定 video4、仅把 action solver 提至 LingBot
LIBERO 官方50步，作为独立采样敏感性复核。需要注意 `action_train_shift` 同时改变噪声采样与
官方 loss weighting，因此本轮只能归因于“训练 scheduler”，不能单凭结果宣称高噪声覆盖就是
唯一机制。预注册 dossier：
`experiments/evidence/h3_lingbot_action_shift{1,5}_s100_v1.json`。

两条 s100 canary 均完成。固定 raw val40 的 video/action：shift1 为
`0.139244/1.243479`，shift5 为 `0.139366/1.230748`；相对 E30 shift0.05 的
`0.139230/1.264393`，raw action 分别改善约 `1.65%/2.66%`。但统一4步无泄漏因果
video/action 为 shift1 `0.596848/1.288768`、shift5 `0.592029/1.304205`；shift1 仅比 E30
action `1.295219` 改善约 `0.50%`，未过预注册 `1.282267` 门，shift5 则退化约 `0.69%`。
因此两臂均不续训到500步，尤其不能依据 raw 指标选择 shift5；训练期高噪声覆盖不是越大越好。
固定 video4、action50 的采样复核仍在运行。

同一时段，legacy tail-2 长线累计 step3000 的 raw video/action 已达
`0.098415/0.839149`，因果 video/action 达 `0.268833/1.099539`；因果 action 相对 step2500
的 `1.231785` 再改善约 `10.7%`。这支持“训练预算不足是重要因素”，但不替代 LIBERO 闭环
success；step3000 是下一固定 task3/trial0 闭环的优先候选。

step3000 的固定 Goal task3/trial0、replan32 闭环随后完成：`0/1`，80步、3次重规划，动作
绝对值均值 `0.318`、饱和率 `7.64%`，但最大物体/抽屉位移仍仅约 `1.45e-16`。轨迹显示
末端并非静止：相对初始位姿移动约 `(+0.120,-0.127,+0.044)m`，只是没有形成物体接触；
32步块的预测方向会反转。由于该 checkpoint 的离线因果动作显著强于 E30，补做同 checkpoint、
同 task/trial、仅将执行周期改为 LingBot LIBERO 对应的16动作，检验更强模型与上游执行节奏的
组合；该部署消融不改变训练结论。

step3000 的 replan16 配对闭环同样为 `0/1`，80步、5次重规划，动作绝对值均值 `0.333`、
饱和率 `9.69%`，最大物体位移约 `1.48e-16`。因此更强 checkpoint 与上游16动作执行周期的
组合仍不能产生接触，执行周期再次排除。下一结构性整改应对齐上游 rolling KV/已执行动作历史，
并增大或预训练动作 carrier；不再把 scheduler shift 或 replan cadence 作为主线。

shift1 固定 video4、action50 的 sample40 最终为 `0.596650/1.286845`，相较 action4 的
`0.596848/1.288768`，action 仅再改善约 `0.15%`，仍未过 `1.282267` 门。官方50步求解器
不是主要失败原因，且成本约为4步的12.5倍，不作为当前默认部署配置。

为确认 task3 是否属于单任务偶然失败，在空闲 `32611` 上启动同一 step3000、同一 Goal suite
的 task `0/3/7/8`、各 trial0、replan16、80步探索性闭环。该评测不用于选择 checkpoint，也不
冒充完整 benchmark；它只回答“当前模型是否在任一不同场景形成物体接触/成功”。若四任务仍为
零成功且轨迹保持末端运动、物体零位移，则停止 scheduler、solver 和 replan 微调，把下一训练
预算用于上游已明确存在而本地缺失的跨 replan observation/executed-action rolling history/KV
合约及更强动作 carrier。长线 tail-2 和 frame-indexed epoch2 均继续独立运行。

累计 step4000 的配对结果已完成：teacher-forced video/action 为
`0.096700/0.573319`，无泄漏 causal video/action 为 `0.265205/0.831797`。相对 step3000，
causal action 从 `1.099539` 再改善约 `24.35%`，causal video 改善约 `1.35%`；这排除了
“离线改善已在 step3000 饱和”，也进一步说明训练预算确实制约动作预测。按预注册的闭环门，
step4000 晋级同一 Goal task3/trial0/replan16 的一次固定 canary，并排在上述四任务诊断之后自动
执行。闭环成功前仍保持 `NO_GO_LONG`，不能用 MSE 代替成功率。

step4000 固定 task3/trial0/replan16 闭环最终仍为 `0/1`，80步、5次重规划，动作绝对值均值
`0.263`、饱和率 `4.79%`，物体最大位移约 `1.11e-16`。随后累计 step5000 的配对离线结果为：
teacher-forced video/action `0.095529/0.483439`，无泄漏 causal video/action
`0.259440/0.719067`。相对 step4000，causal action 再改善约 `13.55%`、video 改善约
`2.17%`，因此按同一门控晋级不改任务、不改 trial、不改执行周期的固定闭环 canary；成功前仍为
`NO_GO_LONG`。

## 2026-08-13 executed-action history 动作生成线

step3000–5000 的因果 action MSE 持续改善而固定 task3 仍无接触，说明单纯延长冷启动训练不足。
官方 LingBot-VA 固定 commit `7c6ffa9` 在执行后把真实 observation 与 executed actions 回写到
rolling KV；本地此前每次 replan 冷启动。为隔离这个差异，新增16步已执行动作history：训练时只
读取窗口起点之前的真实动作，作为固定clean token，后32步才计算action loss；部署时传入最近16步
真实环境动作。当前阶段显式clean token重算attention，尚不声称等价于持久化KV的计算实现。

history sidecar覆盖train+val全集1712 episodes、277713 actions、222929 windows，仅11MB；对实际
train+val评测清单检查200819 windows，缺失为0。真实H3 2-step smoke的H3/action梯度分别为
`1.608/51.239`、`2.397/62.664`，峰值reserved `41.06 GiB`；history checkpoint保存后严格restore
并完成真实val forward。故放行100-step、global batch8 canary。其父权重固定为step5000，唯一
变量为history条件；未满足因果action再改善至少2%、video退化不超过2%并产生真实物体接触前，
不得扩到用户要求的3000-step长训。
### 2026-08-13 — executed-action history budget ladder

- The first 100 updates improved teacher-forced action MSE from `0.613764` to `0.524416`
  (`+14.56%`) but regressed 4-step causal action MSE from `0.633375` to `0.695905`
  (`-9.87%`); causal video MSE changed by only `+0.65%`.
- This is evidence of train/inference mismatch, not evidence that 100 updates are a universal stopping
  point. Continue the same single-variable intervention to 3000 cumulative updates, checkpointing every
  500 updates and evaluating the full causal learning curve.
- Training permission is `GO_LONG`; effectiveness remains `NOT_EVIDENCE_READY` until causal metrics and
  a fixed LIBERO closed-loop gate pass.

### 2026-08-13 — executed-action history step500 gate

- 累计 step500 的固定 episode-disjoint val40 teacher-forced video/action MSE 为
  `0.093675/0.448326`，4-step causal video/action MSE 为 `0.258240/0.540989`。
- 相对未训练 history 父基线 causal action `0.633375` 改善 `14.59%`；相对 step100 的
  `0.695905` 改善 `22.26%`。causal video 相对父基线 `0.257647` 仅变化 `+0.23%`。
- 因此 step500 通过预注册离线门，已启动与父 checkpoint 完全相同的 Goal task3/trial0、
  replan16、80-step 固定闭环。唯一变量是 checkpoint 中新增并训练的 16-step executed-action
  history；显式 clean history 重算 attention 仍是对 LingBot-VA rolling KV 合约的
  `INTENTIONAL_DEVIATION`，不能称为等价复现。
- 训练许可保持 `GO_LONG`，效果结论保持 `NOT_EVIDENCE_READY`。只有固定闭环出现 success 或
  至少真实物体接触，才扩大 trial/task；若仍无接触，则继续消费 step1000/1500 曲线点，以区分
  history 学习不足与仅改善离线因果误差但不改善控制语义。

固定闭环结果随后为 `0/1`：80步、5次重规划，动作绝对值均值 `0.248`、饱和率 `3.44%`，
但所有物体与抽屉的最大位移仍约 `1.46e-16`。轨迹中末端相对初始位姿移动约
`(+0.086,-0.053,+0.031)m`，说明 history 分支被使用且策略不静止，只是未形成任务相关接触。
因此不扩大该 checkpoint 的 trial；继续到 step1000/1500 获取预注册学习曲线。

同期无 history 的累计 step7000 causal video/action MSE 达 `0.255289/0.518924`，相对 step5000
action `0.719067` 再改善 `27.83%`，触发同一固定闭环。结果仍为 `0/1`，动作绝对值均值
`0.205`、饱和率 `1.25%`，物体最大位移仅 `5.20e-17`。这进一步证伪“继续压低当前 causal MSE
即可自动产生接触”的假设。主线仍按用户要求保留到 step10000，但 step8000/9000 只消费离线
causal 曲线，避免重复同一失败闭环；终点 step10000 再做固定闭环确认。

累计 step8000 的 teacher-forced video/action MSE 继续降至 `0.093060/0.353209`，但 causal
video/action 为 `0.252260/0.732903`：causal action 相对 step7000 的 `0.518924` 反弹
`41.24%`。这是同数据、同 sampler、仅预算增加下的明确训推偏移/过拟合曲线点。当前主线的
causal 最优 checkpoint 暂锁为 step7000，不能以 `latest` 或 teacher-forced loss 自动选择
step8000。step9000/10000 继续完成预注册学习曲线；只有 causal 与固定闭环证据能改变选择。

### 2026-08-13 — executed-action history step1000 gate

- 累计 step1000 的固定 val40 teacher-forced video/action MSE 为
  `0.092413/0.443828`，4-step causal video/action MSE 为 `0.260282/0.604701`。
- causal action 相对 history step500 的 `0.540989` 退化 `11.78%`；相对无 history 父权重
  `0.633375` 仍改善 `4.53%`。causal video 相对父权重退化 `1.02%`，仍在原离线视频门内。
- 该点没有超过 step500，也没有比当前无 history 主线 step7000 的 `0.518924` 更强，因此不重复
  task3/trial0 闭环；step500 已证明该强度下策略会移动末端但不能产生物体接触。继续自动消费
  step1500/2000/2500/3000，以判断 history 曲线是否再次反转，而不是按 latest 选择 checkpoint。
- 训练许可仍为 `GO_LONG`；效果结论仍为 `NOT_EVIDENCE_READY`。当前 history 最佳 checkpoint
  保持 step500。

### 2026-08-13 — frame-indexed second-epoch step1400 probe

- 第二个完整 frame-indexed epoch 在内部 step1400（本轮 `179200` samples，累计约
  `1.645` effective epochs）做了一次只读、episode-disjoint val40 提前探针；action loss 为
  `0.089823`。
- 相对第一 epoch 终点 step2170 的 `0.109190` 改善 `17.74%`，相对第一 epoch step1600 的
  `0.108532` 改善 `17.24%`。因此“第二 epoch 已明显过拟合/无继续价值”被当前证据否定，保持原定
  2170-step 预算与终点 watcher。
- 该探针没有重复闭环：第一 epoch step200/400/800/1600/final 各10次、累计50次固定 task3
  rollout 已全部失败。第二 epoch 终点才复用同一10-trial协议；此前仍为
  `NOT_EVIDENCE_READY`。

### 2026-08-13 — deeper DoT carrier implementation gate

- FastWAM 官方代码固定 commit `45d8e1458921d83f8ad6cf9ce993d371208dabd0` 的 resolved model
  config 使用 30 层、hidden1024/FFN4096/24头×128 的 ActionDiT，并由 Wan2.2 video DiT 对动作
  backbone 做顺序线性插值和 alpha scaling；只把 action encoder/head 保持随机。
- 当前 H3 frame-indexed DoT 父线是 hidden1024/FFN4096/24头×128，但只有1层且随机初始化。
  因此后续分成两个受控变量：先比较 `1层随机 → 4层随机`（容量），若机制门通过再比较
  `4层随机 → 4层H3插值`（初始化），不得合并归因。
- 已实现 checkpoint 自描述的任意 DoT 深度，以及 H3 50层到浅层 carrier 的均匀深度采样；4层固定
  选取 `[0,16,33,49]`。插值复用现有 H3 ActionDiT/FastWAM 的逐维线性规则与 alpha scaling，
  动作 encoder/head 保持随机。静态形状、每层梯度和初始化映射测试已通过；真实H3 2-step及严格
  restore 仍是长训放行前的未解决门。
- 1层 carrier + KV fusion 为 `72.67M` 参数，4层为 `135.67M`。本实验是 H3 backbone port 的
  `controlled_ablation`，不是 FastWAM 官方复现；当前只具备 `GO_CANARY` 实现许可，效果仍为
  `NOT_EVIDENCE_READY`。

真实 H3 depth4-random 2-step smoke 随后完成：global batch8、16 samples，两步 loss finite，梯度
范数 `3.328/3.609`，每卡峰值 reserved `17.516 GiB`。保存的 stage 自描述
`action_layers=4`；恢复命令不传 `--action-layers`，仍严格装载4层并完成真实 val forward，action
loss `1.013943`、峰值 reserved `14.225 GiB`。因此容量线通过 finite-gradient、显存和 restore 门，
训练许可提升为 `GO_LONG`，效果仍为 `NOT_EVIDENCE_READY`。正式父/子比较保持同一 v8 manifest、
global batch128、2170 steps、277760 samples（`1.000169` epoch）、LR `1e-4` cosine、H3冻结，
每200步保存；唯一架构变量为动作 carrier/KV fusion 深度1→4。

### 2026-08-13 — executed-action history step1500 gate

- 累计 step1500 的固定 val40 teacher-forced video/action 为 `0.091506/0.381671`，4-step causal
  video/action 为 `0.255432/0.471264`。
- causal action 相对 history step500 的 `0.540989` 改善 `12.89%`，相对 history step1000 的
  `0.604701` 改善 `22.07%`，相对无 history 主线最佳 step7000 的 `0.518924` 改善 `9.18%`；
  causal video 同时比 history step500 改善 `1.09%`。
- 它成为当前所有 shared-H3 里程碑的最优 causal checkpoint，并通过预注册离线晋级门。排队执行
  与 step500 完全相同的 Goal task3/trial0/replan16/80-step 固定闭环；在真实 success 或至少物体
  接触出现前，效果仍为 `NOT_EVIDENCE_READY`。

固定闭环最终仍为 `0/1`：动作绝对值均值 `0.239`、饱和率 `3.54%`，末端沿约 `0.383m` 路径
从初始位姿净移动约 `(+0.338,+0.004,+0.027)m`，但最大物体关节位移仅 `8.82e-17`，没有接触。
这说明 history 条件下的策略并非静止，且当前全局最佳 causal MSE 仍未转化为任务相关控制。
step2000/2500/3000 继续用于完成预算学习曲线；除非出现新的机制变化，不逐点重复同一闭环。

2026-08-13 代码审查确认该闭环把 LIBERO environment-domain history 直接用 dataset-domain
quantile 归一化，夹爪 open/close 条件反转；episode 开头的左 padding 也被当成真实动作。因此上述
`0/1` 与“无接触”只保留为旧实现实际输出，撤销其对 executed-history 机制的否定性推论。离线
teacher-forced/causal 数值不经过 environment codec，可作为旧 checkpoint 的描述保留，但该 shared
checkpoint 同时受 replicated-gradient 未同步影响，不能支持 global-batch-8 方法结论。

同一时段完成 depth4-H3init 的真实机械 gate。首次运行在构造 `H3DoTWAM` 后才读取 H3 源 block，
而构造函数已把源 block 移入 hub wrappers，导致源深度为0；修正初始化顺序并增加回归测试后重跑。
修正后的 `[0,16,33,49]` 插值在2步内无 NaN/OOM、显存仍为 `17.516 GiB`，但第一步 action loss/
gradient norm 已为 `2.5846/604`，第二步爆到 `382.7455/9920`；同一数据、LR与容量的随机父线仅为
约 `1.22–1.28/3.33–3.61`。因此 residual output scale `0.01` + LR `1e-4` 的直接 H3 插值初始化
判定 `NO_GO`，不放行长训。任何 `0.001` 缩放或更低 LR 都必须另立稳定化消融，不能覆盖这条负结果。

### 2026-08-14 — shared-H3 distributed contract v2

- 旧 shared/history 训练均已到达预注册终点：tail-2 step10000 的 teacher-forced
  video/action 为 `0.091899/0.285537`，旧 sampler causal video/action 为
  `0.251560/0.464778`；history step3000 分别为 `0.090237/0.346457` 与
  `0.257231/0.632286`。这些 checkpoint 继续标记
  `TAINTED_FSDP_REPLICATED_GRAD`，不能成为修复版父权重或证明 global-batch-8 方法有效。
- 修复版 shared-H3 从干净初始化在8×A800完成真实2-step：16 samples，H3/action梯度均finite且
  非零，step1跨rank replicated参数一致性断言通过，每卡峰值reserved `40.934 GiB`；step1/final
  v2 checkpoint均落盘。final恢复后的两次val40具有完全相同的40样本loss序列，save/restore门通过。
- 第一遍真实forward发现 trainer 漏传 `clean_video_valid`；attention正确拒绝了半套validity合约，
  没有生成checkpoint。补齐成对mask并新增cold-start双流隐藏测试后，云端50项定向测试及真实硬门
  全部通过。
- 修复后的step0 teacher-forced video/action为 `0.170026/1.304397`，严格隐藏尚未生成clean token的
  causal sample40为 `0.695563/1.306988`。s2 causal为 `0.706956/1.305335`：action仅改善
  `0.127%`，video退化 `1.64%`，所以机械门通过但效果仍为 `NOT_EVIDENCE_READY`。
- 已预注册 `h3_lingbot_shared_sync_v2_s1000_v1`：1000 steps、global batch8、8000 samples、
  `0.039845` effective epochs、每200步保存。晋级门为 causal action至少改善3%且causal video
  退化不超过5%；训练许可 `GO_LONG`，效果结论仍 `NOT_EVIDENCE_READY`。
- 配对 adapter-only 机械门也已通过：冻结 shared H3/proj_out，仅训练同步后的 action adapters；
  2-step action梯度 `1.4526/1.7654`、post-step1跨rank差为0，显存峰值reserved `35.73 GiB`。
  checkpoint现显式记录并严格校验 `freeze_shared_blocks`；8-rank恢复后差为0，val40
  video/action为 `0.170031/1.303723`。因此放行同seed、同8000 windows、同预算的1000-step
  配对对照，唯一变量是是否更新H3 tail-2；这条对照同样为 `GO_LONG / NOT_EVIDENCE_READY`。

### 2026-08-15 — C17/C18 frozen-H3 progress shadow

- C17复用FACT commit `618a6c16868699b6d4138941de6a863589ac00dd` 的time-to-go意图，但
  offset32是本项目适配，不是官方复现。4000 train/2000 episode-disjoint val上，task+step+H3相对
  task+step MAE改善29.3%，只证明冻结H3 K/V含专家阶段信息。
- 固定D0-H32-s14000/replan8的16条闭环shadow中，首动作块和outcome均16/16复现；C17最终值AUROC
  `0.1875`，失败轨迹被absolute-step时间捷径压到0，判定`FAIL_PROGRESS_SHADOW_GATE`。
- C18唯一变量是删除absolute-step。离线MAE `0.21545→0.09952`、R² `0.00418→0.74192`；同一
  16条闭环AUROC提高到`0.546875`，仍未达到`0.65`，判定`FAIL_PROGRESS_SHADOW_GATE`。
- 两轮均未更新任何动作/H3参数，不涉及global batch、effective epoch或训练checkpoint；各16 episode、
  最多6400环境步、实测344秒。结论是成功专家time-to-go不足以训练闭环停滞critic；停止该标签路线，
  action-conditioned ranking继续`NO_GO`，直到具备failure onset或counterfactual action outcome。

### 2026-08-15 — C19/C20 counterfactual branch restore contract

- C19证明旧轨迹的LIBERO flattened state（time/qpos/qvel）能max-abs0写回，但恢复观测不等于旧轨迹
  缓存；旧轨迹不能支持“原观测精确接管”声明。
- C20首跑发现LIBERO `seed()`修改process-global RNG，两个env顺序reset会得到不同world layout；该
  harness失败保留在`c20-libero-branch-repeatability-v1`，不计方法结果。
- v2在每次reset前重设seed42：四suite各3个状态、每状态两独立env执行同一8步chunk；起点和终点
  双相机图像逐像素一致，proprio/state差≤`1e-10`，steps/success一致，通过
  `PASS_MULTISUITE_BRANCH_REPEATABILITY_GATE`。
- 训练许可仅为`GO_CANARY`采集小规模规范化分支outcome；效果仍`NOT_EVIDENCE_READY`。在真实
  alternative-action outcome、episode-disjoint split和ranking gate完成前，critic/best-of-N仍禁止。

### 2026-08-15 — C21 same-state alternative-action outcome canary

- 可证伪假设：固定规范LIBERO state、环境seed42、`D0-H32-s14000/replan8/no-ensemble`父模型，仅改变
  policy diffusion-noise seed，必须使四组首动作块全部不同，并至少产生一组同状态混合成败结果。
- 冻结选择为四suite各1个state、每state 4个noise offset；两波各占8张A800，最多6400环境步。
  唯一变量是policy noise；checkpoint、H3 INT8权重、cache、task/trial、branch state、环境seed、
  replan、horizon和无ensemble均固定。
- 真实命令：
  `nohup bash /mnt/h3-wam/candidate-d0-rollout-96976ce/project/scripts/h3wam/launch_c21_counterfactual_outcome_canary.sh > /mnt/h3-wam/logs/c21-counterfactual-outcome-canary-v1-launch.log 2>&1 &`
- 结果：Goal `4/4`、Object `3/4`、Spatial `0/4`、LIBERO-10 `4/4`，合计`11/16`；Object是唯一
  mixed-outcome组。各组最小首动作块RMS为`0.22723/0.14624/0.27201/0.09239`，全部通过
  `>1e-6`动作多样性门；墙钟`280s`。
- 原始判定为`PASS_COUNTERFACTUAL_OUTCOME_CANARY / GO_DATASET_EXPANSION / NOT_EVIDENCE_READY`。
  后续代码审计发现base seed同时改变所有后续replan，故`GO_DATASET_EXPANSION`被收窄为
  `GO_ENTROPY_CALIBRATION_ONLY`；本轮不能构成首动作的因果outcome监督。
- artifact：`/mnt/h3-wam/eval/c21-counterfactual-outcome-canary-v1/COMPLETED`；SHA256
  `c258cb829c45e504e03e5a183008d2820a1a32152bb8ff70723ba1acf8895f8c`。

### 2026-08-15 — C22 multisuite counterfactual entropy sweep（预注册）

- 父模型固定为`D0-H32-s14000/replan8/no-ensemble`。选择四suite各2个成功源episode，在距离原成功
  终点`1/3/5`个replan处建立24个规范state组；每组4个预注册noise offset，共96条分支。
- 组内唯一变量为policy diffusion-noise seed；环境seed42、checkpoint、INT8 H3、cache、task/trial、
  state、replan和horizon固定。跨组改变suite/source/distance只是用于定位高熵采样层，不作模型对比。
- 可证伪门：24组首动作块必须全部pairwise RMS>`1e-6`，且至少4个mixed-outcome组、覆盖至少2个
  suite。通过仅放行`GO_TARGETED_COUNTERFACTUAL_DATASET`；失败则`NO_GO_UNIFORM_DATASET_EXPANSION`。
- 预算：4节点×8 A800，计划每节点24条/3 waves；最多38400环境步，预计新增约270MB。
  后续数据划分必须以源episode为单位，严禁同一源轨迹的相邻state跨train/validation。
- 效果状态预注册为`NOT_EVIDENCE_READY`；无论本轮是否通过，都不能直接宣称critic或best-of-N有效。

### 2026-08-15 — C21/C22 causal-label boundary audit 与 C23机制

- 审计`rollout_libero.py`确认旧调度为`episode_seed + replans`。C21/C22对base seed加offset会让首动作和
  所有后续replan同时变化，适合测state-level stochastic continuation entropy，但最终成败不能因果
  归给`first_environment_action_chunk`。
- C22继续完成其已预注册的高熵时间带校准；不修改运行中代码、不更改阈值。其artifact即便输出
  `GO_TARGETED_COUNTERFACTUAL_DATASET`，解释也被本审计收窄为`GO_CAUSAL_FIRST_ACTION_CANARY`。
- C23新增两段seed合同：`first_policy_noise_seed`只用于replan0；从replan1起使用固定的
  `continuation_policy_noise_seed_base + replans - 1`。同组候选仅首动作seed不同，后续seed逐值一致。
- 本地机制门：项目`.venv`运行`tests.test_h3_dreamwam_kv_rollout_adapter`共10项通过，包括两个不同
  first seed在replan1/4解析到完全相同continuation seed。真实闭环仍等待C22完成后选择高熵state。

### 2026-08-15 — C22/C23 闭环结果

- C22完成96/96：`71`成功、`7/24`同状态mixed group，覆盖全部四suite；动作多样性24/24通过。
  高熵层为LIBERO-10 d1/d3/d5、Goal d3、Object d3/d5、Spatial d5。正式报告判定
  `PASS_COUNTERFACTUAL_ENTROPY_SWEEP`，但按因果审计只放行C23，不直接形成critic标签。
- 30234的C22首波因`/tmp/h3-wam-libero-site`缺失，8条在import阶段0结果退出；从共享固定
  `/mnt/h3-wam/runtime/libero-site.tar`恢复并通过真实LIBERO import后，保留traceback并只重跑shard3。
- C23 preregistration SHA256为`ffd760d84c7f2937ee5258fb8b1da0d2b8d7319620e2008322b7f90bd7330492`，
  selection SHA256为`c7b1bd905f379b072d449a2972c4245e7009d73565759742e824feb940b298b4`。
- C23机械smoke：首动作与C22 bit-exact、后续seed为固定`20000000..20000011`，但相同首动作的结果
  从C22成功变为C23失败，证实C21/C22完整noise schedule标签不能冒充首动作因果标签。
- C23正式32/32完成：`18`成功；只有Spatial task0/trial3/d5为`2/4` mixed。全部首动作bit-exact、
  全部continuation schedule合法、8/8组动作不同；四shard墙钟`137/139/160/161s`，判定
  `PASS_FIRST_ACTION_CAUSAL_CANARY / GO_EPISODE_DISJOINT_CAUSAL_DATASET_CANARY`。
- 效果仍为`NOT_EVIDENCE_READY`。下一数据门必须扩大源episode覆盖并按源episode隔离split；训练
  critic后还需held-out within-state ranking与固定LIBERO闭环胜率两级门。
- C22/C23 artifacts分别为`/mnt/h3-wam/eval/c22-counterfactual-entropy-sweep-v1/COMPLETED`和
  `/mnt/h3-wam/eval/c23-first-action-causal-canary-v1/COMPLETED`；SHA256分别为
  `05f18c76e9c460e06f8e4290f7a3332d2cf784ee35022dcb1973f63644cc3978`、
  `11dfef6ce6523ac58f8cb8aae7166e81173b2e3e7724b95332b9ab0348b4143f`。

### 2026-08-15 — C24 first-action execution-horizon sweep（预注册）

- 动机：C23固定后续随机性后仅`1/8`组mixed，可能是父策略每8步重规划把首动作差异快速纠正。直接
  扩大几百条数据前，先测动作候选需要执行多长才产生可学习的因果结果差异。
- 父项为C23 horizon8；复用同8个state、32个first seed和逐值相同的continuation seed schedule。
  两个独立challenger分别只把首chunk执行长度改为16或32；从第二次replan起仍固定执行8步。
- 每个challenger 8组×4分支、32条，合计64条、最多25600环境步；四节点各8 A800，按challenger
  串行跑，避免GPU oversubscription。
- 可证伪门：首动作必须32/32 bit-exact于C23、continuation seed合法、8组动作均不同；至少一个
  challenger达到`>=3` mixed groups并覆盖`>=2` suites。通过才按胜出horizon扩episode-disjoint数据；
  失败则停止“单纯延长首动作”路线。
- 效果状态固定为`NOT_EVIDENCE_READY`；该轮只比较标签可辨识性，不宣称LIBERO成功率提升。

### 2026-08-15 — C24 结果

- horizon16：`17/32`成功，Object d5为`2/4`、Spatial d5为`3/4`，共2个mixed组覆盖2 suites；
  未达到至少3组门。四shard墙钟`141/158/140/141s`。
- horizon32：`16/32`成功；LIBERO-10 task3/d1为`3/4`、task5/d3为`2/4`、Spatial d5为`3/4`，
  共3个mixed组覆盖2 suites，通过预注册门。四shard墙钟`139/156/139/139s`。
- 两候选首动作均32/32 bit-exact于C23，continuation seed schedule和动作多样性均通过；唯一变量确为
  首chunk执行长度。判定`PASS_FIRST_ACTION_HORIZON_SWEEP`，选择h32用于因果数据集canary。
- 边界：h32仅提高标签可辨识性，尚未证明用h32部署、训练critic或best-of-N可提高LIBERO成功率。
  artifact SHA256为`a85d7d0dd03c355906cfa5f8277b8abf3e1d2675b5291153a1fae03e8b53f54e`。

### 2026-08-15 — C25 episode-disjoint causal dataset canary（预注册）

- 固定C24胜出的首chunk执行32、后续replan8与同组continuation seed；组内唯一变量仍为first seed。
- 选择14个新的/隔离源episode、32个state组、每组4候选，共128分支；train22组、held-out val10组。
  Goal/Object/Spatial各4个源episode，LIBERO-10因父策略仅2个成功源episode而各分配train/val一个。
- 同一源episode的所有distance/state/branch强制留在一个split；Goal/Object/Spatial的val还选择不同
  task以加强泛化审计。C23/C24用过的Goal/Object/Spatial源episode不进入C25。
- 预算为4节点×8 A800、每节点32条/4 waves、最多51200环境步。
- 放行门：机械合同全过，mixed group总数`>=8`、train `>=4`、val `>=2`，且覆盖`>=3` suites。
  通过才允许冻结H3/action父策略训练小critic canary；失败不训练通用critic。
- 效果仍固定`NOT_EVIDENCE_READY`；该门只判断数据是否足够支持可审计训练。

### 2026-08-15 — C25 结果

- 128/128完成，`70`成功；32组中9组mixed，train6、held-out val3，覆盖LIBERO-10、Object、
  Spatial三suite，达到预注册`8/4/2/3-suite`门。Goal 8组为5个4/4、3个0/4，无mixed。
- mixed明细：train为Object task2/d3 `3/4`、task4/d3 `2/4`，Spatial task3/d5 `2/4`，
  LIBERO-10 task5 d1/d3/d5 `2/4,1/4,1/4`；val为Object task3/d5 `3/4`与LIBERO-10
  task3 d1/d5 `3/4,1/4`。
- 14个源episode的split完全隔离；所有seed schedule合法，32/32组首动作不同。四shard各32条，
  墙钟`512/506/514/515s`；新增artifact约324MB。
- 判定`PASS_EPISODE_DISJOINT_CAUSAL_DATASET_CANARY / GO_FROZEN_H3_ACTION_CRITIC_CANARY`。
  下一步只允许冻结父策略训练小critic并做held-out within-state ranking；critic、best-of-N与闭环效果
  继续`NOT_EVIDENCE_READY`。
- preregistration/selection SHA256为
  `2b8adcb97ebe01be70c1fee47a5f19e54d549d6019b1ebe31b801e832185a3ae`、
  `73ae9d08675756c145cdc13d749d14fbca84b01e12c1ec3094950b5921d14b5a`；完成报告SHA256为
  `641ce0fb53853c555da2ad69cbe1b2d6451faec470c362c6db1ad7aef3fa4165`。

### 2026-08-15 — C26 frozen-parent causal action critic

- 训练前重新拉取FACT官方main：远端从固定`618a6c1`前进到`9427ea4`，唯一diff为README增加
  Hugging Face live-demo链接，没有训练、value、failure或best-of-N代码变化。因此实验继续固定
  `618a6c16868699b6d4138941de6a863589ac00dd`，不把README更新冒充方法更新。
- C25原始结果再次做输入合同冻结：128/128的`results.json`首动作块与轨迹第0行逐字节一致；32/32
  组的起始双相机、sim state与proprio组内逐字节一致。后续replan行明确排除，防止结果后状态泄漏。
  冻结dataset SHA256为`4cde59a044b6fb2a27b7f1e6c2f3a01dd394c147d730e31b987c90268b61e785`。
- 对32个规范起点用真实在线INT8 H3链路重算两类冻结输入：D0 layer49 K/V 512D compact和FACT训练
  合同的layer49 hidden `1×32×5376`；峰值显存`29.679GiB`、耗时`15.78s`，feature SHA256为
  `ead7842d58317a3193b8863fd8b65b0f17bd8d46d3e486a01e1e974b4f0d1af3`。
- 三臂均只训练线性pairwise head，H3、D0 action parent和FACT consequence expert全部冻结。每臂21个
  正pair/42个平衡样本，1000 full-pair steps=`1000` effective pair epochs、`42000` examples seen；
  三臂训练都到21/21，排除“没收敛”。
- untouched C25 validation结果：action-only `0/9, top1 0/3`；H3×action `4/9, top1 2/3,
  exact p=0.6875`；FACT consequence `0/9, top1 0/3`。判定
  `FAIL_C26_HELDOUT_WITHIN_STATE_RANKING / NO_GO_BEST_OF_N`，完成报告SHA256为
  `4b5319bc679928125b45b33637336f3d6c25f2689af62d426152da9d3213ce27`。
- 失败后的调参只在6个train mixed groups做leave-one-group-out，固定grid为steps
  `{10,30,100,300}`、weight decay`{0.03,0.3,3.0}`；H3×action选中`10/3.0`，LOO为`18/21`、
  top1 `6/6`，但已看过的C25 val仍仅`4/9`，只允许作为C27预选配置，不能回写C26结论。

### 2026-08-15 — C27 fresh expanded causal dataset 与 C28确认

- 从固定父策略`D0-H32-s14000/replan8`的57个成功源episode中，在查看新分支结果前排除C22/C25
  用过的18个源episode，剩余39个全新源episode：Goal5、Object22、Spatial12。LIBERO-10父策略仅有
  两个成功源且均已被旧轮使用，因此本轮不伪造“fresh LIBERO-10”声明。
- 每源episode固定距离成功终点3/5 replans两个state、每state四个first-noise候选；首chunk执行32，
  后续replan8且continuation seed逐值固定，共78组/312分支、最多124800环境步。
- split按suite内source key哈希在结果生成前冻结：29 train source、10 untouched val source；同源所有
  state/branch不跨split。放行门为train mixed>=10、val mixed>=4且覆盖Goal/Object/Spatial三suite。
- 四台8×A800分别运行一个noise-offset shard，避免一台机器同时跑同状态四候选；preregistration与
  selection SHA256分别为`6711ee8b30f9c85410e598255e9aea2e741269ce9d4594a563a898e794719a39`、
  `c4712c6d0dbbd4477c1fac2861643387ca279a2ed7bb158eb98c5a3632d1107c`。结果生成期间未改门槛。
- C27最终312/312完成、198成功；78组中17组mixed，train13组/42对、fresh val4组/12对，train和val
  mixed均覆盖Goal/Object/Spatial。机械合同、源episode隔离、seed schedule与动作多样性全部通过，判定
  `PASS_C27_EXPANDED_CAUSAL_DATASET`。完成报告SHA256为
  `d2c33b99b7281a2ad8f6310def9e482bf4a1d8c3235f89ed3bfafa923d6fd6f0`。
- 冻结dataset重新审计312条结果/轨迹及78个bit-exact起点，SHA256
  `585d3060095c3a797304eae3e81971b4aa2991bc22e48f072b1b123b6de253d0`。78个真实在线INT8 H3 layer49
  特征在32611重算38.24秒、峰值29.68GiB，SHA256
  `9a3b2c51746968edbf8f8695491960721919afd773b501729c461f34ee1c875b`；没有使用ComfyUI或旧结果缓存。
- C28严格复用C25 train-group LOO预选的10 steps/LR0.03/WD3.0，不读取C27 val调参。C25全部转train后
  与C27 train合计22个mixed组/72对；每臂每步144个平衡样本，10 steps=`10` effective pair epochs、
  1440 examples seen，H3/action父模型均冻结。
- C28训练上H3×action为65/72、action-only为53/72，但fresh C27 val反转为H3 `6/12`、top1 `2/4`、
  exact permutation `p=0.5859375`，action-only为`7/12`、top1 `2/4`。七项预注册门仅score variation和
  shuffle margin通过，判定`FAIL_C28_FRESH_HELDOUT_WITHIN_STATE_RANKING / NO_GO_BEST_OF_N`。
  artifact/checkpoint SHA256分别为
  `718abe0b47e47b7b456c19205f6c2a807b3c312f1ab88fb1ac14f3e732be0558`、
  `f0cbf0836c4e90553ceaf1586a38fc0bc6f552caf14a4dc56669346a4d749f11`。
- 该失败排除“C26只有21对/训练不足”作为主因：监督增至72对后训练拟合更强、fresh泛化仍失败。停止用
  已消费C27 val调静态H3×action critic；下一条 consequence 路线必须显式学习候选动作导致的短期动态，
  且重新采集未消费源episode作验证。
- 基础设施事件：首次手工特征命令遗漏正式rollout脚本固定设置的
  `LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64`，导致32409上的INT8 H3前向报cuBLAS
  初始化错误；不带该环境的探针在30907/32409/30234也失败。补齐环境后，三节点最小BF16 matmul全部
  通过，故不是GPU/节点故障。C27特征与C28训练已在32611完成，模型结果不受此启动合同事件影响。

### 2026-08-15 — C29/C30 action-conditioned consequence 数据转向

- C28失败后重新逐行对齐官方代码：FACT `618a6c1`在teacher forcing和stage2 inference中把clean action
  作为future state/value/video的K/V-only条件；MiniWorld `e484206`把每4个原始动作对齐到一个视频
  latent frame，并经action encoder与AdaLN-LoRA注入。当前C28仅有静态state×action特征，不等价于
  action→future机制，因此停止给静态critic增加步数。
- C29保持`D0-H32-s14000/replan8/no-ensemble`完全不变，完整补跑四suite、tasks0..9、trials4..7。
  160/160完成、61成功：Goal8、Object30、Spatial17、LIBERO-10 6；全部预注册源门通过。四节点各40条，
  墙钟`989/980/950/984s`，artifact SHA256
  `132c2409753f15a75f5c836590d8a760d2a31e9e913422f7973e11fb654868c6`。这只是新鲜源库存，不是候选对比。
- 旧branch轨迹只保存重规划前观测，C27有122个首chunk内成功分支仅1行，若训练future target会系统性
  丢掉最快成功样本。后续rollout新增独立terminal双相机/proprio/sim-state/step/action字段，不改变旧行轴、
  动作或控制；本地12项rollout合同测试通过。C30强制每branch存在terminal字段：有row1时使用start+32
  观测，否则只允许首chunk内terminal作为直接动作后果。
- C30在查看branch outcome前冻结C29的61个成功源：每源距离终点3/5两个state，共122组/488分支；
  suite-stratified source split为45 train/16 val。组内唯一变量仍为first action seed，continuation seed逐值
  相同；数据门为train mixed>=10、val mixed>=4、覆盖>=3 suites且488/488动作后观测合法。
  preregistration/selection SHA256为
  `3fc656ef9e24abdb5be843ee199d16bee8a9f915b1c9666f6c5600a23cf14c78`、
  `d2cc42f650e482fc09651ecd5f463fcc34cb3d02a51a8b98bd2ab8d4bb5c1e8a`；训练前仍为`NOT_EVIDENCE_READY`。
- 已实现新的temporal consequence adapter：32步动作按MiniWorld合同变为8个有序token，future query按
  FACT意图只读clean-action K/V与冻结当前H3/state；candidate action在模块边界detach。flattened与
  temporal两种模型可使用相同future-H3 trainer和三臂控制。顺序敏感、动作梯度隔离、finite与旧future-H3
  测试共6项通过；只有C30数据门通过后才预注册同数据同预算训练。
- C30运行期间进一步审计出首chunk内提前成功的动作合同：policy虽一次提出32步，但环境只执行到
  terminal step，后半段不能称为clean executed action。C31因此同时保留完整proposal用于候选身份，并按
  `future_step-current_step`记录`executed_action_steps`；未执行尾部显式zero-mask。该修正不改变正在运行
  的C30 rollout或门槛，只修正随后冻结的数据语义；本地相关9项测试通过。
- C31训练器已在结果未知时完成：对每个source-disjoint train/val branch使用真实current H3、masked
  executed action和真实future H3，future仅作label；三臂为正确动作、同一bit-exact state内循环错配动作、
  zero action。flattened与MiniWorld对齐的8-token temporal使用同样的10000 steps/batch64、每1000步
  checkpoint，并做fresh restore。只有correct arm在未见source上同时击败两个控制且同状态shuffle使其
  退化至少1%，才放行后续value ranking；C30未完成前状态仍为`HOLD_DATA_GATE`。
- 为避免同一validation被连续用于“选consequence结构”和“证明动作ranking”，C31在未读取C30 outcome时
  又冻结了三级source用途：只从原C30 train源按suite哈希划出约20%作consequence validation；其余作
  consequence train；原C30的16个val source全部标记`reserved_ranking_val`，C31 future-MSE训练与选型
  均不读取。这样后续value head可用原train mixed groups训练，并仍有完整的fresh-source ranking终验集。
- C30最终488/488完成、317成功，122组中28组mixed并覆盖四suite；动作多样性、source split、seed
  schedule和488个post-action consequence全部通过，147条首chunk内成功使用独立terminal观测。但预注册
  要求validation mixed `>=4`，实际train/val为`25/3`，故严格判定
  `FAIL_C30_ACTION_CONDITIONED_CAUSAL_DATASET / NO_GO_ACTION_CONDITIONED_CONSEQUENCE_TRAINING`，不把门槛
  事后从4降到3。artifact SHA256为
  `758b24be6197a520850ff3ec6190139eda13737c03d92a9a8eeddbe491d28578`；自动C31冻结已按设计停止。
- 新建C32而不消费C30 val调参：固定同一D0-H32-s14000/replan8/no-ensemble父策略，完整枚举从未进入
  C21–C30的trials8..11、四suite×10 tasks×4 trials=160 episodes。C32仅建立新鲜source库存；成功数
  与per-suite门在结果前固定，后续新的causal branch仍需独立预注册。四个trial可在四节点各8卡并行。
- C33采集器与审计器已提前实现但保持`HOLD_PARENT_GATE`：只有C32通过后，才从其成功源各取距成功
  d3/d5 state、每state四个first-action seed，后续seed逐值固定。C33全部source标记
  `fresh_ranking_val`，不得训练或选择consequence结构；放行门固定为fresh mixed groups `>=8`且覆盖
  `>=3` suites，同时要求动作/seed/terminal consequence机械门全过。
- C32已完成160/160并通过：52成功源，其中LIBERO-10/Goal/Object/Spatial为`4/9/25/14`；四节点
  trial8/9/10/11均完整，artifact SHA256
  `158b57c012e956a44a542e61e58238aa640b6f5d3abbd35d7d9389d6f62dc3e2`。C33据此冻结52源、104 state、
  416 branches；preregistration/selection SHA256分别为
  `94aeb3b69d7b03fd3852c6ced1654012be14379c9707da1350c573907c4a7481`、
  `19ebca8a5af70971404a03af6ca16681c16f4ca32011fde45fcfc92b73ff15dc`。
- C33自动prepare watcher的shell内联Python引号被远端shell剥离而报SyntaxError；C32 artifact本身正常，
  随后用相同已提交脚本手工执行并得到上述固定哈希。该事件发生在任何C33 outcome前，只影响自动化，
  不改变selection、数据门或模型结论。
- C33运行期间提前完成C34组合冻结器：C30只保留45个原train source/90 states/360 branches，并继续按
  结果未知时固定的37/8 source划分训练与consequence validation；C30原16个旧val source全部排除。
  C33若通过，其52个source/104 states/416 branches全部只作fresh ranking validation。组合数据预计为
  194 states/776 branches，在线INT8 H3需抽取970个current/future样本；C34格式、特征路由、masked
  executed-action loader及现有future-H3模型共12项测试通过。C33未通过前保持`HOLD_C33_GATE`。
- C35训练矩阵也已在C33 outcome未知时固定：flattened/temporal各跑seed42与314159，共4个独立job；
  每job内部为correct、同状态错配、zero-action三臂，steps10000、batch64、LR3e-4、WD1e-2，每1000步
  保存并做fresh restore。四job分散到四节点GPU0，避免共享CPU projection争用；C34真实hash和long
  dossier未通过前launcher硬拒绝启动。
- C33运行到ordinal300时，32611的policy server随机IPC端口发生`Address already in use`，该branch在环境
  rollout前退出，归类为infra而非失败outcome。原残缺目录完整移到
  `/mnt/h3-wam/incidents/c33-port-collision-20260815-ordinal300/`；launcher补充严格result-based resume：
  已有`results.json+trajectory`的branch保留并跳过，残缺目录必须先显式归档，禁止静默覆盖。shard0随后
  用完全相同selection/seed/state恢复，其他三shard未重启，数据合同不变。

### 2026-08-16 — C33–C41 consequence机制、fresh ranking与扩数据决策

- C33最终完成`52 sources / 104 groups / 416 branches`，其中279条成功、24个fresh mixed group，覆盖
  四个LIBERO suite；动作多样性、逐值相同continuation seed、terminal consequence与结果/轨迹动作一致性
  全部通过。完成artifact SHA256为
  `0bee853840820ba8fdeb5635759a8dea6fe7b1c39bfcac4e2bac7a5766929514`。
- C34冻结组合数据为194 states/776 branches；37个source作consequence train、8个source作已消费的
  consequence validation、C33的52个source保持ranking-only。dataset与真实在线INT8 H3 feature SHA256
  分别为`2a6c9252b8e77975f58920425bc18110fa8ea63bdc12c4c15571cfffeb9f7459`和
  `4655d0783a443b4817f93f5368f7cace9996be7163dfa49b2080033b3b176f69`。
- C35/C36/C37严格门均FAIL：独立null模型不稳定、delta按训练std缩放没有解决、10% condition dropout使
  shared-null支路过强且跨seed不稳定。因此不继续围绕归一化或dropout做局部调参。
- C38用四个全新seed复现无dropout temporal consequence：最小same-model true-vs-null增益`12.629%`、
  最小shuffle退化`1.765%`、最小shuffled-train增益`7.481%`。restore初审的假失败来自原模型`.eval()`而
  restored模型留在`.train()`；修复后四个state dict逐tensor相同、eval-vs-eval输出max-abs0。机械修正
  artifact SHA256为`c5709b8d2009deebff42a801592618305d30887ae17ed6fccf2d8a7d862818f5`。
- C40只用22个train mixed groups/74 pairs拟合，首次且仅一次消费C33 final：consequence ensemble fresh
  pairwise为`54.321%`、top1为`58.333%`、permutation `p=0.303297`，未过60%/60%/0.05门；但相对
  action-only的`40.741%`高`13.58`个百分点。结论是动作进入H3 consequence的方向有信号，但价值排序
  泛化和统计功效不足；禁止在已消费C33上继续选配置。report SHA256为
  `9a3148b04d412ee0d3307bf352249c2a762e8def56693926daa23fac48b1235c`。
- C41随后完整采集未消费trials12..15共160个固定父策略episode并PASS：63个成功来源，Goal/Object/
  Spatial/LIBERO-10为`7/28/22/6`，所有预注册suite门通过。trial12..14保持train-source角色，trial15
  保持future-final角色；artifact SHA256为
  `c0adf899674e9b52fe3c90c164b57a1b994c78784a64f2894a491946cf501c8a`。
- 为避免只有一个final trial，C42在查看新增outcome前把角色扩成各五个trial：train为
  `12,13,14,16,17`，final为`15,18,19,20,21`。新增trials16..21已由四节点自动接力；只有每个角色
  `>=36`成功source且覆盖`>=3` suites才允许C43。
- C43也已在C42 outcome前预注册：所有成功source取d3/d5状态、每状态4个first-action分支；组内仅首动作
  diffusion seed变化，后续seed逐值相同。train与fresh-final各需`>=24` mixed group且覆盖`>=3` suites，
  才允许C44重新训练和一次性验证powered consequence-value ranker。整个接力逐级FAIL即停止，不绕门。

### 2026-08-16 — C42–C46 powered ranking与闭环否证

- C42完成固定父策略trials12..21共400/400 episode、141成功；预先冻结的ranker-train/future-final各为
  `70/71`个成功source，四suite均覆盖。完成artifact SHA256为
  `86146292dd51df967ebb06a4624a784d0ebc98d937d18010f5d814fefa9df072`。
- C43从141个成功源冻结282个bit-exact状态和1128个first-action分支，最终716成功；ranker-train与
  fresh-final分别产生25/30个mixed group，所有机械动作、seed和source-role门通过。ordinal624曾在环境
  rollout前因随机IPC端口冲突失败，残缺目录完整归档后按同state/seed结果式恢复；artifact SHA256为
  `6c160030c3e8082a9a0b879a33be692303da21bdc1c878f9e66dc5959fb8944b`。
- C44组合C34历史train与C43新train，共47个mixed group/154对拟合；C43 fresh-final 30组/98对只评一次。
  四个冻结C38 temporal consequence ensemble取得fresh pairwise `67/98=68.367%`、top1 `23/30=76.667%`、
  permutation `p=0.00271`，显著优于action-only `60/98=61.224%`及C40 `54.321%`，全部离线门PASS。
  report/ranker SHA256为`bc73ba59db3e8c56bc5bf0d7b5bd02e3fe157d3955c5de0a92fe606c2e6bebcf`/
  `cf583ac3a9205bfb38b63342970e27b8a44cb6a901bedaeb84b28959b9473e39`。结论边界仍只是独立源离线排序。
- C44上线评分模块先对冻结final逐组回放，严格复现`67/98`和`23/30`，最大score-range误差
  `1.43e-6`；随后C45在新trial22做四suite×tasks0..4的20对闭环。候选0为20/20精确父策略、916次
  决策全部有分差且74.9%选择非0候选，但父策略`8/20`、best-of-4仅`3/20`，配对1胜6负，严格FAIL。
  report SHA256为`cb69b94760b71d8c099c2643e8e70d0f0dd589dd21b2b85634c642aee80483eb`。
- C45失败后查出两项真实合同错位：C43 seed偏移为`0/1e6/2e6/3e6`而C45为相邻seed；C43每状态只
  干预一次，C44组合数据中943/1424分支完整执行32步，C45却每8步重排、最多50次。C46因此在新trial23让两臂step80
  前完全相同，step80均执行32步；候选仅此处用原C43四seed排序，随后恢复replan8/sample1。
- C46机械门全过：20/20 step80前状态及candidate0动作精确（1条在80前成功）、19次排序均有分差，
  14/19选择非0；合同对齐消除了C45严重退化，但对照与候选同为`6/20`，配对1胜1负18同，严格FAIL。
  report SHA256为`e8b3a3ed337c5440938c77e0f33cbbed547d971fcdc01b01c3aba39ed3822690`。因此C44线性
  binary-success ranker正式`NO_GO`在线控制，不再换trial刷结果；consequence仅保留为离线机制证据。
- 回看官方FACT `618a6c1`确认关键差异：其value target是每个future state的连续归一化time-to-go，失败
  active段加penalty；value/future-state/action由同一非线性Transformer联合预测，而非64D线性二分类头。
  C47据此在结果前固定：C42的400条完整轨迹仅作dense-value train，新trials24/25作validation、26/27
  作one-shot final。四节点已各自采集一个trial；只有160/160轨迹完整且每角色成功/失败与suite覆盖过门，
  才允许并行生成真实INT8 H3特征并训练FACT式稠密value expert。

### 2026-08-15 — C47–C52 FACT式稠密价值路线

- C47完成新trials24..27的160/160条父策略轨迹。validation为27成功/53失败、3133个replan row；
  final为30成功/50失败、3009个row，两个角色的成功均覆盖四suite，全部数据门PASS。完成artifact
  SHA256为`96d27ac76865edffbaf0ee0adde7c572b2c09a081d05fb38cb7009dd61585364`。
- C48把C42的400条训练episode及C47的validation/final按每个replan展开。每个32步动作窗口只拼接
  实际执行的replan8前缀，终止尾部才置零并显式mask，拒绝把未执行proposal tail当因果动作。最终train/
  validation/final分别为`15417/3133/3009`样本，共21559样本、22119个去重观测；dataset SHA256为
  `d416d86c09ba334fae449a131510b84fa1d111e665a77eabfb248f1c79a5bc61`。
- C49用32张A800逐观测在线运行INT8 H3固定特征合同，22119/22119个`[1,32,5376]` BF16结果均有限且
  observation_id精确覆盖一次；随后沿C38固定随机投影冻结为256维，projected feature SHA256为
  `a09917cea43072a85d4cb8dcb06441dae907832eff9b2728419cbf4cdf8cdcb2`。H3始终冻结，没有缓存标签泄漏。
- C50对四个C38 seed各跑joint/frozen-consequence两臂，共8个10000-step任务；每批32成功/32失败，
  联合拟合future-H3、future-state和连续value。只看validation后选择joint seed8675309 step10000；其
  checkpoint SHA256为`d2f3a812eb1d4921efd6f2f9ee6f7f4f2736c516d338168b8110df281960907c`。
- C51仅一次读取final：value MSE `0.188720`，相对train-mean baseline `0.258225`降低26.9%；rank
  correlation `0.539721`，失败-成功value margin `0.656632`，shuffle action MSE `0.192992`高于clean
  `0.188720`。所有预注册门PASS，但结论严格限于held-out trajectory value，尚不代表会选动作。
- C52在C51 checkpoint冻结后，从C47 final的30条成功父轨迹取d3/d5共60个状态；每状态按C43原合同
  新执行四个offset，共240个从未存在的反事实outcome。source observation曾用于C51 final，因此这不是
  全新视觉源；但动作结果保持未见且不得调参。机械/产量门为至少10个mixed group覆盖3个suite；固定排序
  门为pairwise/top1均至少60%、组内精确单侧permutation `p<=0.05`及所有组score range大于`1e-6`。
  PASS也只放行新的闭环canary，不直接宣称在线收益。
- C52实际完成240/240分支、148成功，机械门全过；得到17个mixed group、覆盖LIBERO-10/Object/
  Spatial，56个成功-失败对中排对`37/56=66.071%`。mixed-group top1为`14/17=82.353%`，同组
  candidate0为`76.471%`、随机期望为`58.824%`；精确组内单侧permutation `p=0.038920`，最小组内
  score range `0.059310`，所有预注册门PASS。outcomes/ranking SHA256分别为
  `a894dd73767d5d27aaa2daaa24ac10547fca7ec46871bdb72ff17bb7a59a236f`和
  `d6de1463dce1e7e172836d871c639c8f934ba82a719ad910853096f8cd0127a8`。这首次证明冻结稠密value能对
  未见反事实动作结果做显著排序，但source observation曾用于C51 final，且仍不等于闭环提升。
- C53因此在任何trial28 outcome前冻结：四suite×tasks0..4共20对；两臂step80前完全同轨，均在step80
  执行32步，candidate只在此处按C52四offset排序一次，之后恢复replan8。效果门复用C46：candidate至少
  比control多2成功、至少3胜、净胜至少2且任何suite退化不超过1/5。首次启动在仿真前暴露外层CLI未转发
  dense checkpoint/report，未产生candidate outcome；错误日志原样归档，补齐纯机械转发后按同一预注册
  trial/state/seed恢复，不改变门槛。
- C53最终40/40闭环、20/20可排序，所有step80 prestate、candidate0 chunk、四seed、冻结C51身份及
  score variation机械门全过，15/20选择非0。control为`7/20`、dense为`8/20`，配对`1胜0负19平`；
  唯一差异为正向且四suite均未退化，但没有达到预注册的`+2/3胜/净胜2`效果门，因此严格FAIL，权限
  `NO_GO_DENSE_VALUE_ONLINE`。report SHA256为
  `ae9b2f421d51193882a91c9a3e7546d03cacdbc712d274c34e44035621571045`，不得靠增加trial重复同一检验刷门。
- 失败后确认一个预先存在而非事后造出的合同差异：C52所有状态是父策略成功前d3/d5，C53却固定step80；
  前者是近成功状态分布，后者不是。C54因此在看新outcome前冻结trials29..32共160条全新父轨迹；若至少
  40个合格成功源覆盖3个suite，则全量取d3/d5并做candidate0 vs dense单次32步干预、共同replan8续跑。
  效果门为绝对成功率`+5pp`、净胜至少5、精确单侧McNemar `p<=0.05`且suite退化不超过5pp。即使PASS，
  因触发位置使用父轨迹 hindsight，也只放行可部署progress trigger研究，不恢复在线部署权限。
- C54完成trials29..32共160/160条全新父轨迹，64个合格成功源覆盖四suite；全量冻结d3/d5共128组、
  256条candidate0/dense配对，selection SHA256为
  `20a8e4931896680b122dd7306f0b1e34412f08fef6c96c5427273dd9f243c9b2`。四节点动态队列完成256/256，
  所有start state、candidate0 chunk、seed、C51身份及score variation机械门PASS，96/128选择非0动作。
- 效果却明确FAIL：candidate0 `87/128`、dense `86/128`，配对`3胜4负121平`，绝对增益`-0.781pp`，
  精确单侧McNemar `p=0.773438`；四suite安全门虽过，但所有预注册效果门失败。final report SHA256为
  `e96f2d58a4f40ede0ca70061282f55843f8ab98bd82d4db7951a6e512ec0c595`。因此C50–C54稠密value只保留为
  H3 representation/trajectory-value机制证据，禁止继续作为online reranker；C52的mixed-group显著性没有
  在全新视觉源和全量配对中复现。
- 失败原因不是步数或触发时机：scorer大量选择非0且机械合同精确，但其训练只见behavior-policy动作的
  time-to-go/failure标签，没有直接学习候选动作的反事实优势；冻结原动作生成器再外挂value选择，偏离官方
  FACT把action、future-state、value置于同一非线性Transformer联合训练的核心。下一条主线应回到动作生成：
  H3冻结，联合训练小型action expert与future/value auxiliary，并对action-only D0做严格配对；不得再对
  C53/C54调ranker阈值或补trial。

### 2026-08-16 — C55共享动作块的FACT式联合辅助路线启动

- 官方FACT身份重新核验：本地训练代码固定`618a6c16868699b6d4138941de6a863589ac00dd`且clean；远端
  main为`9427ea451e806220742148049ef0576e43ef7382`，两者唯一差异是README增加live demo，模型、trainer、
  transform和config无变化。论文为`arXiv:2608.10232v1`。
- C55不再使用C50外置value scorer。`H3DreamWAMKVCarrierPolicy`新增显式`forward_hidden`训练hook，原
  public action forward保持不变；`H3FactJointAuxPolicy`以干净执行动作的隔离第二次forward复用完全相同
  ActionDiT blocks，再预测future-H3、future-state和value。15项云端单测PASS：包括父动作bit-exact、
  future target不可进入action forward、aux-only loss对共享block和aux head均产生finite非零梯度。
- C48数据重审：train/validation/final为15417/3133/3009 rows，分别400/80/80 episodes，三者overlap0；
  train成功/失败rows为2467/12950。训练时按每8卡4成功+4失败平衡；失败动作mask。由于轨迹没有FACT的
  failure-active onset，失败value也mask，不伪造penalty；两类row仍用实际future H3/state作后果监督。
- 环境动作到D0训练动作的gap已显式修复：motion保持不变，gripper执行映射严格取逆
  `dataset=(1-env)/2`，再走D0原min-max且不clamp。单测与真实`rollout_000000` loader smoke都通过；该样本
  得到actions范围`[-0.64453125,1.0]`、future-H3 256维、future-state 8维。
- C55 K/V smoke真实运行INT8 H3：层9/19/29/39/49各有独立K/V，shape均`[32,56,128]` BF16且finite；
  单文件4,592,421 bytes。正式目标为train+validation 18550个current observation，约85.2GB。
- 32卡正式缓存首轮只有32611正常；30907/32409/30234在首个H3 Linear前报
  `CUBLAS_STATUS_NOT_INITIALIZED`，没有写入正式item，归类infra。三节点将PyTorch wheel cu13 runtime置于
  系统CUDA前后，各自真实单样本在1.45–1.53秒通过，随后仅重启shards8..31；shards0..7保持运行且不覆盖。
- 预注册两臂预算为各8卡、6000 steps、每step 8 demo+4 success rollout+4 failure rollout，global
  batch16、96000 rows；相当于demo 0.239 epoch和balanced rollout 3.113 effective epochs。正式训练在
  全缓存审计及两臂真实BF16 step/restore前仍为`NO_GO`；当前只有cache prep许可，效果
  `NOT_EVIDENCE_READY`。
- C55缓存最终审计PASS：train+validation精确覆盖`18550/18550`个观测、32/32 shard、85,189,409,550
  bytes，missing/extra/partial均为0。第一轮joint机械canary暴露未经标准化的随机投影H3 target RMS
  `66.8458`，future-H3 MSE约`4480..4709`，使总loss约225；该v1只保留为scale-audit失败，未进入长训。
- 对照官方FACT可见其visual target是在缩放VAE latent上做flow velocity，并非未缩放隐藏投影。C55 v2
  因此只用C48 train样本分布逐维z-score未来H3，validation/final不参与统计；15417个train target归一化
  后RMS=`1.0`、最大绝对均值`1.14e-6`、逐维std=`0.99999994..1.0`，统计SHA256为
  `e9d404b4edfb8cff0b9ebcc3570d330141b4c8cc67adc1befd0013c034eb2618`。
- v2 action-only/joint-aux各自完成8卡10-step及严格恢复，耗时`20.25s/23.02s`，两者restore probe
  max-abs均为0。joint future-H3 loss降到`1.13..1.56`，共享block梯度从v1最高约28降到最高4.20；两条
  deployment-only导出又被既有balanced-80 evaluator以两个独立实例strict restore，固定噪声max-abs0。
  10-step物理action MSE为`0.024191/0.024195`，只说明动作路径未破坏，不作效果结论。
- C55由`NO_GO`提升为`GO_LONG`：两臂各6000步、每1000保存并严格续训；canary曾保守估算每臂4.5小时，
  真实step1000摊销后action-only/joint仅耗时`204.68s/256.53s`，故总预算修正为约0.6小时（含分段I/O）。
  mechanism signal和fresh LIBERO closed-loop仍为FAIL/未测，长训结果不能自动升级为`EVIDENCE_READY`。
- 在读取任何step1000评估结果前固定里程碑门（此时只看过训练机械报告）：joint相对同step action-only的balanced-80 normalized与physical
  action MSE均至少改善1%，gripper macro-F1下降不超过0.005；同一固定256条C48 validation上future-H3
  clean MSE相对step10至少下降5%，且shuffled-action MSE至少比clean高0.01。只有全部满足的里程碑才有资格
  进入fresh闭环，多个合格点按physical MSE最低选择。step10基线future-H3 clean/shuffle为
  `1.338880/1.343398`（差`0.004518`），明确不通过。
- step1000固定机制评估随后完成：future-H3 clean MSE=`0.287156`，比step10下降78.55%；相同当前观测但
  circular无自映射动作shuffle后为`0.373049`，退化`0.085893`。future-state/value shuffle退化也为
  `+0.131194/+0.087338`。这通过机制半门，证明共享块学到了动作相关后果；仍须paired action MSE和fresh
  closed-loop，不能单独升级为效果成功。
- step1000 paired balanced-80也通过预注册半门：joint相对action-only的normalized action MSE从
  `0.062424`降到`0.061413`（改善1.62%），physical MSE从`0.025925`降到`0.025540`（改善1.49%），
  gripper macro-F1从`0.931255`升到`0.933214`；双臂fresh restore max-abs均为0。因此step1000是首个完整
  eligible milestone，但仍按预注册跑完2k..6k，从所有eligible点中只选physical MSE最低者。
- 两臂均已完成6000步并产生1k间隔的全部checkpoint，native final restore max-abs均为0。2k..6k的
  future-H3 clean MSE继续从`0.224781`降至`0.182070`，shuffle退化从`0.108615`升至`0.139360`；但动作
  收益不随机制指标单调增长。2k/3k/5k/6k没有同时达到两个1%动作门，4k虽有1.71%的normalized改善，
  physical仅改善0.82%，也不合格。因此冻结选择仍是唯一eligible的step1000，而不是事后选择单项
  physical MSE最低的4k。
- 同一balanced-80上的原擂主D0为normalized/physical MSE `0.056571/0.024190`、gripper F1
  `0.942075`；selected joint step1000分别为`0.061413/0.025540/0.933214`，即仍比D0差
  `8.56%/5.58%/-0.00886`。所以离线结论仅是FACT式联合辅助优于匹配的action-only续训且机制有效，尚未
  证明替换擂主。最终报告为`PASS_C55_OFFLINE_GATE / GO_FRESH_CLOSED_LOOP`，artifact：
  `/mnt/h3-wam/outputs/c55-fact-joint-action-long-v2/offline_final.json`。
- fresh闭环在里程碑选择前固定为三臂，而不是只比两个continuation：selected joint、同step action-only、
  原擂主D0-H32-s14000。先用trials33..36只做harness机械canary，不据其效果早停；通过后消费LIBERO四
  suite×10 tasks所有剩余未见init-state trials33..49，即每臂680 episode。主joint-vs-action-only门为
  `+3pp`、净胜至少20、exact one-sided McNemar `p<=0.05`、任一suite退化不超过3pp；升级擂主还要求joint
  overall不低于原D0且任一suite相对D0退化不超过3pp。这样不能把“两条续训都退化但joint少退一点”称为成功。
- step1000两臂已导出为D0兼容deployment-only权重并记录source/parent/output SHA256；fresh机械canary清单
  冻结为trials33..36、四suite×10 tasks×三臂，共480 episode，manifest SHA256
  `ad5829e5478ee2a37dcd9ecc52803e99046c72fb0fed3b3d1d4f923d556d9f57`。2026-08-16已在四节点32张
  A800启动；canary阶段只读取完整性、初态、checkpoint与seed合同，不聚合success，机械通过后才生成
  trials37..49的1560条剩余任务并最终一次性统计33..49的680组三臂配对。

### 2026-08-16 — C55 fresh闭环否证与四条完整机制支线

- C55 fresh闭环最终一次性聚合680组同初态、同seed、同replan8三臂结果：joint为
  `231/680=33.97%`，同预算action-only为`234/680=34.41%`，原擂主D0为
  `270/680=39.71%`。joint相对action-only为36胜39负，exact McNemar
  `p=0.6778`；相对D0绝对退化`5.74pp`。因此C55判定
  `FAIL_CLOSED_LOOP / NO_GO_FUSION`，不允许用后果预测改善覆盖动作成功率退化。最终artifact为
  `/mnt/h3-wam/outputs/c55-fresh-closed-loop-v1/FINAL.json`，SHA256
  `72a840eba03e0f79ff3a8568153adc9c5fe72165f4c8f2bec932aba390e4c799`。
- 该结果否定的是“在D0后追加浅层future/state/value head即可把H3世界知识转成动作收益”，不是否定
  H3作为世界骨干。后续停止该类补丁，改为四线并行：C56 FACT共享causal backbone、C57 LingBot真实
  observation/action persistent KV生命周期、C58 FastWAM完整30层ActionDiT与C58b逐层H3 carrier、
  C59-C61真实失败/反事实数据合同。
- C56a只验证官方`[P,A,G,V,I]`顺序、causal mask、两阶段不泄漏、失败动作mask和future/value梯度；
  因其仍是D0五层后的独立4层causal trunk，明确标为`INTENTIONAL_DEVIATION / PROBE_ONLY / NO_GO_LONG`。
  8卡机械aggregate为8/8 PASS、D0 parity和restore max-abs均0，峰值reserved 4.62GiB；真正C56b必须把
  P/A/G/V/I置入C58b的30个逐层shared blocks。
- C57按LingBot官方5000-step/global80/AdamW1e-5/warmup10/save200配方启动8卡长训；冻结sequence
  manifest有200779窗口/1542 episode、0缺失、0 future leakage，持久窗口最大536 token小于540。
  真实8卡canary 10/10 finite，严格恢复max-abs0；该支线完整移植persistent lifecycle，但承载体仍为
  D0五层，禁止称为官方30层shared backbone复现。长训目录为
  `/mnt/h3-wam/outputs/c57-lingbot-persistent-kv/long5000`。
- C58从固定FastWAM commit `45d8e1458921d83f8ad6cf9ce993d371208dabd0`直接加载30个官方
  ActionDiT block，function-preserving地把D0从5层扩为30层。8卡真实probe中30/30 block梯度非零、
  D0 step0 parity与严格恢复max-abs均0，峰值reserved 24.27GiB；10000-step长训每1000原子保存。
  但C58把H3 layer49重复给全部30层，只能作为`CONTROLLED_ARM_REPEATED_LAYER49`，不是完整
  FastWAM世界—动作联合训练。
- C58b固定H3 50层到ActionDiT 30层的单调映射
  `(0,2,3,5,7,8,10,12,14,15,17,19,20,22,24,25,27,29,30,32,34,35,37,39,41,42,44,46,47,49)`；
  每个action block消费对应H3层K/V。80样本cache canary约2.202GB，80k训练slice纯tensor约2.202TB；
  先过真实cache/no-alias/layer49退化等价/真实layerwise非零delta/30层梯度/恢复门，再扩全量。
- C59 outcome-only overlay冻结560 episode/362 failure/21559 samples，不伪造failure onset；C60从成功父
  轨迹exact state restore得到83条失败分支/3115样本，按父episode split-disjoint。C61进一步从141条
  train-only成功父轨迹的d3/d5状态生成1128个四seed分叉；只保留终局失败分支，干预边界作为显式onset，
  action imitation全mask，真实post-intervention future和value仍监督。

### 2026-08-16 — 停止大缓存并回归四线训练主线

- C58b 的80样本逐层K/V parity canary已完成其唯一用途：online H3与磁盘K/V在30个映射层逐tensor
  bit-exact，缓存/在线ActionDiT输出也bit-exact；单卡真实online H3 + 30层ActionDiT一步训练峰值
  allocated/reserved为`41.46/42.21GiB`，热态H3抽取约`0.225s`、动作前后向与AdamW约`0.793s`，
  总计约`1.021s`。因此正式训练改为每rank驻留冻结INT8 H3并在线抽取，不再生成80k约2.2TB的K/V缓存。
- 按可追溯删除边界清理约182GB失效派生物：C56a structured5缓存约15GB、两个已判NO_GO的历史
  DreamWAM/StarWAM缓存约165GB、C58b parity缓存约2.1GB及小型机械canary。保留原始windows、manifest、
  数据集、事故日志、checkpoint，以及仍被C57/C61在途任务读取的既有dense K/V。删除不改变任何训练样本
  或已发布artifact；共享存储当前约49TB总量、24TB可用。
- C58b在线8卡DDP门禁使用80个唯一样本完成10步：30/30 ActionDiT层每步均有非零梯度，单rank峰值
  allocated/reserved为`42.401/43.742GiB`，12.18GB checkpoint独立恢复
  `restore_max_abs=0.0`，checkpoint SHA256为
  `c94aa0b3f38efcdcefb1d5cbbb8d588da8289e82968ec7025798efaff15e67fa`。门禁状态为
  `PASS_ONLINE_DDP_CANARY / GO_ONLINE_10000`，但仅是机械证据；n0已启动online 10000步、每1000步保存。
- C58重复layer49对照臂在s8000前因probe sample identity检查严格停止。根因是fresh probe offset变更及
  online新合同字段误入legacy cached合同，不是checkpoint损坏。兼容修复保持旧合同字节不变后，s7000
  独立恢复`restore_probe_max_abs=0.0`，现从s7000严格续跑s8000，之后仍需s9000/s10000和final restore。
- C56b online共享30层机械门虽证明8/8 rank、30层梯度、H3冻结和恢复均正确，但首批全为C60失败样本，
  按合同action loss恰为0，且未标准化7168维future-H3目标使loss约2520–2868。因此它仍是
  `NO_GO_LONG`：先只用train split拟合target normalization，再以expert/success/failure混合批验证成功
  action mask非零、失败action mask为零及raw/normalized梯度尺度；通过前不写candidate checkpoint。
- C57继续按LingBot官方预算运行到5000步；step200固定held-out已明确退化：C57/D0 loss为
  `0.133945/0.076288`，相对差75.58%，样本胜率5/80，四suite均退化。该早期点为`NO_GO`，但不据此
  中止预注册5000步；只允许step5000通过离线门后进入真实persistent LIBERO canary。

### 2026-08-16 — C56b完整在线训练门与四线自动接力

- C56b future-H3 scale只消费train split的512个在线样本，固定构成为expert/success/
  observational-failure/causal-failure=`256/128/64/64`。7168维raw target RMS为`53.49984`，逐维
  z-score后RMS为`1.0`；std最小/中位/最大为`5.90/19.62/169.08`，0维触发clamp。统计SHA256为
  `95df1f65eba1b1c3bfb9cebea90983ca54dffa69f60e6135354eb67e8551d000`，未写任何K/V或feature cache。
- 随后的8卡mixed balance gate固定4 expert、2 success、1 observational failure、1 causal failure。
  success action loss全部非零、两类failure action loss严格为0，替换future target对Stage-1动作输出
  max-abs为0，30个shared block最小归一化梯度norm为`5.1686`，完整state restore max-abs为0。
  raw future loss `1608.54..3657.31`降为normalized `1.684..2.455`；aggregate SHA256为
  `d083fe955266b8f56da2a39f250f7bdb7e63d6fbb463984b417e8302ea08ac7f`。
- 完整online optimizer canary使用FACT固定loss权重`10/1/0.4/0.4`、base/action LR
  `2e-5/2e-4`、每步新的4/2/1/1样本和online current+future H3。10/10步finite、每步30层梯度
  全正、future leak为0；完整model+optimizer+scheduler独立恢复max-abs为0，峰值reserved
  `57.49GiB`，热态约`1.587s/step`。`GO_LONG.json`状态为`GO_LONG / NOT_EVIDENCE_READY`，checkpoint
  SHA256为`2d302b2cef4a2e9d1eb371a480cb235e50113c73267ae675de3ea5f735f39521`。正式10000步必须等待
  C58b online s10000及其final strict READY，不得用canary或D0替代父模型。
- C58b online长训已跨过s1000/s2000原子checkpoint并通过下一段strict load，继续s3000；为修复原launcher
  只训练不发布final restore的问题，新增后台finalizer。它只在s10000 report/checkpoint完成、训练退出后
  运行独立8卡restore，审计completed steps、模型身份、30层映射、final1000步梯度、文件大小及三个SHA，
  仅`restore_probe_max_abs=0`才原子写`online-long10000/READY.json`。
- C57固定held-out趋势为：s200相对D0 `-75.58%`/样本胜率`6.25%`，s400为
  `-36.35%/16.25%`，s600为`-24.82%/25.0%`；均为`NO_GO`，但连续改善支持按预注册预算继续到s5000。
  n2在C58重复layer49对照s10000及final restore完成后自动补每200步评测；C58b final READY出现时可原子
  抢占单次约70秒评测并让位给C56，不提前封锁空闲GPU。
- C61的1128分支采集保持在n3；已部署严格后台finalizer。只有node completion marker、1128个精确result与
  trajectory、冻结job/FROZEN SHA及每条branch身份全部通过，才生成C60兼容的failure dataset；未完成期间
  不写正式dataset/COMPLETED。C61完成后另开matched C56+C61 arm，禁止把新数据热插入在途C56 run。

### 2026-08-16 — C58b完整逐层H3→FastWAM首次正向闭环点估计

- C58b online 10000步最终checkpoint为
  `/mnt/h3-wam/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt`，
  SHA256 `2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541`；30/30层末段梯度非零、
  fresh restore max-abs为0。online-H3/no-disk-KV balanced80通过6项门：normalized/physical MSE
  `0.0593315/0.0254498`、gripper macro-F1 `0.936444`、prediction std `0.478793`、language delta
  `0.231742`、visual-shuffle delta MSE `0.0316270`。
- 首版trial33误用了`wait_steps=0`和显式seed `330042`，D0前31条归零。该轮已fail-closed并完整归档为
  `fresh-libero-trial33.invalid-wait0-20260816T093403Z`，不计模型效果。corrected合同严格恢复历史
  `wait_steps=30`、`episode_seed=42+task_id*100000+trial*1000`。固定D0的40条结果不仅复现
  `16/40`，而且success、steps、replans、seed schedule、初始物体关节、首动作、首32×7 chunk及所有
  replan首动作与C55历史逐值完全相同，40×10字段零mismatch。
- corrected同初态、同seed、同replan8的40对结果：C58b `18/40=45%`，D0 `16/40=40%`，差`+5pp`；
  discordant pair为C58b赢5、D0赢3，exact one-sided McNemar `p=0.36328125`、two-sided
  `p=0.7265625`。suite分解为spatial `6/10 vs 7/10`、object `8/10 vs 6/10`、goal
  `3/10 vs 1/10`、LIBERO-10 `1/10 vs 2/10`。最终报告：
  `/mnt/h3-wam/outputs/c58b-fastwam-layerwise-v1/online-final-eval-v1/fresh-libero-trial33/RESULTS.json`，
  SHA256 `f7e9c8f65c177d33a3b168d0e0a47e79034d0054c99866a66ba09f82ee916ab3`。
- 这是四条完整机制支线中首个“离线世界机制不塌缩且闭环点估计正向”的候选，结论为
  `GO_EXPANDED_PAIRED_EVAL / NOT_PROMOTED`。40对中仅8个discordant，尚无统计显著性；必须扩大到多trial
  配对闭环并检查spatial/LIBERO-10退化是否持续，不能把`+5pp`单trial结果写成泛化或新擂主结论。

### 2026-08-16 — C58b trials33..49 扩展配对评测启动

- 扩展评测预注册为四suite×10 task×trials34..49的640条新C58b candidate episode，和corrected
  trial33合并为680对；模型、wait30、replan8、horizon32、eval10、default task-specific seed和
  pre-clamp均不变。最终门为overall至少`+3pp`、净胜至少20、exact one-sided McNemar `p<=0.05`且
  任一suite不低于D0超过3pp；不按中间success早停或选trial。
- C55历史D0控制复用前完成fail-closed冻结审计：trials34..49共640/640条result及trajectory通过
  checkpoint/合同/seed/动作finite/完整初态/C55 FINAL outcome/content hash，成功`254/640`，manifest
  SHA256 `e20a6f6c479d7b57dda106cf0e2d73f2ce41305d8926223aea685114ee0a7a0a`。加上本轮trial33已逐动作exact
  复现的`16/40`，完整控制精确恢复D0历史`270/680`；任一控制审计失败原计划为双臂重跑。
- 第一次expanded v1启动时，目的snapshot仍被`cp -a --reflink=auto`写入，违反immutable source gate。
  该launcher、8 rollout和8 policy已立即TERM且无survivor；当时尚无正式result episode，整个root保留为
  `expanded-paired-trials34-49-v1.invalid-snapshot-copy-race-20260816T1056Z`，永不计入效果。
- 复制完成、overlay后重新逐文件hash并将snapshot冻结为mode555，正式v2从
  `/mnt/h3-wam/runtime-snapshots/project-3e39368-c58expanded`启动，root为
  `expanded-paired-trials34-49-v2`，launcher PID `963748`。8 GPU各运行一个固定80-episode segment；首个
  candidate object/task0/trial34 trajectory与冻结D0的8项完整初始状态digest逐字节一致，证明多trial
  单server分段没有引入初态gap。
- 独立只读finalizer snapshot为
  `/mnt/h3-wam/runtime-snapshots/c58-expanded-finalizer-229d990`，PID `966603`。它只等待candidate
  `COMPLETED.json`，随后一次性重验640条candidate与640条control的result/trajectory hash、完整初态及
  trial33桥接，输出680对overall/per-suite/per-trial、Wilson95、paired-delta95和exact McNemar；全量完成前
  不生成效果结论。

### 2026-08-16 — C58b expanded v2 初态门禁否决与独立进程 v3 修复

- v2完成640条candidate后，`229d990` finalizer在读取任何聚合效果前fail-closed：首个错误为
  `libero_10/task0/trial35`初态不一致，因此没有生成`FINAL.json`。全量只读诊断确认严格initial-state
  mismatch为`532/640`：spatial/object/goal各`140/160`，LIBERO-10为`112/160`。每个80条进程的
  首trial（34或42）均exact，其后7个trial系统性偏离；这不是可选择剔除的随机坏样本，v2整批标记无效。
- 根因为评测进程边界未对齐：历史C55/D0每个suite/task/trial由独立`rollout_libero.py`进程和新MuJoCo
  env执行，v2却让每个task的8个trial复用一个env。以LIBERO-10 task0为例，trial34的8项轨迹初态及
  object joints均exact；紧随其后的trial35虽然图像、EEF、gripper和previous_action仍exact，但sim_state
  出现`1.35e-14`差异且object joints不再bit-exact。episode seed、replan seed及policy的独立episode key
  均正确，故不能用放宽浮点容差掩盖process-contract污染。
- corrected trial35 canary改为一个episode一个全新simulator+policy进程；其8项完整初态digest恢复为
  `02c1e0c596848f4d9e3aa778ec970aa8646db694c8edf48a3a6e0bd62563b8bc`，与冻结D0逐字节一致，object
  joints、episode seed `35042`、environment seed和replan seed schedule也全部exact。v3固定640个
  one-episode jobs，由8个GPU队列各顺序消费80个；checkpoint、wait30/replan8/horizon32/eval10及统计门
  全部不变，不读取中间success、不复用v2子集，也无需重跑已通过hash冻结的D0。
- 首个isolated v3运行时快照仅由`git archive`构造，遗漏未被主仓库跟踪的固定StarWAM vendor；8个policy
  均在ready前因缺`wan_block.py`退出，产生`0`条formal result。finalizer随即停止，root写入
  `INVALID_NEVER_USE`（SHA256 `06ea7cf9691868dbe264dc74645c63985766fe9930fa1614573e8015fd50cd2c`），
  没有在partial root续跑。该失败属于infra packaging，不涉及模型效果。
- v4从已跑通v2的完整553MB复合runtime复制，等待`COPY_DONE`后再覆盖commit `eecfde3`，并包含
  StarWAM `cd76d96f`、DreamWAM `6e989fac`及FACT `618a6c16`。真实policy restore+H3 inference smoke产生
  finite首动作，trial35完整初态仍byte-exact；之后快照冻结为0个writable file，`SNAPSHOT.json` SHA256为
  `90dcd111b6e6ca8d30c141f77987b8d2f658d0b41468377517d5e304634e9cfb`。正式v4固定640个
  one-episode jobs，manifest SHA256 `ff5b57a374dc8b96343fdb359b7d7a7ca9e9ab33a73a0563c52ae660f188b7a6`。

### 2026-08-17 — C58b 680对最终晋级与独立复核

- v4全部640条candidate完成后与已验证trial33桥接合并为680对；C58b为`295/680=43.382%`，固定D0为
  `270/680=39.706%`，绝对提升`3.676pp`。149个discordant pairs中C58赢87、D0赢62，净胜25；
  exact one-sided McNemar `p=0.0244576`，paired delta 95%区间为`[0.166pp,7.186pp]`。
- suite结果为Spatial `92:80`（`+7.059pp`）、Object `131:125`（`+3.529pp`）、Goal `44:40`
  （`+2.353pp`）、LIBERO-10 `28:25`（`+1.765pp`）。overall gain至少3pp、净胜至少20、one-sided
  `p<=0.05`、无suite低于`-3pp`四项promotion门全部PASS；完整680 pair、trial33 bridge、严格初态和
  one-episode-per-process机械门也全部PASS。
- 正式`FINAL.json` SHA256为
  `53a06ac5c3c36298ed2ee397688eb03e6918219d32f469897e9139530d954f88`；正式
  `PAIR_EVIDENCE.jsonl` SHA256为
  `e44a32833c1d9f71485f3cca37785b5d813f59c7af4eea12311dfd1ed14f1e3c`。从只读snapshot在新目录独立
  重跑全部checkpoint/source/result/trajectory hash与初态审计，PAIR_EVIDENCE逐字节相同，去掉独立输出
  路径后的FINAL逐字段相同。
- 结论更新为`EVIDENCE_READY / CARRIER_TRACK_CHAMPION`：C58替换D0成为后续融合父节点，但不是最终
  全赛道冠军。新的lineage固定为`C58 carrier -> + temporal/context winner -> + consequence/ranking
  winner`。完整trials0..49报告尚待补齐；其中0..32已被历史研发消费，只能作为descriptive benchmark，
  不允许重新用于confirmatory promotion。

### 2026-08-17 — C60 FACT失败分支融合终局与C61停止扩展

- C60将C58 carrier与C56b FACT online objective结合，固定四suite×10 tasks×trials33..49、
  one-episode-per-process、wait30/replan8/horizon32/eval10，与C58做680对配对闭环。候选checkpoint
  SHA256为`d6659c6b387f062a99f670a1d902b56df71a6bf1472aa4e46e56c9213ba75a36`，固定C58父节点SHA256为
  `2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541`。
- C60为`313/680=46.029%`，C58为`295/680=43.382%`，点估计提升`2.647pp`；discordant pairs为
  C60赢63、C58赢45，净胜18，exact one-sided McNemar `p=0.0507164`，paired-delta 95%区间
  `[-0.344pp,5.638pp]`。suite为Spatial `98:92`、Object `139:131`、Goal `44:44`、LIBERO-10
  `32:28`，suite安全门通过。
- 预注册门要求overall至少`+3pp`、净胜至少20、one-sided `p<=0.05`且suite安全；前三项分别因
  `2.647pp/18/0.0507164`全部失败。因此正式结论为`FAIL_C60_FACT_EXPANDED_PAIRED / NOT_EVIDENCE_READY /
  KEEP_C58_PARENT`。`313/680`只能称为当前最高完整点估计，不能称为晋级或冠军。
- 正式`RESULTS.json` SHA256为
  `d9280c5ad4aeac231a8da793ac5f5d667f005dbc8c5cfe3657b93a4895483ec3`，`PAIR_EVIDENCE.jsonl`
  SHA256为`b96421a1e5c6d6ff8fe729f5cf3128560a00e64eac3983d20ef505346f3e9b05`，`COMPLETED.json`
  SHA256为`5eb9623c570c371062ed8d8b8262bec90eec3973c88f3feafe4c48904a3c349d`。从独立只读finalizer重新读取
  640条新证据并桥接trial33，format/status/permission/effect、overall/per-suite/per-trial、gates与pair
  outcomes逐字段一致。
- C61额外失败分支版本仅做预注册trial33 canary：C61 `17/40`、C60 `20/40`、C58 `18/40`；相对C60为
  `-7.5pp`且0胜3负，相对C58为`-2.5pp`。判定`NO_GO_EXPANSION`，不得扩大到trials34..49，也不得以
  “更多负例”名义热插入C60或替代C58。
- C60 s1k..s10k固定balanced80共完成800个模型-样本评测，10/10 strict restore和conditioning gate通过，
  数据/噪声/solver/normalization身份一致。晚期s8k–s10k相对中期s4k–s6k的physical/normalized平均MSE
  均改善，但s10k physical MSE `0.0252567`差于s5k `0.0252263`，失败唯一预注册续训门；正式判定
  `NO_EVIDENCE_FOR_S20K_CONTINUATION`。RESULTS SHA256为
  `2008293c4cc11ccfb333c67aaf72dd888920b59c1e1ebeb2ddb343a8268e325e`，独立JSON重算与正式结果一致。
  s10k scheduler已到LR=0，故不创建新s20k dossier、不启动长训，n2八卡释放；该诊断也不改变
  `KEEP_C58_PARENT`。

### 2026-08-17 — C57收口与C62 MiniWorld/C58 context route

- C57完整跑完5000步/400000样本，对200779个windows为`1.9922398249` effective epochs，最终strict
  restore checkpoint SHA256为`4df57b0cd4f053f4cba064ad5a34adb3ec9ce26b3ffb3071885bb0423066ac58`。
  固定80样本最终C57/D0 MSE为`0.07717749/0.07628779`，相对改善`-1.166%`、win rate43.75%，失败
  +3%/55%双门；正式`C57_FINAL_OFFLINE_NO_GO`，不做LIBERO或追加步数。
- C62从官方MiniWorld `e484206`只移植real-observation sink+FIFO和4-action alignment，以已晋级C58
  s10000为冻结父策略。纠正H3 gradient boundary后，真实机械报告SHA256
  `0adfbf47a821606cab8be8d416bcb975918701cc8706fdb4fb46d6d86412a54b`：父strict restore、
  default-off exact、30/30梯度和runtime restore均PASS。
- C62短canary固定8卡100步、global batch8、800 unique train samples/800 episodes/1.0 epoch，64
  heldout来自32个episode，耗时189.98秒。所有optimizer/restore/safety机械门通过，但clean/shuffle MSE为
  `0.0705387413/0.0705377535`，相对改善`-0.001400%`，且clean比context-off差3.378%。正式判定
  `FAIL_C62_CAUSAL_OPTIMIZER_CANARY / NO_GO_C62_TRAINING`。report SHA256
  `ab289a4f34794f024f03dabd67f1f5c44e852c8a7279bcdf47b0d34740078084`，delta checkpoint SHA256
  `86d795d010bdacd95fee660e46354314302d85678a5f475d0c508f3cc6cda3c6`；仅审计保留，不长训、不rollout、
  不融合，n1释放。

### 2026-08-17 — C58 full50、C65/C66诊断与C69同预算动作归因

- C58完成四suite×10 tasks×trials0..49的描述性full50：`846/2000=42.3%`，D0为
  `734/2000=36.7%`，差`+5.6pp`，discordant为284胜/172负，one-sided exact McNemar
  `p=8.787e-08`。其中仍未消费的正式confirmatory部分trials33..49保持既有结果
  `295/680`对`270/680`、`+3.676pp`、87胜/62负、`p=0.0244576`；trials0..32只作补充描述，
  不重新产生promotion claim。`FINAL_DESCRIPTIVE.json` SHA256为
  `866cf335b1dca2c097c482d09db42f58815740829deb3cde55fcb950312eeac0`。
- C65 Stage-2 pair收集完成`3072/3072`，但预注册data gate fail-closed：四suite满足mixed-source
  success/failure pair的数量仅Spatial17、Object2、Goal11、LIBERO-10 13，均未达到每suite20；正式
  permission为`NO_SCORE_DATA_COVERAGE_GAP`，没有运行score，也没有降低阈值。Object大多为全成功轨迹，
  是当前同状态成功/失败pair构造的主要覆盖瓶颈。
- C66固定64条配对context-length诊断表明问题首先是结构性前缀干扰：未经训练的C58从context-off
  MSE `0.0616758`恶化到k1/k3/k7的`0.117637/0.173727/0.248364`（`+90.7%/+181.7%/+302.7%`）。
  C66-s100将对应值修复到`0.0775573/0.0816324/0.0904292`，但仍差于自身off `0.0677915`；因此只选择
  k1作为下一 bounded mechanism test，不给C66长训或LIBERO许可。RESULTS SHA256为
  `50a726dd6bc69fa185c9c9bf17cac9ed138d9d8ef6a229b886d44af76c241237`。
- C66-k1按该诊断做了严格单变量fresh-parent s100：同一train800/heldout64、seed、AdamW与8卡预算，
  仅把实际committed history从7个完整chunk改为最近1个。clean/shuffle/off MSE为
  `0.08219094/0.08289697/0.07801624`；clean-over-shuffle仅`0.8517%`（门`>=1%`），clean相对off
  退化`5.3511%`（门`<=5%`），正式report SHA256
  `70975e1b9de6612f6bdb65ff8d0bbeb9fdff3530b82e6b22cc4a7c781aba908a`，结论
  `FAIL_C66_K1_BOUNDED_MECHANISM / NO_GO_C66_K1_LONG_OR_ROLLOUT`。只读restore v2诊断进一步证明正式
  `runtime_restore_exact=false`是eval精度作用域假阴性：clean在BF16 autocast内而restore重算在外；64/64
  state snapshot K/V和15/56/14坐标精确，同精度作用域original/restored与重复forward最大差异均0，只有跨
  作用域比较64/64非零、最大`0.0234375`。诊断SHA256
  `f2bc3344ec7dded536605d5bb935f4fdbfc821296e284b56df14c21ccc416019`；这排除了serialization、k1
  prefix和机制非确定性缺陷，但不改变两个efficacy门均失败，仍禁止长训、改阈值或rollout。
- C69补齐历史缺失的严格world-objective-off对照。它与C67保持同一C58 parent、30层joint-token forward、
  seed、4/2/1/1 rank样本顺序、失败action mask、base/action LR和20k cosine；唯一变量为loss权重从
  `[10,1,0.4,0.4]`变成`[10,0,0,0]`，并冻结/排除六个auxiliary encoder/decoder。CPU单测证明
  DDP action分量逐步等于C67的`10×global masked action mean`，不把两个failure ranks偷换为expert。
- C69真实8×A800十步canary的8项门全部PASS：10/10 finite、30/30 global shared-block gradients、未来
  target到action泄漏0、辅助head冻结、在线INT8 H3无disk K/V、strict restore max-abs0；checkpoint
  SHA256为`af29173c780691f3f1a6f8d7efef1a49e24d349c2c938796a66d08e5865d4b07`，GO_LONG SHA256为
  `5d448cbf94bc9820a66d148e294f133b469fa5bb66913b389b5456376b4c89a5`。这只给机械训练许可，不是效果证据。
- C69正式20k从只读commit `a60b056`与SOURCE_FREEZE SHA256
  `a9197001d9b545ba7542dccc864c104ac2ee99a6defbd4c50bb9acab9ef66d68`启动，固定每1000步保存并严格恢复；
  最终只允许C69-s20与C67-s20做同预算归因，C58在fresh闭环被击败前仍是carrier champion。
- C69只读异步balanced80评测队列固定在commit `55b622f`和SOURCE_FREEZE SHA256
  `9e6cae8f01af159b9214428815ca4e226272044f2c739594916b9a26fb24ca78`，在30907按checkpoint/report/
  strict-restore三件套消费，不调用trainer、不早停、不选点。首个s1000机械与conditioning门通过：
  normalized/physical MSE `0.0587997/0.0256934`、gripper macro-F1 `0.939408`；同一步C67为
  `0.0583450/0.0253304/0.940178`。单点仅证明评测链与动作专用合同可用，不构成FACT方向结论。
- C67最终证据另部署独立只读复核watcher：snapshot commit/tree为`15615527b5dcfd2ee0f4e2fa4347b5beefa25447`/
  `0c9dfdf45bb3691bc4a2432a66f2a631dc19e298`，SOURCE_FREEZE SHA256为
  `7eaa799f3124fb7253cd9ae96f55e15126b6bb5ee5eb5d47c6f9d48ee2ff7fad`，全树`5478`文件验证通过。它只在
  C67 `TRAINING_COMPLETE`、20点preview seal和固定aggregate三者齐备后复算证据，不训练、不重评模型、
  不选择checkpoint且不自动rollout；部署时C67已进入固定`s19000→s20000`最后一段，输出根尚未创建。
- C67最终完成20,000步/160,000 samples/aggregate `0.733522` epoch，20/20训练段、加载恢复、独立strict
  restore及conditioning点门均完整。固定s10→s20的normalized/physical MSE却分别恶化`1.526%/1.729%`，
  physical逐样本胜率仅`52.5%`；gripper与language守住，但visual response只保留`81.13%`。late-window
  normalized改善`1.902%`，physical仅`0.956%`，故正式判定
  `FAIL_C67_BUDGET_BALANCED80_GATE / NO_C67_PAIRED_680_ROLLOUT`，不启动680对闭环。独立审计逐字段复算
  同一结果，`AUDIT.json` SHA256为
  `21a3c28567d04116770c01c2f092b15276b8bcc47b183da4a6a97c8bbe2a7b58`；C58继续作为carrier champion。
- C70把C67唯一变量改为确定性两步周期sampler：每步平均6 expert、1 success、0.5 observational failure、
  0.5 causal failure，保持C58 parent、30层joint tower、FACT `10:1:0.4:0.4`、GB8、seed、LR和20k
  scheduler不变。首个非保留probe实际没有写checkpoint，但暴露训练器把缺失save path写成字符串`"None"`
  的审计缺陷；提交`45ee644`修正为JSON null并在新目录完整重跑。probe2六项门全部PASS，PROBE.json
  SHA256为`f07cd31de2a637361c5cdb1653325053afe198a6820212c3411ebb4ad8a61b36`。
- C70真实8×A800十步canary全部有限、30/30 shared gradients持续为正、future leak为0；checkpoint
  SHA256为`9bfd7ebed7069929b007336017fe2b87d6ef18a67aa822987cdcfc0a4b9b1ca1`且strict restore max-abs0。
  `GO_LONG.json` SHA256为`6076d37050e0cd05f9a580b5b21bb5b3a86061cba4b1bf0f48748c0c084f70f3`，状态仅为
  `GO_C70_LONG / MECHANICAL_PERMISSION_ONLY / NOT_EVIDENCE_READY`；长训固定每1k保存/恢复，最终离线只比较
  C70-s20与C67-s20，未过门不得rollout。
- C70长训已从只读commit `7323607`、tree `804d4d8a`、完整SOURCE_FREEZE SHA256
  `dcb14f9737b81c79f7d6e63ffe68f371761b12b8c3ccb337af1f0cc4d71422c4`启动；manual release SHA256为
  `3984770c55b1bd5e6823239281f656ea134b1b366028510680c30e4a20d851e3`。固定20k/GB8/每1k原子
  checkpoint+strict restore，首段运行时确认`--rank-schedule c70_6_1_half_half`和joint objective均生效。
- C70异步只读preview队列固定commit `be5867a`、SOURCE_FREEZE SHA256
  `49e449f31542aab5221bc67290b15888db459db51c9891c66a47e21c7e72f476`；最终sealing/offline watcher固定
  commit `34b81b8`、SOURCE_FREEZE SHA256
  `56cc29a1c3da55f61bee64451297758b903354bba413fed697843ee2a25788e9`。watcher只在20个preview和
  `TRAINING_COMPLETE`齐备后做零模型forward的重绑定，再按预注册八门比较固定C70-s20/C67-s20；不自动
  启动LIBERO，避免offline门失败仍越级消费闭环预算。

### 2026-08-17 — C69/C70 长线运行检查点（23:22 CST）

- C69动作专用同预算对照运行到约`s7917/20000`，已原子生成并异步评测`s1k..s7k`七个预注册里程碑；
  七点strict restore及prediction/gripper/language/visual四个conditioning gate全部PASS。其
  normalized/physical MSE在`s1k..s7k`依次为`0.058800/0.025693`、`0.062500/0.026422`、
  `0.063105/0.026201`、`0.059873/0.024875`、`0.065754/0.026547`、`0.067132/0.027459`、
  `0.058213/0.024634`，gripper macro-F1依次为`0.939408/0.931968/0.929235/0.933211/0.921429/`
  `0.922724/0.937251`。曲线非单调且尚未到固定终点；全部报告仍为
  `PREVIEW_NOT_EVIDENCE_NOT_FOR_EARLY_STOPPING`，不得据此选`s7k`或停止20k。
- C70 sampler-coverage运行到约`s2029/20000`，`s1000` checkpoint、训练报告和独立strict restore三件套
  已被只读队列审计并完成fixed balanced80。结果为normalized MSE `0.0628116`、physical MSE
  `0.0272833`、gripper macro-F1 `0.933863`、end-to-end language relative delta `0.896045`、visual
  shuffle delta MSE `0.0314490`；四个conditioning gate全部PASS。状态保持
  `PREVIEW_ONLY_PENDING_TRAINING_COMPLETE_REBIND / NOT_EVIDENCE_READY`，不能与固定C67终点做跨步效果结论。
- C70 `s2000`随后也完成fixed balanced80：normalized/physical MSE为`0.0597471/0.0252303`、gripper
  macro-F1 `0.934842`、language relative delta `0.889947`、visual shuffle delta MSE `0.0331694`，四个
  conditioning gate继续全部PASS。相对自身`s1000`，两类MSE分别改善约`4.88%/7.52%`，但这只是候选内
  学习曲线，不是相对C67父对照的机制归因。checkpoint SHA256为
  `30ae03e3bddf865832db38d0186ccdd86c579e3ae88d1050e115a059d19e2955`，正式preview report SHA256为
  `9b460779a2ae06b9dd13a3c0477d1b3ac57632bb2c274dcb220bd40ddef57804`。
- 两条训练每步future-to-action leak继续为0；C70八卡显存约前六卡45.8GB、后两卡55.7GB，GPU均有计算
  活动。共享存储剩余约24TB，五个节点分别承担C69训练、C69 preview/终点归因、C70训练、C70 preview和
  C70最终封印，没有空闲节点或重复训练进程。
- 用相同里程碑的C67父报告做只读paired preview后，C69在`s1k..s6k`未形成稳定优势；`s7000`首次同时
  获得normalized/physical均值改善`1.086%/1.204%`，但physical逐样本胜率仅`53.75%`，低于最终门的
  `55%`，且这不是固定`s20`。C70则从`s1000`相对C67恶化`7.66%/7.71%`，到`s2000`转为改善
  `3.78%/3.84%`，normalized/physical逐样本胜率`57.5%/55.0%`，gripper提升`0.00486`，language/visual
  response分别保留`101.16%/99.94%`。这种跨里程碑反转只支持继续完整学习曲线；不允许选择`s2k`、提前
  宣称sampler有效或启动闭环，最终仍固定比较C70-s20与C67-s20。

### 2026-08-18 — C69/C70 固定终点与新闭环归因放行

- C69和C70均完成`20000 steps / global batch 8 / 160000 samples / 0.733522 effective epoch`，20个
  1000-step checkpoint、训练报告和strict restore全部齐备。仅计20段trainer invocation的实测墙钟分别为
  C69 `25102.06s (6.973h)`、C70 `24550.71s (6.820h)`；preview/seal另计且不包含模型训练。
- C70固定s20相对C67-s20的normalized/physical均值分别为`0.0601827/0.0250890`对
  `0.0606277/0.0254389`，但逐样本胜率仅`47.5%/52.5%`；normalized均值改善门也失败。gripper、language、
  visual安全门通过仍不能替代动作门，正式状态为`FAIL_C70_SAMPLER_BALANCED80_GATE / NO_C70_VS_C67_PAIRED_680_ROLLOUT`。
  C70停止，不选中间checkpoint、不加步数。最终`RESULTS.json`记录于
  `/mnt/h3-wam/outputs/c70-sampler-coverage-v1/fixed-s20-offline-34b81b8-v1/RESULTS.json`。
- C69的零重评封存因首次远程后台会话在哈希第14个大checkpoint后退出，未发布结果；保留partial故障证据后，
  从同一只读`e1872dc`源码重新执行完整20点SHA绑定。恢复结果
  `/mnt/h3-wam/eval/c67-vs-c69-fixed-s20-attribution-reseal-e1872dc-v3/RESULTS.json` SHA256为
  `12fb56ed96da82fdb232e7184648b2a7dd454eddd6344024a7e96f493ede12f9`，十项身份/完整性/conditioning门全部PASS。
- 固定s20离线端点非常接近：C67-joint与C69-action-only normalized MSE为`0.0606277/0.0606874`，physical
  MSE为`0.0254389/0.0254129`；C69逐样本赢`44/80` normalized和`45/80` physical。离线结果不宣布赢家，
  只放行`GO_C67_VS_C69_FIXED_S20_PAIRED_LIBERO_ATTRIBUTION`。
- 新闭环协议固定两臂s20000、四suite×10 tasks×新trial `50..66`，共680初始状态配对/1360个隔离进程；
  每对同环境seed、同policy noise、wait30、max400、replan8、horizon32、10次模型求解。可按pair-id在多A800节点
  分片，但任何shard不读取成功率；全部完成后一次性聚合。该结果只归因FACT consequence objective的增量价值，
  无论支持C67、支持C69或不显著，C58在单独被击败前仍是唯一carrier champion。
- C71 Light-WAM三层state-fusion的首个A800启动在模型加载前fail-close：只读snapshot验证通过，但launcher
  对无`.git`元数据的正常archive仍调用云端缺失的`git`命令。GPU、optimizer、checkpoint均未产生；修复后必须
  从新commit和新只读snapshot重跑，不能改旧snapshot。
