# H3-WAM context / consequence 历史探索总审计

审计日期：2026-08-17（Asia/Shanghai）  
审计性质：只读历史汇总；未启动训练、未修改任何在途实验、未重解释已冻结结论。

## 1. 审计口径与总判断

本审计交叉读取：

- `docs/H3_WAM_EXPERIMENT_LEDGER.md`；
- `docs/H3_WAM_CANDIDATE_REGISTRY_2026-08-14.md`；
- `experiments/dossiers/` 中 context、FACT、MiniWorld、ranking、failure 相关 dossier；
- 必要时读取 `experiments/evidence/` 中已冻结的只读数据审计。

字段冲突时采用最新、证据门更严格的记录。例如 C21 artifact 曾写
`GO_DATASET_EXPANSION`，后续代码审计证明它改变了整条 noise schedule，因此最终权限收窄为
`GO_ENTROPY_CALIBRATION_ONLY`。未在历史材料中记录的日期、实测墙钟或 effective epoch 一律写
`UNKNOWN`，不根据 GPU 数或 steps 反推。

当前结论：

1. **context 赛道尚无冠军。** 早期 executed-history 的离线改进没有稳定转成接触；C14 的窄任务收益
   没有跨 trial 复现且伤害 Object；完整预算 C57 反而略差于 D0；C62 正确/乱序上下文不可分；C64 只过
   机械门，尚未取得优化许可。
2. **consequence 机制本身成立，但外挂选择器路线没有转成稳定闭环收益。** E45/E48/E49、C38、C44、
   C51/C52 均提供不同层级的动作条件后果或价值信号；C45/C46、C53/C54 则连续否证这些外挂 ranker 的
   在线增益。
3. **联合动作—后果训练比外挂 scorer 更接近官方 FACT，但仍未晋级。** C55 离线优于匹配 action-only，
   fresh 680 组闭环却退化；C60 在 C58 上达到当前最高完整点估计 `313/680`，但 `+2.647pp`、净胜18、
   `p=0.0507164` 同时错过预注册门，因此仍须 `KEEP_C58_PARENT`。
4. 稳定失败模式不是“训练步数普遍太少”，而是 **训练/部署条件合同不一致、行为策略数据缺少候选动作
   反事实优势、时序结构被平均或错误 RoPE 抹除，以及 future/value 目标虽可学却不能约束动作生成**。

## 2. 官方来源身份

| 来源 | 固定 revision | 2026-08-17 本地身份 | 本审计使用的机制 | 不能越过的边界 |
|---|---|---|---|---|
| FACT | `618a6c16868699b6d4138941de6a863589ac00dd` | clean，`TRAINABLE`；远端 `9427ea4` 仅 README live-demo 变化 | clean-action K/V、`[P,A,G,V,I]` causal mask、future state/value、failure mask、Stage-2 argmin | H3 端口均为 `backbone_port`，不是官方 Wan2.2 模型复现 |
| MiniWorld | `e484206bbd4360ae56ed8abad51c83f2457ac092` | clean，world-model 训练代码可用；无 license file | 每4 action 对齐一个 latent frame、real-observation sink/FIFO、condition dropout、rolling KV/RoPE | 无动作输出头；不能把它直接称为 robot policy |
| LingBot-VA | `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb` | clean，`TRAINABLE` | 共享 video/action blocks、预测 cache rollback、真实 observation/executed-action persistent KV | C14 浅 history adapter 与 C57 五层 D0 port 都不是官方30层共享模型复现 |
| FastWAM | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` | registry 记录为固定 commit 的 dirty 只读来源 | 30层 ActionDiT、逐层 carrier、动作 flow 与 LIBERO evaluator | C58/C60 使用 H3，不是官方 Wan2.2 联合训练复现 |
| DreamWAM | `6e989facc0c452fd3488d75f60bc36411005558c` | clean，`TRAINABLE` | D0 五层动作父模型、layer-wise K/V | D0 为本项目父基线，不等同官方 DreamWAM 完整 structured-future 训练 |

## 3. temporal / context 路线

### 3.1 已执行实验账

| ID / 日期 | 父模型与唯一变量 | 预算：steps / global batch / samples / effective epochs / wall | 离线与闭环结果 | 最终 permission / effect | 失败原因与可复用资产 |
|---|---|---|---|---|---|
| E17–E32，2026-08-13～14 | clean shared-H3/LingBot 四流父结构；依次只改训练预算、adapter-only、无泄漏采样、quantile、timestep、noise、LR、WD | 单项多为100 steps、GB8、800 windows；E22累计2400 windows；其余总样本/epoch见 ledger；wall 多数 `UNKNOWN` | E30 无泄漏 video/action 改善 `14.020%/0.900%`，但固定 task3 `0/1`、无接触；E32 对齐 WD0.1 后 action 退化 `2.836%` | 各子项最终均未获效果许可；E30只保留优化配方 | 证明“共享世界流可学”不等于动作控制；复用资产：四流 mask、causal sample40、严格 restore、LR/WD 单变量结果 |
| E46/E47/E50，2026-08-13～15 | shared-H3 或 history16；唯一变量为预算或已执行动作历史 | E47 fresh 5000 steps、GB8、40000 samples、`0.199223` epoch，原估17.5h；E50 为预注册 history s2500 checkpoint | E47 s1000 causal action较step0改善17.79%，task3仍`0/1`无接触；E50 causal action/video `0.376101/0.255801`，仍`0/1` | `PASS_OFFLINE_GATE / FAIL_CLOSED_LOOP / STOP_BENCHMARK` | 早期历史实现后来发现 env/dataset codec 和夹爪方向错误，且旧 shared checkpoint 有 replicated-gradient 污染；这些闭环只保留实际输出，不用于否定官方 LingBot |
| C14，2026-08-15 | D0-H32-s14000/replan8；只增加零初始化16步 executed-action progress adapter，父模型全部冻结 | 3000 steps、GB8、24000 samples、`0.119534` epoch、14m46s；每500保存 | s14500 trials0/1窄门有长程 `2/10 vs 1/10` 且Object `5/8 vs 5/8`；独立trials2/3长程 `0/10 vs 0/10`，合并Object `9/16 vs 10/16`；s17000 Object `2/8 vs 5/8` | `MECHANISM_EVIDENCE / NO_GO_FUSION`；s17000 `FAIL_PAIRED_GATE / NOT_EVIDENCE_READY` | 输出确实依赖历史，但收益集中在单task/前两trial，不稳定且扰乱接触；保留 action-history codec、valid mask、adapter restore 与固定18-episode复核协议 |
| C57，2026-08-16～17 | D0-H32-s14000 五层父模型；加入 LingBot predicted-cache rollback + real observation/executed-action persistent KV 生命周期 | 5000 steps、GB80、400000 samples、200779 windows、`1.9922398249` epochs、34646.34s；8 GPU | fixed80 C57/D0 MSE `0.07717749/0.07628779`，相对改善 `-1.166%`、win rate43.75%；未做LIBERO | `C57_FINAL_OFFLINE_NO_GO / NO_GO / NOT_EVIDENCE_READY` | 完整预算已排除“只是不够久”；承载体仍为D0五层。保留200779-window/1542-episode无泄漏sequence manifest、536-token上限、persistent lifecycle及checkpoint restore |
| C62，2026-08-17 | 已晋级C58 s10000；只加 MiniWorld-style real-observation sink+FIFO、将两个4-action组共享平均后调制 context K/V | 100 steps、GB8、800 unique train samples/800 episodes、1.0 epoch、189.98s；heldout64/32 episodes | clean/shuffle MSE `0.0705387413/0.0705377535`，改善 `-0.001400%`；clean 比 context-off `0.0682336408`差3.378%；未rollout | `FAIL_C62_CAUSAL_OPTIMIZER_CANARY / NO_GO_C62_TRAINING / NOT_EVIDENCE_READY` | 两个时序组均值化且共享时间坐标抹掉顺序；保留C58 default-off exact、30/30 bridge gradient、真实restore与delta checkpoint |
| C64，2026-08-17 | C62合同；分两级只恢复一观测↔一4-action组和H3 temporal key RoPE reindex | 0 optimizer step；GB/samples/epoch不适用；real mechanical wall `UNKNOWN` | C64A交换后逆置action，K/V max-abs `0/0`；仅加RoPE的C64B为 `K=3.4609375,V=0`；13项测试及真实C58/H3机械门过；无heldout/rollout | `PASS_REAL_MECHANICAL_NO_GO_OPTIMIZER / NO_GO_OPTIMIZER / NOT_EVIDENCE_READY` | 相对C62同时改变 cadence/mean 与 learned-null，尚非元素级单变量；未冻结sequence manifest。保留framewise bridge、null sink、精确K re-rotation和source-audit文档 |

### 3.2 context 路线冻结判断

- 当前擂主仍是 C58 carrier，本表没有 `temporal/context winner`。
- C57 不能用“更完整地照搬 lifecycle”掩盖五层承载体差异；它已经用完整 5000-step 官方预算得出负结果。
- C62 不能追加 steps：correct/shuffle 已不可辨，且 context-off 更好。
- C64 可复用但还不能训练。其下一合法实验必须把 `cadence-only`、`+ learned-null`、`+ temporal RoPE`
  拆成三个父子臂，或直接移植 MiniWorld 内部 pre-QKV 6D modulation；不能把三项一起训练后归因。

## 4. consequence / ranking 路线

### 4.1 早期机制与标签诊断

| ID / 日期 | 父模型与唯一变量 | 预算 | 结果 | 最终状态 | 复用价值 |
|---|---|---|---|---|---|
| E45 / E48，dossier未写日期（已在2026-08-15前完成） | action-independent future-proprio父头；correct action vs deranged vs zero；E48只把100→500 steps | batch16；s100 1600 samples、1.5625 selected-subset epochs；s500 8000、7.8125 epochs；s500预计3min | s500 val MSE `0.012862` vs shuffled-train `0.068679` vs independent `0.073846`；clean模型shuffle后`0.212931` | `PASS_MECHANISM_GATE / EVIDENCE_READY`，**只限 action→future-proprio 机制**；不授权策略 | 三臂干预、episode-disjoint 1024/256 split、restore diff0 |
| E49，dossier未写日期（已在2026-08-15前完成） | E48；唯一把target换成start+32 H3 layer49固定256D投影 | s100/s500，batch16；s500 8000 samples、7.8125 selected-subset epochs | s500 `180.116` vs shuffled-train `194.065` vs independent `193.492`；eval shuffle `212.738` | consequence机制通过；`NO_GO` value/best-of-N，直到failure/counterfactual data门 | 固定随机投影、future-H3三臂trainer；证明H3后果表征可学 |
| C16–C18，2026-08-15 | 成功专家time-to-go probe；C18唯一删除absolute-step | 4000 train/2000 val；无动作/H3更新；两轮各16 shadow episodes、最多6400 env steps，344s | C17 AUROC0.1875；C18离线MAE `0.21545→0.09952`、R² `0.00418→0.74192`，shadow AUROC0.546875<0.65 | `NO_GO_POLICY_INTEGRATION / NO_GO_BEST_OF_N` | 保留冻结40-context ridge/17KB restore、只读shadow；失败说明成功demo time-to-go不能区分闭环停滞 |

### 4.2 因果数据合同 C19–C34

这些阶段主要产出**可训练数据与审计合同**，不是候选策略效果。

| ID / 日期 | 父与唯一变量 | 预算 / 产量 / wall | 数据或机械结果 | 最终 permission / effect | 失败原因或资产 |
|---|---|---|---|---|---|
| C19/C20，2026-08-15 | 旧state恢复；C20改为两个新env同state同8步动作 | 12 states；训练0；wall `UNKNOWN` | C19 state可写回但旧RGB/proprio不exact；C20 v2 四suite×3 state起终双相机逐像素一致，数值≤`1e-10` | `GO_COUNTERFACTUAL_COLLECTION_CANARY / NOT_EVIDENCE_READY` | canonical branch restore harness；首版process-global RNG失败归为infra |
| C21/C22，2026-08-15 | D0-H32-s14000；只改整条policy noise schedule | C21 16 branches/280s；C22 96 branches，四shard457/453/458/460s | C21 11/16成功、1 mixed；C22 71/96、7/24 mixed、四suite覆盖 | `GO_ENTROPY_CALIBRATION_ONLY`；C22只放行C23 | 后续replan seed被一起改变，不能把outcome归因给首动作；保留高熵state清单与事故恢复流程 |
| C23/C24，2026-08-15 | C23仅首seed变化且固定continuation；C24只改首chunk执行8→16/32 | C23 32 branches、shard137–161s；C24每臂32 branches、shard约139–158s | C23 18/32、1 mixed；C24 h16有2 mixed，h32有3 mixed/2 suites并过门 | `GO_EPISODE_DISJOINT_CAUSAL_DATASET_CANARY / NOT_EVIDENCE_READY` | 两段seed合同、first-action bit-exact审计；h32只改善标签可辨识性，不证明部署优越 |
| C25，2026-08-15 | C24 h32；增加14个source并按source隔离split | 128 branches；约506–515s/shard；训练0 | 70/128；9 mixed=train6/val3，覆盖3 suites | `GO_FROZEN_H3_ACTION_CRITIC_CANARY / NOT_EVIDENCE_READY` | 32 exact states、128轨迹、source-disjoint split，约324MB |
| C27，2026-08-15 | C25后更换为39个未消费source | 312 branches；特征38.24s，峰值29.68GiB | 198成功；17 mixed=train13/val4、42/12 pairs，3 suites | `PASS_DATA_GATE`，只放行一次C28 | dataset/feature SHA、78个bit-exact states；LIBERO-10无fresh source，明确不冒充覆盖 |
| C29/C30，2026-08-15 | C29只换trials4..7；C30仅首seed且强制真实post-action terminal consequence | C29 160 ep，约950–989s/node；C30 488 branches、32GPU、估0.6h | C29 61/160成功；C30 317/488、28 mixed、train25/val3、147 terminal fallback | C29 PASS；C30 `FAIL_DATA_GATE / NO_GO_ACTION_CONDITIONED_CONSEQUENCE_TRAINING` | 预注册要求val mixed≥4，实际3，未事后降门；保留terminal字段、executed-tail mask与61-source资产 |
| C31 | C30数据父；flattened vs MiniWorld 8-token action-conditioned future-H3 | 计划10000 steps、batch64；**实际0 steps** | 11项代码/loader测试过，C30门失败后未启动 | `HOLD_DATA_GATE` | 保留temporal consequence adapter与proposal/executed mask，不产生效果结论 |
| C32/C33/C34，2026-08-15～16 | 只换fresh trials8..11并冻结rank-only角色；C34只组合不泄漏split | C32 160 ep；C33 416 branches；C34 194 states/776 branches；训练0 | C32 52成功；C33 279/416、24 mixed/四suite；C34 dataset+online H3 feature均冻结 | `PASS_DATA_GATE`；放行C35/C40一次性使用 | 可复用的C34 dataset/feature SHA；C33 IPC端口事故按结果式恢复，未污染outcome |

### 4.3 静态 critic 与 action-conditioned consequence

| ID / 日期 | 父与唯一变量 | 训练预算 | 离线 / 闭环结果 | 最终 permission / effect | 失败原因与资产 |
|---|---|---|---|---|---|
| C26，2026-08-15 | C25；action-only、H3×action、FACT consequence三种冻结表征 | 每臂1000 full-pair steps、GB42、42000 samples、42 unique、1000 pair epochs；单GPU约0.0002h | train均21/21；heldout为0/9、4/9、0/9，H3 top1 2/3、p=.6875 | `NO_GO_BEST_OF_N / NOT_EVIDENCE_READY` | 小样本跨episode反转；保留dataset/features/critic和train-only LOO配置 |
| C28，2026-08-15 | C26预选配置；唯一加入fresh C27 split | 10 full-pair steps、GB144、1440 samples、144 unique、10 epochs；单GPU估0.005h | H3 train65/72但fresh6/12；action-only fresh7/12；H3 top1 2/4、p=.58594 | `NO_GO_BEST_OF_N` | 排除“C26只是训练不足”；静态state×action不泛化 |
| C35，2026-08-16 | C34；flattened vs 8-token temporal，seed42/314159，correct/shuffle/zero三臂 | 每job每臂10000、batch64；四job×三臂合计GB记账768、7.68M samples、296 unique、25945.95 aggregate epochs；估1h | temporal两seed均优于flattened，shuffle伤5.1–6.1%，但一seed独立null门差0.38% | strict FAIL / `NOT_EVIDENCE_READY` | 独立null模型不稳定；保留temporal/raw结构与每1k checkpoints |
| C36，2026-08-16 | C35；唯一按train-only future-current std缩放delta | 同C35；估0.1h | 仍只有一seed越过独立门 | strict FAIL | target scaling未解决跨seed不稳 |
| C37，2026-08-16 | C35；唯一增加10% structured zero-action dropout并用same-model null | 同C35；估0.1h | dropout让shared-null过强且跨seed不稳 | strict FAIL | 保留“condition dropout有害”的单变量负证据 |
| C38，2026-08-16 | C37；去掉dropout、固定temporal/raw、换四个全新seed | 4 jobs×10000、batch64；dossier aggregate GB768/7.68M/25945.95 epochs；估0.1h | 最小true-vs-null增益12.629%、shuffle退化1.765%、shuffled-train增益7.481%；restore max-abs0 | mechanism PASS；只授权C40 fresh ranking，不是策略效果 | 当前可靠action→future-H3 ensemble；保留4 checkpoint、eval-mode restore修复 |
| C40，2026-08-16 | C38 ensemble；相对action-only只增加预测consequence | 100 steps、GB148、14800 samples、74 pairs、100 epochs、估0.1h | fresh pairwise54.321%、top1 58.333%、p=.3033；比action-only高13.58pp | `FAIL / NO_GO_BEST_OF_N` | 方向有信号但样本/泛化/显著性不足；C33已消费，禁止继续调参 |

### 4.4 powered binary ranker 与 dense continuous value

| ID / 日期 | 父与唯一变量 | 预算 | 离线 / 闭环 | 最终状态 | 失败原因与资产 |
|---|---|---|---|---|---|
| C41–C43，2026-08-16 | 只扩fresh source和first-action outcomes，不改D0 | C41 160 ep；C42新增240 ep、最终总400；C43 1128 branches；训练0 | C42 141成功；C43 716/1128，train/final 25/30 mixed | 数据门PASS | 141 source、282 exact states、1128分支，train/final role预先冻结 |
| C44，2026-08-16 | C40模型族不变，只扩C43 train/final来源 | ranker最多100 full-pair steps；47 mixed/154 train pairs；单GPU估0.5h | fresh 67/98=68.367%、top1 23/30=76.667%、p=.00271；action-only60/98 | `PASS_OFFLINE_RANKING`，仅放行闭环canary | 可靠的独立源离线binary ranking资产；不是在线成功证据 |
| C45，2026-08-16 | D0；每8步用C44从4个相邻seed候选重排 | 20 paired episodes；估2h | parent8/20，best-of-4 3/20；1胜6负 | strict FAIL | 与C43的offset和单次32步干预合同错位；保留916次score与合同诊断 |
| C46，2026-08-16 | C45；唯一对齐step80单次32步、C43四seed，之后replan8 | 20 pairs；估2h | control/candidate均6/20，1胜1负18同 | `NO_GO` online binary ranker | 合同对齐消除退化但仍无增益；正式停止C44外挂binary selector |
| C47–C49，2026-08-15～16 | 只扩完整父轨迹、按每replan密集展开，并在线冻结H3 feature | C47 160 ep；C48 15417/3133/3009 rows、21559总；C49 22119 features，32GPU、估1h | 数据/feature完整门全过 | 放行C50 | 可复用dataset SHA、22119个`[1,32,5376]` BF16及固定256D投影；H3始终冻结 |
| C50/C51，2026-08-16 | C38四seed；joint small consequence vs frozen consequence；C51一次读取final | 8 jobs×10000 steps、batch64、每job640000 balanced examples；GB按job64；wall `UNKNOWN` | C51 final value MSE .188720，比mean baseline .258225低26.9%；rank corr .539721，failure-success margin .656632，shuffle MSE .192992>.188720 | `PASS_HELDOUT_TRAJECTORY_VALUE`；只放行fresh counterfactual ranking | 价值预测成立但action generator absent/frozen；保留seed8675309 s10000 checkpoint |
| C52，2026-08-16 | 冻结C51；只在60个近成功state执行四个未见offset | 240 branches；训练0；wall `UNKNOWN` | 148成功、17 mixed；37/56=66.071%、top1 14/17、p=.03892 | offline ranking PASS，只放行C53 | source observation曾用于C51 final，不是fresh视觉源 |
| C53，2026-08-16 | C52 scorer；step80只重排一次32步 | 20 pairs；训练0 | control7/20、dense8/20，1胜0负，未达+2/3胜/净胜2 | `NO_GO_DENSE_VALUE_ONLINE` | 点估计正但功效门未过；step80与训练近成功分布错位 |
| C54，2026-08-16 | C53；唯一改为fresh trials29..32父轨迹的d3/d5 state | 160 parent ep→128 states/256 pairs；训练0 | candidate0 87/128、dense86/128；3胜4负、`-0.781pp`、p=.773438 | `FAIL / NO_GO_DENSE_VALUE_ONLINE` | 排除触发时机和步数；behavior-policy time-to-go没有学习候选动作反事实优势；保留全新视觉源配对闭环否证 |

### 4.5 联合动作—后果与完整 FACT port

| ID / 日期 | 父与唯一变量 | 预算 | 离线 / 闭环 | 最终 permission / effect | 失败原因与资产 |
|---|---|---|---|---|---|
| C55，2026-08-16 | D0五层；相对action-only只在相同ActionDiT block增加clean-action第二forward与future-H3/state/value auxiliary | 每臂6000 steps、GB16、96000 rows；demo0.239 epoch、rollout3.113 balanced epochs；实测约0.6h；每1k保存 | selected s1000 相对action-only normalized/physical改善1.62%/1.49%，机制shuffle退化0.085893；但680组三臂 joint231、action-only234、D0 270，joint vs action-only 36胜39负/p=.6778 | `PASS_C55_OFFLINE_GATE` 后被闭环否证为 `FAIL_CLOSED_LOOP / NO_GO_FUSION` | shallow auxiliary能学后果但未约束动作，且两续训臂都低于D0；保留18550-observation K/V、train-only z-score、动作codec修复、1k–6k学习曲线 |
| C56a，2026-08-16 | D0；增加独立4层FACT causal trunk | 1 step、GB8、8 samples、8GPU、0.021633h | 8/8机械、parity/restore0、future/value梯度过；无效果 | `PROBE_ONLY / NO_GO_LONG / NOT_EVIDENCE_READY` | 与官方共享backbone偏差太大；保留`[P,A,G,V,I]` mask、failure action mask实现 |
| C56b/C60，2026-08-16～17 | C58 s10000；在同30层中加入FACT clean-action/future/state/value/failure mask，online冻结H3 | 10000 steps、GB8、80000 samples、218125 unique、aggregate `0.366761` epoch；rank mixture4/2/1/1；8GPU；稳态估4.41h，每1k保存 | balanced80 restore/conditioning 10/10过；680对 C60 `313/680` vs C58 `295/680`，`+2.647pp`，63胜45负，p=.0507164；suite安全过 | `FAIL_C60_FACT_EXPANDED_PAIRED / NOT_EVIDENCE_READY / KEEP_C58_PARENT`；`NO_EVIDENCE_FOR_S20K_CONTINUATION` | 差一点但同时错过+3pp、净胜20、p≤.05三门；scheduler已到LR0。保留当前最高完整点估计、C60 checkpoint、680 pair evidence与future target normalization |
| C61 matched failure，2026-08-17 | C60；唯一将51-episode C60 failure pool换成更广C61 exact-state failure | 计划10000/GB8/80000；实际 matched long 未获许可；C61 collection 1128 branches | trial33四suite×10 task canary：C61/C60/C58=`17/40,20/40,18/40`；C61相对C60为0胜3负 | `NO_GO_EXPANSION / NOT_EVIDENCE_READY` | 更多负例并未改善；C61仍提供1128 exact jobs、387 failures、48 mixed groups供只读诊断，但分布是D0+h32首chunk而非C60/replan8 |
| C63，2026-08-17 | 冻结C60 Stage-2；同state只换successful-parent/failed-counterfactual action | 0训练；32 pairs，11 parent trajectories；suite严重偏Spatial30/Object2 | v1/v2在评分前发现value shape广播错误并修正；最终1/32出现exact BF16 tie，违反all-nonzero机械门 | immutable FAIL；`NO_GO_C60_STAGE2_RANKING_KEEP_C58` | 保留官方`value[:,0,0]`、raw=normalized+1、argmin、同噪声/逆序不变性合同；不能用FP32消除真实部署tie |
| C65，2026-08-17 | 新候选，拟用fresh C60/replan8八候选同state pairs做四suite确认 | 当前0训练、0 score；目标80 pairs（每suite20独立mixed source） | 只读C61审计：1128 jobs、741/387成功失败、48 mixed；严格独立source Spatial21/Object15/Goal2/L10=0，数据门失败 | `BLOCKED_C61_DATA_GAP_NO_SCORE / NO_SCORE_COLLECT_C65_DEPLOYMENT_DISTRIBUTION_PAIRS` | C61来源/执行合同与C60部署不一致，不能凑pair。复用资产是C65 auditor、冻结hash和完整fresh collection/score预注册合同 |

## 5. 失败原因归因矩阵

| 失败类型 | 直接证据 | 被排除的简单解释 | 后续必须继承的约束 |
|---|---|---|---|
| 离线误差改善不转闭环 | E30/E47/E50、C14、C55 | 不是单纯“没训练”：C57完整1.99 epochs仍负；C55跑满6k且有完整曲线 | context/consequence必须同时过paired action与fresh闭环，不能用future MSE替代 |
| 静态state×action不泛化 | C26/C28 train强拟合、fresh反转 | C28把监督扩到72 pairs仍失败 | consequence必须显式建模action→future dynamics，而非线性乘积shortcut |
| 外挂value/ranker训推错配 | C45/C46、C53/C54 | 已分别对齐seed/执行32步和触发到d3/d5，仍无增益 | 价值监督进入动作生成共享块；不再调外挂阈值或补trial刷门 |
| 时序合同被抹除 | C62 clean≈shuffle；C64B证明RoPE位置可辨 | 不是bridge不可达：30/30梯度与restore均过 | 保留每4 action一个frame slot、real sink/FIFO、连续time reindex；单变量拆null/cadence/RoPE |
| failure标签不完整 | C16–18、C55 observational failure、C61/C65 | 成功time-to-go和terminal outcome不足以等价failure-active onset | 失败动作不模仿；onset未知时mask value；反事实pair按source隔离且执行合同对称 |
| 训练/执行源污染 | C22缺site、C33端口冲突、C55 CUDA库、早期shared replicated grad | 均有0-result或结果式恢复证据，不能算policy失败 | 固定runtime、明确LD_LIBRARY_PATH、one-result identity、事故目录永不混入正式结果 |

## 6. 可复用资产索引

### context

- C57 sequence manifest：200779 windows / 1542 episodes / future leakage 0；最大536 tokens。
- C57 persistent lifecycle：predicted-cache rollback、real observation/action commit、严格restore。
- C62/C64 C58-compatible bridge：default-off exact、30层梯度、real-H3 restore。
- C64 source audit与 temporal key delta rotation；C64A/C64B pair-swap falsification。
- C14 environment action history codec、left-padding valid mask和父模型冻结 adapter-only 训练器。

### consequence / data

- C20 canonical branch restore；C23 first/continuation 两段seed；C24 h32 label-identifiability合同。
- C34：194 states / 776 branches 的train/selection/ranking-only冻结角色及在线H3 feature。
- C43：141成功source、282 exact states、1128 first-action branches、train/final预分工。
- C48/C49：21559 dense replan rows、22119去重观测、在线INT8 H3 feature与固定256D投影。
- C38 四seed temporal consequence ensemble及eval-mode严格restore修复。
- C44、C51、C52 的离线正证据，连同 C45/C46、C53/C54 对应在线否证，必须成对保留。
- C55/C56b 的环境↔dataset gripper round-trip、train-only target z-score、failure action mask、
  `[P,A,G,V,I]` causal token mask、online no-cache H3路径。
- C58/C60 680对 one-episode-per-process、严格初态、pair evidence与McNemar聚合器。
- C63/C65 official Stage-2 value shape/solver/argmin审计；C65 四suite80-pair新数据门。

## 7. 对下一轮研发的约束性结论

1. fusion lineage 仍是 `C58 carrier champion -> + context winner -> + consequence winner`；后两项为空，
   C60 只能称“最高完整点估计”，不能填 consequence winner。
2. context 下一步若继续，优先做 C64 三臂元素级 ablation；没有 clean-vs-shuffle 机制门前不得长训或rollout。
3. consequence 下一步不再训练外置线性 ranker。可行路线只有：
   - 以 C58 为父，在共享30层动作块内训练 source-faithful FACT Stage-2/action joint objective；或
   - 先按 C65 合同采集 C60/replan8、每suite≥20独立mixed source，再只读检验当前Stage-2是否真的会选动作。
4. 新数据必须来自候选部署分布、同state、候选只改首动作seed、后续replan8和noise逐值相同；C61的
   D0/first32近邻数据不能替代。
5. 任何新长训必须同时报告 steps、global batch、samples、effective epochs、墙钟和checkpoint曲线；
   `UNKNOWN` 字段先做吞吐/数据审计，不能用估算冒充实测。

## 8. 冻结结论清单

- context：`NO_TRACK_CHAMPION`；C57、C62均`NO_GO`；C64仅机械PASS。
- consequence mechanism：E49/C38可称动作条件后果机制有效；不能称策略有效。
- offline rank/value：C44、C51/C52 PASS；它们的在线父子 C45/C46、C53/C54 FAIL。
- joint auxiliary：C55 offline PASS、fresh closed-loop FAIL。
- full shared FACT：C60 `NOT_EVIDENCE_READY / KEEP_C58_PARENT`；C61 `NO_GO_EXPANSION`。
- Stage-2 selector：C63 FAIL；C65因四suitefresh数据缺口而`NO_SCORE`。
- 唯一已晋级父节点仍为 C58 `EVIDENCE_READY / CARRIER_TRACK_CHAMPION`。

## 9. 2026-08-17 追加执行状态：C66 否证、C67 长预算与下一诊断

- C66 以 C58 s10000 为唯一父模型，把 LingBot committed observation/action K/V 放入同一 30 层
  ActionDiT。固定 8 卡、100 steps、global batch 8、800 unique samples、1.0 epoch；heldout 为四套件
  64 条，墙钟 891.502 秒。clean/shuffle/context-off MSE 分别为
  `0.10568063/0.11943121/0.07991016`：正确历史相对乱序改善 `11.513%`，证明历史内容进入模型；但
  clean 相对 context-off 退化 `32.249%`，正式结论为
  `FAIL_C66_PAIRED_CANARY / NO_GO_C66_LONG_TRAINING / NOT_LIBERO_EVIDENCE`。因此不得通过追加 steps
  掩盖结构/优化干扰。
- C66 的 `runtime_restore_exact=false` 与主要 effect 失败分开处理。候选原因是同一 BF16
  FlashAttention forward 重算的非确定性使 bit-exact 门过严；即使后续证实，也不改变 clean 比
  context-off 退化 32.249% 的停止结论。
- 新增 analysis-only 的 `evaluate_c66_context_length_diagnostic.py`：在完全相同 heldout/noise 上比较
  C58 parent 与 C66 s100，并各测 context-off、最近 1/3/7 个 committed chunks。假设是该配对能分离
  structural-prefix harm、100-step optimization harm 与 excessive-history harm。该脚本 optimizer steps=0，
  只能选择下一次 bounded mechanism candidate，不能授权长训、rollout 或晋级。
- C67 是 C60 的唯一变量训练预算消融：保持 C58 parent、FACT objective、数据、global batch 8 与
  20k cosine trajectory 不变，从 fresh trajectory 训练到 20000 steps（160000 samples，aggregate
  effective epoch `0.733522`），每 1000 步 checkpoint+strict restore。只有固定 balanced80 里程碑门通过
  才允许 680-pair rollout；训练进行中始终为 `NOT_EVIDENCE_READY`，C58 仍是 carrier champion。
- C67 里程碑评测改为异步但不自适应：每个已完成 checkpoint 先复用 finalizer 的同一训练/restore 审计，
  然后在固定 balanced80 上产生隔离的 preview report；preview 明确禁止 early stopping、checkpoint
  selection 与 rollout。20k 完成后，sealer 逐 checkpoint SHA256 比较 preview audit 与
  `TRAINING_COMPLETE.json` 内最终 audit，完全一致才无模型重算地绑定最终证据并交给原20点聚合器。
  该调度只减少训练结束后的评测等待，不改变模型、数据、160000 samples 或任何效果阈值。
