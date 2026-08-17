# C60 终局与训练预算诊断

日期：2026-08-17。本文先冻结诊断合同，再运行任何 s1k..s10k milestone offline evaluation。

## 已完成的闭环结论

C60/C58 的 680 对结果位于
`/mnt/h3-wam/outputs/c56b-fact-online-v1/c60-expanded-paired-trials34-49-v1-isolated/RESULTS.json`，
SHA256 为 `d9280c5ad4aeac231a8da793ac5f5d667f005dbc8c5cfe3657b93a4895483ec3`。独立重聚合重新读取
640×2 个 result/trajectory、trial33 桥接和全部初态摘要，得到同一结果：

- C60 `313/680=46.0294%`，C58 `295/680=43.3824%`，点估计 `+2.6471pp`；
- paired wins `63:45`，净胜18；单侧 exact McNemar `p=0.0507164`；
- Spatial `98:92`（`+3.529pp`）、Object `139:131`（`+4.706pp`）、Goal `44:44`、
  LIBERO-10 `32:28`（`+2.353pp`）；无 suite 触发 `-3pp` 安全退化；
- 完整性、trial33 桥接、640 对 full initial-state exact 和两臂 fresh-process 合同全部通过；
- 但 `+3pp`、净胜20、单侧 `p<=0.05` 三项效果门均未过。因此严格结论为
  `FAIL_C60_FACT_EXPANDED_PAIRED / KEEP_C58_PARENT / NOT_EVIDENCE_READY`。`313/680` 只能称最高完整
  评测点估计，不能称 FACT 已晋级。

C61 matched 在固定 trial33 为 `17/40`，同批 C60 为 `20/40`、C58 为 `18/40`；C61-C60
`-7.5pp`、C61-C58 `-2.5pp`，配对 C61-vs-C60 为0胜3负。因此 C61 已按预注册门判
`NO_GO_EXPANSION`，不进入更大闭环，也不能与 C60 事后择优后再对 C58。

## s1k..s10k 固定 offline 曲线预注册

一句话假设：如果 s10000 仍是训练不足而不是饱和/退化，那么相同 balanced-80 上的动作误差应从中期
持续改善到末期，同时 gripper、语言和视觉条件响应不能坍塌。

本诊断不选择既有 checkpoint 做闭环，也不改变上述 680 对结论。十个 checkpoint 全部评测，禁止依据
closed-loop success 或中间 offline 值停队列：

- checkpoint：s1000..s10000，每1000一步，共10个；每个都须已有独立 restore `max_abs=0`；
- 数据：相同 episode-disjoint balanced-80，40 task×2，selected IDs SHA256
  `26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42`；
- H3、动作 normalization、FlowMatch shift5、10 solver evaluations、seed42和逐 sample noise完全固定；
- 每个 checkpoint 报 normalized/physical MSE、gripper macro-F1、prediction std、language replacement、
  无 self-map visual shuffle，以及逐 sample paired error；总计800次模型-样本评测；
- n2 八卡并行，GPU0/1各跑两个checkpoint，其余各一个，预计约10–15分钟；执行源必须为新只读快照。

只有以下预注册门全部通过，结果才允许写“可起草 s20k 续训 dossier”，并不自动启动训练：十个
conditioning gate全过；s8k–s10k平均 normalized/physical MSE不差于s4k–s6k；s10k两种MSE不差于
s5k；s10k gripper不低于s5k超过0.005；s10k语言与视觉响应各至少保留s5k的90%。未全部通过则记
`NO_EVIDENCE_FOR_S20K_CONTINUATION`。

## 当前预算事实

C60 global batch8，s10000共见80000样本、218125 unique windows，总混合 effective epoch
`0.366761`。分流暴露为 expert `0.199224`、success rollout `8.107013`、observational failure
`0.772201`、causal failure `5.184033`。训练最后三段 action loss 均值为s8k `0.064754`、s9k
`0.066642`、s10k `0.063429`，没有仅凭训练 loss 证明继续的依据；且原 cosine scheduler 在s10000
已经到 LR=0。任何 s20k 都必须视为新优化合同，而不是原 checkpoint 的无变化续跑。

## 固定曲线结果与预算结论

十个checkpoint全部完成，共`10×80=800`个模型-样本评测；每个checkpoint都通过conditioning与
strict restore，selected IDs、H3、数据、噪声、solver和normalization身份完全一致。正式结果为
`/mnt/h3-wam/outputs/c56b-fact-online-v1/milestone-balanced80-s1k-s10k-v1/RESULTS.json`，SHA256
`2008293c4cc11ccfb333c67aaf72dd888920b59c1e1ebeb2ddb343a8268e325e`。不导入正式聚合模块的独立JSON
重算验证了10个report hash、80个逐样本身份、全部指标与门禁，结论一致。

- 中期s4k–s6k与晚期s8k–s10k平均physical MSE为`0.0256052 -> 0.0252404`，normalized MSE为
  `0.0616219 -> 0.0601616`，两项窗口门通过；
- s5k physical/normalized为`0.0252263/0.0611622`，s10k为`0.0252567/0.0602009`。s10k normalized
  较好，但physical差`0.00003042`，所以`s10_physical_not_worse_than_s5=false`；
- s10k gripper `0.933197`、language delta `0.224284`、visual-shuffle MSE `0.0370162`，均通过相对s5k
  的保真门；因此不是conditioning collapse，而是动作误差曲线没有给出继续训练的完整证据；
- 逐样本paired字段中`left=s1k/right=s10k`，较小误差记胜：normalized上s10k为49胜、s1k为31胜，
  但`s10k-s1k`均值仍为`+0.00229842`；physical上s10k为51胜、s1k为29胜，均值仍为
  `+0.0000383514`。这表示多数样本小幅变好、少数退化幅度更大，不能把“49/51胜”误写成总体MSE改善。

预注册门要求全部通过；physical s10k-vs-s5k一项失败即得到
`NO_EVIDENCE_FOR_S20K_CONTINUATION`。因此不创建s20k dossier、不重启scheduler、不启动长训；n2评测进程
已自然退出并释放8张GPU。C60的闭环结论仍为`KEEP_C58_PARENT`。
