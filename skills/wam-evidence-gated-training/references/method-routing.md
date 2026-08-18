# WAM 方法路由与证据边界

本文件用于选择该查哪套实现，不替代每次重新浏览论文和官方仓库。

| 项目 | 一手来源 | 可支持的结论 | 不能直接支持的结论 | 审计重点 |
|---|---|---|---|---|
| Fast-WAM | arXiv:2603.16666；`yuantianyuan01/FastWAM` | video co-training 可在推理时不生成 future；官方 LIBERO 数据/动作/评测口径；动作目标是 shifted flow | 冻结任意视频 backbone、只训浅 head 也会成功；官方含 clean-action regression 或 policy teacher roll-in | 33-step span、stride4 得 9 视频帧/32 actions、10 epochs、30-layer ActionDiT、动作归一化与 rollout；从源码区分官方 flow 与本地 novel composition |
| StarWAM | `shaohua-pan/StarWAM` | Wan feature-conditioned 30-layer ActionDiT、last-layer pooled visual context、shifted flow、可训练视觉主干 | 冻结 INT8 H3 last-layer cache 等价于官方 joint backbone training | launcher resolved config、feature timestep、context 顺序、backbone optimizer group、10-step Euler、pre-denorm clamp |
| Faster-WAM / DoT | arXiv:2608.02365；截至审计时未确认作者官方代码 | 单层 action head 可从视频 backbone 所有层 K/V 获得表示；需要 RoPE realignment | 任意自写 KV fusion 都等价；论文重实现可作为训练主线；冻结 backbone 足够 | 没有作者代码时标 `paper-only backbone_port`；可做有界诊断长训，但不能冒充官方复现；测试 all-layer K/V、mask、RoPE、gradient，并以机制与闭环结果单独决定 `EVIDENCE_READY` |
| Faster-WAM future conditioning | arXiv:2608.04404 | SparseMoT + Interval KV-Fusion 在推理保留 future 表征 | 与 DoT 同一项目；head-only 等于 future conditioning | 区分两篇同名 Faster-WAM；确认 future 是怎样计算/复用和推理成本 |
| Light-WAM | arXiv:2606.08242；`L1ziang/Light-WAM` | 冻结小型视频 backbone、全层 LoRA、多层 adapter、learned-query state fusion 和浅 action trunk 的可执行组合 | INT8 H3 冻结 K/V 自动等价；官方 LoRA 可与浅 head 同时移植且仍是单变量 | 固定 launcher 的 steps/global batch；核对 adapter 层、query pooling、prefix action weighting 与 LoRA optimizer groups；H3 端口优先分开验证 shallow fusion 和 backbone LoRA |
| WLA | arXiv:2606.05979；`SJTU-DENG-Lab/WLA` | 官方同预算 action-only 与 image-action 配置可用于严格归因 world objective；history/meta-query 路由可执行 | 不同 backbone 的官方指标可直接横比；任意 future loss 都具备动作收益 | 固定 LIBERO-all 两臂的 steps、batch、history、数据和 evaluator；优先移植 matched world-on/off 配对实验设计 |
| GAM | arXiv:2606.17046；`cvlab-kaist/Geometric-Action-Model` | geometry foundation representation、因果 future predictor、depth/geometry 辅助和 action history 的完整训练实现 | pooled H3 K/V 是显式几何表示；只加 depth loss 即复现 GAM | 审计冻结 block 边界、future predictor 深度、depth target、history/future horizon mixture 和 resolved global batch；作为结构化 future 独立赛道 |
| SelfWAM | arXiv:2608.00725；截至审计时未确认作者官方训练代码 | clean-action-conditioned future 与 robot self-mask 可提出降低无关视觉变化的受控假设 | 论文摘要足以放行主训练；需要 SAM3D 才能生成 self-mask | 标 `paper-only`；LIBERO 优先使用 simulator robot mask 做单变量实验，先验证 mask/target 和因果梯度再占用长训资源 |
| DreamWAM | arXiv:2608.04996；`hustvl/DreamWAM` | RGB+motion 联合 latent denoise、geometry/semantics supervision；action Q 在每层读取该层 video K/V | 只加 RAFT loss 60 steps即复现；把同一个 last-layer context 重复注入每层等价于 DreamWAM carrier | 官方 config 21,700 steps、LR/warmup、逐层 K/V 身份与 alias、flow init、DINO/depth routing、joint vs uncond |
| MiniWorld | arXiv:2608.01127；`zhao-yian/MiniWorld` | action-conditioned、block-causal、diffusion-forcing streaming video WM；多阶段长度 curriculum | 它是可直接 rollout 的机器人 action policy；可直接替换 H3 head | DROID action keys/camera、6→16→32→64 latent-frame stages、Muon、rolling KV cache；若接入 WAM 要另建 policy bridge |
| FACT | `Bariona/FACT` | causal act-then-imagine；pred action 不看未来，clean action 条件 future state/value/video；失败样本屏蔽 imitation；best-of-N value ranking | 冻结 H3 action head 天然具备 consequence/value；没有失败 canonical 数据也能声称 failure-aware | teacher-forcing mask、两阶段推理、failure onset/action mask、future/value target、BoN 排序；新仓库/单 commit 时提高审计强度 |
| ImageWAM | arXiv:2606.19531；`yuyangalin/ImageWAM` | text-guided target edit/cache 可能比密集视频更聚焦任务相关变化 | H3 I2V 天然等价 image-editing prior | source-conditioned target frame、KV cache 接 action expert、冻结理解模块、single-step inference；适合作为目标/语言绑定独立路线 |
| RepWAM | arXiv:2606.13674；`wdrink/RepWAM` | semantic visual-action tokenizer 与 latent action pretraining 可作为长期 representation 候选 | 当前仓库一定包含可直接运行的完整 LIBERO 训练配方；少量 policy SFT 等于 tokenizer pretraining | 先确认真实 dataloader/optimizer/restore/evaluator 的开源完整度；若缺失则标 `PARTIAL`，不能用论文预算放行训练 |
| LingBot-VA | `Robbyant/lingbot-va`；LeRobot `policies/lingbot_va` | Wan2.2 双流视频/动作 flow matching、block-causal streaming、真实观测与执行动作回灌 KV cache | 只训浅动作头能获得同样长上下文能力 | launcher 覆盖、预计算 latent 的 frame ids、30D action mask、两套 scheduler、完整 transformer 反传、LIBERO client |
| DiT4DiT | `Mondo-Robotics/DiT4DiT` | future-video loss 更新视频 backbone；独立 16 层 ActionDiT 从指定中间层取特征 | action loss 直接更新视频 backbone；默认 YAML 就是实际运行配置 | hook 中 `detach`、launcher 将 logit-normal 攓为 uniform、8-action/9-frame窗口、80k steps、64 GPU evaluator |
| Motus | `thu-ml/Motus` | 三专家 MoT、视频和动作联合 flow matching；Stage-2 latent-action 预训练后 Stage-3 embodiment SFT | 直接从 H3 原始权重做少量 LIBERO SFT等价于 Motus | Stage-2 checkpoint、30层 1:1 joint attention、动作/视频频率、全参数训练、RoboTwin evaluator |
| Cosmos Policy | `NVlabs/cosmos-policy` | 将 proprio/action/future state/value 编码进视频 latent slots，LIBERO 四 suite 完整训练/评测 | 其 2B 配方可原样缩放到 32B H3 | demo+成功/失败 rollout mixture、latent mask、chunk16、global batch1920、40k step、action L1 和 device-sensitive eval |
| RoboTTT | NVIDIA project/paper | 长上下文 rollout 的 test-time adaptation | 能修复第一步就选错目标的 policy | 仅在静态/短程闭环已有正例后评估，严格区分在线适应与训练增益 |

## 代码优先选择规则

1. 优先选择包含 dataloader、真实训练 loop、官方 config、checkpoint restore 和闭环 evaluator 的仓库。
2. 只有模型定义或伪代码的仓库不能作为训练配方来源。
3. README 参数与代码默认值冲突时，以 launcher 展开后的 resolved config 和执行代码为准。
4. 有官方 checkpoint 时先在官方 evaluator 上复现至少一个已报告基线，再移植 H3。
5. H3 替换实验必须保留同仓库、同数据、同 evaluator 的原 backbone 对照；否则无法归因。

## 条件坍塌时的方法路由

当 physical MSE/ADE 随训练改善，但 visual-feature shuffle、language replacement 或 gripper
指标持续坍塌时，先按 `conditioning collapse` 而不是“训练不足”路由：

1. 回到执行代码检查 feature 是否真正进入 forward、是否被 mask/detach、loss weight 是否为零、
   optimizer 是否覆盖 projector，以及 AMP 下是否发生精度归零。
2. 固定 evaluator、样本、噪声、solver 和 checkpoint 身份，比较至少一个早期与一个晚期点；视觉
   shuffle 必须是无 self-map 置换，语言测试只能替换指令。
3. 若 paired sensitivity 近乎消失且预期视觉梯度归零，停止同配方增步，下一候选必须用单变量机制
   阻止 action-prior bypass；不得先堆 world loss、history、LoRA 或多种正则后再归因。
4. 以固定 commit 的真实 forward/loss/optimizer/evaluator 和原始实验 artifact 决定是否放行；论文
   只用于从官方方法中选择下一单变量候选，不能替代上述证据。
5. 新增辅助损失必须与未启用它的同 step 父配置做 paired comparison；候选自身早晚曲线不能归因。
   反事实响应同时看绝对值和相对输出尺度，避免把整体动作幅值变化误判成条件依赖增强。

## 项目替换规则

将 Wan2.2 换成 MiniMax-H3 时，逐项重新证明：

- latent/video token layout 和条件首帧语义；
- text encoder/token 与 action head 接口；
- backbone layer 数、hidden/head dimensions 和 K/V 投影；
- 3D RoPE 到 action 1D RoPE 的对齐；
- first-frame causal mask 与 future leakage；
- flow timestep/velocity target；
- VAE 时空压缩和 future frame indices；
- 哪些 H3 参数被 video/action/aux loss 实际更新。

只要其中一项没有直接测试，就标 `UNKNOWN`，最多允许工程 smoke，不允许长训。

## 因果动作 critic 与 best-of-N 数据门

动作候选排序的有效样本量不能用总 branch 数表示。若每个固定状态采样多个动作，只有同时含成功和
失败的 `mixed group` 能提供 within-state 排序监督；其正 pair 数为
`n_success × n_failure`。预注册、训练预算和放行结论必须同时报告：源 episode 数、mixed group 数、
train/validation pair 数与 suite 覆盖。

1. 同组必须固定 checkpoint、规范化恢复状态、环境 seed、后续 policy-noise schedule、执行 horizon
   与 evaluator；只改变首动作生成随机性。起点观测/状态和候选动作要逐字节或按预注册容差审计。
2. critic 输入只能含决策时可见的第0行状态和候选动作。后续 replan observation、最终状态、steps、
   success predicate 都属于标签侧，进入输入即为泄漏。
3. 至少保留 action-only shortcut 与 state-only/tie control。state-conditioned critic 必须在未见源
   episode 的 within-state ranking 上击败 shortcut，才能进入 best-of-N；训练 pair 满分不算机制证据。
4. 当训练 pair 接近100%而 episode-disjoint ranking反转或退到随机时，分类为小样本/动作捷径泛化
   失败，不靠增加同一批 pair 的 epochs 修复。先用 train-group-only leave-one-episode-out选容量与正则，
   再扩大新的因果源 episode。
5. validation outcome 一旦被读取，就永久降为 exploratory。任何在该结果之后选择的步数、正则、
   投影、特征或融合，都必须在结果生成前冻结一批全新 source episodes 才能恢复 confirmatory 身份。
6. best-of-N 仍是独立闭环效果门：held-out ranking通过只允许固定父策略的 `N=1` 对 `N>1` canary，
   不能直接宣称 LIBERO 成功率或通用 WAM 能力提高。

## Action-conditioned consequence 数据与模型门

静态 `current-state feature × flattened action` critic 不等价于 action-conditioned world model。
当它在训练 pair 拟合更强、fresh episode ranking却不优于action-only时，停止增加同批epochs；下一机制
必须显式建模 `clean action -> future state/value/video`，并重新冻结未消费source episodes。

1. 代码对齐应区分三类路径：DreamWAM的future supervision通过共享video/action attention塑造动作；
   MiniWorld把固定数量原始动作对齐到video latent time并逐层调制；FACT先生成动作，再把clean action
   作为K/V-only条件预测future/value。三者不能用一个无时间结构的action flatten head互相冒充。
2. action-conditioned adapter必须保持动作时间顺序。若video VAE每个latent frame对应`k`个动作，就预注册
   `k`和token count；至少保留flattened等预算基线与action shuffle/independent控制。
3. future/value loss不得回传到候选动作生成器：用因果attention mask或模块边界detach直接测试。报告
   action encoder非零finite梯度，同时断言输入candidate action/upstream policy梯度为零。
4. rollout日志必须同时覆盖成功和失败的动作后果。只在replan前存观测会遗漏“首chunk内成功”样本，
   导致成功标签选择偏差；应另存post-execution terminal observation，不要为补终态而扩展旧replan行轴。
5. future target优先取执行完整首chunk后的下一replan观测；若episode在首chunk内成功，可取terminal
   observation，但必须记录实际terminal step且证明`start < terminal <= start+horizon`。超出首chunk的
   terminal混有continuation policy，不能当首动作直接后果。
6. 当前状态与候选动作是在线输入；future image/state、terminal、steps和success永远是标签侧。数据冻结
   artifact应分别保存current/action/future/value及hash，训练loader必须有防future泄漏测试。
7. future prediction门只证明动作影响后果表示；必须再过fresh within-state value ranking，最后才允许
   `N=1`对`N>1`闭环。任一级失败都不能用上一层的MSE改善宣称动作生成或LIBERO成功率提高。
