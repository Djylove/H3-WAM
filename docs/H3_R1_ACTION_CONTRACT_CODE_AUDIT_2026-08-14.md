# H3 R1 ActionDiT 官方代码合同审计

日期：2026-08-14
范围：只审计 R1 `action-only-on-frozen-features`，不启动训练、不占用 GPU。
证据优先级：作者官方仓库实际代码 > 官方配置/launcher > 本地实现；论文仅作解释，不用于覆盖代码。

## 结论

R1 的 **动作目标、噪声混合、uniform-shifted timestep、shift=5、加权 masked MSE、
10-step Euler、7D×32 action/state min-max 训练合同**与 pinned StarWAM `cd76d96` 一致，
并被独立的 DreamWAM 官方代码再次交叉验证。当前没有证据支持继续修改 action flow 数学。
但审计发现一处实际部署差异：offline evaluator 按 StarWAM 在 min-max 反归一化前 clamp
`[-1,1]`，当前 rollout helper 未做 pre-clamp，而是在反归一化与夹爪转换后才 clip 物理动作。

最主要的未决问题不是 scheduler，而是 ActionDiT 是否真实使用 H3 视觉条件：R1 把冻结的 INT8 H3
第 49 层 observation token 缓存成 32 tokens，再作为最后一组 cross-attention context；StarWAM 官方
feature 路线默认让 Wan feature backbone 接受 action loss，DreamWAM 则在每一层用 video K/V 与
action token 融合。MiniWorld 不预测动作，它用已知 action 条件预测视频，因此其 diffusion-forcing、
动态 shift 和 block-causal video mask 对 R1 action flow 均为 `N/A`，不能直接移植。

因此，在 frozen held-out 基线完成后，优先并行做两个**不训练、单变量、same-noise**的条件因果实验：

1. `R1-COND-VIS-SHUFFLE`：只在 held-out 样本间置换 H3 visual tokens；
2. `R1-ROLLOUT-PRECLIP`：只在 rollout 中加入 StarWAM 的 normalized-action `[-1,1]` pre-clamp。

它们分别用于判断“模型是否依赖 H3”和“离线/部署 normalization mismatch 是否伤害闭环”，
不把任一离线结果直接宣称为闭环成功。
原 evaluator 已有 same-noise wrong-language 对照，held-out 基线应同时打开；它属于固定评测合同，
不占用这两个新增单变量名额。

## 来源身份

| 来源 | 官方远端 | 固定 revision | 本地状态 | 可用范围 |
| --- | --- | --- | --- | --- |
| StarWAM | `https://github.com/shaohua-pan/StarWAM.git` | `cd76d96f273f81e228a05f40f9697fe2514e2356` | clean detached HEAD | `TRAINABLE`；R1 的直接父方法 |
| DreamWAM | `https://github.com/hustvl/DreamWAM.git` | `6e989facc0c452fd3488d75f60bc36411005558c` | clean `main` | `TRAINABLE`；交叉验证 action flow，提供更强 video/action carrier 参考 |
| MiniWorld | `https://github.com/zhao-yian/MiniWorld.git` | `e484206bbd4360ae56ed8abad51c83f2457ac092` | clean `master` | `TRAINABLE` world model；不是 action policy |
| 本地 R1 | 当前工作树 | 未声称官方复现 | dirty，H3 backbone port | `action-only-on-frozen-features` |

StarWAM 是本轮对齐父实现。DreamWAM 与 MiniWorld 只作开源实现交叉审计，不能把三个项目中彼此
不兼容的字段静默拼接成“官方配方”。

## 可执行差异矩阵

状态是“本地 R1 相对其声明父合同”的判断；`N/A` 表示上游解决的问题不同，不是待修 bug。

| 字段 | 本地 R1 | StarWAM `cd76d96` | DreamWAM `6e989fa` | MiniWorld `e484206` | 状态与处理 |
| --- | --- | --- | --- | --- | --- |
| action 的角色 | 预测未来 32×7 action | 预测未来 32×7 action | 预测未来 32×7 action | 已知 action 是视频预测条件 | `EXACT`（Star/Dream）；MiniWorld `N/A` |
| flow target | `noise - clean_action` | `noise - clean_action` | `noise - clean_action` | 视频 target=`clean_video-noise` | `EXACT`；MiniWorld 的符号配合其 `z -= dt*v`，不可单独搬运 |
| noisy action | `(1-sigma)*clean + sigma*noise` | 同左 | 同左 | 同式用于视频 | `EXACT` |
| timestep 采样 | seeded uniform `u` 后做 shift | uniform `u` 后做 shift | uniform `u` 后做 shift | per-chunk diffusion forcing + logit-normal warp | `EXACT`（分布一致）；MiniWorld `N/A` |
| train/infer shift | action shift=5 | action shift=5 | action shift=5 | world-model shift 由配置解析，不是固定 action shift | `EXACT`；没有依据再扫 action shift |
| loss weighting | StarWAM bell-shaped normalized importance weight | 同左 | 同左 | clean-context masked video MSE | `EXACT`（Star/Dream） |
| padding loss mask | `action_is_pad=True` 不计 loss | 同左 | 同左 | clean-context/video mask | `EXACT`（Star/Dream） |
| action 初始状态 | Gaussian noise | Gaussian noise | Gaussian noise | Gaussian video latent | `EXACT` |
| 推理积分 | sigma 1→0，`sample += velocity*delta` | 同左 | 同左 | 视频采用与其 target 匹配的反向更新 | `EXACT`（Star/Dream） |
| denoise steps | 10 | 10 | 10 | world-model streaming 配置 | `EXACT` |
| action normalization | train min/max→`[-1,1]`、clamp5；offline infer pre-clamp `[-1,1]`；rollout **无 pre-clamp**，只在物理域最终 clip | train clamp5；infer pre-clamp `[-1,1]` 后反归一化 | 同一 LIBERO affine；release policy 无 pre-clamp | DROID q01/q99 clip `[-1,1]` | train/offline 对 StarWAM `EXACT`；rollout `MISMATCH`，必须单变量验证/修正 |
| LIBERO gripper | server 做 `-(2g-1)`，默认 sign binarize | 官方通用 policy 只反归一化，benchmark adapter 自行负责 | release policy 做 `-(2g-1)` 与 sign | DROID action condition | 转换与 DreamWAM `EQUIVALENT`；R1 另有最终物理 clip，闭环必须固定全部 flags |
| chunk horizon | 32 | 32 | 32 | action 与 latent video transition 对齐，不是预测 horizon | `EXACT` |
| tail padding | 重复 episode 尾 action；前 6 个 delta 维在 pad 位清零；mask loss | 同左 | 同左 | 不同 DROID clip contract | `EXACT`（Star/Dream） |
| proprio | 当前 8D state，min/max，投影成一个 token | 取序列 `[:,0]`，一个 token | 取 `proprio[:,0]`，追加到 text context | action-conditioned path 无 proprio | `EXACT`（Star/Dream） |
| text | H3/Qwen 原生 5120-d cached context | Wan/T5 4096-d context | Wan/T5 4096-d context | action-conditioned DROID 路线无 text | `INTENTIONAL_DEVIATION`：语义接口相同但 encoder/宽度不同；保留 wrong-language gate |
| visual 输入内容 | 当前双目首帧；H3 clean-observation tokens | 当前 observation 首帧、无 feature noise | uncond 用首帧；joint 用生成未来视频 | 历史视频 + action 条件预测未来视频 | R1 对 StarWAM `EQUIVALENT`；H3 clean timestep=1、Wan clean timestep=0，语义相同 |
| visual 层/token | H3 layer49，adaptive pool 到 32，5376→5120→ActionDiT | Wan last layer，adaptive pool 到 32，3072→4096→ActionDiT | 每层 video K/V 进入 action attention | action 进入每层 timestep/AdaLN stream | `INTENTIONAL_DEVIATION`；这是首要机制缺口 |
| context 顺序 | `[text, proprio, visual]`，feature-t token disabled | `[text, proprio, visual]`，release 禁用 feature-t | proprio 追加 text；video 经 per-layer K/V 单独融合 | 无 ActionDiT context | 对 StarWAM `EXACT` |
| attention mask | text mask + proprio/visual 全 valid；32 action token 双向 self-attention | 同左 | action-action 全 visible；joint action 可看所有 video | block-causal 是 video 时间 chunk mask | 对 Star/Dream `EXACT`；MiniWorld mask `N/A`，禁止直接抄到 action token |
| visual backbone 梯度 | INT8 H3 frozen cache，无 action 梯度 | release `feature_condition_train_backbone: true` | joint video/action carrier 共同训练 | world model 本体训练 | `INTENTIONAL_DEVIATION`；R1 必须保持当前命名，不能称 joint fine-tuning |
| ActionDiT 初始化 | 随机（StarWAM parent） | release `action_expert_init_from:null`、random head | 从 `ActionDiT_linear_interp...pt` 初始化 | 无 ActionDiT | 对 StarWAM `EXACT`；DreamWAM init checkpoint 本地缺失，不列为当前可执行消融 |
| 当前预算 | 100 steps，global batch 8，800 windows，0.0032 effective epoch | release 配方是长训练/50 epochs | 21,700 steps，batch16 | 6→16→32→64 latent-frame curriculum，100/50 epochs +30k+30k | `INTENTIONAL_DEVIATION`；100-step 只能做机械与早期 held-out gate，不能判断收敛 |

### 关键代码证据

- R1 action flow 与 mask：`scripts/h3wam/train_h3_int8_starwam_action.py:327-378`。
- R1 normalization/data/context：同文件 `:134-139`、`:289-303`；
  `src/fastwam/models/h3wam/starwam_feature_action.py:142-223`。
- StarWAM scheduler/target/Euler：`third_party/StarWAM/starwam/modules/scheduler.py:17-18,35-36,50-105`；
  weighted masked loss：`third_party/StarWAM/starwam/training/loss.py:27-50`。
- StarWAM release 配置：`third_party/StarWAM/examples/libero/configs/recipes/starwam_libero_feature_conditioned_wan22_5b.yaml:74-105,148-167`。
- StarWAM 首帧/feature/上下文：
  `third_party/StarWAM/starwam/wam/feature_conditioned_action_model.py:108-141,170-231,270-305,357-397`。
- StarWAM dense window/pad/normalization：`third_party/StarWAM/starwam/data/lerobot.py:658-678,699-793`。
- DreamWAM scheduler：`third_party/DreamWAM/dreamwam/scheduler.py:4-104`；action loss：
  `third_party/DreamWAM/dreamwam/model.py:665-688,713-825,890-992`。
- DreamWAM per-layer carrier/mask：`third_party/DreamWAM/dreamwam/mot.py:114-142,194-250,252-326`；
  dense window/normalization/config：`dreamwam/preprocessing/libero.py:51-66,143-180`、
  `dreamwam/normalization.py:7-65`、`configs/dreamwam_joint.yaml:13-34,56-100`。
- MiniWorld action 是 condition：`third_party/MiniWorld/miniworld/data/droid.py:247-280`、
  `miniworld/conditioning/actions.py:22-36`、`miniworld/miniworld.py:404-432`；diffusion forcing 与 mask：
  `miniworld/denoiser.py:354-499`、`miniworld/miniworld.py:50-95`；curriculum：
  `third_party/MiniWorld/scripts/train_droid.sh:18-29,56-80`。
- R1 offline evaluator 已固定 same-noise Euler 和 wrong-language：
  `scripts/h3wam/evaluate_h3_int8_starwam_action.py:612-650,979-1015`。
- R1 server 的 gripper 转换：`src/fastwam/models/h3wam/deployment.py:160-197`；server 默认
  `--binarize-gripper`：`scripts/h3wam/serve_rollout_policy.py:94-101,1190-1200`。

## Held-out 后并行的两个单变量实验

两个实验共享父基线：split-native R1 `s1/s50/s100`，固定 episode-disjoint validation manifest，
每任务 2 个窗口（80 windows），固定 selection salt、相同 checkpoint、相同 action initial noise、
shift=5、10 Euler steps、相同 batch/order。先完成原始 held-out + wrong-language；只有 source identities、
sample-id hash、checkpoint hash 完全一致的报告才允许配对比较。

### E1：`R1-COND-VIS-SHUFFLE`

可证伪假设：如果 R1 ActionDiT 使用了 H3 中的样本特定视觉信息，那么只打乱 visual feature 与
样本的对应关系会显著改变动作，并使 held-out action 指标恶化；若几乎不变，则当前 H3 carrier
可能被忽略，继续盲目增加 action steps 不能证明“最大化利用 H3”。

- 唯一变量：`h3_features[i] <- h3_features[perm(i)]`。
- permutation：对冻结的 80 个 selected sample IDs 用固定 salt
  `h3-r1-visual-shuffle-v1` 排序后循环右移 1；禁止同 ID 映射、禁止改 text/proprio/action/noise。
- 必报：baseline↔shuffle normalized/physical action delta、原 action MSE/ADE/endpoint/gripper-sign
  的变化、逐任务分布；同一 checkpoint 的两次 restore 仍须 `max_abs=0`。
- 晋级信号：至少 s50 或 s100 的 shuffle-vs-baseline action delta 明显高于 fresh-restore 数值噪声，
  且真实 held-out metric 向坏方向移动；具体效应量先记录，不在看结果前发明阈值。
- 负结果解释：如果 s1/s50/s100 均近乎不敏感，优先审计/改造 visual carrier；不能用训练 loss
  下降宣称 H3 有用。
- 实现状态：`READY_NOT_LAUNCHED`。`evaluate_h3_int8_starwam_action.py` 已增加
  `--visual-feature-shuffle`；只接受 `--samples-per-task 2` 的 40×2 frozen selection。实现按固定
  salt 对 80 IDs 做 SHA256 排序并循环右移 1，同时校验无 self-map、mapping hash 与真实消费顺序。
  报告包含 normalized/physical `baseline_vs_shuffle_action_delta`、shuffle-vs-target 原指标和
  `metric_change_shuffle_minus_baseline`。为避免与正在准备的 baseline 冲突，本轮没有启动真实评测。
- 测试：`.venv/bin/python -m pytest -q tests/test_h3_int8_starwam_action_evaluator.py
  tests/test_h3_starwam_feature_action.py tests/test_h3_int8_starwam_action_trainer.py`，结果
  `29 passed, 2 subtests passed`。

### E2：`R1-ROLLOUT-PRECLIP`

可证伪假设：如果训练/离线与 rollout 的 normalized-action clamp 不一致正在放大 flow sample 的
越界输出，那么只在 rollout min-max 反归一化前增加 `normalized_actions.clamp(-1,1)`，会降低
动作饱和/异常首步并提高固定 LIBERO canary 的成功率或目标接触质量。

- 唯一变量：`libero_environment_actions` 的输入是否先 clamp 到 `[-1,1]`；其余 checkpoint、
  observation、text、proprio、policy seed、环境 seed、replan、action scale、median window、gripper
  binarization 全部固定。
- 配对协议：先用同一 held-out batch 报告 normalized 越界率、越界幅度及两种解码后的 per-dim delta；
  再在相同固定 LIBERO 初始状态上做 A/B rollout。不得让 E2 与 H3 carrier 改动同跑。
- 晋级信号：pre-clamp 降低物理动作异常/饱和且闭环 predicate 改善；若解码动作完全一致或闭环无改善，
  记录为排除 normalization mismatch，不继续围绕 clamp 调参。
- 这是上游 StarWAM 推理代码支持的合同修正候选，不是凭直觉新增结构。

E1 的 deterministic permutation 已实现并通过 CPU 合成测试，但按调度要求暂不启动；E2 仍需要给
deployment helper 增加显式、可记录的 pre-clamp 开关与 round-trip 单测。本文不通过构造临时 cache
或人工改 manifest 绕过身份门禁。
已有 `--language-sensitivity` 应在父 held-out 命令中开启，它保持 visual/proprio/noise 不变，只替换语言。
proprio shuffle 保留为 E1 无视觉敏感性时的第二层诊断，但优先级低于已确认的 rollout 合同差异。

## 决策门

- 训练许可：`NO_GO`（仅针对本审查线；用户明确要求先不启动训练，且 held-out 基线尚未消费）。
- 效果结论：`NOT_EVIDENCE_READY`。
- 当前允许动作：baseline agent 完成冻结 held-out 后，用已测试的 E1 开关消费相同 selection；另行
  实现/测试 deployment pre-clamp 开关；记录 s1/s50/s100 配对曲线与固定 rollout A/B。
- 当前禁止动作：改 action target、重扫 shift、直接抄 MiniWorld mask、同时加入 DreamWAM per-layer
  fusion 与 H3 解冻、把 100-step loss 下降称为 H3 WAM 有效。
- 后续选择：若 visual 因果信号存在，再按原合同扩大预算；若 visual 被忽略，单独开一个有官方代码
  依据的 carrier ablation。DreamWAM 的预训练 ActionDiT checkpoint 未在本地取得并验证身份，
  暂不列为可执行实验。
