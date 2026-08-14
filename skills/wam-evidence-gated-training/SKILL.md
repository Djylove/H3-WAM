---
name: wam-evidence-gated-training
description: 审计并演化 World Action Model 研究：对齐作者官方代码、固定 commit、数据/动作合同、训练预算与评测，组织多开源方法的单变量淘汰赛和胜者融合，分级放行昂贵 GPU 训练。用户提到 WAM/H3/FastWAM/StarWAM/DreamWAM/MiniWorld/FACT、论文或开源实现对齐、模型替换、并行试验、方法融合、“练蛊/蛊王”、效果退化或准备启动新实验时使用。
---

# WAM evidence-gated training

把开源实现转成可审计的实验合同和可进化的候选池。证据门用于控制风险、区分“训练许可”和“效果结论”，不能把
任意固定步数当作研究规律。机械 smoke、诊断性探索、规模化确认和对外结论使用不同门槛。已明确
要求保留的长线实验继续运行。论文只作为方法意图、消融和代码缺失细节的辅助证据，不能代替
可执行代码。“练蛊”表示让独立机制在相同合同下竞争，再融合胜者；不表示把所有技巧一次性堆入模型。

## 工作流

### 1. 固定问题和实验类别

只写一个可证伪问题，并将实验标为以下一种：

- `reproduction`：同模型、数据、代码和协议的复现；
- `backbone_port`：保留方法接口但替换基础模型，例如 Wan2.2 → H3；
- `controlled_ablation`：仅改变一个已声明因素；
- `novel_composition`：组合多个项目思想。

模型替换不得称为“官方复现”或“完全对齐”。给每个实验指定一个不变父基线和唯一主变量。

### 2. 组织开源方法淘汰赛

先按机制而不是项目名分赛道：

- representation/carrier：视觉层、K/V、cross-attention、structured future；
- action objective：flow、regression、repeat、mask、normalization；
- temporal/context：history、executed-action feedback、persistent KV、replan continuity；
- consequence/ranking：future state/representation、value、failure-aware、best-of-N；
- deployment contract：sampler、clip、gripper、execution horizon、latency。

为每个候选写卡片：官方来源和 commit、可迁移的最小机制、父基线、唯一变量、不变量、所需数据、
训练样本/effective epochs/墙钟/存储、机制指标、闭环指标和停止条件。按以下阶段晋级：

1. `SOURCE_GATE`：确认实际代码、训练级别和方法边界；
2. `MECHANICAL_GATE`：形状、梯度、恢复、默认关闭兼容；
3. `PAIRED_GATE`：同 split/初始化/噪声/预算的单变量 held-out；
4. `ROLLOUT_GATE`：固定初始状态和 success predicate 的闭环 A/B；
5. `FUSION_GATE`：只让不同赛道的胜者融合；每轮相对父候选只新增一个已独立验证机制。可在
   二元胜者上再加第三个胜者，但每一轮都必须有去掉新增机制的父对照；
6. `CHAMPION_GATE`：统一 benchmark、多 seed、部署成本与消融齐全后选最终候选。

保留至少一个不变基线和一个有充分预算的长线。不要用一批短 smoke 淘汰所有架构，也不要用不同
数据、seed、预算的最高分拼“蛊王”。相同数学效果的候选应合并，例如额外 clean-x0 MSE 若可化为
原 velocity error 的 timestep 重加权，就按 weighting ablation 处理，不声称新增能力。

### 3. 建立来源身份

必须浏览当前的一手资料，并检查本地仓库；论文、仓库可能更新，不能凭记忆。按以下优先级取证：

1. 作者/机构官方开源仓库中的实际训练、数据和评测代码；
2. 官方 resolved config、启动脚本、发布 checkpoint 及其模型卡；
3. 固定 commit 的本地干净镜像和真实运行日志；
4. 论文正文/附录与作者项目页；
5. 第三方复现、博客或聚合列表仅用于发现线索，不用于放行。

执行时：

1. 先读实际 dataloader、model forward、loss、optimizer、launcher 和 evaluator，再读 README；
2. 记录仓库 URL、commit、branch、dirty diff、发布 tag/checkpoint，再记录论文版本；
3. 区分 `code-backed` 与 `paper-only`。找不到作者代码时明确写 unknown，不拿第三方重实现
   冒充官方代码；
4. 对本地 vendor/fork 运行只读 `git status --short`、`git diff --stat`、`git rev-parse HEAD`；
5. 不修改 dirty vendor 仓库来“清理”对齐问题。

把开源程度再分三级：

- `TRAINABLE`：包含真实 dataloader、forward/loss、optimizer、launcher、checkpoint restore 和 evaluator；
- `PARTIAL`：只缺其中一到两项，可作为模块实现参考，不能直接提供完整训练配方；
- `INFERENCE_ONLY`：只有模型定义、推理或 README，不作为训练预算依据。

必须读取 launcher 对 config 的最终覆盖；默认 YAML/py config 不能代表真实实验。检查 forward hook 的
`detach`、`no_grad`、`requires_grad` 和 optimizer param groups，因为“联合 forward”不等于“联合反传”。

若论文与官方代码冲突，以官方代码作为“可复现实验”的执行基准，同时记录冲突；只有在单独的
controlled ablation 中才能切换到论文描述。`paper-only` 方法可以进入明确标记的架构探索，但不能
被称为官方复现，也不能仅凭训练 loss 获得效果结论。是否扩大预算由学习曲线、机制评测、资源机会
成本和用户目标共同决定，而不是来源标签单独决定。

WAM 项目的方法边界见 [references/method-routing.md](references/method-routing.md)。每次使用时仍需
重新核验上游，不把该文件当作永远最新的事实。

### 4. 四层逐字段对齐

对每项同时核对：官方执行代码 → 官方 resolved/default config → 发布 checkpoint/model card → 本地
实际启动命令；最后用论文正文/附录解释差异。至少覆盖：

- 架构：backbone/checkpoint、输入 token、attention mask、融合方向与层、RoPE、动作头深度、
  trainable/frozen 参数；
- 数据：suite/task/episode、去 noop、camera、分辨率、窗口跨度、逐帧 sample stride、视频 stride、
  action horizon、padding、split、normalization；
- objective：flow parameterization/timestep shift、RGB/action/aux loss、权重、teacher 或 ranking；
- 优化：global batch、gradient accumulation、optimizer、每组 LR、weight decay、warmup、scheduler、
  steps/epochs、实际样本数；
- 推理/评测：环境版本、wait/max/replan steps、denoise steps、RNG device/seed、action clip、
  gripper conversion、task/trial 数和 success predicate。

每个字段只允许 `EXACT`、`EQUIVALENT`、`INTENTIONAL_DEVIATION`、`MISMATCH`、`UNKNOWN`。
`EQUIVALENT` 必须给等价性测试；`INTENTIONAL_DEVIATION` 必须给父基线、假设和独立消融；其余必须
给文件行号、配置输出、URL 或日志证据。

### 5. 做数据与动作合约审计

训练前输出以下可复核量：

- episode/task/suite 数、原始帧数、有效 window 数和每任务分布；
- 一个完整 window 的 frame/action indices，证明没有把逐帧 dense sampling 误写成少量抽帧；
- train/val episode-disjoint 检查及 manifest hash；
- action/state 每维含义、单位、min/max/quantile、delta/absolute、gripper 编解码 round-trip；
- 用专家 demo 在对应初始状态 replay，验证数据动作能完成任务。

若上述任何一项未知，不进入长训练。

#### State-coverage gate

不能用总 window 数掩盖每条轨迹的稀疏性。训练动作策略或准备闭环前，额外报告每 episode window 数的
min/median/mean/max、相邻 start gap、首末 start span，以及 grasp/contact/release/success 前后的覆盖；
至少展示一条完整 episode 的所有 start，不只展示全局直方图。把“每 episode 固定 K 个均匀快照”标为
`sparse_state_coverage`，即使 episode/task 数很多也不能称为 dense。

稀疏子集可以用于架构机械门或同数据的 representation/carrier 消融，但不能单独放行闭环 benchmark。
进入策略规模化训练前，必须使用每个合法 start、或给出 stride 足以覆盖动作 horizon 与接触阶段的等价性
证据。若稀疏模型离线改善而闭环无接触/目标错误，优先固定架构和推理合同，将数据密度作为唯一变量：
先保存与稀疏父模型相同见样本数/steps 的 dense checkpoint，再继续预注册的 dense effective epoch；
不能同时改模型、LR 或 sampler 来模糊归因。

### 6. 验证架构与梯度路径

不能只看模块名称。执行最小测试证明：

- 形状、mask、时间/空间位置和 RoPE 对齐；
- action loss 能到达预期 action/fusion/backbone 参数；
- video/structured-future loss 能到达声称在训练的 backbone；
- frozen 参数确实无梯度，trainable 参数非零且 finite；
- 保存、恢复后同输入推理一致。

如果 backbone 冻结且 world loss 无法更新它，把实验命名为 `action-only-on-frozen-features`，不能称为
video-action co-training。H3 的 DoT 移植还必须证明 all-layer K/V 汇聚和 RoPE realignment，而不是
仅证明代码能 forward。

### 7. 分离训练许可与效果结论

从 [references/dossier-template.json](references/dossier-template.json) 复制 dossier，填写真实证据，
运行：

```bash
python scripts/validate_dossier.py DOSSIER.json --target canary
python scripts/validate_dossier.py DOSSIER.json --target long
python scripts/validate_dossier.py DOSSIER.json --target claim
```

`PROBE_ONLY` 是辅助状态，不是训练许可：允许为解决 `UNKNOWN` 做只读审计、单元测试、合成输入
forward/backward、显存/吞吐探针或最多一个不保留权重的 optimizer step；不得消耗正式数据预算、
产生候选 checkpoint 或作效果结论。validator 因 `UNKNOWN` 拒绝 canary 时，可用 `PROBE_ONLY`
补证据，但不能绕过 validator 启动探索训练。

- `GO_CANARY`：允许完成机械 smoke 和最小可解释训练。`100 step` 只能是常用 smoke 尺度，绝不是
  上限或统一停止条件；如果一个 epoch、调度 warmup 或信号出现本来就需要更多 step，应按样本数、
  effective epoch 和墙钟预注册实际预算，并密集保存 checkpoint。
- `GO_LONG`：允许诊断性或长程探索训练。需通过 finite gradient、checkpoint restore 和可信父基线，
  并预先声明 checkpoint/评测节奏、存储上限和停止条件；**不要求在训练开始前已经出现机制提升或
  闭环成功**。训练过程中在多个 checkpoint 上测学习曲线，允许先排除“训练不足”。
- `EVIDENCE_READY`：只有固定数据切分上的机制指标和固定闭环评测均 PASS，才能宣称方法有效、优于
  父基线或值得最终规模化。官方复现类还必须有固定 revision 的官方训练代码；backbone port 和
  novel composition 可基于清楚标记的本地实现给出项目内结论，但不得冒充官方复现。
- `NO_GO`：只用于会让实验不可解释或不安全的阻断项，例如动作语义/单位未知、数据泄漏、梯度断路、
  NaN、checkpoint 无法恢复或没有可比较父基线。早期效果差是负结果，不自动等于禁止继续研究。

离线总 MSE 下降不等于机制指标。按假设选择指标：语言路线看同状态 correct/wrong 指令差异；motion
路线看真实 motion loss 与 action 分支梯度；闭环路线看目标 predicate，而非任意物体位移。

#### Conditioning-collapse gate

对 feature/language-conditioned 动作模型，在预注册的早期和晚期 checkpoint 上使用同一
episode-disjoint 样本、噪声、solver、normalization 和 seed，同时报告 physical error、gripper
macro-F1、无 self-map 的 visual-feature shuffle 和只替换文本的 language sensitivity。若晚期
checkpoint 的 generic regression error 改善，但视觉/语言反事实响应接近消失，并伴随 gripper
退化或预期视觉路径梯度归零，标记 `FAIL_CONDITIONING_COLLAPSE / NOT_EVIDENCE_READY`。

单个 batch 的零视觉梯度不能单独证明机制坍塌；必须先排除无效 feature、零 loss weight、mask、
AMP underflow 和 infra，再与至少一个配对反事实趋势或多个 checkpoint 互证。一旦互证成立：

- 不把低 MSE 解释成更强条件策略，也不把 shuffle 不敏感解释成 robustness；
- 不用增加 steps、选择 latest checkpoint 或把 finite/nonzero gradient gate 改成 warning 来晋级；
- 不进入 rollout，直到新单变量合同在相同协议下恢复视觉/语言依赖并守住 gripper；
- 先引用固定 commit 的执行代码、resolved command、原始 checkpoint/log/evaluator JSON；论文只用于
  提出修复假设或解释机制，不能覆盖负实验或单独放宽门。

#### Paired-mechanism attribution gate

新增正则、辅助损失或条件路由时，候选自身的早晚 checkpoint 曲线只证明“训练发生了”，不能证明
新增机制有效。必须另取未启用该机制的直接父配置，在相同 completed steps、样本/噪声/solver/seed、
normalization 和 evaluator 下做 paired comparison；若训练样本顺序不能完全相同，至少固定 manifest、
global batch、采样器合同和总见样本数并明确记录差异。

对视觉或语言依赖类机制，同时报告反事实绝对 delta 和相对输出尺度的 delta。若绝对响应变大但按输出
尺度归一后不变，或 physical、gripper、动作范围中任一关键指标退化，则不能把候选内学习曲线归因于
新机制，标记 `FAIL_PAIRED_GATE / NOT_EVIDENCE_READY`。不通过 paired gate 时不靠增加 steps、挑选
不同 checkpoint 或直接闭环来绕过归因；需要修改单变量假设后建立新候选。

不要用单个早期 checkpoint 的效果阈值作通用停止门。出现 teacher-forced 改善而 causal/closed-loop
退化时，先把它记录为 exposure bias 或 train/inference mismatch 证据；可以继续同变量的预算阶梯，
或启动直接针对该错配的对照实验。只有学习曲线在多个预注册 checkpoint 上稳定饱和/恶化，或达到
资源上限，才停止该路线。

### 8. 调度昂贵算力

先保留至少一个长线基线，再用其余节点并行验证不同机制，不让多个节点重复同一超参：

1. 评测节点及时消费 checkpoint；
2. 新路线先做机械 smoke；随后可直接进入有界诊断训练，不把 100 step 指标当作扩训硬门槛；
3. 训练预算用样本数和 effective epochs 表示，不能只写 steps；
4. 诊断训练默认每 500–1000 step 保存，并让评测节点异步消费 checkpoint；checkpoint 间隔应结合
   每步墙钟调整，目标是至少得到 3 个可比较的学习曲线点；
5. 到期前一天停止新长任务，固化 checkpoint、commit、resolved config、manifest hash 和评测 JSON；
6. 记录负结果和适用预算，避免无意重复；负结果不禁止有新假设或更长预算依据的后续实验。

有 N 个同级节点时默认最多让 N-1 个节点同时长训，至少保留一个节点或等价 GPU 配额做异步评测、
数据审计和恢复验证；只有存在独立评测资源时才可占满全部节点训练。吞吐尚未测得时先运行探针，
用 `steps(E)=ceil(E×unique_train_windows/global_batch)` 和实测秒/step 计算墙钟。

### 9. 回写经验并进化 skill

每轮结束先把原始命令、commit、manifest/checkpoint hash、指标和失败分类写进项目 dossier；只有满足
以下任一条件才更新本 skill 或 `references/method-routing.md`：

- 多个实验重复暴露同一可迁移风险；
- 官方代码/commit 改变了方法边界；
- 新机制补上现有赛道空缺；
- 现有规则错误阻断研发或允许了不可解释实验。

不要把单任务分数、偶然阈值或某次 GPU 性能写成通用规则。更新后运行 skill validator；若改动会
改变实验放行行为，用没有结论提示的原始 artifact 做一次独立 forward-test。

持续保留这些已验证反模式：

- `detach()` 不会打断 storage alias；缓存中间层时必须 clone 并验证层间不相同；
- “全量 feature cache”不等于全量微调 backbone；两者在文档和资源预算中分开命名；
- 训练集 loss/机械 restore 不等于 episode-disjoint 泛化，更不等于闭环；
- evaluator 与 rollout 的 normalization、pre-clamp、gripper 和 horizon 必须逐项相同或做 A/B；
- 依赖、PythonPath、device-current 等启动失败单列 infra，不计作 policy trial；
- failure-aware 训练必须保存可复现的失败 observation/action/predicate/onset，不能只用 mp4 或把失败动作当 expert imitation。
- “每 episode 五个均匀 start”可用于载体消融，不等于接触策略的 dense 训练；总样本数和 episode 数都
  不能替代 per-episode state/contact coverage 审计。

## 输出要求

每次给用户的方案必须包含：

1. 一句话可证伪假设；
2. 来源/版本身份；
3. 逐项差异矩阵及未解决项；
4. 实际 global batch、样本数、effective epochs、预计墙钟；
5. 父基线、唯一变量、晋级/停止门槛；
6. 训练许可写 `GO_CANARY`、`GO_LONG` 或 `NO_GO`；效果结论另写 `EVIDENCE_READY` 或
   `NOT_EVIDENCE_READY`；
7. 真实启动命令或明确说明尚未放行。

若用户尚未提供数据规模、microbatch 或吞吐，相关字段写 `UNKNOWN`，同时给计算公式和取得真实值的
探针；禁止为了满足输出格式编造数字。trial 数、置信区间和晋级阈值按 benchmark、基线方差和预算
预注册，不把某次项目的阈值写成通用规律。

不要用“参考了论文”“基本对齐”“应该有效”代替证据。若论文与官方代码冲突，默认复现官方代码
行为并报告冲突；论文版本只能作为独立消融，不静默混用两个版本。
