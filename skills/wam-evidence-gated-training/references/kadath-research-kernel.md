# KADATH 研究内核到 WAM 的迁移边界

## 目录

1. 来源身份
2. 代码支持的机制
3. WAM 迁移
4. 不直接照搬的部分

## 来源身份

- 官方仓库：https://github.com/i3T4AN/KADATH
- 审计 commit：`db7a6438d98c18d590b78b2146dc3bcd2c4ea0ef`
- 审计日期：2026-08-18
- 开源级别：`TRAINABLE`（包含运行状态机、评分、选择、变异、Git 谱系、存储、容器和测试）
- 许可证：Apache-2.0

重新使用本参考时先核对官方 HEAD；若变化，固定新 commit 并重新读取真实执行代码，不能只看 README。

## 代码支持的机制

### 固定内核与可变 genome 分离

`kadath/engine.py` 拥有目标锁、执行、证据冻结、评分、选择、恢复和导出；`kadath/gitstore.py` 把每个候选
完整框架提交进 run-local Git；运行时 worktree 只读。`engine._verify_locks` 重新核验 objective、runtime、
tool manifest、Architect output 和容器镜像身份。

### benchmark 先锁定、分数由内核计算

`kadath/specialists.py` 的 Architect 合同要求有限 score range、正权重且总和为 100%、机器可读 measurement、
required evidence、failure/anti-fraud 和 tie-break。Grader 只从冻结证据抽取 typed facts；数值公式由 kernel
计算，agent 自报分数不生效。

### 执行、评分、变异为三个持久边界

`engine._execute_epoch` 在中断后恢复 epoch 前快照并丢弃部分分数；`_freeze_attempt` 停止执行后生成带 hash
的 evidence manifest；`_grade_epoch` 可以复用输入 hash 相同的评分 checkpoint；`_select_and_birth` 只在评分
结束后创建后代。选择前另做完整 snapshot，失败时整体回滚。

### 内容寻址、去重与 lineage

`gitstore.py` 保存永久 genome commit/tag；engine 组合完整 Git tree、prompt 和 runtime 生成 genome 身份并
拒绝重复内容。mutation 只允许有界文件操作，禁止越界和修改 `.git`。后代显式记录 parent genome、child
genome 和 epoch。

### 经过验证的记忆遗传

`store.py` 按 owner、record type 和内容 hash 去重 knowledge，以 link 表示继承而非递归复制。检索结合文本
相关性、证据、记录类型、peer rating 和“产生该记忆的 epoch 中来源候选的验证质量”。agent 不能给自己的
记忆投票。

### 群体选择与故障隔离

官方实现保留成功 cohort 的 top 30%，middle 自反思，失败优先淘汰并由 elite 后代补位；若全员没有 verified
success，则停止而不是从失败者繁殖。单个 agent 崩溃不会终止整个 population，infra 与得分阶段分开。

## WAM 迁移

| KADATH 概念 | WAM 对应物 | 迁移规则 |
|---|---|---|
| kernel | benchmark/evaluator/调度器/证据审计 | 候选代码不能修改；固定 hash |
| genome | 模型代码+resolved config+父 checkpoint+数据/动作/运行时身份 | 用完整内容签名去重 |
| epoch | 一轮固定父基线和锁定 benchmark 的候选 cohort | 完成冻结评分后才产生下一轮 |
| frozen attempt | checkpoint、日志、评测 JSON、轨迹和 manifest | 停止写入后 hash，再只读评分 |
| grader facts | paired metrics、success、置信区间、机制门 | 由确定性脚本计算；LLM 只解释 |
| elite | 通过当前赛道门的候选 | 原样保存，不能一边保留一边热改 |
| middle mutation | 证据支持的单变量修订 | 新 candidate ID、新内容签名 |
| culled memory | 负结果与 infra 经验 | 保留适用范围和原始 evidence refs |
| heredity | fusion lineage、父 checkpoint、canonical memory | 只继承已验证机制和可追溯经验 |

对 H3-WAM，固定内核至少锁定：H3 SHA、父动作专家 SHA、官方来源 commit、train/val manifest、action/state
语义、normalization、horizon/replan/sampler、LIBERO 版本和成功判据。候选只能改变预注册的一个机制。

## 不直接照搬的部分

- 不固定使用 30% elite/cull。GPU 实验按统计门、候选成本和资源期限设置 cohort。
- 不让 LLM 直接决定 LIBERO 分数。success、McNemar、置信区间、hash 和预算由确定性程序计算。
- 不让候选自动修改 benchmark、数据 split、evaluator 或晋级阈值。
- 不为模仿 population 而复制昂贵近邻超参；优先跨机制赛道和严格父子对照。
- 不把 prompt/source 自变异直接等价为模型训练。WAM mutation 必须重新过 source、mechanical、paired 和
  rollout gates。
- 不把 KADATH 的容器 CPU/内存默认值当 GPU 作业资源配方；只迁移身份、隔离、恢复和遗传机制。
