# H3-WAM 审查报告复核与整改记录（2026-08-13）

## 可证伪问题

在不把审查报告当作事实源的前提下，当前代码是否存在足以破坏 shared-H3 训练/部署解释的确定性
合同错误；修复后能否通过真实 torch 单测和 2-rank 参数一致性检查？

审查报告固定在项目 commit `fc56c8e81863dbce5b726b2904bbb75f93337691`；本次复核起点为
`91b0fdd88bf11979e63e5f12a129919b36f129a2`。两者间没有修改报告涉及的 shared trainer、history
codec、anchor 或 DoT server，因此这些代码指控仍需逐项核验。上游 LingBot-VA 使用本地只读镜像
commit `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`。
2026-08-13 通过 GitHub `ls-remote` 复核，LingBot-VA、FastWAM、DreamWAM、MiniWorld 的远端 HEAD
仍分别与 `docs/UPSTREAM_SOURCES.lock.json` 中的固定 commit 一致。

## 复核矩阵

| 报告判断 | 复核结论 | 实际依据与处理 |
|---|---|---|
| shared/non-shared four-stream 复制参数未同步 | **正确** | ignored 的 trainable adapter/projection 在 optimizer 前无 gradient all-reduce；新增显式平均、正确的 mixed shard/replica norm clipping、启动及 step1 rank parity 断言。DoT 自己已有 I/O all-reduce，不受此项影响。 |
| history 夹爪域反转 | **正确但影响范围被扩大** | 训练 sidecar 是 dataset domain，离线数值不受此 codec bug；在线 history 是 environment domain，原服务直接套 dataset stats。现先 env→dataset，再 normalize，并用 valid mask 隔离左 padding。旧 history 闭环推论撤销。 |
| anchor 应改成 4 observed + 32 predicted | **修复建议错误** | pinned LingBot 训练确实在总长度内前置零动作；server 只在 `frame_st_id==0` 固定该帧，client 只首次跳过，随后依赖持久 KV。本地每次 replan 冷启动不等价。暂时禁用 trainer flag、拒绝旧 anchor stage，而非机械扩成36。 |
| DoT READY 前访问已删除的 `stage` | **正确** | 删除前保存 `checkpoint_steps`；当前仍需做真实 H3 `start→ready→predict→close` 生命周期 smoke。 |
| 当前树含 W&B 凭据 | **正确** | 已从当前跟踪树清除硬编码 key，增加 tracked-secret scan。远端 token revoke 与 Git 历史重写不属于本地代码修复，仍需账户所有者完成。 |
| shared text-only flag 没有生效 | **正确，尚不能证明旧 cache 被污染** | shared trainer/server 改为无条件校验 `text_only=True` 和全部 `token_tags==1`；payload 有 task 字段时再校验 task。 |
| M11 val40 是 held-out | **错误命名，报告纠正正确** | v8 使用 `--all-train`，M11 对 v7 val40 的评估属于训练 population；只保留为训练样本指标，不作泛化证据。 |
| train+val quantile stats 泄漏 | **合同错误成立** | 当前 JSON 明写 277,713 全帧来源；对 all-train M11 不构成额外 split 泄漏，但对有 val split 的 shared 实验不满足 train-only stats 合同。需在新 split 上重算，不反向篡改旧实验。 |
| 所有旧 shared checkpoint 均完全无信息 | **过度结论** | 不能证明声明的 global-batch-8 算法或作修复后 resume 父点；但其实际 rank0 混合输出、学习曲线和短闭环仍是可复现的诊断数据。标 `TAINTED_FSDP_REPLICATED_GRAD`，不删除。 |
| 立即停止所有现有长线 | **不采纳** | 训练在整改时已接近预注册终点；保留日志/终点可排除预算因素。禁止从旧 shared checkpoint 接修复后长训，新的 shared 实验从共同初始化开始。DoT 不受 replicated-grad bug 影响。 |
| 必须先出现非零成功才允许任何长训 | **不采纳** | 与 evidence-gated 研发规范冲突。`GO_LONG` 可以是有界诊断训练；只有效果声明才要求机制指标和完整闭环。 |

## 已落地整改

1. shared/four-stream 所有 FSDP ignored 且 trainable 参数在裁剪前做 `SUM/world_size`；global norm 不再把
   replicated 参数重复计算 world-size 次，并在启动/首步后断言跨 rank 参数一致。
2. 新增批量 `environment→dataset` action codec 与 history 归一化；padding 保持零且不进入 clean/noisy
   attention key，也不被 sampler 生成。
3. initial anchor 路线 fail closed，等待真正的 LingBot `frame_st_id + rolling KV` port。
4. DoT READY 修复；shared launcher 显式传 layer/action/latent geometry；shared context 强制 text-only。
5. rollout 使用 `running partial → complete` 原子状态；只有 task/episode 数精确完整的 result 才能被新
   watcher 识别为完成。
6. 当前树移除 W&B secret，增加 `scripts/security/check_tracked_secrets.sh`。

## 验证

- 本地：全部一方 `src/scripts/tests` static compile 通过；所有 H3 shell 语法通过；`git diff --check` 通过；
  tracked-secret scan 通过。
- 云端隔离目录 `/mnt/h3-wam/audit-remediation-20260813`：真实 torch 环境运行65个相关测试全部通过。
- 同一云端真实 `torch.distributed` 2-rank gloo parity：rank0/rank1 分别产生梯度1/3，平均后均为2，
  SGD 后参数均为0.5，跨 rank max diff=0，PASS。

这些测试证明 codec、mask、同步原语与 CPU 模型合同；尚未证明33B H3 的完整 2-GPU FSDP
`train→save→restore` 或 LIBERO 闭环效果。

## 当前放行状态

- 修复后的 shared-H3：`GO_CANARY`，`NOT_EVIDENCE_READY`。必须从共同干净初始化跑2-GPU真实H3
  两步 parity/save/restore，随后才可 `GO_LONG`。
- 旧 shared/history/anchor checkpoint：`TAINTED_FSDP_REPLICATED_GRAD`；anchor 额外
  `INVALID_STREAMING_CONTRACT`。仅作历史诊断，不作 resume 父点或效果结论。
- DoT：不受 shared replicated-gradient 问题影响；训练 checkpoint 可保留。部署需补真实
  `start→ready→predict→close` smoke 后才称 deployable。

## 未完成且不能假装完成的事项

- 账户所有者在 W&B 后台 revoke/rotate 已泄露 token；协调 Git 历史重写和远端 force-update。
- 建 episode-disjoint train/dev/sealed-test，并以 train-only stats 重算 normalization。
- CheckpointContractV2、严格 optimizer/RNG/sampler resume、完整 content-addressed cache。
- 精确匹配 dataset episode 初态的 expert replay，以及 released FastWAM 同环境 gold control。
- LingBot persistent observation/action KV 生命周期；在此之前不再声称 initial anchor 与上游等价。
