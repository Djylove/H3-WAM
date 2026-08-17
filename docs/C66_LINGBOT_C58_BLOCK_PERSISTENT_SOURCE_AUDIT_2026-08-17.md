# C66 LingBot × C58 block-persistent：历史复用、候选选择与源码差异审计

状态：`SOURCE_AUDITED / UNIT_PASS / REAL_MECHANICAL_PENDING / NO_GO_OPTIMIZER`

## 结论

context 线不再给冻结 H3 K/V 叠加可学习的外部变换。最短且不是 post-hoc adapter 的候选是
`C66_LINGBOT_C58_BLOCK_PERSISTENT`：保留已晋级的 C58 full-30 ActionDiT，把 LingBot-VA 的
“预测 cache 可回滚、真实 observation/action 以 clean token 提交、下一次所有 block 读取持久前缀”
生命周期移到 C58 的 **30 个 self-attention block 内**。当前 C66 没有新增参数；启用前后的
`state_dict` 与空历史输出都必须逐值等于 C58。真正训练时，仍然训练 C58 ActionDiT 本体，不训练一个
补丁 bridge。

C57 的 5000-step 无效不能证明 persistent-KV 思路无效。重新逐行检查后发现两项训练/部署差异：

1. C57 rollout 在 `serve_c57_lingbot_policy.py:183-185` 用绝对 `frame_position` 改写 H3 layout；训练在
   `train_c57_lingbot_persistent_kv.py:135-155` 将多个“都作为本地首帧编码”的 K 直接拼接，没有 temporal
   reindex。
2. C57 训练先把 history 最后的真实观测提交到 state，随后 forward 又传入相同 current carrier；
   `lingbot_persistent_kv.py:465-498` 会把两份都加入 attention。官方 server 在 feedback 时用真实观测
   替换预测 cache，下一次推理不重复追加当前帧。

因此 C57 的负结果首先是合同错位，不是“训练步数不够”。C66 关闭这两个 gap：每个独立 H3 观测的 K
按逻辑 frame 位置精确重旋，V 不变；只要 state 已有 feedback，forward 明确忽略冗余 current carrier。

## 可直接复用的历史资产

| 资产 | 固定身份/结果 | C66 如何复用 |
| --- | --- | --- |
| C57 sequence manifest | `/mnt/h3-wam/data/c57-lingbot-replan8-v1/manifest_train.jsonl`；SHA256 `8f95005ac66fd89ca3a22a80d75480e9792b09f976e928f2eb70d4f08680049f` | 直接复用 episode-disjoint 的 replan8、observe-every4 真实序列，不重造样本 |
| C57 data audit | `AUDIT.json`；SHA256 `a383bd0d201b8eb9e3ed52b93bad712a96a4e74ecc063d8d71fcc000e46329fb`；200779 rows、1542 episodes、4 suites、missing/leakage=0 | 复用数据门；probe 仍逐行核验选中的 history IDs/action indices/window shape |
| C57 mechanical | `real_mechanical.json`；SHA256 `28a5720e96048ba2389fd2582f434daadac1203f4e192f680abf8f8955ef1104` | 复用 state snapshot/FIFO 经验，但不继承“已通过”结论，因为 parent 和 block 路径已改变 |
| C57 effect | final SHA256 `84bf74812dabf3ec520deed88d1b7497d8ac16bbc95bab678b4aa00f170c999e`；heldout SHA256 `05210e92106fd6b32eef14657036d1d16e351cd360b18f957e7a979c38c76ce4`；相对 D0 `-1.166%`，win 43.75% | 作为明确失败基线；禁止把 C66 描述成已有能力提升 |
| C58 promoted parent | `c58b_online_s10000.pt`；SHA256 `2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541`；paired LIBERO 295/680 vs D0 270/680 | C66 唯一 parent；strict restore；不回到 D0/5-layer |
| C64 mechanical | report SHA256 `f460a1c8443b32beee6fd68e34c446fde73e347541225c661de64cce5f7dcaf9` | 只复用 H3 temporal K 精确 reindex 数学和 source probes，不复用 post-hoc modulator |
| C62 | 猜测路径下没有完整 frozen data artifact | 不作为可复用放行证据；只保留历史失败解释 |

## 两个候选的比较

| 候选 | 忠实度 | 新参数/训练对象 | 最短有效门 | 决策 |
| --- | --- | --- | --- | --- |
| A：C64 cadence-only → +learned null → +RoPE | 对 MiniWorld 的 cadence/null/RoPE 可做元素级单变量，但 action modulation 仍发生在 H3 已产生 K/V 之后 | 外部 shared/per-layer bridge | 需先冻结三份独立 data/optimizer arms，之后才有 clean-vs-shuffle | 保留为对照，`NO_GO_OPTIMIZER` |
| B：C58 full30 + LingBot committed observation/action persistent-KV | lifecycle 直接发生在 C58 每个 self-attention block；clean executed-action K/V 由同一 block 在 `t=0` 产生 | 无新参数；继续训练 C58 ActionDiT 本体 | 真实 H3/C58/sequence 的零优化器 block/mechanical gate | **选中 C66** |

选择 B 不是因为代码更短，而是它删除了外部 adapter 这个结构性 gap，并直接作用于当前唯一有正向闭环
证据的 C58 parent。A 即使通过仍只能证明外部 K/V bridge 的某个元素有效，不能回答“上下文是否进入动作
生成 block”。

## LingBot 官方代码逐项对齐

固定官方 checkout：`third_party/code_audit/lingbot-va` commit
`7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`，clean。

| 合同 | LingBot-VA 官方实现 | C66 | 判定 |
| --- | --- | --- | --- |
| 训练 token streams | `modules/model.py:702-779` 将 noisy/clean video、noisy/clean action 拼入同一 30-block tower | H3 observation K/V 与 noisy/clean action token 在 C58 每个 block 的 self-attention 中相遇 | `ARCHITECTURE_EQUIVALENT_WITH_FROZEN_H3_BOUNDARY` |
| causal semantics | `modules/model.py:93-200`：clean 历史因果可见，noisy 当前只读严格过去 clean 与当前 noisy | persistent state 只 materialize committed observation/action；当前 noisy action K/V 只参与本次 block，不写 state | `SOURCE_EQUIVALENT` |
| cache placement | `modules/model.py:414-459`：K/V 在 attention 内产生、append、read、temporary restore | C66 在 C58 block 的 `q/k/v` 后合并 state prefix；不是 parent 输出后的 adapter | `SOURCE_EQUIVALENT` |
| predicted rollback | `modules/model.py:333-339`、server `572-600`：feedback 前先清 predicted，再提交 real | state 保留 `clear_predicted`/fail-closed；首次 C66 只启用 committed feedback，不伪造 predicted-video stream | committed path `EXACT`；predicted-video `INTENTIONAL_OMISSION` |
| real feedback commit | server `572-604`：真实 keyframes 与实际 executed action 以 timestep0 写 cache | `commit_executed_feedback` 先 append 真实 observation，再以 `t=0` 重跑同一 C58 30 blocks 得 executed-action K/V 并原子 replace | `SOURCE_EQUIVALENT` |
| observation cadence | LIBERO client `102-124`：一个 action chunk 每 4 actions 留一个 keyframe，整段实际 action 回传 | 直接消费 C57 manifest 的 observe-every4；3 observations + 8 executed actions 是一个 feedback update | `DATA_EQUIVALENT` |
| current observation | server 下一次 `_infer(... frame_st_id=self.frame_st_id)` 不再附加 feedback 中的最后一帧 | state 非空后忽略 API 中冗余 current carrier | `SOURCE_EQUIVALENT` |
| temporal position | server `_prepare_latent_input:265-312` 用持续增长 `frame_st_id` 构造 video/action grid | H3 observation K 做精确 temporal delta reindex；C58 action 用自身 1D absolute RoPE | observation `H3_EXACT_REINDEX`；action geometry `INTENTIONAL_DEVIATION` |
| joint world prediction | 官方同时 denoise video 与 action，并可暂存 predicted-video K/V | H3 只作为冻结 world carrier；C66 没有等价的 joint video denoise token stream | `NOT_REPRODUCED`，禁止称完整 LingBot 复现 |

## 唯一变量与训练边界

C66 相对 promoted C58 的机械唯一变量是：**是否启用 committed observation/action persistent prefix**。
模型参数集合、权重值、30-layer mapping、action flow loss、H3 INT8 权重和 normalization 均不变。启用且
history 为空时也必须与 C58 bit-exact。history 非空时，所有 30 个 block 才读取 committed prefix。

这不是相对 C57 的单变量实验；C57 parent、layer mapping、temporal contract 和 duplicate-current 都不同。
C57 只能用于定位 bug 与复用序列数据。

真实机械门必须同时满足：

1. LingBot source commit/三文件 SHA、H3 SHA、C58 SHA、C57 manifest/AUDIT SHA 全部 fail-closed；
2. C58 strict restore，C66 无新增 state keys；disabled 与 active-empty 相对 C58 `max_abs=0`；
3. 真实三个 observation 的 K 从共同 local phase 重排到三个 logical frames，V bit-exact；
4. commit 后每层恰为 `96 observation + 8 action = 104` tokens，`frame_st_id=3`、`action_st_id=8`；
5. 改变冗余 current carrier 不改变 history prediction，证明当前观测没有重复；
6. action loss 到达 C58 30/30 block 的 `self_attn.k`，H3 无 gradient；
7. model + runtime state strict restore `max_abs=0`；全程零 optimizer step、零训练 checkpoint。

机械通过后也只允许一个小预算、episode-disjoint 的 paired canary。必须固定 C58 control，比较 context-on、
context-off 与 history-shuffle；没有 clean-over-shuffle 机制信号就停止，不能用更多 steps 掩盖。最终晋级仍只看
同 seed/init 的 paired LIBERO success，而不是 train MSE。

## 当前实现与资源状态

- 实现：`src/fastwam/models/h3wam/c66_lingbot_fastwam_persistent.py`；无新 learned parameter；生命周期在
  C58 30 个 attention blocks 内。
- 测试：C66/C64/C57 共 `17 passed`，覆盖 state-key/parity、H3 reindex、no-duplicate、30/30 gradient、
  FIFO/restore。
- GPU：2026-08-17 检查时 n0 在跑 C58 full50 eval，n1/n2 在跑 C65，n3 也有既有任务。C66 不抢卡；
  真实 probe 保持 `PENDING_RESOURCE`，不会因此越门直接训练。

## 真实 probe 运行记录

- `mechanical-v1` 永久标记为 `INFRA_FAIL`：新节点使用 lean `conda-py311`，launcher 只导出了
  `${project_root}/src`，在第一次 online H3 forward 前因找不到
  `diffusers.modular_pipelines.minimax_h3` 退出。没有 report、optimizer step 或模型判定。
- 修复只改变执行环境：launcher 显式加入固定 vendored
  `${project_root}/third_party/diffusers_h3/src`；新增测试防止再次遗漏。模型、数据、seed 和全部阈值不变。
- `mechanical-v1` 不得改写或复用；后续必须写入全新 `mechanical-v2` 目录，并重新完成所有身份、数据、
  real H3、C58 block 和 restore gates。
- `mechanical-v2` 仍从旧 d2 snapshot 启动，重复同一个 missing-diffusers 错误；同样是 `INFRA_FAIL`、
  无 report、无 verdict。
- `mechanical-v3` 改用 `h3-int8-native` 和有效 vendor 后进入首个 H3 linear，但缺少云端
  `/usr/local/nvidia/lib{,64}` 动态库路径，以 `CUBLAS_STATUS_NOT_INITIALIZED` 退出；仍为 `INFRA_FAIL`。
- 正式重跑前 launcher 现在强制：`h3-int8-native`、pinned `before_denoise.py` SHA256
  `530b007c1d689c3ee1fc1690527f5253522d2da6b44dd326bec99faaf9f72fff`、CUDA library path，以及
  BF16/FP16 matmul preflight。前三个目录永不晋升为证据。
