# C62 MiniWorld → C58 rolling-context source audit and mechanical gate

状态：`SOURCE_READY / LOCAL_MECHANICAL_PASS / NO_GO_LONG`

## 结论先行

MiniWorld 官方 `master` 与本地固定版本均为
`e484206bbd4360ae56ed8abad51c83f2457ac092`。它是可训练的 action-conditioned streaming
video world model，但没有 action prediction head、语言目标、proprio policy 输入或 LIBERO policy
evaluator。因此 C62 不会把 MiniWorld 当作动作策略，也不会用视频生成质量冒充执行成功率。

C62 的直接父模型固定为已经过 680 对闭环晋级的 C58：checkpoint
`/mnt/h3-wam/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt`，
SHA256 `2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541`。唯一变量是：训练和
rollout 都把真实历史观测按 MiniWorld 的 action-aligned chunk、persistent sink、bounded FIFO
生命周期送入 C58 的 30 层 world K/V 前缀。H3 仍冻结，C58 仍负责生成 32×7 action。

## 官方代码实际合同

| 环节 | 官方执行路径 | 审计结论 |
| --- | --- | --- |
| dataloader | `miniworld/train.py:76-101` → `miniworld/data/droid.py:105-285` | 只保留 success episode；每条样本是 RGB clip 和 `T-1` 已执行动作；q01/q99 归一化并 clip 到 `[-1,1]` |
| action alignment | `miniworld/conditioning/actions.py:17-31` | Wan VAE 的每个未来 latent frame 对齐4个 raw action；第一个 seed latent 使用 null action |
| model forward | `miniworld/miniworld.py:581-1006` | video token 使用 chunk-block-causal attention；action 通过 shared AdaLN 与每层低秩 refinement 调制 video blocks |
| loss | `miniworld/denoiser.py:446-499` | 非递减 chunk timestep 的 rectified-flow video velocity MSE；clean context 不计 loss；没有 action loss |
| launcher | `scripts/train_droid.sh` | 6→16→32→64 latent frame 四阶段；前两段100/50 epoch，后两段各30k step；8卡BF16+Muon |
| checkpoint | `miniworld/train.py:131-194` | 保存 model/EMA/optimizer/meta；curriculum load 会按 shape 过滤并 `strict=False`，optimizer不兼容时跳过 |
| streaming | `miniworld/denoiser.py:527-1043` | 已完成 world chunks 写入逐层K/V；保留1个sink，其余FIFO；eviction后对K做RoPE shift；in-flight异步去噪 |
| evaluator | `miniworld/sample.py` | 生成和吞吐路径；没有 action-policy rollout、LIBERO success、动作恢复或 policy checkpoint exporter |

开放性分类为 `TRAINABLE_WORLD_MODEL_ONLY`。仓库无 LICENSE 文件，因此 C62 只在研究环境中从固定
vendor checkout 导入并验证源，不复制其实现到发布包。

## 与 C58 的差异矩阵

| 维度 | C58 carrier冠军 | C62唯一变量 | 对齐状态 |
| --- | --- | --- | --- |
| action policy | 30层官方FastWAM ActionDiT，H3 50→30逐层K/V | 完全相同父模型和权重 | `EXACT` |
| current world input | 当前真实观测的32个H3 K/V token/层 | 当前观测前追加真实历史world K/V | `INTENTIONAL_COMPOSITION` |
| action/context coupling | 历史动作不进入world prefix | 每4步动作成组，shared modulation + per-layer low-rank refinement | `EQUIVALENT_ORGANIZATION`，不是MiniWorld权重复现 |
| cache lifecycle | 每次replan冷启动 | 第1真实观测永久sink，剩余按FIFO；最多15个历史chunk，当前chunk为in-flight | `EQUIVALENT_LIFECYCLE`；15而非官方24是数据窗口约束 |
| world model | 冻结INT8 H3在线抽取 | 仍冻结；不训练/部署MiniWorld VideoDiT | `EXACT_PARENT_INVARIANT` |
| objective | C58 action flow MSE | 完全相同action flow MSE；梯度额外进入context modulation | `EXACT_ACTION_OBJECTIVE` |
| deployment | wait30/replan8/horizon32 | 相同；每次只提交真实观测及实际执行的8步动作 | `EXACT`，真实runner尚未接线 |

## 单变量假设

在固定 C58 checkpoint、H3、action flow、数据、normalization、noise、optimizer、replan8 与 evaluator
时，使当前动作查询读取 action-aligned 的真实 world history，会提高 Goal/LIBERO-10 长程成功率，并且
Object/Spatial 任一 suite 相对 C58 不退化超过3pp。

## 已实现的机械门

- `src/fastwam/models/h3wam/c62_miniworld_context.py`
  - 对6个MiniWorld关键执行文件逐文件SHA256 fail closed；
  - 实际导入并执行官方4-action alignment和block-causal mask；
  - disabled与empty-history路径逐值等于C58；
  - real observation only，显式action-before-observation，4n对齐；
  - one-chunk sink + FIFO eviction，不保留predicted/future entry；
  - shared action modulation + 30个layer-local low-rank refinement；
  - runtime state可独立snapshot/strict restore。
- `tests/test_c62_miniworld_context.py`：5项本地测试通过，覆盖source、parity、eviction、shape、action-loss
  gradient与model/runtime restore。
- `scripts/h3wam/probe_c62_miniworld_context.py`：只加载真实C58 s10000，验证30层gradient和严格恢复，
  写report但不写candidate checkpoint。

## n1机械门与后续放行顺序

在 n1 (`30907`) 的只读复合snapshot上执行：

1. official source hash/import/action alignment/block-causal mask；
2. C58 s10000 SHA、合同和30层 strict restore；
3. disabled/empty context `max_abs=0`；
4. sink+FIFO后每层token shape、update order、4n action alignment；
5. action loss必须到达30/30 C58 blocks、shared modulation和30/30 layer refiners；
6. frozen world K/V不得带gradient；bridge+runtime restore `max_abs=0`；
7. 仍写 `NO_GO_LONG`。只有再补齐 episode-disjoint sequence manifest、真实rollout commit trace、匹配
   C58 optimizer-step/optimizer-restore canary和 clean-vs-action-shuffle mechanism gate，才能升级小规模训练。

后续训练也不得一次放到长训：先固定 C58 为 parent 跑等预算 context-off/context-on 两臂，短→长上下文
阶段只改变历史长度；每个里程碑先做 held-out action-shuffle 因果门，再做新trial paired LIBERO。世界预测
或离线MSE改善不能替代动作成功率门。
