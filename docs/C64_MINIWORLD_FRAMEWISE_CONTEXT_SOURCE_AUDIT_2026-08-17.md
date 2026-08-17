# C64 MiniWorld framewise context：逐行源码差异审计与机械门

状态：`SOURCE_AUDITED / REAL_MECHANICAL_PASS / NO_GO_OPTIMIZER`

## 结论

C62 的失败不能归因于 100 steps 太少。它先违反了 MiniWorld 最关键的时间合同：C58 每次执行
8 个动作后才提交一次观测，C62 将这 8 个动作 reshape 成两个 4-action latent group 后再做
`mean(dim=1)`；与此同时，每个独立提取的 H3 首帧 K 都保留相同的 temporal RoPE 坐标。结果是
“哪个 4-action group 对应哪个观测”和“历史观测的先后位置”都被抹弱。C62 canary 中 clean 与
action-shuffle MSE 几乎相同，正符合这个实现缺陷，不构成“多训就会好”的证据。

C64 只做源码能够直接支持的合同替换：每个真实观测严格对应此前 4 个已执行动作；真实全局首帧走
learned null；保留 raw H3 K/V，并在每次构建 rolling view 时将 K 的 H3 temporal RoPE band 重编号为
连续位置；sink 固定，FIFO 幸存帧重新落到连续逻辑位置。C58 仍是动作生成器，H3 仍冻结。

这不是 MiniWorld VideoDiT 的权重复现。MiniWorld 在每个 video block 的 QKV 之前用 6D AdaLN
shift/scale/gate 调制 video hidden states，并重新计算逐层 K/V；C64 仍是在外部对已经产生的 H3 K/V
做 4D shift/scale。这个差异必须标为 `INTENTIONAL_COMPOSITION`，不能把机械通过描述成 MiniWorld
训练方案已经完整移植。

## 固定源码身份

- MiniWorld 官方与 vendor：`e484206bbd4360ae56ed8abad51c83f2457ac092`，checkout clean。
- 执行文件 SHA256 继续由
  `src/fastwam/models/h3wam/c62_miniworld_context.py:26-34` fail closed。
- C58 parent：`c58b_online_s10000.pt`，SHA256
  `2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541`。
- H3 INT8：`minimax_h3_fl2va_pruned_int8_convrot.safetensors`，SHA256
  `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a`。

## 官方执行路径逐项对齐

| 合同 | MiniWorld `e484` 的真实执行 | C62 | C64 | 判定 |
| --- | --- | --- | --- | --- |
| raw action→latent frame | `conditioning/actions.py:22-36`：`4n` actions reshape 为 `n` 个独立的 `4A` slot，前置一个 null seed；无时间均值 | `c62_miniworld_context.py:318-324`：允许 `4n`，但随后跨 `n` 求 mean | 每个 observation 只接受 `[B,4,A]`；无 mean | C62 `MISMATCH`；C64 `SOURCE_EQUIVALENT` |
| learned null | `miniworld.py:621-623,659-660,798-815`：只有 global frame0 替换为 learned null，后续 streaming window 首帧保留真实 action | `None` 直接 identity，不经过 action encoder | `None` 替换为 trainable `[1,4,A]` null，再走同一 encoder | C62 `MISMATCH`；C64 `SOURCE_EQUIVALENT` |
| frame→token | `miniworld.py:815-834`：每帧 action embedding/modulation 通过 `frame_ids` 广播给该帧所有空间 token | 跨两个时间 group 平均后广播给整个 observation K/V | 一个 4-action embedding 广播给该 observation 的32个池化空间 token | C64 `EQUIVALENT_LAYOUT` |
| action modulation | `miniworld.py:404-432,472-506,541-556`：`to_emb` 加入 timestep stream，`to_mod` 加入 shared 6D AdaLN；每 block 再用低秩 6D refinement，调制 attention 和 MLP 的 shift/scale/gate | shared+per-layer low-rank，但直接产生 K/V 4D shift/scale | 保留 shared+30 layer refiners，仍直接调 K/V | C62/C64 均为 `INTENTIONAL_DEVIATION` |
| per-layer KV update | `miniworld.py:979-988`：每个 block 读取该层 past KV；当前 token 经本 block hidden update 后产生该层 new KV | 独立 H3 forward 后的逐层 K/V 外接拼接 | 同左；不伪称复现 VideoDiT block lifecycle | `NOT_REPRODUCED` |
| temporal position | `miniworld.py:301-361,970-977`：3D RoPE 使用连续 frame position；streaming 通过 offset 放置当前帧 | 独立 H3 首帧都在同一原始 temporal coordinate，直接拼接 | 只对已旋转 K 的 H3 temporal pairs 做精确 delta rotation；V、spatial bands、tail 不变 | C64 `H3_EQUIVALENT_REINDEX` |
| sink/FIFO | `denoiser.py:546-593`：sink K/V 原样固定；丢最老 post-sink；幸存 K 做负向 RoPE shift，布局连续 | sink/FIFO 身份正确，但未重旋幸存 K | state 存 raw K/V；每次 view 将 sink/FIFO survivors 重算到 `[0..N)` logical deltas | C64 `LIFECYCLE_EQUIVALENT` |
| commit | `denoiser.py:896-939`：完成的 clean frame/chunk 以 `t=0` 重跑所有 blocks，append per-layer KV，再 eviction/reindex | 提交真实观测 H3 K/V，不重跑 MiniWorld blocks | 同左，但明确列为外部 frozen-H3 composition | `INTENTIONAL_DEVIATION` |
| output/objective | MiniWorld 是 action-conditioned video velocity predictor，没有 action output | C58 action-flow policy | C58 action-flow policy | 必需的角色组合，不得以 video metric 代替 LIBERO success |

## H3 自身位置合同

对 released H3 layout 的真实 shape probe 得到：v7 双视角首帧 latent 为 `14×28`，固定窗口有14个
text rows，2×2 patch 后有98个 condition video rows，98 行的 temporal position 全为 `14`。这个
base position 随前置 text rows 数变化，但同一 observation 的全部空间行共享一个 phase；同一任务的
rolling observations 复用相同文本合同。H3 `rope.inv_freq` 为 `[16]`；
`int8_backbone.py:101-109` 将 time/height/width 三个 16-wide band 拼成 48-wide half，再复制为96个
rotary dims；`int8_backbone.py:132-145` 证明 capture 的 K 已经过 RoPE，V 未旋转。

因此 C58 的 sequence-only adaptive average pooling 不妨碍时间重编号：被池化的98行共享相同 time phase，
线性 pooling 与同一个 temporal rotation 可交换。为了保持 C58 已训练的 current-only 分布，C64 不把
base coordinate 从14改成0，而是让 sink 保持14，后续 logical frame 使用 `14+1,14+2,...`。这样
empty context 与 C58 逐值相等，同时历史帧具有连续相对时间。

H3 head dim128 的精确 temporal pairs 是：first half `0:16`，second half `48:64`；height/width bands
和 `96:128` tail 不动。公式是：

`x' = x*cos(delta*inv_freq) - y*sin(delta*inv_freq)`，
`y' = y*cos(delta*inv_freq) + x*sin(delta*inv_freq)`。

## 分阶段、可证伪机械假设

为避免把 cadence 和 RoPE 两项混成不可解释的效果实验，机械门拆成两个相邻阶段；两者都不做 optimizer：

1. `C64A_FRAMEWISE`：相对 C62 只替换 action/observation cadence——一个观测严格配一个4-action group，
   移除 shared mean。预期：交换完整 observation-action pairs 后，K/V 只是 chunk permutation；将 chunks
   对齐回原顺序后应逐值相同。
2. `C64B_TEMPORAL_REINDEX`：在固定 C64A 上唯一增加 exact H3 temporal K reindex。预期：交换两个
   post-sink pairs 后，即使把 chunks 对齐回同一 raw frame，K 也应因 logical position 不同而非零变化；
   V 应仍逐值相同。

若 C64A 不满足 permutation equality，说明 action/frame 实现仍混入其他变量；若 C64B 的 aligned K
仍相同、或 V 变化，说明 RoPE band/公式错误。任一失败都停止，不训练。

这里的“单变量”只对 A→B 的 RoPE 机械差成立。相对 C62 而言，C64A 同时改了 cadence/mean 和
learned-null 两个元素，不能作为严格的效果单变量。learned-null 在 zero-init 机械点不改变输出，但一旦
optimizer 更新 modulation head 就可能产生效果。因此后续若要做 optimizer canary，必须再显式拆成：
`cadence-only` → `+learned-null` → `+RoPE` 三臂；当前实现和 report 不授权直接训练。

## 已实现的机械边界

- `src/fastwam/models/h3wam/c64_miniworld_framewise_context.py`
  - exactly-four action contract，无 temporal mean；
  - learned null；
  - H3 temporal-only K reindex；
  - raw-state sink+FIFO，rolling view 重新连续编号；
  - context default-off 直接调用未修改 C58。
- `tests/test_c64_miniworld_framewise_context.py`
  - 4项 C64 定向测试，加 C62 回归共13项通过；
  - 覆盖 RoPE 数学公式、default-off、拒绝8-action frame、pair order、action-loss gradient、sink/FIFO。
- `scripts/h3wam/probe_c64_miniworld_framewise_context.py`
  - 读取真实 H3 `rope.inv_freq` 与 released layout；
  - strict restore 真实 C58；冻结 C58，只验证 action loss 到达 bridge 30/30 refiners；
  - C64A→C64B 单变量 permutation/reindex falsification；
  - zero optimizer step，报告固定为 `NO_GO_OPTIMIZER`。

## n1 真实机械结果

报告：`/mnt/h3-wam/outputs/c64-miniworld-framewise-context/mechanical-v2/report.json`，SHA256
`f460a1c8443b32beee6fd68e34c446fde73e347541225c661de64cce5f7dcaf9`。结果为
`PASS_MECHANICAL_GATE / NO_GO_OPTIMIZER`：

- MiniWorld、H3 INT8、C58 三重身份通过；C58 30层 strict restore；
- released H3 layout 为98行，共同 temporal position `14`；
- default-off 与 empty-context 相对 C58 的 `max_abs` 都为 `0`；
- C64A pair-swap 后对齐的 K/V `max_abs=0/0`；
- 唯一增加 temporal RoPE reindex 的 C64B 对齐 K `max_abs=3.4609375`，V仍为 `0`；
- C58 parent 无 gradient；action loss 到 shared head 与30/30 refiners；
- sink/FIFO IDs `[0,3,4]`；bridge/runtime restore `max_abs=0`；
- elapsed `62.01s`，峰值 allocated/reserved `4.747/4.842 GiB`；n1 随后释放为8卡全空闲。

第一次 `mechanical-v1` 在 shape gate 以零 optimizer step fail closed：探针使用4个 synthetic text rows，
所以 condition anchor 合法地为4而不是固定真实窗口的14。`mechanical-v2` 将 source probe 修正为真实窗口的
14 text rows；没有改模型、权重或判定门。v1 没有 report，不能作为通过证据。

## 后续放行规则

真实机械 report 只有全部满足以下条件才允许设计数据 canary：

1. MiniWorld/H3/C58 三重身份和 strict restore 通过；
2. H3 layout 为98 condition rows、共同 temporal coordinate14；
3. context-off 与 empty-history 相对 C58 均 `max_abs=0`；
4. C64A aligned K/V 均 `max_abs=0`；
5. C64B aligned K `max_abs>0` 且 V `max_abs=0`；
6. C58 无 gradient，action loss 到达 shared head 与30/30 layer refiners；
7. sink/FIFO IDs 为 `[0,3,4]`，bridge/runtime strict restore `max_abs=0`。

机械通过仍不等于有效，当前明确 `NO_GO_OPTIMIZER`。只有先把 cadence、learned-null、RoPE 做成三个
显式可切换且逐臂 parent parity 的合同，才可重新申请 episode-disjoint 等预算 canary。即使完成拆臂，
clean-vs-action-shuffle 仍是机制门，context-off 是安全门，最终只能由 paired LIBERO 晋级。若不愿继续
外部 K/V bridge，则应转向在 H3 block 内部实现 pre-QKV 6D action modulation 或显式 world-prediction
auxiliary，而不是继续给 C62 打补丁。
