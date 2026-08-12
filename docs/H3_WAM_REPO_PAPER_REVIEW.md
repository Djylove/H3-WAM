# H3-WAM 仓库与论文对照整改结论

更新时间：2026-08-12

## 2026-08-12 数据采样纠偏

前期每条 episode 固定取 5 个窗口的方案已经停止。这里的“5”是窗口起点数，
不是一个视频 clip 的图像帧数，但它仍然把 1,712 条示范压缩成 8,560 个窗口。

逐行复核 FastWAM `RobotVideoDataset`、`BaseLerobotDataset` 和 DreamWAM
`iter_libero_windows` 后确认公开协议为：

- 四套 LIBERO 共 40 tasks、1,712 episodes、277,713 原始 frame；
- 每个原始 frame 都是窗口起点，`global_sample_stride=1` / `sample_stride=1`；
- 动作为当前帧起连续 32 step；RGB 在 33-step span 上取 9 帧（Wan 路径 stride 4）；
- 轨迹末尾不删除，而是复制边界帧/动作构造定长张量，并用 `image_is_pad`、
  `action_is_pad` 从损失中屏蔽 padding；
- 全部 1,712 demonstrations 用于训练，batch16×8 GPUs、10 epochs，约 21.7k step。

H3 路径保留相同 33-step/1.6 s 语义，把 33 个 20 Hz 源帧最近邻对齐为 H3
合法的 39 个像素帧，再经 H3 VAE 得 12 个 latent frames。新增实现按 H3 发布 VAE
的 17-frame chunk、4× temporal compression、末尾 drop 3 token 的精确几何，把
pixel padding 映射成 latent loss mask。

当前三阶段门禁：40-task dense canary（80 step）→ 全部 277,713 frame-indexed
样本一 epoch（2,170 step）→ 闭环转正后扩到 10 epoch / 21,700 step。

### 2026-08-12 实际执行结果

- 固定 val40 动作损失：旧 sparse head `0.259033`，dense canary step40
  `0.240706`（改善 7.1%），step80 `0.212337`（改善 18.0%）。说明扩大窗口起点
  覆盖对动作拟合有稳定收益，但该指标仍不能替代闭环成功率。
- 同一 LIBERO Goal task3、相同 4 个 trial 的闭环结果，step40 和 step80 均为
  `0/4`。step80 的 4 次 rollout 中抽屉最大位移仅 `0.0017/0/0/0`；有盘子或碗
  位移，但未完成“打开顶层抽屉并放入碗”的组合任务。因此 canary 只通过健康检查，
  没有通过可执行策略 Gate 2。
- 已生成纯新增 dense 中间候选：40 tasks × 256 windows = 10,240 windows，
  `new_dense_train_windows=10240`，不允许旧 sparse 窗口补位。从 step80 续训
  160 steps，global batch 128；首轮 optimizer 冷启动波动后，step5–15 action loss
  回到约 `0.22–0.29`、gradient norm `0.49–1.19`，无 OOM/NaN。
- cache 编码改为三节点共 24 个 episode-disjoint shards，原子写入共享目录；已有缓存
  全部跳过复用。有效 horizon cache 目标 222,929，完成后继续三节点编码 54,784 个
  padded-tail windows，最终审计必须精确等于 277,713。
- 旧 sparse joint 训练在 step432 停止，保留 step420 等已有 checkpoint，不再为已淘汰的
  5-start sampling 继续消耗 8 张 A800。

## 结论

下一阶段不再优先增加 motion、memory 或对象教师，而是先完成一次数据与优化配方都可比的
H3-DoT 训练。当前连续闭环失败还不能证明 H3 不适合作为 WAM 基模，因为现有实验有两个
主导性偏差：训练数据每条 demonstration 只保留 5 个窗口，以及联合训练的 H3/action
学习率分别只有 `1e-6/1e-5`。

四套 LIBERO 原始数据有 1,712 episodes、277,713 frames。现有 candidate 只有 8,560
windows，其中 episode-disjoint train 为 7,710；这相当于每个 episode 固定抽 5 个窗口，
只覆盖官方 frame-indexed loader 约 2.8% 的样本。按“不使用尾部 padding”的保守口径，
仍有 222,929 个完整 horizon-32 windows，train/val 分别为 200,779/22,150，是现有缓存的
26 倍。

因此，当前 602 steps、global batch 128 虽被称为 10 epochs，实际只是对 7,710 个稀疏窗口
循环 10 次；它不能与 FastWAM 的 20k steps、DreamWAM 的 21.7k steps 或 Faster-WAM 的
dense-data 10 epochs 等价。

## 逐项目对照

| 项目 | 核心可借鉴点 | 与当前 H3-DoT 的差异 | 现在如何使用 |
|---|---|---|---|
| FastWAM | 训练期 RGB future co-training；推理只保留首帧；30 层、约 1B 的 ActionDiT 从 Wan 视频层插值初始化 | 当前推理首帧路径和因果 mask 已对齐；动作头只有 1 层且不是由 H3 插值初始化；数据量和训练步数远低 | 固定数据、动作归一化、10-step sampler 和闭环协议；作为效果基线 |
| Faster-WAM / DoT | 单层动作头；全层 K/V fusion；3D RoPE 逆旋转后映射到 action 1D RoPE；去掉 action text cross-attention | 核心结构基本对齐；但论文统一 `1e-4`、cosine、logit-normal、dense 10 epochs，当前主线是 action `1e-5`、H3 `1e-6`、shifted-uniform | 保留 DoT 架构；先用 dense 数据和动作 `1e-4` 做受控复现，再逐级提高 H3 LR |
| DreamWAM | RGB + optical-flow 联合去噪；DINO/depth residual 仅训练期使用；新增 flow I/O 用 0.1×随机初始化；训练约 21.7k steps | 已验证 flow loss 可从 2.0 降至 1.80，但 sparse-data 60-step 下动作 MSE和闭环未收益 | 暂不扩 602 steps；dense RGB-DoT 出现闭环正例后，再做同数据 motion A/B |
| ImageWAM | 更换 world backbone 仍保留深 ActionDiT；动作模块从视觉 backbone 插值初始化；`1e-4`、10 epochs；大规模机器人预训练显著提升 C2R | 强调“更强 H3 不会自动变成更强动作策略”，动作初始化和机器人数据规模同样关键 | 若 dense DoT 仍失败，优先做 H3→ActionDiT 初始化或接入 FastWAM action expert，而非继续加视觉目标 |
| MiniWorld | block-causal video DiT、chunk-wise diffusion forcing、长上下文 curriculum、rolling KV | 是 action-conditioned causal world model，不是直接 policy；替换当前 H3-DoT 会同时改变训练和推理范式 | 第二阶段用于 RobotTTT/上下文记忆；先做独立 action-following world-model canary |
| MotuBrain | 独立 text/video/action 三流；中间 50% 层 H-Bridge；V2A 非对称注意力；大规模 embodied pretraining | H3-DoT 通过视频 hub 间接路由语言，缺少独立 text stream；当前数据远小于其预训练规模 | dense DoT 若语言路由仍错，再做“中层 K/V 或 text stream”消融；不先照搬全架构 |
| BadWAM | world-action drift：想象保持不变时动作仍可漂移；提供 action-only / imagination-preserving attack 诊断 | 它不是性能提升模型，而是暴露 world representation 与 action 对齐脆弱性 | 借鉴为评测：同世界表征下测 action sensitivity，不作为下一条训练主线 |

## 当前实现中已排除的问题

- 推理时 H3 只接收文本、首帧与 state，不实例化未来 RGB/audio 噪声；随机未来模态不是闭环
  失败原因。
- 训练时 observation mask 禁止文本/首帧查询读取未来 video/audio 行，未发现 target leakage。
- 两摄像头按 `224×448` 横向拼接、action horizon 32、10-step flow sampler、min/max
  normalization 均与 FastWAM 主协议一致。
- Faster-WAM 的 K/V channel mapping、head-wise layer mixing、inverse 3D RoPE、key norm、
  action 1D RoPE 和无 action-text cross-attention 已实现。论文没有公开官方代码，RoPE 中 fused
  video token 的 1D 位置 `b_j` 也未给出具体映射；当前将首帧所有空间 token 设为 position 0，
  属于合理但未被官方代码验证的实现选择。

## 新证据

- frozen H3 motion checkpoint 上，仅把 ActionDiT/KV fusion/state 的学习率提高到 `1e-4`
  训练 60 steps，固定 val40 MSE 达到 `0.202317`：优于 motion base `0.217259` 和 RGB
  step360 `0.214504`。但 Goal 0/3/7/8 canary 仍为 `0/4`；这支持“动作侧欠训练”，也说明
  sparse 60-step 的数值收益尚未转成可执行策略。
- RGB 主线 step360 是当前联合训练离线最好点，val40 `0.214504`；截至本文更新，主线运行至
  step384/602，训练稳定。
- paper-I/O motion step60 的 flow loss 从 `1.999966` 降至 `1.799538`，证明 motion 通道
  能学；但 val850 比 RGB step300 退化约 1.36%，Goal canary 仍为 0/4，所以不能仅凭
  motion loss 下降扩长程训练。

## 执行顺序

### P0：完成现有证据

1. sparse 主线已在 step432 停止；只保留已有 checkpoint ladder 作为历史对照。
2. C 节点完成 action-LR `1e-4` 最终 checkpoint 的 Goal 0/3/7/8 canary；若出现成功，扩到
   task3 seed42 ×10 和四 suite canary。
3. B 节点完成 RGB step360 的跨 suite canary；离线 MSE 不作为成功替代。

### P1：dense H3 缓存与训练（新的主线）

1. D 节点生成 222,929 个完整窗口的 H3 VAE cache；复用 8,560 个旧缓存硬链接，新增磁盘
   预计约 100GB。入口：`scripts/h3dreamwam/prepare_h3dotwam_dense_libero.sh`。
2. dense split 保持 episode-disjoint，避免把同一 episode 的相邻窗口分到 train/val。
3. 第一条训练线先冻结 H3，只训练 action/KV/state：global batch128、action LR `1e-4`、cosine，
   跑一个 dense epoch（约 1,569 steps），每 200–300 steps 做 val40/闭环门禁。
4. 只有 head-only 至少出现一个标准闭环 success，才启动联合 H3：从该 action checkpoint
   开始，H3 LR 先用 `1e-6`；若稳定但收益饱和，再做 `3e-6` 和 `1e-5` 短 A/B。不能直接把
   32B H3 提到 `1e-4`。
5. 联合训练目标保持 RGB/action flow，各自权重 1.0；先不加 motion、ranking、history 或
   SAM3D，保证闭环收益可归因。

### P2：只有 P1 仍失败才做的结构消融

1. 对照 ImageWAM/FastWAM，给单层 action block 做 H3 权重插值初始化，或接入已预训练的
   FastWAM ActionDiT，再比较随机初始化。
2. 对照 Faster-WAM/MotuBrain，比较 all-layer KV、middle-50% KV 和 final-layer KV；同时输出
   layer-mix 分布，检查是否学到中层信号。
3. 增加 BadWAM 风格 world-action drift 和 task-language counterfactual，不把单纯 action MSE
   当作闭环代理。
4. 只有 RGB dense 主线已成功，才把 DreamWAM motion 加回做严格同数据 A/B；MiniWorld
   memory/RobotTTT 留到基本策略可用之后。

## 晋级门槛

- Gate 1：固定 val40/val850 相对 sparse baseline 明确改善，只是健康检查。
- Gate 2：Goal task3、seed42、10 trials 至少 1 次成功，证明动作链可执行。
- Gate 3：四 suite 各 4-task canary 至少出现跨任务成功，不接受只会一个 task 的专用头。
- Gate 4：再扩到 40 tasks × 50 trials 的标准 LIBERO benchmark，并同步测试 LIBERO-Plus。
- 任何新模块都必须对同一 dense split、同一 normalization、同一 rollout seeds 做 A/B。
