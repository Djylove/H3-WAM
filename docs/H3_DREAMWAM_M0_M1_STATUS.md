# H3-DreamWAM 快速验证状态

更新时间：2026-08-10

## 当前结论

H3 替换 DreamWAM 的 Wan2.2 在结构、真实显存、反向传播、联合采样和 Flow teacher
训练上均可行。M2 已确认 Wan 与 H3 的关键差异：H3 必须为 proprio 预留真实 packed
text row；Flow 输出不能零初始化；world teacher 必须做逐样本标准化和 timestep
weighting。修正后 motion-20 在同结构 held-out 基线上将 10-step 动作 MSE 改善 6.10%，
且 100-step 闭环首次产生强物体交互，但仍把酒瓶当成目标、未打开抽屉，因此只能证明
DreamWAM 表征路线有增益，尚不能证明 task policy 成功。当前最佳保留为 M2 motion-20。

## 真实 8×A800 结果

- 33.12B H3 + 3.37B ActionDiT 完整 FSDP 一步通过；BF16 参数版本峰值
  23.79 GiB/卡。
- H3→ActionDiT 直接全层更新失败：即使先校准输出头、FP32 FSDP 参数存储且学习率
  1e-7，action loss 仍从 4.816 上升到 14.54、147.52。
- 失败梯度来自 Action 主体：首步 group norm 约 5.97e5；输出头仅 23.3，H3 为 0。
- 输出头在 task3 的 8-rank 多窗口 warmup 中，50 步 action loss 从 18.168 单调下降到
  4.359；checkpoint 仅 17 KB。
- FP32 分片参数存储已跑通，峰值 41.46 GiB/卡、reserved 58.39 GiB/卡；CPU 端直接
  构造八份 FP32 模型会触发容器 OOM，因此必须先 BF16 构造、FSDP 分片、再转 FP32。
- 尾 1 层续训 10 步稳定，action loss 从 4.816 降到 4.235；可恢复 stage checkpoint
  310 MB。
- 从该 checkpoint 扩至尾 2 层后正式训练 10 步，action loss 从 4.197 降到 3.385；
  第 2 步短暂升到 4.250，随后连续下降，未发生深层爆炸。
- 尾 4 层若直接沿用同一学习率会不稳定；旧参数保持 1e-7、新增两层使用 1e-8 后，
  固定窗口 10 步 action loss 从 3.328 降到 3.006。
- 训练清单已改为跨 rank、跨 step 轮换 45 个窗口。继续轮换 18 步后，在完全未见的
  episode 47（5 个窗口）上，平均 action velocity loss 从 14.6271 降到 12.0310，
  改善 17.75%；video loss 保持 0.468094，符合 H3 冻结的预期。
- 真实 H3 + ActionDiT 的 10-step 联合 Euler sampler 已跑通。相同 held-out 数据、噪声
  和采样配置下，训练前 checkpoint 的动作 MSE 为 12.7518，轮换训练后的动作 MSE 为
  11.8940，改善 6.73%；两者 video MSE 都是 1.36603。说明训练收益已传递到最终动作
  生成，而不只存在于训练目标上。
- 对齐 DreamWAM 官方 timestep weighting 后继续训练 50 步，held-out 10-step 动作 MSE
  从 11.8940 降到 7.9193；在线动作饱和率从 61.46% 降到 44.79%。
- optimizer 审计发现 `time_projection` 一直保持全零，且时间分支未进入参数组。修复后
  新增 7.61M 优化参数，首步 action-time grad norm 为 12.71；再训练 50 步后 MSE 降到
  7.2640，相对最初 12.7518 累计改善 43.04%。
- 训练域投影可把闭环环境动作饱和率从 47.03% 降到 0%，平均动作幅值从 0.712 降到
  0.537，但 task3 的 100-step rollout 仍未移动抽屉或物体；视频显示机械臂轨迹仍不指向
  目标，因此不能用动作裁剪代替训练收敛。
- 将其余 46 层以 1e-9 纳入的 20-step 全层探针训练稳定、训练 loss 下降，但 held-out
  sampler MSE 反而从 7.2640 退化到 8.1722。该候选被拒绝，13 GB checkpoint 已删除。
- 最佳 `timecond50` 在 5 个 held-out 窗口上的 10-step sampler MSE 为 7.497
  （6.896～8.421），确认单窗口结果不是偶然；继续训练 100 step 后同一首窗口退化为
  8.416，因此新 checkpoint 不进入闭环。
- 复核 DreamWAM 数据协议后发现旧 v1 context 含约 100 个示范图像 token；在线又固定
  复用该 context，造成当前观测与旧示范图像同时条件化。该训练/部署错配是闭环随机运动
  的首要代码级根因。M2 已改为 manifest `context_id` + text-only 强校验。
- 10-step 联合采样耗时约 10.2 秒/批（8×A800 当前研究实现，尚未做缓存和推理优化）。

## 云端有效产物

- 根目录：`/home/h3wam_finetune`（未写 `/root`）
- head warmup：`outputs/h3dreamwam_m1/task3_head_warmup50.pt`
- 尾 1 层可恢复点：`outputs/h3dreamwam_m1/task3_tail1_train10.pt`
- 尾 2 层可恢复点：`outputs/h3dreamwam_m1/task3_tail2_train10.pt`
- 尾 2 层探针：`outputs/h3dreamwam_m1/task3_tail2_resume_probe3.json`
- 尾 4 层分层学习率点：`outputs/h3dreamwam_m1/task3_tail4_layerwise_train10.pt`
- 当前最佳点（尾 4 层 + DreamWAM weighting + 时间条件）：
  `outputs/h3dreamwam_m1/task3_tail4_timecond50.pt`
- 训练前采样对照：`outputs/h3dreamwam_m1/sample10_task3_tail4_before_rotate.json`
- 当前最佳采样：`outputs/h3dreamwam_m1/sample10_task3_tail4_timecond50.json`
- 100-step 闭环：
  `outputs/h3dreamwam_m1/libero_task3_timecond50_clip_canary100/results.json`
- 失败全层候选仅保留 JSON 报告，13 GB checkpoint 已删除。
- 清理前 H3-DreamWAM 输出约 7.2 GB；清理结果见下方 M2 状态。

## M2：DreamWAM 精确对齐起跑

- 已修复 manifest `context_id` 路由并增加 text-only 强校验；v2 task3 为 150 个 train
  窗口（30 episodes）和 15 个 val 窗口（3 个完全未见 episodes）。
- 已加入 DreamWAM/FastWAM 的 full-attention-width Q/K RMSNorm 和 ActionDiT 插值
  alpha scaling，并写入 checkpoint architecture 元数据；旧 M1 checkpoint 仍兼容。
- 全新 M2 初始化在 15 个 val 窗口上的 10-step sampler MSE 为 20.8438。仅校准随机
  action output head 50 step 后降至 14.0012，改善 32.83%；训练 raw loss 的前/后 10 步
  均值从 24.3364 降至 16.9235。
- 回退为 H3 风格 norm/初始化并继续使用严格 text-only context 后，head-only 的 5 个
  held-out 10-step sampler MSE 为 11.937、12.076、13.658、10.785、11.940，均值
  12.0790；比 Wan exact norm + alpha 的 14.0012 再低 13.73%。这证明 text-only 修复
  应保留，但 Wan 专属 ActionDiT 尺度规则不应机械移植到 H3。
- FP32 head backward 因 FSDP 预取达到 78.44 GiB/卡并 OOM；BF16 模型存储 + FP32
  master head 可稳定完成，峰值 76.66 GiB/卡。这个内存现象只影响 head-only FSDP
  布局，尾层训练继续用已经验证更省显存的 FP32 sharded 路径做 capacity probe。
- exact norm + alpha 的尾 1 层探针未通过：前 3 步 gradient norm 为 3.97M、15.42M、
  6.07M，第 2 步 loss 跳至 55.91；任务在第 4 步后主动终止且未保存 checkpoint。
  DreamWAM 从 Wan 复制的 time projection、block modulation 和 cross-attention 均有预训练
  权重，而 H3 没有一一对应模块；只移植 alpha 会破坏尺度，不能称为“精确复现”。
- 已删除 7 个被覆盖或明确退化的 M1 大 checkpoint，只保留 M1 最佳
  `task3_tail4_timecond50.pt`、M1/M2 JSON 证据和 KB 级 M2 head；云端可用空间恢复至
  212 GB。
- 已按论文优先级实现 RAFT color-wheel motion 预计算入口和 H3-VAE latent shape 校验；
  trainer 已支持 RGB/Flow 同 timestep 联合 flow matching、Flow loss 0.5 和独立 H3 I/O
  解冻开关。推理仍保持 flow 全零，因此不增加 RAFT 在线依赖。进入训练前先做 1-window
  云端 artifact smoke，防止直接生成 dense cache。
- H3-style head 后接 tail-1 训练 10 步，覆盖 80 个不同训练窗口；5-window sampler 均值
  12.0433，相对 head-only 12.0790 仅改善 0.30%，且 2/5 窗口变差，未达到晋级门槛。
  310 MB 候选 checkpoint 已删除，仅保留训练/评测 JSON。
- task3 的 165 个稀疏 motion artifact 已全部生成：missing/extra 均为 0，全部为 finite
  `[1,24,12,14,28]`，总计 72 MB。单窗口 RAFT + H3 VAE 平均约 2.5 秒/卡。
- 首次 RGB+Flow 联合 backward 通过：RGB/Flow/Action loss 分别为 0.3565/2.6647/12.4172，
  H3 gradient norm 651.24，峰值 41.46 GiB/卡。参数审计同时发现 FSDP wrapper 导致
  H3 tail 名称筛选漏选；该 probe 只更新了 H3 I/O，已修复后重跑，不能作为训练候选。
- 源码逐行对照发现三项关键偏差并已修复：官方 Flow 输出行以源输出 std 的 0.1 随机
  初始化，而旧实现为全零；官方 proprio 同时进入世界和动作 expert，而旧实现只进入
  ActionDiT；官方 world teacher 做逐样本标准化并使用 timestep weighting。
- Wan 使用独立 cross-attention context，H3 则把文本/视频/音频放入同一 self-attention
  packed sequence。H3 适配因此在 layout builder 中新增一个 proprio text row，并同步
  获得正确 RoPE、timestep index 和 observation causal mask，不能只拼接 context tensor。
- 仅加入共享 packed proprio、尚未 motion 训练时，5-window sampler 从 12.0790 降到
  11.5317，改善 4.53%，说明机器人当前状态确实应进入 H3 世界分支。
- 修正后的首步 Flow loss 为 1.9961，H3 gradient norm 从旧探针约 646 增至 781；峰值
  41.47 GiB。motion-20 的同结构 held-out sampler 为 10.8282，相对 11.5317 净改善
  6.10%，5 个窗口中 4 个改善；相对旧 text-only 12.0790 累计改善 10.36%。
- motion-20 的 task3 100-step canary 为 0/1 success：top drawer 最大位移 0.00153，但
  wine bottle 位移 0.756。旧 M1 相同 100-step rollout 所有物体和抽屉几乎零位移，因此
  Flow 表征已转成更强环境交互，但对象/任务对齐仍错误，不能把“会碰物体”算作成功。
- 部署 server 已修复只加载 Action stage、漏加载 H3 I/O/tail 的问题，并在 FSDP FP32
  materialization 后精确恢复所有 stage tensor；rollout 子进程也已隔离 simulator 的
  CPU torch 与 policy server 的 CUDA torch。
- H3/Action 分 expert clipping 已打通。Action LR 1e-5 在第 2 步把 action loss 推到
  414，已主动终止且未产 checkpoint；降到 Action I/O 1e-7、新 tail 1e-8 后 50 步稳定，
  Flow loss 从约 1.986 降到 1.958。但 sampler 为 10.8995，比 motion-20 退化 0.66%，
  5.50 GB checkpoint 已删除，仅保留 JSON。说明联合训练机制可用，但必须更密 checkpoint
  early stop，不能用固定 LR 盲跑。

## M3：四套 LIBERO 数据扩展

- 已将 `libero_10`、`libero_goal`、`libero_object`、`libero_spatial` 合成新的 episode-disjoint
  candidate：1,712 条真实 episode、40 个任务、8,560 个唯一窗口；train/val 为
  7,710/850 个窗口、1,542/170 条 episode，episode overlap 为 0。
- 最初拟议的 task3 6 倍采样方案在训练启动前被否决，旧自动启动器已停止。正式 v4
  candidate 中 7,710 个训练窗口均只出现一次，40 个任务没有 task-specific weight；task3
  checkpoint 也不作为初始化，避免把单任务偏置带入通用模型。
- 旧 H3 RGB latent 和 40 个 text-only context 只读复用；归一化已经改为仅从 7,710 个
  unique train 窗口计算，验证数据不再参与 stats。四个 suite 的真实 artifact loader smoke
  已通过。
- 全量 RAFT color-wheel/H3-VAE motion latent 正在 8 卡并行生成；已有 task3 的 165 个
  artifact 以硬链接复用。生成结束后自动做 8,560-window 全量审计；通过后从原始 H3 派生
  ActionDiT，先用所有 7,710 个唯一窗口完成 964-step uniform head epoch。只有最后 50-step
  action loss 均值低于最初 50-step 才进入 100-step uniform RGB+Flow 联合 canary（H3
  `2e-6`、Action `1e-7`、独立 clipping）。
- 全量 motion cache 已于 2026-08-08 完成并通过 8,560-window finite/shape/full-loader 审计，
  最终占用 3.7 GB。uniform head epoch 已从无 task checkpoint 的原始 H3 派生初始化起跑；
  前 115 步覆盖四个 suite，前/最近 50 步 action loss 均值为 24.89/20.62（下降 17.2%），
  梯度范数均值为 376/237，训练稳定。
- uniform head 已完成全部 964 步并覆盖 7,710 个唯一 train 窗口；最初/最后 50 步 action
  loss 为 24.89/13.19。随后 100-step RGB+Flow canary 覆盖 800 个窗口。严格分层验证集从
  40 个任务各取 1 个未见 episode 窗口（四个 suite 各 10 个），相同 10-step sampler 下
  fresh/head/joint100 action MSE 为 19.8969/8.8162/8.6472：head 相对 fresh 改善 55.69%，
  joint100 相对 head 再改善 1.92%，5 个跨任务 batch 中 4 个改善；video MSE 从 1.2104
  降至 1.1951。joint100 因此晋级。
- 已从 joint100 继续累计 200 step，训练 manifest 向后旋转 800 行以避免重跑首批窗口；
  H3 LR 从 `2e-6` 降至 `1e-6`，累计联合阶段将覆盖约 2,400 个不同窗口。首两步 finite，
  action loss 为 13.55/14.14、flow loss 为 1.990/1.997。
- 累计 joint300 已完成并按同一 40-task sampler 复测：action MSE 为 8.7850，比 joint100
  的 8.6472 退化 1.59%，5 个跨任务 batch 仅 2 个改善，因此不晋级；video MSE 从
  1.1951 改善到 1.1744。该结果表明世界分支继续学习，但当前 loss/LR 比例开始损伤动作
  生成，joint100 仍是 M3 最佳 checkpoint。
- joint100 已恢复完整 H3 I/O 和尾 2 层并完成 LIBERO-goal 闭环 canary。task3“打开顶层
  抽屉并放入碗”、task0“打开中间抽屉”和 task7“打开炉灶”均为 0/1；机械臂最大运动
  约 26.5/16.5/48.2 cm，但目标对象关节位移均为 0。闭环 0/3 说明 40-task 离线 sampler
  的相对改善尚未进入可执行区间；当前 checkpoint 没有训练任何 ActionDiT block
  （`loaded_action_layers=[]`），下一阶段应冻结已晋级 H3 世界分支，按低 LR 分阶段解冻
  通用 ActionDiT 尾层，而不是继续降低 Flow loss。
- 已源码审计 MiniWorld（固定提交 `e484206b`）。它是动作条件的流式视频世界模型，不是
  动作生成策略，不能直接替换 ActionDiT；但其零初始化 AdaLN-LoRA 动作调制、H8→H16→
  H32 的短到长 curriculum（映射到本项目 horizon）和因果流式训推对齐可直接指导下一轮
  Action tail 稳定解冻。详细结论与独立 0.5B canary 门槛见
  `docs/MINIWORLD_H3WAM_STUDY.md`。

## M4：MiniWorld 式 ActionDiT curriculum

- 已在每个 Action block 增加零初始化、逐 attention-head 的 video residual gate；旧
  checkpoint 缺少 gate 时可严格迁移，step-zero 输出保持兼容。训练器支持 H8/H16/H32
  horizon、gate/tail 独立学习率、H3/Action/gate 独立裁剪以及冻结 H3 后只保存继承状态。
- gate-only H8 warmup 完成 100 step、覆盖 800 个跨 suite 窗口。最后两层 gate 的平均
  绝对值为 `1.84e-4/3.58e-4`，证明梯度和更新链路有效；但相同 val40/10-step sampler
  action MSE 为 `8.7768`，相对 joint100 的 `8.6472` 退化 `1.50%`，故未晋级，5.4 GB
  权重已删除，仅保留训练和评测 JSON。
- gate warmup 参数审计发现原实现还更新了 46,080 个 shared state-embedding 参数；已新增
  `--freeze-shared-state` 并通过 19 个 H3-DreamWAM 单测，之后的纯 gate 实验可做到真正
  只更新 output + gate。
- 当前改为从已接受的 joint100 重新起步：H3/Flow 完全冻结，最后 2 个 Action block 与
  gate 联合适配，H8 先跑 100 step；Action I/O/gate/tail/modulation 学习率分别为
  `1e-7/1e-6/1e-8/1e-8`。只有固定 val40 优于 joint100 才扩到 H16/H32。
- 上述 full-tail H8 100-step 实测 val40 action MSE 为 `9.2341`，相对 joint100 退化
  `6.79%`；video MSE `1.1943` 与基线基本不变。原因是直接更新 1.343 亿 Action tail
  参数破坏了 joint100 输出头已经适配的冻结特征；该 5.4 GB 权重已删除，JSON 保留。
- 已改成更接近 MiniWorld AdaLN-LoRA 的零初始化低秩路径：每层增加 rank-16 video
  residual adapter，末端 projection 为零，step-zero 输出不变；当前冻结 H3、原 Action
  block 和 gate，只训练最后两层约 26.2 万 adapter 参数与输出头，H8 100-step canary
  已启动。该实现通过本地 20 项、云端 7 项单测。
- adapter H8 训练后完整候选 val40 为 `9.4544`；恢复 joint100 I/O 的 adapter-only 为
  `9.4738`，证明该注入位置会破坏动作生成。反之，清零 adapter、仅保留继续训练的
  output head 为 `8.6110`，相对 joint100 的 `8.6472` 改善 `0.42%`。已将 adapter 输出
  严格归零并固化为 `miniworld_output_h8_s100.pt`，删除会被误加载的坏 adapter 权重；
  output-only 候选正在三个固定 LIBERO-goal 任务上做同配置闭环 canary。
- output-only 的固定闭环最终仍为 `0/3`：任务 0/3/7 的目标抽屉、碗和炉灶按钮位移均为
  0（其余物体只在 `1e-13` 数值噪声量级）。但末端执行器最大位移分别达到
  `0.319/0.632/0.470 m`，说明执行链路和动作幅度正常，失败是目标方向/接触对齐而非
  “机器人没动”。三个任务生成动作都存在相近的负 Y 偏置，且相对 joint100 的动作相关
  系数仅 `0.39~0.49`；离线 MSE 的 0.42% 改善不足以约束可执行语义。
- 训练数据方向审计进一步定位为任务条件塌缩：task0/task3/task7 的示范平均 XY 动作约为
  `(+0.05,+0.05)/(+0.10,0.00)/(-0.22,+0.20)`，差异明显；output-only 闭环策略却分别
  输出 `(-0.23,-0.41)/(-0.11,-0.51)/(-0.35,-0.32)`，三个任务共享强负 Y 偏置。下一轮
  应显式增强 language/object-conditioned action routing，并用跨任务方向对照作为训练期
  门槛，不能继续只优化全局 flow-matching MSE。
- 参数路径审计发现更直接的原因：ActionDiT 的 50 个 language cross-attention `to_out`
  在 H3 初始化时全零，而 joint100 只优化了 7,175 个 action output 参数、没有保存任何
  Action block，因此直接语言通路从未被打开。当前已启动最后两层 cross-attention output
  的零初始化 H8 warmup（约 1468 万参数，`1e-7`）；H3、Action 主体、adapter、gate 和
  已接受的 output head 均冻结，避免再次混淆归因。

## 2026-08-10 外部训练路线复核

- DreamWAM/FastWAM 的公开训练路线都不是“冻结随机/零路由 ActionDiT，只训练 7,175 个
  output 参数”：它们先把 Wan VideoDiT 插值成完整 ActionDiT backbone，再联合训练视频、
  动作及共享注意力。DreamWAM 公布配置为 30 层 ActionDiT、21,700 steps、AdamW
  `1e-4`、video/action loss 各 1.0。该差异解释了当前 output-only 离线 MSE 可下降、
  但任务语义和闭环接触没有形成的现象。
- 最新 Faster-WAM 的 DoT 结果显示，动作模块不必复制视频基模的全部深度：单层 action
  head 通过 docking interface 汇聚所有视频层 K/V，也能在 LIBERO/RoboTwin 上达到竞争
  性能，同时显著降低时延。因此若当前 cross-attention warmup 仍不能形成 language
  sensitivity，下一架构 canary 将改为“冻结 H3 + 1~2 层多深度 docking action head”，
  而不是继续解冻 50 层 H3-ActionDiT。
- Fast-WAM 的核心证据仍支持 video/action co-training，而不是推理时必须生成未来视频。
  对本项目的含义是：先让 action routing 可用，再用低学习率恢复 H3 world loss 作为表征
  正则；不能从 joint300 的 video MSE 改善推断策略也在改善。
- OpenPI 的公开实现把图像、语言放在可被 action expert 访问的 prefix，并明确从 task 字段
  生成 prompt；这进一步支持把“同一观测更换指令时动作应显著改变”设为晋级测试，而不只
  看 aggregate action MSE。
- SAM3D 路线按当前决定暂停。近期不再增加 geometry/semantic teacher；先解决已确认的
  language route、动作头结构和闭环选择指标三个主矛盾。

## M5：Faster-WAM / Dock-of-Transformer 路线

- 2026-08-10 决定停止最后两层 language cross-attention warmup，不再修补 50 层
  ActionDiT。该任务在 28/100 step 主动终止，未产生需要保留的 checkpoint，云端 8 卡
  已释放。原因是 Faster-WAM 的消融明确表明：语言应通过 language-conditioned video
  hub 进入动作头，删除 action text cross-attention 反而更好。
- 已逐公式实现论文核心：缓存 H3 每层 conditioning-frame K/V；对旋转后的 H3 key 做
  3D RoPE 逆变换；分别用 full-width K/V channel map 投到动作空间；使用每个 action
  layer、每个 attention head 独立的跨层融合信号；key normalization 后改用 action 侧
  1D RoPE；最后由无 text cross-attention 的单层 ActionDiT 生成动作。
- 新增 `docking.py` 和 `dot_model.py`。单层 H3 action head 保持 `hidden=1024`、
  `ffn=4096`，最终按论文使用 action `24x128`，由矩形 channel map 从 H3 `56x128`
  映射；相比复制 50 层 ActionDiT，动作专属深度降到 1 层。DoT action loss 可反传到
  layer-mixing、K/V channel map 和 H3 K/V projection。
- 本地 16 项相关测试通过（1 项因本地无 diffusers 跳过）；云端真实 diffusers 环境 13
  项全部通过，包括 tiny H3 的联合 forward/backward、RoPE 逆变换精确恢复、逐 head
  layer mixing 和无 action text cross-attention 结构检查。
- 第一版真实训练已完成 100 step，8 卡每步 global batch 为 8，action loss 从 `1.1769`
  降至 `0.3063`；40 个 episode-disjoint 验证窗口的 10-step sampler MSE 从首步点的
  `1.1258` 降至 `0.2821`，改善 `74.94%`。但 LIBERO-goal task0 的 100-step 闭环为
  `0/1`，目标抽屉位移为 0；该点只看过 800 个窗口，不能据此宣称策略可用。
- 闭环后再次逐式核对发现 v1 把 action attention 错误扩成了 H3 的 `56×128`。论文的
  单层 action head 实际为 `24×128`，KV-Fusion 正是用于从 hub attention width 映射到
  action width；论文的跨层矩阵也是无约束线性矩阵，不是 softmax 权重。v1 checkpoint
  已删除，只保留 JSON 与失败 rollout，防止把错误结构继续加训。
- v2 已改为 H3 `56×128` → rectangular K/V channel map → action `24×128`，action head /
  KV-Fusion 参数分别为 `28.62M/44.04M`，总可训练参数从 `148.23M` 降为 `72.71M`。
  云端真实 tiny-H3 7 项测试和 8 卡 first-step/gradient-accumulation smoke 均通过。
- `global batch=128`、AdamW `1e-4`、weight decay `0.01`、cosine schedule、H32 的 v2
  head-only 训练已完成：150 optimizer steps 共消费 19,200 个窗口，action loss 从
  `1.4042` 降至 `0.2576`。25/50/75/100/125/150 的 val40 MSE 为
  `0.2735/0.2505/0.2311/0.2155/0.2106/0.2104`，step150 最优。
- 推理实现已切换为 Faster-WAM 的 representation-only 路径：每个在线观测只运行一次 H3
  并缓存 50 层融合 K/V，10 次动作去噪只重复单层 action head；同时移除被 observation
  mask 隔离的 future-video/audio rows。在线 model inference 从 `4.91s` 降至 `0.489s`，
  约 10 倍加速，100-step rollout 总时长从 110 秒降至 66.6 秒。
- v2 step150 的 task0 闭环仍为 `0/1`，但不再是无接触：机械臂稳定越过柜子，将 wine
  bottle 推动 `0.657`，目标 middle drawer 位移仍为 0。视频确认是强 wrong-object
  interaction，不是动作幅度、仿真 transport 或随机抖动问题。
- 同一 task0 观测仅把语言替换为“turn on the stove”时，整段动作余弦相似度仍为
  `0.9939`、RMS 差仅 `0.0450`，证明冻结 H3 的 head-only 策略主要走视觉捷径，语言
  条件对动作影响过弱。该 counterfactual 成为联合 H3 阶段的硬晋级指标。
- 50 层 H3 全解冻的真实探针已通过，32.35B 参数首步峰值为 `37.24/46.31 GiB`
  allocated/reserved，显存不是问题。机械照搬 5B Wan 的统一 `1e-4` 会使 action loss
  在 step2 从 `0.256` 跳到 `1.781`；改为 action `1e-5`、H3 `1e-6` 后 5 步稳定，action
  loss 最终为 `0.207`。
- 已实现 same-world-size 的 8-rank H3 分片 stage，最后 1 层 smoke 共 1.4GB，保存和重载
  后真实采样通过。正式 M1 使用 50 层、global batch 128、10 optimizer steps，最终仅
  保存一次约 64GB H3 stage 和 145MB action stage；训练后自动跑 val40、counterfactual
  和 task0 闭环。
- M1 已于 2026-08-10 完整跑完：10 optimizer steps 共消费 1,280 个窗口，32.35B
  trainable parameters，action/H3 LR 分别为 `1e-5/1e-6`。平均 action loss 为
  `0.26349`，除一个被 `clip_grad_norm_(1.0)` 截断的 pre-clip 尖峰外无发散；8 个 H3
  rank 分片各约 8.07GB，action stage 145MB，真实重载评测通过。
- 固定 val40 MSE 从 head-only 的 `0.210443` 小幅改善到 `0.209437`（0.48%），但语言
  反事实余弦仅从 `0.993899` 降到 `0.993680`，没有解决语言捷径。task0 闭环仍为 `0/1`：
  末端路径为 0.341m、动作幅度正常，但视觉确认机械臂从 wine bottle 偏到 bowl，所有
  物体和 drawer 关节位移均近零。结论是普通 RGB+action 联训只改变了视觉偏好，没有建立
  task text 到目标物体的可执行绑定。
- 已新增跨任务错误指令 margin-ranking：同一观测、同一 noisy action 下，错误指令的动作
  flow MSE 必须至少比正确指令高指定 margin。正确分支先 backward 释放 H3 激活，再运行
  错误指令分支，避免双图同时驻留；支持 ranking weight/margin/frequency 和 video loss
  weight。云端真实 H3 smoke 得到 correct/wrong MSE `0.26770/0.27353`、ranking loss
  `0.04417`、梯度 `1.20`，证明变长负文本、FSDP backward 和指标链均有效。
- M2 从同一 head-only step150 重新出发做公平 A/B：full50、global batch128、5 steps，
  Action/H3 LR `1e-5/1e-6`，video weight `0.25`，language-ranking weight `0.5`、margin
  `0.05`、每个 microbatch 生效。2026-08-10 最后可见进度为 step4/5；四步
  correct MSE 为 `0.25340/0.25903/0.25599/0.26919`，ranking loss 为
  `0.04712/0.04654/0.04823/0.04683`，梯度稳定。随后云实例 SSH 端口进入拒绝连接状态，
  step5、分片保存和自动评测需在实例恢复后核验，不能提前记为完成。

## 下一执行门槛

1. 云实例恢复后先核验 M2 step5、训练报告和 8 个 H3 rank 分片是否完整；任何缺失都按未
   完成处理，不在残缺 stage 上评测或续训。
2. M2 必须同时满足：val40 不显著劣于 `0.210443`；固定 task0 正确/错误语言动作余弦相对
   M1 的 `0.993680` 有明确下降；task0 至少朝 cabinet/drawer 产生目标接触。仅 ranking
   训练 loss 下降不算晋级证据。
3. task0 首先要求 middle drawer 出现有效位移；若仍只碰 wine bottle/bowl，不通过“会交互”
   代替任务成功。只有目标接触后才扩到 task3/task7。
4. 三个固定 rollout 首先要求至少一个任务出现目标物体或关节的非零有效位移；
   wrong-object 位移单独报告，不计成功。对象教师路线暂缓，不用它掩盖语言路由问题。
5. M4-A 出现闭环正例后再启动 MiniWorld-0.5B causal-world canary；它必须通过真实动作与
   零/打乱/反向动作的 counterfactual action-following 对照，才允许进入融合实验。

## M6：论文配方 H3-DoT 全量联合训练

- 2026-08-11 重新对齐 Fast-WAM、DreamWAM 与 Faster-WAM 后，停止 history delta、
  history adapter 和语言 ranking 修补路线。正式目标固定为 H3 全 50 层、DoT 全层 K/V
  fusion、单层 ActionDiT，以及权重各 1.0 的未来 RGB flow/action flow 联合训练；不启用
  ranking、phase、history 或 regression 辅助目标。
- full50 canary 使用四套 LIBERO 的 7,710 个 episode-disjoint 训练窗口、global batch 128，
  从 head-only step125 初始化动作侧，H3/action LR 为 `1e-6/1e-5`。10 steps 消费 1,280
  个窗口，32.35B 参数参与更新，平均 action loss `0.26314`，峰值显存为
  `38.24/46.79 GiB allocated/reserved`，无 NaN、OOM 或失控发散。
- canary 联合分片位于
  `/mnt/h3-wam/outputs/h3dotwam/m3_paper_joint_full50_gb128_s10_joint`，8 个 H3 rank
  shard 加 action stage 共约 61GB，文件与 manifest 完整。固定 val40 MSE 为 `0.218792`，
  相对同链路 head-only `0.217331` 退化 `0.67%`；语言反事实 cosine 为 `0.992009`，
  相对 head-only `0.991760` 未改善。该点仅通过稳定性门槛，不作为能力 checkpoint。
- 正式 M6 已从相同 head-only 基线重新启动，避免继承短 canary：602 steps，global batch
  128，约 10 epochs，cosine LR，四套数据统一训练。首步 loss/action/video 分别为
  `0.695792/0.275908/0.428317`，8 卡利用率 100%，训练稳定。
- 为避免只看 final checkpoint，训练器已支持 rank-sharded periodic joint checkpoint；正式
  run 每 60 steps（约一 epoch）保存完整 H3+action 点，保留 60..600 的十个中间点以及
  final 602，预计总占用约 0.7TB。正式输出前缀为
  `/mnt/h3-wam/outputs/h3dotwam/m4_paper_joint_full40_10ep`。
- 评测原则：先对 step60/180/300/420/540/602 跑固定 val40 和语言反事实；闭环选择以
  LIBERO 四套统一协议为准，离线 MSE 或语言 cosine 只做诊断，不能代替成功率。

## M7：三集群并行与 DreamWAM motion 对照

- 新增 30907、32409 两台 8×A800 80GB，三台通过同一 `/mnt/h3-wam` 共享模型、数据和
  checkpoint。基础 Conda 运行时已从 32611 固化到 `runtime/conda-py311`，附加包只读使用
  `.venv/lib/python3.11/site-packages`；新容器不再依赖失效的 `.venv/bin/python` 链接。
- 30907 已完成 8 卡 collective、Diffusers-H3、LIBERO/MuJoCo/OSMesa 和真实 5-step
  rollout 烟测。M3 checkpoint 的 val40 精确复现为 `0.2187924549`；闭环服务加载
  34.05 秒，模型 inference/策略往返均值为 `0.958/1.754` 秒。该线已等待 M6 step60，
  checkpoint 完整后自动跑 val40 和 Goal task 0/3/7/8 trial0。
- M3 同协议四任务闭环对照已跑完，Goal task 0/3/7/8 trial0 为 `0/4`，共用时
  459.77 秒。task0 错误推动 wine bottle `0.283`，task3/7 主要推动 bowl/plate，task8
  未完成 bowl-on-plate；这组结果固定为 step60 的直接闭环基线。
- M3 完整 850-window validation 也已完成（8-rank × 107 batch，末尾 6 个样本为整除重复），
  10-step sampler mean action MSE 为 `0.258914`；head-only step125 同协议为 `0.256766`，
  因而 M3 退化 `0.84%`。这与 val40 的 `+0.67%` 方向一致，说明短联合训练未改善动作采样；
  该指标仍不替代闭环成功率。
- 32409 已用官方 RAFT `raft-things.pth` 和 H3 VAE 完成 8-window motion smoke，全部
  artifact 与 RGB latent shape 对齐，稳态约 `2.05 s/window`。全量 8,560-window cache
  已启动，8 卡各负责约 1,070 windows；cache 完成后自动启动 full50 1-step 和
  global-batch-128 10-step canary。
- 当前 DoT 训练器不再把 motion 当 feature residual，而是严格采用 DreamWAM 核心：
  per-sample 归一化后的 motion latent 独立加噪、H3 RGB+motion 双流输入、RGB/motion
  双 velocity 输出、motion loss 权重 `0.5`，并可启用 shift=12 的 world timestep weight。
  扩展的 H3 `proj_in/proj_out` 会参与 FSDP 同步和 joint checkpoint；若不训练这两个零初始化
  flow 投影，motion 分支无法学习，因此训练器对错误配置直接报错。
- 相关 motion/DoT 测试在共享云端环境为 `11 passed`。完整操作口径、自动编排脚本和存储
  约束见 `docs/H3_WAM_PARALLEL_DEVELOPMENT.md`。
- M6 step60 已完整保存并在独立 30907 重载成功。固定 val40 MSE 为 `0.216276`，相对
  head-only `0.217331` 改善 `0.49%`，相对 M3 `0.218792` 改善 `1.15%`；Goal
  task0/3/7/8 trial0 闭环为 `0/4`。task0 错误推动 wine bottle `0.308`，其余任务也未
  完成 predicate；step60 不作为能力点。32611 不等待评测，继续 step61+；30907 正补跑
  val850 和 trials1–4 共 16 episodes。
- step60 完整 val850 mean action MSE 为 `0.255086`，相对 head-only `0.256766` 改善
  `0.65%`，相对 M3 `0.258914` 改善 `1.48%`。这确认离线改善不是 val40 抽样偶然，但
  trial0 仍是 `0/4`。补跑 trials1–4 后再得 `0/16`，合并为四任务 `0/20`，因此结论升级
  为“数值改善、闭环无收益”，step60 明确不晋级。30907 已转去补齐 motion 10-step 的
  val850，并用自动 ladder 等待 RGB step120..602。
- Motion cache 已完成 `8,560/8,560`，各 rank 稳态约 `1.77 s/window`；32409 已自动进入
  full50、H3 I/O 可训练、DreamWAM world weighting 的真实 1-step backward gate。该 gate
  已通过：loss/action/RGB/flow 为 `1.3765/0.3475/0.4029/1.9959`，gradient norm
  `2.563`，峰值显存 `32.26/42.48 GiB allocated/reserved`，无 NaN/OOM；10-step、
  global-batch-128 canary 已完成。该 canary 覆盖 1,280 samples，mean action/flow loss
  为 `0.272353/1.999704`，峰值显存 `38.28/46.76 GiB`，全程 finite；约 61GB 的 8-rank
  joint stage 已原子保存，包含 `train_h3_io=true`。独立 8 卡重载成功，val40 MSE
  `0.222776`，相对 head-only `0.217331` 退化 `2.51%`，但仍通过 10% 安全门槛；Goal
  task0/3/7/8 trial0 闭环为 `0/4`。随后已从同一 head-only step125 基线启动独立 motion
  60-step，避免把 cosine 已降到零的 10-step canary 当作续训点；该 60-step 完成后自动跑
  val40 和相同四任务闭环，与 RGB step60 做同步长度公平比较。
- Motion 10-step 的完整 val850 MSE 为 `0.259202`，相对 head-only `0.256766` 退化
  `0.95%`，相对 RGB step60 `0.255086` 退化 `1.61%`，与 val40 方向一致。30907 已利用
  step120 前空窗补跑 RGB step60 的 Spatial/Object/10 三套 12-episode canary，同时保留
  自动 ladder 等待后续 RGB checkpoints。

## M8：motion 新通道配方纠偏

- 原 motion 60-step 已完成，mean action/flow loss 为 `0.264387/1.999141`，val40
  `0.218063`，Goal task0/3/7/8 为 `0/4`。flow loss 从首步到末步都约 `2.0`，说明新增
  motion 输出基本停留在零预测附近，继续按旧配方堆步数没有价值。
- 与 `third_party/DreamWAM/configs/dreamwam_joint.yaml` 和初始化代码逐项对照后找到根因：
  DreamWAM 用 RGB 权重标准差 `0.1×` 随机初始化 flow 输入/输出通道、统一 LR `1e-4`、
  21,700 steps；旧 H3 适配为零初始化，并把新增 I/O 与 H3 blocks 一起使用 `1e-6`。
  motion-60 checkpoint 的 flow `proj_in/proj_out` 权重范数仅 `0.00443/0.02005`，不足以
  形成非零 flow 预测。
- 训练器现将 H3 blocks、H3 flow I/O 和 action/KV 分为独立 optimizer groups，新增
  `--h3-io-learning-rate` 与 `--flow-channel-init-scale`；motion 默认 `1e-4/0.1`，而
  RGB-only 默认仍为 `h3-learning-rate/0.0`。回归测试 `11 passed`。
- 32409 已启动 paper-I/O 20-step 门禁：H3 blocks/action/I-O LR 分别为
  `1e-6/1e-5/1e-4`，flow 初始化 `0.1×`，constant LR，global batch128。其目的先验证
  flow loss 能否明确低于 2.0。前四步已得到
  `1.99997/1.99648/1.99169/1.98952`，旧配方同期无下降，说明新 I/O 已开始学习。
  60-step 独立 epoch 编排器已挂起：20-step 的末四步相对首四步至少改善 `0.005` 且
  val40/闭环完整后，自动从原 head-only 基线重训并评测。
- RGB 主线已到 step271，离线当前最好为 step180：val40/val850
  `0.215161/0.253201`，但 Goal canary 仍为 `0/4`。step60 在四个 suite 的固定 canary
  合计 32 episodes 全失败，30907 正补跑 step180/240 的跨套件 canary。
- paper-I/O 20-step 已完成：flow 首四步/末四步均值为 `1.994413/1.937348`，末步
  `1.931418`；val40 `0.220007`，Goal 四任务 `0/4`。门禁确认 motion 已开始学习，但尚无
  闭环收益。独立 60-step 已自动从原 head-only 基线启动，当前 step24 的 flow loss
  `1.918774`，趋势继续向下。
- RGB 主线已到 step324，step300 的 val40/val850 为 `0.215509/0.252340`，完整验证集
  优于 step180，但 Goal 仍 `0/4`；step180/240 的 Spatial/Object/10 共 24 episodes 也为
  `0/24`。30907 已启动 step300 跨套件 canary。
- 为核对历史成功协议，step300 随后会在 Goal task3、seed42、trial0→9 连续 seeded-env
  下跑诊断；motion-60 完成后也会跑同协议。该结果只判断 seed/protocol sensitivity，不能
  与历史 task-specific action head 的 4/10 或 FastWAM 7/10直接当作同模型对比。
- paper-I/O motion step60 已完成：flow 从 `1.999966` 降到 `1.799538`，确认新 motion
  通道学会去噪；但 val40/val850 为 `0.217259/0.255773`，后者比 RGB step300 的
  `0.252340` 退化约 `1.36%`，Goal canary 仍为 `0/4`。因此不直接扩成长程 motion。
- 第四台 8×A800 节点 `30234` 已接入共享盘。新增 frozen-world action-retune 因果实验：
  固定 motion step60 的全部 H3 权重，只以 `1e-4` 重训单层 ActionDiT、KV fusion 和状态
  embedding 60 steps，并保留 20/40/60 action-only checkpoints。训练器允许显式 action
  stage 与 joint H3 stage 解耦重载，相关云端 unittest 为 `13 passed, 1 skipped`。

## M9：仓库/论文深度复核与 dense-data 纠偏

- 逐代码对照 FastWAM、DreamWAM、ImageWAM、MiniWorld、MotuBrain、BadWAM，并逐公式核对
  Faster-WAM 后，确认当前 DoT 架构主干基本正确：首帧全层 K/V、channel/layer fusion、
  逆 3D RoPE、action 1D RoPE、无 action text cross-attention，以及只用首帧的在线推理均
  已实现。训练 observation mask 也阻止首帧读取未来 target，不存在明显 future leakage。
- 找到比 motion 更上游的数据差异：`prepare_libero_full_candidate.py` 固定每 episode 只取 5
  个窗口，导致 1,712 episodes 仅有 8,560 windows、train 7,710。FastWAM 官方 loader
  按每帧构造滑窗；即使暂不使用带 padding 的尾部样本，四套数据仍有 222,929 个完整
  horizon-32 windows，episode-disjoint train/val 为 200,779/22,150。原 602-step run 只是
  对稀疏数据循环，不等于论文 dense-data 10 epochs。
- Faster-WAM 论文配方是全模型 `1e-4`、AdamW、cosine、logit-normal timestep；当前联合
  主线为了 32B H3 稳定性使用 action/H3 `1e-5/1e-6` 和 shifted-uniform。它是保守稳定性
  实验，不是严格 paper reproduction。冻结 H3 后用 action `1e-4` 重训 60 steps，val40
  已降至 `0.202317`，明显优于 motion base `0.217259` 和 RGB step360 `0.214504`；闭环
  正在 32409 补跑。
- 已新增 dense candidate/cache 入口
  `scripts/h3dreamwam/prepare_h3dotwam_dense_libero.sh`，D 节点 `30234` 正生成 222,929
  windows。旧 8,560 个缓存通过硬链接复用；H3 VAE 预计算新增 batch-4 等价编码，逐样本
  posterior seed 与原 batch-1 路径逐元素一致。预计新增 cache 约 100GB，远低于共享盘余量。
- 新主线顺序固定为：dense head-only `1e-4` 一 epoch（约 1,569 optimizer steps）并做闭环
  ladder；出现至少一次标准 success 后，从最佳动作点以 H3 `1e-6` 联合 RGB/action；再按
  `3e-6/1e-5` 做短 A/B。motion、MiniWorld memory、H-Bridge 和 BadWAM drift 分别作为
  后续监督、上下文、结构和诊断线路，不与 dense 基线同时引入。
- action `1e-4` 60-step 的最终闭环已补齐：Goal task0/3/7/8 trial0 为 `0/4`。因此该点只
  保留为“离线动作优化改善”的证据，不晋级。RGB step360 的 Spatial/Object/LIBERO-10
  canary 也均为 `0/4`，继续确认 sparse-data 联合训练尚无跨 suite 闭环收益。
- 完整对照与门槛见 `docs/H3_WAM_REPO_PAPER_REVIEW.md`。
