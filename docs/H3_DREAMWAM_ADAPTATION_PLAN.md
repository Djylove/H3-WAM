# H3-DreamWAM 代码审计与整改方案

更新时间：2026-08-08

执行状态：M0 已完成，M1 已进入 ActionDiT 尾层逐级解冻；真实 8 卡数值、失败归因和
可恢复 checkpoint 见 `docs/H3_DREAMWAM_M0_M1_STATUS.md`。

## 决策

主线采用 DreamWAM 的训练结构，并将 Wan2.2 Video Expert 替换为 MiniMax H3。
这里的“替换”是保持 DreamWAM 的世界目标、动作专家、逐层信息路由和联合采样语义，
按 H3 的 VAE、时序网格和 Transformer 接口重新实现；不是把配置中的模型名机械替换。

## 2026-08-08 官方代码逐项复核后的关键修正

这次复核不再把“能联合 forward”视为 DreamWAM 对齐完成。官方实现与 M1 之间有四项会
直接影响闭环的差异，按优先级排序如下：

1. **旧训练 context 泄漏了示范图像。** `data/v1/cache` 每个 prompt 约含 100 个 image
   token；在线 server 又固定复用某个示范窗口的 context，同时输入当前相机首帧。官方
   DreamWAM 的 context 只有文本和 proprio，首帧只走 Video Expert。M2 已改为读取
   manifest 的共享 `context_id`，并可用 `--require-text-only-context` 硬拒绝非文本 token。
2. **Wan 的 ActionDiT norm 不能直接定义成 H3 的“正确实现”。** FastWAM/DreamWAM 在
   reshape heads 前对完整 attention width 归一化，宽度是 `56*128=7168`；旧 M1 使用
   H3 风格的 per-head 128 维归一化。两者均已实现并写入 checkpoint 元数据。真实 M2
   对照中 H3 风格 head-only held-out MSE 为 12.079，优于 Wan exact 版本的 14.001，
   所以当前默认保留 H3 风格，Wan exact 只作为消融。
3. **H3→ActionDiT 插值与 Wan 不可机械等同。** 官方 FastWAM 对最后一维缩放使用
   `sqrt(source_width/target_width)`，同时复制 Wan 的 time projection、block modulation
   和 cross-attention。H3 没有这些一一对应模块；M2 虽实现同一 alpha 规则，但真实尾层
   探针梯度达到 15.42M 并被否决。alpha 保留为实验开关，不作为 H3 默认初始化。
4. **M1 还不是论文意义上的 DreamWAM。** 当前 Flow 通道为零且无 Flow loss，也没有
   DINO/Depth 门控残差监督；因此它只能叫 RGB+Action MoT baseline。DreamWAM 的核心
   增益来自 dense all-start 数据和 RGB/Flow/DINO/Depth 多世界目标，不能用百步级尾层
   LoRA 替代。
5. **proprio 的共享范围不同。** 官方先把 proprio 投成 text-width token，再同时交给
   VideoDiT 和 ActionDiT；当前 H3 版本只把 state token 交给 ActionDiT。Motion 基线完成
   后增加 shared-proprio 消融，不能与 Flow 同时改动，否则无法归因。

因此接下来的顺序改为：先用 text-only dense 数据验证修正后的 ActionDiT；然后接 Flow，
再接 DINO/Depth。旧 `timecond50` 只保留为 M1 数值对照，不再作为扩大训练的起点。

推荐结构：

```text
首帧 + 未来 RGB 噪声 + 文本
              │
              ▼
       H3 Video Expert（50 层）
       ├─ RGB velocity
       ├─ Flow velocity（完整扩散分支）
       └─ DINO / Depth（门控残差分支）
              │ 每层同深度 K/V
              ▼
       独立 ActionDiT（flow matching）
              │
              ▼
          32-step action chunk
```

H3 原生 audio latent 动作适配不作为主线。已有实验表明该路径难以承担 LIBERO
精细接触控制；DreamWAM 本身也使用独立 ActionDiT，而不是把动作塞入 Wan 的其他模态槽。

## 上游代码固定与审计范围

- 官方仓库：`third_party/DreamWAM`
- 固定提交：`6e989facc0c452fd3488d75f60bc36411005558c`
- 本地体积：约 54 MB，只包含代码、网页演示和两个低秩投影；未下载模型或数据。
- 已执行 Python compile 检查，代码可通过语法编译。
- 上游当前没有 `LICENSE` 文件。研究阶段保留为只读参考目录；我们的实现放在
  `src/fastwam`，不直接复制发布上游文件，后续公开前再确认授权。

本次逐段核对的实现包括：

- `dreamwam/model.py`：联合损失、世界路由、训练加噪和联合采样；
- `dreamwam/mot.py`：VideoDiT/ActionDiT 的逐层共享注意力和首帧 KV cache；
- `dreamwam/experts.py`：两个 expert 的输入、时间调制、RoPE 和输出头；
- `dreamwam/initialization.py`：RGB/Flow 通道扩展和 ActionDiT 预训练初始化；
- `dreamwam/preprocessing/*`：RGB、RAFT、DINOv2、DA3、PCA 投影和时序对齐；
- `dreamwam/train.py`、`scripts/train.py`：BF16 全参数训练；
- `dreamwam/policy.py`、`evaluation/rollout.py`：联合采样和 LIBERO 闭环协议。

## DreamWAM 真正有效的机制

1. RGB 和光流都进入 VideoDiT 的扩散输入/输出。光流不是只挂一个辅助预测头；它与
   RGB 在 channel 维拼接，未来时刻被加噪并预测 velocity。
2. DINO 语义和 DA3 深度不直接做扩散。它们经 rank-8 投影后作为后段层的门控残差监督，
   新分支以零输出、低 gate 初始化，避免一开始破坏预训练视频能力。
3. Video Expert 不读取动作；Action Expert 在每一层读取同深度 Video K/V。
   这是单向耦合。推理时可以先运行 H3 并保留逐层 K/V；训练时为保留 activation
   checkpoint 的显存收益，采用成对 layer checkpoint，在同一重算函数中更新 H3 和
   Action token，同时用 mask 保证动作不反向污染视频 token。
4. joint 推理从首帧、未来 RGB 噪声和动作噪声出发，约 10 次同步 ODE 更新；首帧始终
   固定。Flow/DINO/Depth 分支训练时存在，部署时不需要教师模型，Flow 输出也可丢弃。
5. 官方配置使用 32-step 动作块、执行 10 步后重规划。这和本地已经验证的 H8/H32
   时间一致性结论相符，不能退回只优化首动作的方案。

论文的同协议 LIBERO 消融给出了实施优先级：RGB-only 为 98.00，motion-only 用完整
denoise 为 98.50，而 motion 走 residual 只有 97.85；geometry/semantics 单独走 residual
分别为 98.15/98.10，但把它们也做完整 denoise 都降到 96.80。最终
Motion-denoise + Geometry-residual + Semantics-residual 为 98.90。也就是说，先补真实
Flow 不是工程偏好，而是论文中单项贡献最大且路由结论最清楚的变量。

## 当前 H3WAM 与目标结构的差距

| 模块 | 当前实现 | DreamWAM 化后 |
| --- | --- | --- |
| 视频骨干 | official BF16 H3，支持尾部/FSDP 更新 | 保留 |
| 动作专家 | 50 层 ActionDiT 已接入；H3-style norm 的真实结果优于 Wan exact norm | 保留两种可复现实验结构，默认 H3-style，逐层读取 H3 K/V |
| 动作所见视频 | 已读取全部 noisy video rows，但 context 混入固定示范图像 | context 仅 text+proprio；图像只走 H3 video rows |
| 世界目标 | RGB；Flow 输入为零且无 Flow/DINO/Depth loss | RGB + RAFT Flow 完整扩散，DINO + Depth 门控残差 |
| 推理 | 未来 RGB/动作同步 10-step 已跑通 | 保留，并以严格 text-only context 闭环 |
| 主干更新 | H3 冻结、ActionDiT 尾 4 层 | dense 数据先训完整 ActionDiT，再做 H3 解冻 capacity probe |
| 选点 | validation loss + 少量 rollout | 固定顺序闭环成功率为主，世界 loss 只做健康指标 |

已有 `train_h3_wam_joint_fsdp.py`、FSDP checkpoint、Diffusers H3 loader、LIBERO
rollout 和 attention leakage mask 都可复用。当前脚本不是废代码，它是整改的训练外壳。

## H3 适配细节

### 1. 时序与输入合约

- 当前 LeRobot LIBERO 数据声明为 20 Hz，动作 horizon 32 覆盖 1.6 秒。
- 现有 H3 对齐把 33 个机器人观测重采样到约 38.4 帧后，取合法的
  `17n+5=39` 个 H3 pixel frames，对应 12 个 H3 latent frames；云端训练 cache 的
  `[1,24,12,14,28]` 与此一致。训练前仍逐 shard 检查 `source_fps/h3_frame_count`，
  禁止把其他 cache 版本混入。
- 双相机继续横向拼接为 224x448；RGB latent 为 `[B,24,T,14,28]`。
- 首帧用 H3 VAE 单独编码为一个 condition latent，训练和推理都保持 clean。
- episode 尾部同时维护 `action_is_pad` 和映射到 H3 latent time 的 `image_is_pad`，所有
  future loss 排除 padding。

### 2. RGB + Flow 完整扩散

- 用 RAFT 计算相邻重采样帧光流，沿用 DreamWAM 的 flow color-wheel 表示；为保持帧数先
  复制第一段 transition，联合训练时再把 condition flow latent 置零。
- 用同一 H3 VAE 编码 Flow，得到 24-channel latent。
- 扩展 H3 video input projection：24 RGB -> 48 RGB+Flow；RGB 权重原样复制，Flow
  权重按 RGB 标准差乘小系数初始化。
- 扩展 video output projection为 RGB+Flow velocity；RGB 输出原样复制，Flow 输出小
  初始化/零 bias。
- 训练时 RGB、Flow 共享 video timestep，首个 RGB latent clean、首个 Flow latent 为零。
- 推理时 Flow 输入保持零，Flow 输出丢弃；它是训练世界动态表征，不增加部署教师模型。

云端 Diffusers H3 的真实投影已经核验：patch size 为 `(1,2,2)`，`proj_in` 是
`Linear(96,5376)`，`proj_out` 是 `Linear(5376,96)`。RGB+Flow 扩展后的目标分别是
`Linear(192,5376)` 和 `Linear(5376,192)`；实现仍需输出 shape report 和 step-0 数值
报告，禁止只凭 module name 静默替换。

### 3. 独立 ActionDiT 与共享注意力

- ActionDiT hidden 先用 1024，FFN 4096，层数与 H3 的 50 层对齐。
- 云端真实 H3 config 已核验为 hidden 5376、56 heads、head dim 128，Q/K/V width 7168；
  Action attention 采用同样的 56x128 输出布局，才能拼接同层 K/V。
- action token 输入包含 noisy action、action timestep、位置编码；proprio 作为额外 context
  token，不再用人工 rollout step 作为必要条件。
- H3 官方实现是 AdaLN + packed full self-attention + SwiGLU，没有 cross-attention。
  joint wrapper 复用其 `to_q/to_k/to_v`、RMSNorm、3D RoPE 和 AdaLN：H3 query 只读取
  原 H3 packed rows；Action query 读取选出的 video K/V 加 action 自身 K/V。训练时每个
  H3/Action layer 成对 checkpoint；推理时可缓存每层 video K/V。Action 不直接读取
  H3 的 audio rows，避免再次把机器人控制绑定到音频槽。
- ActionDiT 参考 DreamWAM 的初始化策略：从 H3 block 中可匹配的 attention/FFN 参数做
  维度插值；action input/output、proprio 和新增路由保持新初始化。必须输出逐 tensor
  initialization report，不能静默 missing keys。
- 保留 `joint` 和 `uncond` 两个模式。第一轮只训练 joint；uncond 用于验证收益是否真的
  来自未来世界 rollout。

### 4. DINO + Depth 门控世界路由

- DINOv2 patch token 和 DA3 depth 分别按 H3 latent `T,H,W` 对齐，再用训练集固定 PCA
  投影到 rank 8。
- 路由采用 DreamWAM release 的 parallel fusion 和 local preview 监督。
- Wan 30 层的后段注入范围按比例映射到 H3 50 层：首轮候选为 DINO 43～49、Depth
  35～49；最终由 shape/显存 smoke 后固定配置。
- gate bias 初始化为 -4，残差分支末层为零，保证 step 0 与原 H3 近似等价。
- DINO/Depth loss 从 0.5 余弦衰减到 0.025/0.05；gate L1 为 `1e-4`。

SAM3D 对象中心对齐放在这条主线跑通之后。它可以作为第五个训练期世界分支，但第一轮
同时加入会无法判断收益来自 DreamWAM 还是对象教师。

## 数据与存储方案

本机 `data/h3wam_cache` 已占约 226 GB，主要是重复的逐层 H3 feature cache；新方案训练
时直接运行 H3，不再生成几十 GB 的 feature cache。

第一轮先处理正确映射后的 LIBERO Goal task3 稀疏 165 windows，通过数值和闭环 gate 后
才处理 5646 dense windows：

- 复用已有 RGB H3 latent、action、state、context；
- 新增 Flow/DINO/Depth cache；
- smoke 阶段 Flow 单独存放，便于失败后整目录回收；dense 阶段改用 BF16/FP16 shard，
  避免每窗口一个小文件和重复 context；
- 先做 1-window shape probe 和 165-window 稀疏集，再外推全量；新增 cache 软上限 16 GB，
  超过先停止并核对 dtype/sharding，禁止默认生成 5646 份 FP32 小文件；
- index 带源 episode、start、任务语言、tensor shape/dtype/hash，可断点续算；
- 不下载 DreamWAM 的 Wan、RAFT、DINO、DA3 权重到项目根目录，统一放隔离模型目录。

云端模型与输出继续使用 `/home/h3wam_finetune`，不写 `/root`。云端当前容量不适合保存
完整 Adam 状态：实验阶段保存 model-only rank shard，滚动只保留最近和最佳各一个；
写新 checkpoint 前先清理旧 rolling 点，并把晋级点转移到本地大盘。

实际 probe 已把存储估算校准下来：单个 FP32 flow latent 约 454 KB，task3 稀疏 165
windows 共 72 MB；按相同格式外推 5646 dense windows 约 2.5 GB。先前 8～16 GB 上限仍
作为异常保护，不再是预期占用。

M0 meta-model 探针统计官方 Diffusers H3 Transformer 为 33,122,992,896 参数，BF16
权重约 61.7 GiB；本地 INT8/Comfy 的约 20B 统计不能用于估算 BF16 全量优化器显存。
按 H3 的 56x128 attention 合约实例化 50 层、hidden 1024 的 DreamWAM ActionDiT，
meta 参数量为 3,373,714,439；两者合计约 36.50B，尚未计小型世界路由分支。

## 实施顺序与晋级门槛

### M0：结构探针

1. 固定云端 Diffusers/H3 版本和 transformer config。
2. 导出 H3 input/output projection、head 数、head dim、50 层 block 子模块和 row layout。
3. 用 1 个真实窗口验证 RGB+Flow projection 扩展后，Flow=0 时 RGB 输出与原模型接近。
4. 验证 observation/action mask：当前帧特征对未来 clean target 的梯度必须为零。

门槛：shape、finite forward/backward、step-0 等价和 leakage 单测全部通过。

### M1：DreamWAM 最小闭环（RGB + Action）

1. 实现 50 层 ActionDiT 和逐层 H3 K/V 读取。
2. 实现 joint 10-step RGB/action sampler。
3. 用 100 个窗口、最后 2 个 H3 block 做 20-step FSDP smoke。
4. 用 task3 完整训练集跑短 checkpoint ladder，并固定 trial0/4/5 闭环。

门槛：能完成至少一次 task3；ActionDiT 打乱/置零 H3 K/V 后必须退化。若这一步不成立，
不加入更多教师分支。

### M2：加入 Motion

1. 接入 RAFT Flow latent 和 H3 projection 扩展。
2. 对照相同数据、采样步数、训练步数和 checkpoint 选择协议。
3. 评测固定 seed42 trial0～9，并记录非目标物体位移。

门槛：相对 M1 的闭环成功率净提高，或在成功率持平时明显减少碰撞；仅 Flow loss 下降
不算晋级。

### M3：加入 Geometry + Semantics

1. 接入 DINO/Depth rank-8 target、parallel gated residual 和 local preview。
2. 做三组严格消融：RGB+Action、+Flow、+Flow+DINO+Depth。
3. task3 通过后回归已有 task0 和第三任务，防止世界分支破坏基础控制。

门槛：task3 至少达到官方 FastWAM 同协议 7/10，且两个回归任务不能显著退化。随后再
扩大到多个 LIBERO suite；单任务 recovery gate 的 9/10 只作为上界，不作为泛化基线。

### M4：对象中心与上下文

- DreamWAM 主线稳定后再加入 SAM3D/object mask alignment，针对当前“碰酒瓶、抓错对象”
  失败；
- 再加入 2～4 帧观测历史、previous action 和 RobotTTT 风格在线上下文适配；
- 每次只增加一个变量，并复用 M1～M3 的闭环消融协议。

## 训练配置起点

- 设备：8 x 80 GB 起步，BF16，FSDP per H3 block + activation checkpointing；
- 全局 batch：先 8，显存允许再用 accumulation 到 16；
- 论文正文报告 BF16、batch 16、LR `1e-5`，但 release YAML 当前写 `1e-4`，两者不一致。
  H3 不能照抄任一数值：实测 Action `1e-5` 在第 2 步把 loss 推到 414；当前稳定起点为
  H3 I/O/tail `1e-5`、Action I/O `1e-7`、新 Action tail `1e-8`，并对 H3/Action 独立
  clip。所有扩大以 held-out sampler/闭环决定；
- AdamW betas `(0.9, 0.95)`，weight decay `0.01`，warmup 5%，cosine decay；
- loss：RGB 1.0、Action 1.0、Flow 0.5、DINO 0.5->0.025、Depth 0.5->0.05、Gate `1e-4`；
- checkpoint：短实验每 250/500 step，最多两个；晋级以 rollout 而不是最低 val MSE 决定；
- 主干解冻顺序：最后 2 层 smoke -> 最后 10/20 层 -> 全 50 层。33.12B H3 使用
  FP32 master 和 Adam moments 时，8x80 的全 50 层显存会很紧；扩大前必须做 capacity
  probe，并在 sharded 8-bit optimizer、CPU optimizer offload 或增加 GPU 中选择。
  只有前一档闭环不退化才扩大，避免重演“video loss 变好、动作退化”。

## 不采用的捷径

- 不继续把 action-only LoRA 当主方案；LoRA 只保留为同目标下的参数效率消融。
- 不用 H3 audio latent 替代独立 ActionDiT。
- 不直接复用旧首帧 feature cache 训练 DreamWAM；它缺少未来世界监督和逐层 joint K/V。
- 不以单一离线 MSE、单个 trial 或人工固定阶段切换宣称泛化。
- 不立刻跑完整 benchmark；先用 task3 和两个回归任务证明结构增益，再扩大数据。

## 第一批代码改动

在不改动 `third_party/DreamWAM` 的前提下新增：

```text
src/fastwam/models/h3dreamwam/
  config.py
  action_expert.py
  joint_attention.py
  world_router.py
  model.py
  scheduler.py
scripts/h3dreamwam/
  inspect_h3_contract.py
  precompute_libero_world_targets.py
  train_fsdp.py
  serve_policy_fsdp.py
  eval_libero.py
tests/h3dreamwam/
```

第一提交只完成 M0：模型合约探针、projection 扩展、mask/leakage 单测和 1-window
forward/backward，不启动长训练。这样能够在占用大量 GPU 和存储之前排除结构性错误。
