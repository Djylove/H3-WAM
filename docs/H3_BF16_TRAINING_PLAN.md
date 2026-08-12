# H3-WAM BF16 训练初步方案

更新时间：2026-08-06

## 决策

当前可以进入 BF16 主干适配。依据不是离线 loss，而是三个 LIBERO 任务闭环成功，且
第三任务在固定 10-trial 协议下两次 10/10、zero-feature 0/10。

默认继续采用已经验证的职责拆分：H3 是 video expert / representation hub，轻量 H8
ActionDiT 从 H3 多层特征产生动作。第一轮不把 action token 塞进 H3 audio slot，也不让
action loss 直接大幅更新整个 H3。

## 权重与代码基线

- 使用官方 `MiniMaxAI/MiniMax-H3` 的 FL2VA BF16 transformer，而不是 pruned INT8。
- 官方 transformer 是完整约 33B 参数、BF16 约 61.7 GB；约 13B AdaLN 分支在推理时
  可以预计算，但训练时应保留完整权重。
- 官方公开了权重、配置、VAE 和推理模型实现，没有提供完整训练脚本。训练 loop 由本项目
  基于 diffusers MiniMax-H3 实现补齐，调度参数直接读取官方 scheduler config；不要沿用
  Wan/FastWAM 的 shift=5 假设。
- Qwen3-VL、Visual VAE、Audio VAE 首轮全部冻结；机器人文本 context 与视频 latent 预计算。

官方资料：

- https://huggingface.co/MiniMaxAI/MiniMax-H3
- https://github.com/huggingface/diffusers/pull/14355
- https://arxiv.org/abs/2603.16666 （Fast-WAM）
- https://arxiv.org/abs/2603.10448 （DiT4DiT）
- https://arxiv.org/abs/2608.02365 （Faster-WAM / DoT）

## 四阶段快速流程

### V0：BF16 等价性与反向传播

用同一批 8～16 个缓存窗口比较官方 BF16 与现有 INT8 H3：tensor key/shape、选定层
feature、video flow loss、单步 backward、保存/恢复。要求 loss 有限、目标参数有非零
梯度、恢复后输出一致。此阶段不跑长训练。

### V1：最后层局部解冻

先解冻最后 2 个 block，再扩到最后 4 个 block；其余 H3、ActionDiT、VAE、Qwen 全冻结。
训练目标以 H3 原生 future-video flow matching 为主，并加入小权重 feature anchor，保护
当前已经能控制机器人的中间层表征。每 100～250 steps 保存 checkpoint，不按离线最低点
直接晋级。

默认起始参数：BF16 forward、最后 2 block 使用 FP32 master/Adam state、gradient
checkpointing、lr `1e-6`、weight decay `1e-4`、grad clip `1.0`、500 steps。FP32 master
避免 `1e-6` 更新被 BF16 权重精度舍入为零；参数探针已测得首步最大更新约 `1.01e-6`。
若梯度和 feature drift 稳定，再到 2k/5k steps。

当前 V1 candidate 使用 `libero_goal_500`：10 个 LIBERO Goal 任务、100 个 source episodes、
500 个缓存窗口。全量 500/500 window/context 合约审计通过；按带盐 SHA256 对
`(task_group, episode)` 分组切分为 450 train / 50 val，每任务保留 1 个完整 episode，
不存在同 episode segment 泄漏。训练不额外做 normalization，直接消费已经匹配官方 H3
VAE latent 域的缓存。

当前恢复格式按 8 卡 FSDP rank 分片。使用 FP32 master weights 后每个 checkpoint 约 15GB，
只保存最后 2 个可训练 block、AdamW state 和 RNG state。真实 450/50 多样本 canary 已完成
step 0 基线验证、3 个训练步和 step 3 保存，并从同拓扑恢复到 step 4；恢复后的参数更新探针
仍为非零（约 `9.98e-7`）。它用于训练续跑，不是部署 bundle；部署候选另行汇聚最后 2
block 的完整 delta。

首轮正式 500-step 训练已经完成。验证 loss 按 step 0/100/200/300/400/500 依次为
`0.198597`、`0.189817`、`0.176461`、`0.170910`、`0.164149`、`0.160993`，最终相对
基线下降约 18.9%。5 个 checkpoint 均包含完整 8 个 rank 分片，无残留 `.partial` 文件，
运行日志未出现 NaN、OOM、NCCL error 或 traceback。输出位于云端
`/home/h3wam_finetune/outputs/v1_libero_goal_last2_500`。

### 首个 H3 主干适配闭环增益（2026-08-07）

完整最后层解冻会在少量动作监督下退化，因此先以官方 BF16 H3 最后 2 个 block 的
rank-16 LoRA 做受控验证。LoRA 只使用 task5 的 6 个 FastWAM 连续恢复窗口训练；step50
已将 episode508 动作 MSE 从 `0.005179` 降到约 `0.001559`，后续收益很小且 feature drift
继续增加，因此闭环选择 step50。

部署时保持冻结 H3 特征负责 recovery gate；gate 触发后，对同一观测启用 LoRA 再前向一次，
让首个恢复动作落在训练 roll-in 的起始状态上。若推迟到下一次 replan 才启用，task5 仍为
9/10；修正同帧接管后，原先耗尽 400 步的 trial8 在 102 步成功，黑碗最大位移从约
`0.140` 降到数值噪声。固定 seed42、trial0→9 的正式结果为：

- task5：冻结基线 9/10，条件 H3 LoRA 10/10；
- task0：10/10，所有 episode 均未选择 recovery head；
- 两任务合计：19/20 → 20/20。

晋级 checkpoint：
`/home/h3wam_finetune/checkpoints/h3_tail/lora_task5_ep508_anchor0_lr1e3_r16_blocks2_steps200/step000050`。
结果分别位于云端 `outputs/local_recipe_formal/` 下的
`taskexpert_recovery25_gate099_h3lora_ep508_s50_rerun_early64_task5_trials0_9_seed42`
和 `taskexpert_recovery25_gate099_h3lora_ep508_s50_rerun_early64_task0_trials0_9_seed42`。
这证明 H3 表征适配能产生闭环净增益，但样本仍只有两个任务、20 个 episode，不能替代完整
LIBERO benchmark，也尚不能说明总体优于 FastWAM。

### V2：重缓存与动作头适配

每个候选 H3 checkpoint 都重新缓存三任务 H3 feature，并从同一初始化训练 H8 action head。
这一步不能省略：已有实验已证明 H3 表征变化后直接复用旧动作头会产生错误结论。

### V3：LIBERO Go/No-Go

固定顺序、seed42、trial0→9、400 steps，依次回归：

1. `open the middle drawer of the cabinet`
2. `open the top drawer and put the bowl inside`
3. `push the plate to the front of the stove`

每个候选都与冻结 BF16 base、INT8 当前基线配对；晋级条件是三任务平均成功率不下降，且
至少一个困难任务提升。候选晋级后补 zero-feature 和关闭 finetune 的消融。通过后才扩大到
完整 LIBERO benchmark。

## 多卡建议

- V0/V1：4×80GB 可先打通，使用 FSDP/ZeRO-3、activation checkpointing 和冻结模块预计算。
- 完整 33B AdamW 全参数训练：优先 8×80GB 起步；若保留 FP32 master weights 和完整 Adam
  states，8 卡会比较紧，16×80GB 更稳妥。
- 先做最后 2～4 层解冻。只有它在三任务闭环上优于冻结基线，才启动全参数训练。

## 当前云端 V0 环境

- 节点：8×A800-SXM4-80GB（NVLink 全互联）、124 vCPU、1.7 TiB RAM、124GB
  `/dev/shm`。计算与主存满足 33B H3 的 8 卡 FSDP 验证。
- 隔离根目录固定为 `/home/h3wam_finetune`，权限 `0700`；项目、venv、模型、HF/Torch
  cache、tmp、数据、日志和输出均在该目录。不得在 `/root` 或系统 Python 中写入。
- 当前根盘在模型、V1 数据、canary 和 500-step 局部解冻的 5 个 checkpoint 后仍有约
  250GB；但不适合保存多份 66GB 以上的全量 checkpoint。正式全量训练前需要挂载至少
  1TB、建议 2TB 的持久盘。
- V0 使用 `data/v0/contexts` 中 5120 维原始 Qwen context。ComfyUI 生成的 5376 维
  `refined_contexts` 已经经过 H3 token refiner，不能再次输入官方 Diffusers transformer。
- 入口是 `scripts/h3wam/launch_h3_bf16_v0.sh`。它强制所有 HOME/cache/tmp/output 路径留在
  隔离目录，默认 8 卡、最后 2 个 block、单步 smoke。

## interactive-training 控制面

参考 `/home/ubuntu/interactive-training` 的 `TrainingSession`，在 optimizer step 后设置统一
control point。首版只开放可安全热更新的 knobs：

- learning rate
- feature-anchor weight
- video/action loss weight（action 默认 0）
- gradient clip
- validation/checkpoint interval

`partial_last_blocks`、分辨率、帧数、action horizon 和分布式拓扑属于结构参数，不允许在
运行中热改。控制面支持 pause/resume、save checkpoint、evaluate 和 stop；rank0 接收动作，
再 broadcast/barrier 到全部 rank。监控 train/val video loss、feature cosine drift、梯度
范数、吞吐和显存；LIBERO success 只在 checkpoint 边界异步评测，不混入每步训练。
