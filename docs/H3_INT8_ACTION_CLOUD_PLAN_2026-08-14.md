# H3 INT8 + 动作生成：云端实验计划

更新时间：2026-08-14

## 当前决策

训练主线直接使用云端已经跑通的 native Diffusers/BF16 H3 环境，不依赖 ComfyUI
server、workflow 或整套 Comfy 环境。H3 保持冻结，云端训练 ActionDiT、history encoder、
recovery head 和 gate。这样可以立即使用 8×A800，而不等待量化运行时迁移。

部署线采用本地 RTX 5090 已验证的同一个 MiniMax-H3 逻辑模型及官方量化权重：H3
INT8 diffusion + NVFP4/AWQ text encoder + FP16 VAE。INT8 不是缩小网络，而是压缩
权重表示；本地 H8 推理峰值约 24.96 GiB，已经证明能在 5090 部署。Comfy 代码在这条线
只充当自定义 INT8/ConvRot checkpoint loader，不启动服务，也不参与动作模型训练。

BF16 H3 不再更新权重，也不再把“world prediction 更好”等同于“动作更好”。云端
full-tail 1000-step 因果评测已经出现
video 改善而 causal action 退化；adapter-only 的 causal action 才有小幅正增益。这是冻结
主干、集中优化动作的直接依据。

当前可证结论是：量化 H3 特征在 task0、task3、push-plate 三个任务具有因果贡献，三组
zero-feature 消融均为 0/10。当前不可证结论是：一个统一 H3-WAM 已经具有完整 LIBERO
泛化能力。

## 可证伪目标

统一 INT8-H3 动作策略在固定协议下：

- 三个已验证回归任务合计不低于 26/30；
- 至少两个未用于单任务调参的任务分别达到至少 1/10；
- 相比无 H3/zero-feature 对照，目标推进、接触率和 success 同向改善；
- 5090 单次 replan 峰值不超过 30 GiB，动作计算能被 chunk 执行时间覆盖。

满足前三项才把“多任务 H3-WAM 有效”标记为 `EVIDENCE_READY`。否则只保留“冻结 H3
特征对三个任务有效”的窄结论。

## 上游代码对齐

固定版本记录在 `docs/UPSTREAM_SOURCES.lock.json`。

| 官方实现 | 固定 commit | 采用的思想 | 本轮不照搬的部分 |
| --- | --- | --- | --- |
| FastWAM | `45d8e145` | 独立 ActionDiT、action chunk、flow action、replan | 不把 Wan2.2 专属模块硬塞进 H3 |
| DreamWAM | `6e989fac` | structured future 监督、独立动作载体、四套 LIBERO | 不先训练 RGB/flow/depth 而牺牲 action |
| MiniWorld | `e484206b` | block-causal history、rolling KV、短到长课程 | 不从零重训整个 world model |
| LingBot-VA | `7c6ffa9b` | video/action 解耦噪声、流式历史、交错生成 | 不照搬已被本项目动作指标否决的 WD/配置 |

对应的官方量级只用于预算校准：FastWAM 是 10 epoch；DreamWAM release 是 21,700
steps/8 GPU；MiniWorld 是 6→16→32→64 latent-frame 四阶段；LingBot LIBERO release
是 5,000 steps。我们统一报告 `global_batch × steps / train_windows`，不再只报 steps。

## 数据与运行时合同

- 数据：LIBERO Spatial/Goal/Object/10，当前 train manifest 为 277,713 个 stride-1
  窗口、1,712 个 episode；action horizon 32，不再使用每轨迹 5 帧抽样。
- split：suite-qualified episode 隔离；显式 train/val manifest 优先。旧代码只按整数
  episode split 会让不同 suite 相互碰撞，已在 2026-08-14 修复并加入测试。
- H3 feature：固定层 `(9,19,29,39,49)`，固定模型 hash、ComfyUI commit、timestep、
  action horizon；不混用 BF16-H3 与 INT8-H3 cache。
- 量化模型身份：下载脚本内固定三文件 size/SHA256，完整校验前禁止训练。
- 存储估算：单窗口 feature 约 5.03 MiB，全量约 1.33 TiB。活动 cache 上限 2 TiB；
  大规模生成前先把 277k 小文件改为可索引 shard/mmap，避免共享文件系统 metadata 成为瓶颈。

## 实验矩阵与门禁

### A0：BF16 训练 → INT8 部署兼容（Class A）

- parent：云端 BF16 H3 + 动作 checkpoint；部署候选为本地 INT8 H3 + 同一动作 checkpoint。
- 先用 100 个固定窗口做配对 feature/action 比较，再扩到 1,000 个；不传输 1.33 TiB
  完整 INT8 cache。
- 直换门禁：feature cosine ≥ 0.999，normalized action MAE ≤ 0.02；三任务 10-trial
  success 各自最多相差 1/10。
- 未通过则只训练一个小 feature calibration projector，或在 5090 上做短程 INT8 adapter
  校准；不回退到复制整套 Comfy 环境，不阻塞云端动作训练。

### A1：DoT 动作载体容量（Class C，已启动）

- 配对：depth-1 与 depth-4；唯一变量 `action_layers`。
- 两组均冻结 H3，global batch 128，2,170 steps，277,760 samples = 1.0002 epoch，
  每 200 step 保存。
- 增加 depth-4/action-only 配对，唯一变量 `video_loss_weight: 1→0`；它直接检验视频
  目标是否牵制动作，于 2026-08-14 在 `32409` 启动。
- 晋级：held-out causal action、gripper/contact 指标和闭环 canary 同时不差；video loss
  只作诊断。若 depth-4 只让 video 更好，停止继续加深。

### A2：统一轻量 ActionDiT（Class C）

- parent：A0 的 per-task H8 regression heads。
- 首轮一张 GPU 一个分支，固定 INT8 feature、数据、seed 与样本预算：
  H4/H8/H10；previous-action on/off；history 0/2/4。
- screening：2 epoch；batch 64 时约 8,679 optimizer steps。晋级分支续至 5 epoch，
  约 21,697 steps；每 1,000 step 保存，保留 best/last 和闭环晋级点。
- history 必须来自观测与已执行动作，禁止用未来 demo phase 作为部署不可得捷径。

### A3：动作目标（Class C）

- parent：A2 最佳 regression checkpoint。
- 唯一变量依次测试：FastWAM-style flow；4 个 noise repeats；官方 ActionDiT chunk
  teacher distillation。先 warm-start regression，再切 flow，不混在 A2 架构筛选里。
- gripper F1、chunk endpoint/ADE、causal sampling 和闭环 success 都要报告。

### A4：连续纠错与 learned gate（Class D）

- 在多个任务收集策略失败轨迹，teacher 必须从共同可达状态连续 roll-in 到成功；只保留
  连续 recovery chunks，不做失败点单帧 relabel。
- gate 输入为 H3 feature + proprio + previous action + history，不读硬编码 rollout step。
- 除 success 外记录非目标物体位移，防止通过碰撞换成功。

### A5：评测阶梯

1. 离线 val：action MSE、gripper F1、chunk ADE/endpoint、causal action。
2. 回归集：task0/task3/push-plate，各 10 trials、固定 seed/protocol。
3. canary：至少两个 suite 各两个 held-out task，每任务 10 trials。
4. 只有 canary 通过才跑 LIBERO 40 tasks × 50 trials；此前不浪费 rollout 预算。

## 三台 8×A800 调度

| 节点 | 当前/近期任务 | 原因 |
| --- | --- | --- |
| `30907` | DoT depth-4 跑满 1 epoch | 保留已投入的长线，作为容量上界 |
| `30234` | DoT depth-1 严格配对 | 唯一变量对照，已于 2026-08-14 启动 |
| `32611` | INT8 bundle、100/1,000-window A0 parity | 部署兼容线，不阻塞训练主线 |
| `32409` | A0 闭环/离线评测；之后 A2/A3 独立分支 | 评测不与训练争卡，剩余卡做动作 sweep |

如果 A0 需要云端 INT8 feature，8-way precompute 使用确定性分片：

```bash
for gpu in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$gpu python scripts/h3wam/precompute_h3_video_features.py \
    MANIFEST --cache-root CACHE --comfy-root COMFY \
    --h3-checkpoint H3_INT8 --action-horizon 32 \
    --output-subdir h3_video_features_int8_official \
    --num-shards 8 --shard-index "$gpu" \
    > "logs/features_shard${gpu}.log" 2>&1 &
done
wait
```

这条命令目前只放行小规模 parity/speed probe。全量 277,713 窗口在 shard/mmap
落盘格式完成后再放行，防止制造 277k 个大文件拖垮 UPFS。

## 停止规则

- 连续两个 checkpoint 的 success、接触谓词和目标推进均不改善：停止该分支。
- offline MSE 变好但 causal action/闭环变差：按动作退化处理，不以 world/video 指标覆盖。
- 单任务增益导致任一回归任务下降超过 1/10：不晋级统一模型。
- H3 主干解冻保持 `NOT_RELEASED`；只有冻结动作路线饱和且存在清晰因果瓶颈时重新提案。
