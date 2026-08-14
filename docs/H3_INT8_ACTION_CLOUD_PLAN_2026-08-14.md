# H3 INT8 + 动作生成：云端实验计划

更新时间：2026-08-14

## 当前决策

训练主线切换为云端 standalone INT8 H3：冻结 H3 的 INT8 权重，只训练 BF16/FP32
ActionDiT、history encoder、recovery head 和 gate。不安装或导入 ComfyUI，不启动 server、
workflow 或 node；量化矩阵乘仅调用独立的 Apache-2.0 `comfy-kitchen` CUDA kernel。项目内
原生实现负责 fused QKV、curve AdaLN、packed sequence 和多层特征输出。

采用的 diffusion checkpoint 与本地 RTX 5090 已验证模型完全相同，云端 SHA256 已核对为
`e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a`。INT8 不是缩小网络：
它仍是 50 层 H3，含 200 个 ConvRot INT8 线性层，每层 qkv/out/fc1/fc2 各一个。云端
PyTorch 2.10/cu130 独立环境和项目原生 loader 通过后，才放行动作长训。

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

满足前三项才把“多任务 H3-WAM 有效”标记为 `EVIDENCE_READY`。最终项目目标是对
LIBERO Spatial/Object/Goal/10 共 40 个任务、每任务 50 个固定初态完成 2,000 次 rollout
并报告总成功率；canary 仅用于检查点筛选，不能替代完整 benchmark。

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
- H3 feature：历史回归固定层 `(9,19,29,39,49)`；统一主线固定第 49 层，并严格复用
  StarWAM `adaptive_avg_pool1d` 得到 32 tokens。固定模型 hash、官方 packing revision、
  timestep、action horizon；不混用 BF16-H3 与 INT8-H3 cache。
- 量化模型身份：下载脚本内固定三文件 size/SHA256，完整校验前禁止训练。
- 存储估算：`multi5/full98` 单窗口约 5.03 MiB，因此只生成三任务 18,463-window 子集，
  约 90 GiB；四套 LIBERO 全量使用 `last32`，约 89 GiB。活动 cache 上限 2 TiB，完成后
  校验文件数与 metadata，再打成 indexed shards 供长训读取。

## 实验矩阵与门禁

### A0：standalone INT8 H3 骨干（Class A）

- checkpoint 门禁：size/SHA256、50 blocks、200 quantized linears、925/932 特征路径 tensor
  全部映射；未映射的 7 个 tensor 只允许是当前动作训练不使用的 final video/audio head。
- runtime 门禁：PyTorch 2.10/cu130 + `comfy-kitchen`，真实 qkv ConvRot kernel 输出 shape、
  dtype 和 finite 全通过；环境中不得安装 ComfyUI。
- 数值门禁：用同一 packed input 对 standalone 与本地已验证 H3 跑 block
  `(9,19,29,39,49)`，逐层 feature cosine 与最终动作误差一并报告。未完成 parity 前仅
  `GO_CANARY`，禁止宣称动作效果。
- 真实单层 kernel 与完整 50 层 LIBERO-window forward 已通过：峰值 19.88 GiB，单窗口约
  1.2 秒，输出 `(1,32,5376)`、finite。depth-1/action-only 在未形成 checkpoint 时停止；
  只保留已到 step1600+ 的 depth-4 长线到预定终点。

### A1：DoT 动作载体容量（Class C，冻结历史结果）

- 配对：depth-1 与 depth-4；唯一变量 `action_layers`。
- 原计划三组均冻结 H3、global batch 128、2,170 steps；新决策后不再为完成矩阵而消耗
  两整台服务器。depth-1/action-only 的启动日志保留，但未生成 checkpoint，不作效果结论。
- 晋级：held-out causal action、gripper/contact 指标和闭环 canary 同时不差；video loss
  只作诊断。若 depth-4 只让 video 更好，停止继续加深。

### A2：统一轻量 ActionDiT（Class C）

- parent：A0 的 per-task H8 regression heads。
- 首轮一张 GPU 一个分支，固定 INT8 feature、数据、seed 与样本预算：
  H4/H8/H10；previous-action on/off；history 0/2/4。
- screening：2 epoch；batch 64 时约 8,679 optimizer steps。晋级分支续至 5 epoch，
  约 21,697 steps；每 1,000 step 保存，保留 best/last 和闭环晋级点。
- history 必须来自观测与已执行动作，禁止用未来 demo phase 作为部署不可得捷径。

此前动作路线不原样混跑，按证据分层复验：

| 路线 | 已有证据 | 本轮决策 |
| --- | --- | --- |
| 冻结 H3 + chunk regression | 本地三个任务已有闭环正例，zero-feature 为 0/10 | 作为统一模型 parent，必须复现 |
| FastWAM/DreamWAM ActionDiT flow | 公开实现的主动作目标；本项目曾受主干/数据混变量影响 | 固定 INT8 feature 后做唯一变量配对 |
| DoT 单层/4层 action head | 离线 loss 明显下降，但早期闭环多为 0 | 保留正在跑的一 epoch 容量对照，不单凭 loss 晋级 |
| LingBot executed history | teacher-forced 改善但 causal MSE 曾退化 9.87% | 只有无 history parent 先通过 causal gate 才重验 |
| video/action 联合更新 H3 | 多次出现 video 改善、causal action 退化 | 停止作为主线；仅保留历史反例 |
| H3 全量/尾层解冻 | 云端与本地均显示动作退化风险 | `NOT_RELEASED` |

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

## 四台 8×A800 调度

| 节点 | 当前/近期任务 | 原因 |
| --- | --- | --- |
| `30907` | DoT depth-4 跑满 1 epoch | 保留已投入的长线，作为容量上界 |
| `30234` | 18,463 个历史三任务 `multi5/full98` INT8 feature | 复用本地 26/30 配方所需输入合同 |
| `32611` | standalone INT8 runtime、原生骨干、真实 kernel/parity | kernel/full-forward 已过，继续 reference parity 与 restore |
| `32409` | 277,713 个四套 LIBERO `last32` INT8 feature | StarWAM 风格统一动作主线的共享资产 |

8-way precompute 使用固定的 `/mnt` 环境和确定性分片：

```bash
scripts/h3wam/launch_h3_int8_feature_cache.sh \
  MANIFEST OUTPUT_SUBDIR 49 32 8
```

环境固定在 `/mnt/h3-wam/runtime/h3-int8-native`。启动器显式移除会覆盖 torch-cu130 的
`/usr/local/cuda/lib64`；该污染已在 `30234/32409` 复现为普通 BF16 GEMM/CUBLAS 失败。
当前按原 loader 合同原子写 per-window 文件，完成后校验 count/hash，再打成 indexed shards；
全量 `last32` 约 89 GiB，三任务 `multi5` 约 90 GiB，均低于 2 TiB 活动上限。

## 停止规则

- 连续两个 checkpoint 的 success、接触谓词和目标推进均不改善：停止该分支。
- offline MSE 变好但 causal action/闭环变差：按动作退化处理，不以 world/video 指标覆盖。
- 单任务增益导致任一回归任务下降超过 1/10：不晋级统一模型。
- H3 主干解冻保持 `NOT_RELEASED`；只有冻结动作路线饱和且存在清晰因果瓶颈时重新提案。
