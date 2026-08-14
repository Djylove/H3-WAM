# H3 INT8 WAM 研发计划 V2

更新时间：2026-08-14

## 结论先行

主线改为 **standalone INT8 H3 observation feature + 独立 BF16 ActionDiT**。训练与推理都只
向 H3 输入当前双目首帧、任务文本和固定的 H3 timestep/layout；不运行 ComfyUI，也不要求
先生成未来 RGB。H3 第一阶段完全冻结，优化预算集中到动作专家、视觉 token projector、
proprio 和直接语言条件。

这是在现有证据下最小风险、最接近强开源实现的路径：

- StarWAM 已开源 feature-conditioned 路线：clean observation 单帧、最后层视觉 token、
  独立 ActionDiT、action flow、四套 LIBERO dense 数据；其配置训练 50 epochs，约
  106,850 steps，并报告完整 40×50 rollout。
- FastWAM/DreamWAM 的共同稳定做法是独立 ActionDiT、32-step action chunk、flow matching、
  直接文本/状态条件和完整 dense 窗口；不是把动作塞进视频模型的 audio latent。
- DiT4DiT 同样让一个独立 16 层 action flow head 读取视频模型 hidden state，并使用 4 次
  diffusion-noise repeat，公开配置是 100k steps。
- ImageWAM 和 StarWAM 的结果共同说明“能产生有用当前观测表征”比“部署时必须完整想象未来
  视频”更关键。H3 的首帧条件能力因此有直接价值，但不必绑架动作推理延迟。
- MiniWorld/LingBot 的 block-causal history、rolling KV 和长上下文课程保留为第二阶段；它们
  解决的是持续上下文，不是当前统一动作策略尚未学会目标接触的问题。

## 固定上游版本

以下 revision 于 2026-08-14 再次用官方远端 HEAD 核对；详细锁文件为
`docs/UPSTREAM_SOURCES.lock.json`。

| 项目 | revision | 本轮采用 |
| --- | --- | --- |
| StarWAM | `cd76d96f` | feature-conditioned 主结构、30 层 ActionDiT、50 epoch 预算、完整 LIBERO 协议 |
| FastWAM | `45d8e145` | action chunk、flow scheduler、直接语言/状态、10 epoch 对照 |
| DreamWAM | `6e989fac` | dense 数据、ActionDiT、结构化未来目标只作后续消融 |
| DiT4DiT | `66a6f3a1` | 16 层 action head、4× noise repeat、100k-step 预算参照 |
| ImageWAM | `5d4a341e` | 单图视觉主干也可服务动作，避免把未来 RGB 当硬依赖 |
| MiniWorld | `e484206b` | block-causal/rolling-KV/短到长课程，延后使用 |
| LingBot-VA | `7c6ffa9b` | 真正 persistent history 的训练部署合同，延后使用 |

## 复用已有经验，而不是清零重来

### 已经证明有效，必须保留

- 本地 INT8 H3 多层首帧特征 + H8 regression head 已在 task3/task0/push-plate 达到
  `9/10、9/10、10/10`，对应 zero-feature 均为 `0/10`。这证明 H3 当前观测表征具有真实
  因果贡献，也是云端迁移的首个回归门槛。
- dense stride-1、Horizon8、连续 teacher roll-in/recovery、learned state gate 都有闭环
  正证据；这些模块保留，但 recovery/gate 只能在统一基础策略出现跨任务成功后再接。
- action/状态归一化、LIBERO task-language 映射、episode-qualified split、gripper 单独指标、
  固定 seed/trial rollout 和 zero-feature 消融均沿用。
- video-LoRA 曾给 task3 带来小幅净增益，说明动作对齐的轻量 H3 adapter 有潜力；但它不是
  第一阶段依赖条件。

### 已经证明不值得原样重跑

- H3 audio-slot 动作、全量/大范围 H3 更新、只看 video loss、只延长共享 video-action tail：
  均出现 causal action 或闭环退化。
- 100～1,000 step 的统一模型结果不能再代表“架构无效”；强开源配置是 21k～107k steps。
- 只有 teacher-forced MSE 改善的 history/action-shift/sampler sweep 不晋级；历史必须实现
  LingBot 式训练/部署一致的 persistent state 后才重开。
- phase、硬编码 rollout step 或单任务 mode 不能进入最终统一策略；可作为历史老师或失败
  分析工具，但不能作为 40-task benchmark 的输入。
- 旧的 5-frame episode 抽样不再使用；固定为 33 raw frames、stride-1 window、32 actions。

## 可证伪主假设

在相同的 277,713-window、episode-disjoint split 和 global batch 128 下，冻结的 standalone
INT8 H3 从当前首帧提取视觉 token，配合直接任务文本、proprio 和充分训练的独立 ActionDiT，
能够形成一个统一 40-task 策略；它在三个历史任务上至少复现 `26/30`，并在未调参任务上
产生跨至少两个 suite 的闭环正例。若完成 10 epochs 后仍无跨 suite 正例，说明瓶颈不是
简单训练不足，才允许启动 action-aligned H3 LoRA。

## 数据与 feature 合同

- 四套 LIBERO：277,713 train windows、1,712 episodes，suite-qualified episode split。
- 每个 window：33 raw steps、32 actions、两路相机、当前 proprio、完整任务文本。
- H3 输入：只使用第一帧；FL2VA packed layout 和 timestep 固定，feature 只允许来自当前
  observation rows，禁止 future target token 泄漏。
- 第一轮对比两种 feature：
  1. `multi5`：历史成功层 `(9,19,29,39,49)`；
  2. `last32`：StarWAM 风格最后层 32 个 observation token。
- `last32` BF16 cache 约 89 GiB，而旧 five-layer full-token cache 约 1.33 TiB；`last32`
  必须与 StarWAM 源码一致，先保留完整 observation rows，再用
  `adaptive_avg_pool1d(tokens→32)`，禁止用均匀抽帧/抽 token 冒充。
- 全量只生成 `last32`；`multi5` 仅生成历史三任务的 18,463 个窗口，预计约 90 GiB，
  不生成 1.33 TiB 的四套 LIBERO 全量副本。
- 每个 cache shard 固定 checkpoint SHA256、layout、layer、token index、dtype 和样本 manifest
  hash；INT8/BF16 feature 不混用。

## 动作模型实验树

所有分支使用相同数据顺序、归一化、global batch、seed、验证集和 1,000-step checkpoint。

### R0：云端 INT8 回归复现

- 复用本地三个成功动作 checkpoint/配方，在云端 standalone INT8 H3 上跑固定 30 episodes。
- 门槛：合计至少 `26/30`，每个 zero-feature 为 `0/10`；否则先解决 feature parity，不训练
  新统一模型。
- 状态：`GO_CANARY`，等待真实 H3 block parity 和 rollout restore。

### R1：统一 feature-conditioned ActionDiT 主线

- 结构：32-step action token、直接 text + proprio + H3 visual tokens；动作专家 30 层、
  hidden 1024，BF16；H3 冻结。
- 目标：FastWAM/StarWAM continuous flow matching，train/infer shift 5，10-step Euler。
- 初始化：ActionDiT 主体随机、输出头小尺度随机；这是 StarWAM feature-conditioned 官方
  配置，避免重复旧 H3 interpolation 首步发散。
- 预算：先 10 epochs，约 21,700 steps；通过 canary 后续到 50 epochs，约 108,500 steps。
- 唯一变量配对：`last32` 对 `multi5`。先用 2 epochs screening；胜者跑满 10 epochs。
- 状态：`NOT_RELEASED`，需要完成 action context/attention 与官方代码逐项矩阵。

### R2：动作目标复验

- parent：R1 相同结构、相同初始化与 10-epoch checkpoint。
- 分支 A：继续 flow；分支 B：H8/32 chunk regression warm-start；分支 C：DiT4DiT 式
  4× noise repeat。一次只改变目标或 repeat。
- 先看 causal action、gripper F1、chunk ADE/endpoint，再跑相同 canary；不能用 teacher-forced
  loss 替代闭环。
- 状态：`NOT_RELEASED`。

### R3：动作对齐的 INT8 H3 adapter

- 只有 R1 完成 10 epochs 后仍显示“动作专家容量够、视觉目标绑定不足”才启动。
- 保持 INT8 base 不动，在选定 block 的 qkv/out 或 block output 添加 BF16 low-rank branch；
  只让 action loss 反传，LR 至少比 action head 小 10 倍。
- 与 adapter-off 完全配对；禁止同时加 video loss、history 或 recovery。
- 状态：`NOT_RELEASED`。

### R4：上下文与恢复

- parent 必须已在至少两个 suite 有闭环正例。
- 先加入真实 executed action/observation history，再实现 persistent KV；不再做每次 replan
  冷启动的“伪 LingBot history”。
- recovery 数据必须是连续 teacher roll-in 到成功，gate 只读可观测 state/H3/history。
- 状态：`NOT_RELEASED`。

## 评测和停止门槛

1. 数值/restore：finite、非恒定输出、checkpoint 精确恢复。
2. 离线：teacher-forced 与 causal action 同报；语言替换必须引起动作变化；gripper/contact
   单独统计。
3. 回归：task3/task0/push-plate 各 10 trials，并做 zero-feature。
4. canary：四个 suite 各 2 个未调参任务、每任务 10 trials。至少两个 suite 出现成功才继续。
5. 最终：LIBERO 40 tasks × 50 fixed init states = 2,000 rollouts；wait30、replan10、seed42，
   Spatial/Object/Goal max400，LIBERO-10 max700。

停止规则：连续两个 1,000-step checkpoint 的 causal action、目标接触和 success 均无改善，
停止该分支；但不以固定 100-step 代替训练预算结论。10 epochs 仍无跨 suite 成功时，停止
继续堆 action steps，转 R3；50 epochs 后仍明显不及无 H3 action baseline，则否定本结构。

## 多机调度（2026-08-14 当前）

| 节点 | 任务 |
| --- | --- |
| `32611` | standalone INT8 kernel/backbone 已通过；继续数值 parity、R0 restore |
| `30907` | 保留已到 step1600+ 的 depth-4 长线至 2170；随后转动作消融 |
| `30234` | 8-way 历史三任务 `multi5/full98` cache（18,463 windows） |
| `32409` | 8-way 四套 LIBERO `last32` cache（277,713 windows） |

depth-1/action-only 两条在尚无 checkpoint 时停止并转入 INT8；只保留已投入约 14 小时的
depth-4 到预定终点。新的 R1/R2 **训练**命令仍未发布；feature cache 属于可逆 Class-A 数据
准备，只有实现差异矩阵、feature parity、真实 restore 和数据预算全部通过后才启动动作长训。

## 环境与启动硬约束

- 项目固定为 `/mnt/h3-wam/project`；Python 环境固定为
  `/mnt/h3-wam/runtime/h3-int8-native`；权重、数据和日志也全部位于 `/mnt/h3-wam`。
- 不向 `/root` 或 `/usr/local` 写项目环境。`/usr/local/nvidia/lib{,64}` 只作为容器驱动库读取。
- `30234/32409` 默认环境额外加入 `/usr/local/cuda/lib64`，会令 torch 2.10/cu130 的普通
  BF16 GEMM 报 CUBLAS 错误。所有 INT8 launcher 必须显式设置：
  `LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64`。
- 正式入口为 `scripts/h3wam/launch_h3_int8_feature_cache.sh`；启动前同时跑 BF16 GEMM 和
  ConvRot INT8 kernel smoke，不能只看 `torch.cuda.is_available()`。
