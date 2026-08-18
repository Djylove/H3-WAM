# H3-WAM 云资源停机归档与研发交接

日期：2026-08-18（Asia/Shanghai）

## 一句话结论

当前已经证明的不是“全量微调 H3 有效”，而是：**冻结 MiniMax-H3 INT8，把多层世界特征交给足够强的动作专家，可以在完整 LIBERO 四套件、680 对相同初态闭环中稳定超过历史 D0。** 当前唯一正式晋级模型是 C58b；其余线路保留为对照、失败知识或下一轮架构种子，不得混称冠军。

云服务器关闭前，保留优先级固定为：训练方法与源码身份 > 结果和评测合同 > 数据身份 > 少数关键 checkpoint > 大体积中间缓存/轨迹。checkpoint 不能脱离源码、动作合同和评测证据独立解释。

## 当前整体进度

| 线路 | 训练/评测 | 最终证据 | 结论 |
|---|---:|---|---|
| D0 frozen-H3 5层动作专家 | H32 s14000；680闭环基线 | 270/680 = 39.706% | 历史父模型，保留用于复现谱系 |
| C58b FastWAM full30 + H3逐层K/V | 10k steps、80k samples；680配对闭环 | 295/680 = 43.382%，+3.676pp，87:62，单侧 p=0.02446 | **全部预注册门通过；当前 carrier-track champion** |
| C60 FACT | 10k；680配对闭环 | 313/680 vs C58 295/680；+2.647pp、净18、p=0.050716 | 点估计最高，但三项晋级门都未过；保留 C58 父模型 |
| C67 FACT joint | 20k、160k samples | 324/680 | 只能与 C69 做归因解释 |
| C69 matched action-only | 20k、160k samples | 338/680；C67独胜23、C69独胜37，双侧 p=0.09246 | 当前 FACT consequence 辅助目标无可检测增益；C69是对照，不是新冠军 |
| C70 sampler coverage | 20k | terminal balanced80 失败 | 不做 rollout；方法/负结果比权重重要 |
| C71 Light-WAM shallow state fusion | 9918 steps、79344 samples | normalized MSE好4.70%，gripper好0.49%，但physical差11.49%，语言响应显著弱 | 强视觉 direct-action 架构种子；未授权 rollout |
| C57/C62/C64/C66 context | 多个持久KV/滚动上下文 canary | 机械可用，固定离线/闭环未晋级 | 没有 context winner，不进入融合 |
| BF16/full-H3/LoRA早期线 | 本地与云端多轮 canary | 世界/视频指标可改善，动作闭环可退化 | 当前阶段停止；不要恢复全量H3微调作为主线 |

因此当前 fusion lineage 只有：

```text
MiniMax-H3 INT8 frozen
        |
D0 H32 5-layer action parent (270/680)
        |
C58b FastWAM full30 layer-wise carrier (295/680, promoted)
```

FACT consequence、MiniWorld/LingBot context、C71 Light-WAM state fusion 都尚未得到独立赛道胜者，不能把几个未通过门的组件直接拼成“蛊王”。

## 最重要的训练方法

### C58b：当前成功主线

- 基模：MiniMax-H3 INT8，训练期间完全冻结；本体不做 LoRA、不做全量更新。
- 数据：LIBERO 四套件 dense window，7D absolute action、8D proprio、horizon32、训练集 normalization；首帧/双相机进入 H3。
- carrier：把 H3 的 50 层按深度映射到 FastWAM 30 层 ActionDiT，而不是只重复 layer49。
- 动作目标：shift=5 continuous flow action objective；AdamW、warmup+cosine；global batch 8。
- 预算：10k optimizer steps、80k samples，约 0.398 expert epoch。
- 评测：先固定 episode-disjoint balanced80，再做四套件×10任务×17 trials = 680 对 fresh-process 闭环；horizon32、replan8、10-step solver、max400。
- 晋级门：至少 +3pp、净胜至少20、单侧 exact McNemar p<=0.05、任何 suite 不低于 -3pp。C58b 四项均过。

核心入口：

- `scripts/h3wam/train_h3_fastwam_full_tower.py`
- `scripts/h3wam/launch_c58b_online_long10000.sh`
- `src/fastwam/models/h3wam/fastwam_full_tower.py`
- `src/fastwam/models/h3wam/c58_online_training.py`
- `scripts/h3wam/evaluate_h3_fastwam_full_tower.py`
- `scripts/h3wam/aggregate_c58b_expanded_paired_eval.py`

### FACT归因线：最重要的负结果

C67 与 C69 使用同一 C58b 父模型、同一 20k 预算、数据顺序、30层塔、H3特征、optimizer 和部署合同。唯一主变量是 C67 打开 future-H3 representation/state/value 辅助目标，C69 关闭这些损失并冻结 consequence-only 模块。

结果为 C67 324/680、C69 338/680。这个结果只否定**当前实现的 consequence auxiliary objective 的增量价值**，不否定 H3 世界特征，因为两臂都持续使用冻结 H3 carrier；也不自动把 C69 晋级为冠军。

核心入口：

- `scripts/h3wam/train_c56b_fact_online.py`
- `scripts/h3wam/launch_c67_c60_budget_ablation_20k_8gpu.sh`
- `scripts/h3wam/launch_c69_matched_action_only_20k_8gpu.sh`
- `src/fastwam/models/h3wam/fact_layerwise_tower.py`
- `src/fastwam/models/h3wam/fact_online_data.py`
- `scripts/h3wam/aggregate_c67_c69_attribution_paired680.py`

不要再原样增加 C67 steps。官方 FACT 的 Stage-2/action-conditioned consequence/value best-of-N 在本项目中还没有形成通过门的部署闭环，这是未来可以重新开题的独立变量，不能用 C67 的训练 loss 代替。

### C71：下一轮最值得保留的架构种子

C71 直接借鉴 Light-WAM 的三层 state fusion：H3 相对深度 14/27/41、learned-query pooling、浅动作 trunk。它证明了更轻的动作专家能高效学到强视觉依赖，但语言 conditioning 与物理动作标定退化，所以终点评测失败并停止 rollout。

下一轮若继续，应保持 C71 视觉路径不变，分别单变量验证：语言约束、physical-space action calibration、或 C58→C71 的可恢复蒸馏/初始化。不要把三项一起修改，也不要仅把 9918 拉到更长步数。

## 检查点保留树

完整机器可读谱系、字节数、SHA256 和路径见：

`experiments/archive/h3_wam_shutdown_manifest_2026-08-18.json`

云端建立了零额外空间的 hardlink 归档：

`/mnt/h3-wam/research-archive-2026-08-18/checkpoints/`

优先级如下：

1. P0：C58b s10000——唯一正式冠军，必须优先带走。
2. P0/P1：D0 s14000——C58 的历史父模型和 270/680 基线。
3. P1：C71 s9918——下一代浅动作专家的架构种子。
4. P1：C60 s10000——高点估计、Stage-2/value 研究种子，但不是冠军。
5. P1：C67 s20000 与 C69 s20000——必须成对保存，单独一个会丢失归因意义。
6. P2：C70 s20000——只有 sampler 负对照价值；若空间/带宽不足，可只留方法和报告。

注意：hardlink 只能防止原始 output 文件名被清理，不能防止整个 `/mnt/h3-wam` 共享盘被回收。当前 SSH 实测只有约 20–80KiB/s，大权重无法在停机窗口内可靠回传。本地
`/home/ubuntu/h3-wam-critical-backup-20260818/checkpoints/c58b_online_s10000.pt` 是进行中的不完整文件，只有当大小达到 `12183309492` 且 SHA256 为 `2e6294...e541` 后才能恢复；不能把部分文件当 checkpoint 使用。

## 已保留的非权重资产

- Git 中已有 51 份研发文档、61 份 experiment dossier、379 个 H3-WAM 脚本和 38 个 H3-WAM 核心模型文件。
- 完整实验账本：`docs/H3_WAM_EXPERIMENT_LEDGER.md`。
- 当前候选注册表：`docs/H3_WAM_CANDIDATE_REGISTRY_2026-08-14.md`。
- 代码/来源/训练预算复核：`docs/H3_WAM_PHASE_REVIEW_2026-08-17.md`。
- C67/C69最终归因：`docs/C67_C69_PAIRED_ATTRIBUTION_RESULT_2026-08-18.md`。
- C60失败因果诊断：`docs/H3_C60_FAILURE_CAUSAL_DIAGNOSIS_2026-08-17.md`。
- C67/C69本地逐episode结果和日志：`/home/ubuntu/h3-wam-critical-backup-20260818/eval-summary/`，已含1360个 `results.json`、680对证据和相关日志。
- 核心 JSON/JSONL 结果包：`artifacts/core_results_json_2026-08-18.tar.zst`。
- 完整不可变执行源码包：`artifacts/execution_source_0cc9d9e.tar.zst`。

两个 `.zst` 大包不进入 Git，由 manifest 固定尺寸和 SHA256。Git 保存的是可读结论、方法代码、dossier 和所有身份；二进制包保存在本机 artifacts 目录。

2026-08-18 15:32 对五台 8×A800 节点做了最后审计：40 张卡均为 0 MiB、没有 active compute process，五台看到相同的 `/mnt/h3-wam` 输出目录。因此没有遗漏仍在训练的 optimizer 任务，当前工作应转为归档而不是再开新训练。

## 基模与环境恢复

本机已有完整 H3 INT8：

```text
/home/ubuntu/wan2_2/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
size  = 20970379616
sha256 = e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a
```

路径中虽然有 `ComfyUI`，但它只是历史下载位置。云端训练使用 native Python/diffusers H3 adapter，不依赖 ComfyUI graph。恢复时可把模型放到任意受 manifest 管理的路径，通过 launcher 参数指向即可。

上游固定版本：FastWAM `45d8e145...`、DreamWAM `6e989fac...`、FACT `618a6c16...`、Light-WAM `b2785f66...`、MiniWorld `e484206b...`、StarWAM `cd76d96f...`。完整值见 machine-readable manifest 和 `docs/UPSTREAM_SOURCES.lock.json`。

## 已验证的关键经验

1. H3 作为冻结视觉/世界基模有价值；C58b 的完整闭环提升是当前最强证据。
2. 当前瓶颈主要在动作生成、动作标定、语言/对象选择和长程闭环，而不是继续提高视频重建指标。
3. 全量/BF16 H3 微调可能破坏原有表示并让动作退化；在没有严格 action-only 父对照前不再放行。
4. “更多 steps”不是通用修复：C67/C69 20k 已把 consequence objective 做了同预算归因；C71 是约0.395 epoch，但失败形状是物理与语言约束错位，不是单纯欠拟合。
5. 缓存不是主线。C58b 后期已经用 online frozen H3，避免数TB K/V缓存；本归档也不保留 NPZ/特征缓存。
6. 任何新融合必须先产生单变量赛道胜者，再进入融合；离线 MSE、训练 loss、单任务成功都不能替代完整闭环门。

## 下一次恢复的最短路径

1. 以 Git commit、manifest、H3 INT8 SHA 和 C58b checkpoint SHA 重建环境。
2. 先做 checkpoint strict restore `max_abs=0`，再跑 frozen balanced80；未通过不得开始新训练。
3. 跑一个隔离 LIBERO episode，确认 action normalization、gripper convention、horizon32/replan8/solver10 没有漂移。
4. 若 C58b 权重无法带走，按 C58b 固定方法从 D0 或 fresh function-preserving init 重训 10k；不要改数据、carrier mapping 和评测门。
5. 下一轮只建议两条独立支线：
   - C58b 主线：更完整 expert exposure/数据覆盖，但必须配置 matched continuation control；
   - C71 主线：保留三层视觉 state fusion，单独修复语言或 physical action calibration。
6. FACT Stage-2 best-of-N 只作为独立推理归因实验；先做同状态离线排序门，再决定闭环，不恢复原 C67 consequence 长训。

## 清理边界

可以删除：中间每1k重复 checkpoint、NPZ/KV缓存、已失败线的临时 source snapshot、可重建轨迹视频，以及 `.partial` 文件。

不能删除：本 manifest、Git 方法代码、dossier/结果文档、C58b/D0身份、C67+C69成对归因结果、C71终点报告，以及本机完整 H3 INT8。删除大权重前，必须再次确认是否已有高速对象存储或持久共享盘；不能因为 hardlink 存在就误认为已完成异地备份。
