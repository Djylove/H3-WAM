# H3-WAM 多集群并行开发计划

更新时间：2026-08-12

> 2026-08-12 深度仓库/论文复核发现，原 candidate 每个 episode 只抽 5 个窗口，
> 不是 FastWAM frame-indexed dense loader。当前 7,710 train windows 仅约为四套数据
> 200,779 个完整 horizon-32 train windows 的 3.8%。因此后续主线改为 dense-data
> H3-DoT，完整证据和整改矩阵见 `docs/H3_WAM_REPO_PAPER_REVIEW.md`；本文下方的 sparse
> M6/M8 实验继续作为可比对照，不再代表 H3 上限。

## 目标与统一口径

本阶段只回答一个问题：在同一套四套 LIBERO 数据、动作归一化、采样器和闭环协议下，
全量微调 MiniMax-H3 的 WAM 是否优于冻结 H3 的 head-only 基线，并判断 DreamWAM 的
motion future supervision 是否带来进一步提升。

所有线路共享：

- 数据：LIBERO 10/Goal/Object/Spatial，40 tasks；训练 7,710 windows，验证 850 windows；
- 动作：horizon 32，7 DoF，10-step flow sampler；
- 基线：DoT head-only step125，固定 val40 MSE `0.217331`；
- 主指标：闭环成功率；val MSE 和语言反事实只用于诊断；
- 禁止在本轮加入 history gate、previous action、ranking、SAM3D 或 task-specific loss。

## 集群分工

| 线路 | 入口 | 主要任务 | 当前状态 |
|---|---|---|---|
| A：RGB+Action 主线 | `117.50.181.177:32611` | H3 全 50 层 + DoT，RGB/action 联合 flow，602 steps | step356+，8 卡满载；已保存至 step300 |
| B：独立评测线 | `117.50.181.177:30907` | 检查主线 checkpoint；val40、语言反事实、LIBERO rollout | motion step60 的 task3/seed42 十次闭环 |
| C：Motion 分支 | `117.50.181.177:32409` | 在同一 H3-DoT 上加入 DreamWAM RAFT motion 完整去噪 | paper-I/O step60 完成，Goal canary 0/4 |
| D：动作侧因果消融 | `117.50.181.177:30234` | 冻结 motion H3，单独重训 DoT action/KV fusion | 60-step action LR 1e-4 已启动 |

四台机器均为 8×A800 80GB，并共享 `/mnt/h3-wam`。环境只读共享，以下目录必须隔离：

```text
/mnt/h3-wam/tmp/cluster-32611
/mnt/h3-wam/tmp/cluster-30907
/mnt/h3-wam/tmp/cluster-32409
/mnt/h3-wam/tmp/cluster-30234
/mnt/h3-wam/logs/cluster-30907
/mnt/h3-wam/logs/cluster-32409
/mnt/h3-wam/outputs/h3dotwam/              # A 主线
/mnt/h3-wam/outputs/eval-rgb-dot/           # B 评测
/mnt/h3-wam/outputs/h3dotwam-motion/        # C 分支
```

任何线路不得写入另一线路的 output、log 或 tmp 目录。

## 线路 A：论文配方 RGB+Action H3-DoT

固定配置：

- H3 50 层全部更新，H3 LR `1e-6`；
- DoT action/KV fusion LR `1e-5`；
- RGB video flow loss 1.0，action flow loss 1.0；
- global batch 128，cosine schedule，602 optimizer steps；
- 每 60 steps 保存完整 8-rank H3 + action stage，保留全部十个 epoch 点和 final。

输出前缀：

```text
/mnt/h3-wam/outputs/h3dotwam/m4_paper_joint_full40_10ep
```

A 只训练，不在同机并发评测，保证 step time 可解释。

## 线路 B：不中断训练的独立评测

B 按以下顺序消费 A 的 `step60/120/.../600/602`：

1. 检查 `joint_stage.json`、8 个 rank shard 和 action stage 完整性；
2. 固定 episode-disjoint val40，报告均值和每 suite MSE；
3. 固定 task0 正确/错误语言采样，报告 cosine 与 RMS；
4. 固定 Goal task `0/3/7/8`，每任务 5 trials、replan 10；
5. 只有出现至少一个真实 success 或相对基线明确提升，才扩大到四套完整 benchmark；
6. 最终候选按 40 tasks × 50 trials（2,000 rollouts）报告成功率，使用相同 seeds。

晋级不能只依赖离线 MSE。错误物体接触、动作幅度正常、物体发生位移都不算成功。

## 线路 C：DreamWAM Motion + H3-DoT

依据 DreamWAM 消融，motion 必须作为完整扩散/去噪通道，不能作为 feature residual。
该分支保持 A 的 DoT、数据顺序、LR、batch 和 action objective 不变，只增加：

- 从 `flow_latents` 构造 noisy motion rows；
- H3 输入由 RGB 扩展为 RGB+motion；
- 预测 motion velocity，并加入 `flow_loss_weight=0.5`；
- motion 仅用于训练，推理仍输入零 motion，不依赖在线 RAFT。

执行门槛：

1. 定位或重建全部 8,560 个 RAFT color-wheel/H3-VAE motion artifacts；
2. 全量 shape、finite、ID 对齐审计；
3. tiny 单元测试通过；
4. 8×A800 真实 1-step forward/backward；
5. 10-step、global batch128 canary 无 NaN/OOM，val40 不比 RGB 基线差 10%；
6. 先跑 60-step epoch checkpoint；只有闭环优于 RGB 同步点才扩到 602 steps。

本轮不复用旧 50-layer ActionDiT motion checkpoint；它与当前 DoT 动作接口不同，不能做
公平续训。

## 数据、环境与存储

共享只读资源：

```text
/mnt/h3-wam/project
/mnt/h3-wam/models/MiniMax-H3
/mnt/h3-wam/data/v2_full_cache
/mnt/h3-wam/data/v4_multisuite_uniform_candidate
/mnt/h3-wam/runtime/conda-py311
/mnt/h3-wam/runtime/h3wam-venv
```

共享运行时由已跑通的 32611 环境复制，固定 Python 3.11.13、PyTorch 2.8.0+cu128。
实际布局为：

```text
/mnt/h3-wam/runtime/conda-py311/bin/python              # 基础解释器和 torch
/mnt/h3-wam/.venv/lib/python3.11/site-packages          # 项目附加依赖（只读）
/mnt/h3-wam/project/third_party/diffusers_h3/src        # H3 Diffusers 实现（优先导入）
/mnt/h3-wam/runtime/libero-site.tar                     # 固定 LIBERO 运行时归档
/tmp/h3-wam-libero-site                                 # 评测机本地解包，避免 UPFS 元数据瓶颈
```

新容器不能直接使用旧 `.venv/bin/python`，因为其符号链接指向容器内不存在的
`/opt/conda`。统一通过 `runtime/conda-py311/bin/python` 和固定 `PYTHONPATH` 启动；
`scripts/h3dreamwam/torchrun_shared.sh` 提供同一入口。环境部署后冻结，需要新增依赖时先
复制版本化环境，不在三台机器同时执行 `pip install`，也不允许并发修改共享 venv。

每个完整 H3 joint checkpoint 约 61GB。A 的 11 个点约 0.67TB；C 若完整训练再增加约
0.67TB。当前共享盘剩余约 29TB，不因空间删除 epoch ladder。

## 决策表

| 观察 | 动作 |
|---|---|
| loss 非有限、NCCL 错误或重复 OOM | 停止该线路并保留首个失败日志 |
| val40 比 head-only 差超过 10% | 不进入长闭环 |
| val 改善但 4-task canary 仍 0 success | 不宣称有效，继续下一个 epoch 点 |
| RGB 主线出现闭环正例 | 扩大 RGB checkpoint ladder 评测 |
| Motion 同步点优于 RGB | C 扩到完整 10 epochs |
| Motion 不优于 RGB | 停止 C，不增加 DINO/depth/SAM3D |
| 任一候选在四套 benchmark 明确领先 | 冻结 checkpoint、配置、manifest 和结果作为下一阶段基模 |

## 2026-08-11 执行状态与自动编排

已完成：

1. 30907 和 32409 的 8-rank NCCL/FSDP smoke 均得到 `allreduce=8`；两台均识别
   8×A800 80GB；
2. 30907 精确复现 M3 val40 action MSE `0.2187924549`，排除跨容器数值/加载差异；
3. 30907 的 LIBERO、MuJoCo 3.3.2、robosuite 1.4.0、OSMesa 导入通过；M3 真实 5-step
   闭环烟测完成，H3 服务加载 34.05 秒，平均策略往返 1.75 秒；
4. 官方 RAFT source 和 `raft-things.pth` 已落到共享盘；8-window motion smoke 全部成功，
   实测约 2.05 秒/window；
5. DoT 训练器已接入 motion noisy rows、DreamWAM per-sample latent normalization、
   `flow_loss_weight=0.5`、world timestep weighting，以及可训练/可保存的 H3 RGB/flow
   输入输出投影；相关测试 `11 passed`；
6. 全量 motion cache 已以原子写入方式完整生成 `8,560/8,560`，8 卡各处理约 1,070
   windows，各 rank 稳态约
`1.77 s/window`；motion full50 真实 1-step forward/backward 已通过，loss/action/RGB/flow
分别为 `1.3765/0.3475/0.4029/1.9959`，gradient norm `2.563`，显存峰值
`32.26/42.48 GiB allocated/reserved`，无 NaN/OOM。32409 已自动进入 global-batch-128
的 10-step canary。该 canary 已完成 10 steps / 1,280 samples，mean action loss
`0.272353`、mean flow loss `1.999704`，峰值显存 `38.28/46.76 GiB`，全程 finite 且无
OOM；完整 8-rank joint stage 约 61GB，原子落盘并声明 `train_h3_io=true`。10-step 只用于
验证训练、保存和重载链路，不作为 motion 已收敛的证据。独立 8 卡重载已成功，val40
MSE 为 `0.222776`，相对 head-only `0.217331` 退化 `2.51%`，但通过“不差于基线 10%”
的安全门槛；四任务闭环为 `0/4`，仅作为链路健康检查。60-step 已从相同 head-only
step125 基线独立启动，用同步训练长度判断 motion 是否真正带来收益。

M3 的四任务固定闭环对照也已完成：Goal task `0/3/7/8`、trial0、每个最多 400 步，结果
为 `0/4`。task0 错误推动 wine bottle `0.283`；task3 推动 bowl/plate 但 top drawer 仅
`0.0017`；task7 错误推动 bowl/plate；task8 也未完成 bowl-on-plate。结果保存在
`outputs/eval-rgb-dot/m3_goal_0_3_7_8_trial0`。因此 step60 的硬对照不是“会不会输出动作”，
而是是否从 `0/4` 提升、是否减少 wrong-object interaction，并产生目标 predicate 进展。
另以 107 个 8-rank batch 覆盖完整 850-window validation（末尾 6 个重复用于整除），M3
10-step sampler mean action MSE 为 `0.258914`，head-only step125 同协议为 `0.256766`，
即短 M3 全量更新退化 `0.84%`；step60 将复用同一命令做大样本稳健性复核，val40 仅保留
为快速横向对照。

正式 RGB 主线 step60 的 joint stage 已完整保存（8×约 8.07GB H3 shard + 145MB action
stage）。固定 val40 MSE 为 `0.216276`：相对 head-only `0.217331` 改善 `0.49%`，相对
M3 `0.218792` 改善 `1.15%`。它通过离线健康门槛，但是否有效仍等待同协议 `0/4` 闭环
对照，不能据此提前晋级。四任务 trial0 随后跑完仍为 `0/4`：task0 继续错误推动 wine
bottle `0.308`，task3/7/8 虽产生 bowl/plate 运动但均未完成目标 predicate。因此 step60
不晋级为有效 checkpoint；主训练继续产生 ladder，30907 同时补跑 step60 的完整 val850
和 trials1–4（16 episodes），以排除单 seed 偶然性。

step60 的完整 val850 已完成，mean action MSE 为 `0.255086`：相对 head-only
`0.256766` 改善 `0.65%`，相对 M3 `0.258914` 改善 `1.48%`。大小验证集方向一致，说明
训练数值在改善，但闭环仍未转化。trials1–4 的 16 episodes 已全部完成且为 `0/16`；合并
trial0 后，step60 在 Goal task0/3/7/8 上为 `0/20`。这排除了单个初始状态偶然失败，step60
明确不晋级；30907 已切到 motion 10-step 的 val850，并后台等待 RGB step120 ladder。

motion 10-step 的完整 val850 为 `0.259202`：相对 head-only `0.256766` 退化 `0.95%`，
相对 RGB step60 `0.255086` 退化 `1.61%`。这与 val40 的退化方向一致，再次确认 10-step
仅通过工程链路门禁、不具备能力收益。30907 随后利用 step120 到来前的空窗，补跑 RGB
step60 在 LIBERO Spatial/Object/10 三套的固定 12-episode canary；这能判断 Goal `0/20`
是否只是复杂长程任务的局部现象。

## 2026-08-12 进展与配方纠偏

- RGB 主线运行至 step271，step120/180/240 均已完整落盘。step120 的 val40/val850 为
  `0.217304/0.255541`，step180 为 `0.215161/0.253201`，step240 val40 为 `0.215948`；
  step180 当前离线最好，但 step120 和 step180 的 Goal canary 仍均为 `0/4`。
- RGB step60 的 Spatial/Object/LIBERO-10 固定 canary 均为 `0/4`，连同 Goal 共覆盖
  32 episodes、无成功。30907 正串行补跑离线更好的 step180 和 step240 跨套件 canary，
  自动 ladder 同时等待 step300。
- 原 motion 60-step 已完成：mean action/flow loss `0.264387/1.999141`，val40
  `0.218063`，Goal canary `0/4`。flow loss 全程约 `2.0`，不能视为 motion 已学习。
- 对照 DreamWAM 官方实现后确认配方偏差：官方 flow 新通道按原 RGB 权重标准差的 `0.1×`
  随机初始化、训练 LR `1e-4`、训练 21,700 steps；旧实验使用零初始化、把新 I/O 放入
  `1e-6` 的 H3 参数组且只训练 60 steps。checkpoint 实测 flow `proj_out` 权重范数仅
  `0.0201`，与近零预测一致。
- 训练器已增加独立 `--h3-io-learning-rate` 和 `--flow-channel-init-scale`，motion 默认对齐
  DreamWAM 的 `1e-4/0.1`，RGB-only 仍保持零扩展、不改变原路径；共享环境回归测试
  `11 passed`。32409 已启动 20-step constant-LR paper-I/O 门禁；前四步 flow loss 已从
  `1.99997` 连续下降到 `1.98952`，而旧配方同期基本钉在 `2.0`，说明新增通道终于开始
  学习。门禁完成后自动 val40 和闭环。
- 已挂起独立的 paper-I/O 60-step epoch 编排器：仅当 20-step 末四步 flow loss 相对首四步
  至少下降 `0.005`、val40 不比基线差 10%、闭环 4 episodes 完整时才从原 head-only 基线
  重新训练，不续接短 canary。30907 的通用跨套件脚本新增节点级互斥锁，避免后续两个
  checkpoint 在 suite 切换间隙同时启动服务；当前已启动的 step180/240 任务显存安全，保留
  运行并分别写入独立输出。
- paper-I/O 20-step 已完成并通过门禁：flow 首四步均值 `1.994413`，末四步
  `1.937348`，末步 `1.931418`；val40 `0.220007`，Goal 四任务 `0/4`。这证明纠偏让
  motion 分支真实学习，但尚未转成任务成功。独立 60-step 已自动启动并运行至 step24+，
  flow loss 为 `1.918774`，仍持续下降。
- RGB step300 的 val40/val850 为 `0.215509/0.252340`，刷新完整验证集最佳值，但 Goal
  四任务仍 `0/4`。step180 和 step240 在 Spatial/Object/10 的 24 episodes 也全部失败；
  因此离线改善仍未转化。30907 已启动 step300 三套跨域 canary。
- 历史 H3/FastWAM 数字使用 Goal task3、seed42、trial0→9 的连续 seeded-env 协议；当前
  四任务 canary 使用 seed0、每任务 trial0，不能直接横比。已将 step300 task3 的 seed42
  十 trial 作为诊断排在跨套件 canary 之后；它不替代多任务 benchmark，也不会把历史
  task-specific head 的成功率当作当前 DoT 基线。

已挂起多个低 CPU 自动编排器，不占用正在工作的 GPU：

```text
# 30907：等待 step60 完整落盘 -> val40 -> Goal 0/3/7/8 trial0 闭环
scripts/h3dreamwam/watch_and_eval_m4_step60.sh

# 30907：step120..602 自动 ladder；每 60 步 val40，关键点 val850 + 固定闭环
scripts/h3dreamwam/watch_and_eval_m4_ladder.sh

# 32409：等待 motion cache 完整 -> full50 1-step -> global batch128 10-step canary
scripts/h3dreamwam/launch_h3dotwam_motion_canary.sh

# 32409：等待 motion 10-step joint stage -> val40 -> Goal 0/3/7/8 trial0 闭环
scripts/h3dreamwam/watch_and_eval_motion_m1.sh

# 30907：当前扩展闭环结束后，补齐 motion 10-step 的完整 val850
scripts/h3dreamwam/watch_and_eval_motion_m1_val850.sh

# 30907：利用 step120 前空窗，补齐 RGB step60 的 Spatial/Object/10 跨套件 canary
scripts/h3dreamwam/eval_m4_step60_multisuite_canary.sh

# 32409：等待 10-step 门禁 -> 从相同 head-only 基线训练 motion 60-step -> 自动评测
scripts/h3dreamwam/launch_h3dotwam_motion_step60.sh

# 30907：对任意 RGB checkpoint 跑 Spatial/Object/10 跨套件 canary
scripts/h3dreamwam/eval_m4_multisuite_canary.sh STEP

# 32409：DreamWAM 0.1× 新通道初始化 + 1e-4 I/O LR 的 20-step 纠偏门禁
scripts/h3dreamwam/launch_h3dotwam_motion_paperio_canary.sh

# 32409：20-step 指标通过后，从原基线独立跑完整 60-step paper-I/O epoch
scripts/h3dreamwam/launch_h3dotwam_motion_paperio_epoch.sh

# 30907：RGB step300 的历史同协议 task3/seed42/trial0..9 诊断
scripts/h3dreamwam/eval_m4_step300_task3_seed42.sh

# 30907：motion paper-I/O 60-step 完成后跑 val850 与 task3/seed42/trial0..9
scripts/h3dreamwam/watch_and_eval_motion_paperio_s60_extended.sh

# 30234：冻结 motion step60 的 H3，只用 1e-4 重训 action/KV fusion 60 steps
scripts/h3dreamwam/launch_h3dotwam_motion_action_retune.sh
```

常用只读监控：

```bash
tail -f /mnt/h3-wam/logs/pipeline/m4_paper_joint_full40_10ep.log
tail -f /mnt/h3-wam/logs/cluster-32409/motion_full.log
tail -f /mnt/h3-wam/logs/cluster-32409/motion_eval_orchestrator.log
tail -f /mnt/h3-wam/logs/cluster-32409/motion_step60_orchestrator.log
tail -f /mnt/h3-wam/logs/cluster-30907/m4_step60_extended_orchestrator.log
tail -f /mnt/h3-wam/logs/cluster-30234/m9_motion_frozen_actionlr1e4_s60.log
find /mnt/h3-wam/data/v6_motion_multisuite -maxdepth 1 -name '*.pt' | wc -l
```

下一决策点不再靠人工轮询：C 的 10-step 重载、val40、四任务闭环与 60-step 公平对照
已串成自动门禁。任何一条 canary 出现 NaN/OOM/加载错误都保留首个日志并停止后续长训练；
只有 60-step 与 RGB 同步点的闭环结果决定是否把 Motion 分支扩展到 602 steps。

全量 motion cache 实际仅占 `3.7GB`，远低于预估；共享盘仍余约 `29TB`。不删除任何
epoch checkpoint，且不会因 motion 分支触发存储清理。
