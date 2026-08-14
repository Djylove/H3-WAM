# MiniMax H3-WAM：5090 快速验证记录

更新时间：2026-08-14

> 2026-08-14 决策更新：本文件前半部分保留 5090 探索过程作为证据链；其中
> “进入 BF16 H3 尾层解冻”已被后续云端因果评测否决，不再是当前路线。现行计划见
> `docs/H3_INT8_ACTION_CLOUD_PLAN_2026-08-14.md`：云端 native H3 冻结训练动作模块，
> 本地量化 H3 作为部署目标；把主要预算交给动作生成、历史上下文、连续纠错数据和
> 闭环评测，Comfy/INT8 parity 不再阻塞云端训练。

## 结论

MiniMax H3 的首帧生视频能力可以直接作为 WAM 的视觉条件，因此本项目把
机器人首帧、任务文本、proprio state、未来视频 latent 和 noisy action 一起送入
H3。当前最小链路已经在单张 RTX 5090 上跑通，不依赖 ComfyUI server/workflow；
仅复用本地 ComfyUI 的 H3 checkpoint loader 和 INT8 量化算子。

5090 足够完成数据接入、Action Head、attention LoRA、联合视频/动作训练和 2～4 步
离线采样。它不能进行 H3 BF16 主干训练：本地 H3 有 20.114B 个逻辑参数，仅 BF16
权重就需 37.47 GiB，而 5090 总显存为 31.36 GiB，尚未计激活、梯度和优化器。
下一阶段若要解冻主干，切换 80GB GPU。

## 已完成的实验

数据为公开 LIBERO Goal：10 个任务、100 个 episode、每 episode 5 个窗口，共 500
窗口。按 episode 隔离为 400 train / 100 validation，避免相邻窗口泄漏。

每个窗口包含：

- 224×448 双相机画面，39 帧，对应 H3 video latent `[1, 24, 12, 14, 28]`
- 32 步、7 维 action
- 32 步、8 维 proprio state
- 任务文本、首帧视觉条件和预计算 Qwen context

| 实验 | 训练参数 | held-out 指标 | 5090 资源 |
| --- | ---: | ---: | ---: |
| 小型 ActionDiT 基线 | 4.67M | 4-step MSE **0.2968** | 0.19 GiB，0.0016 s/sample |
| 冻结 H3 + Action Head | head only | flow loss 1.9302；4-step MSE 0.7758 | 20.84 GiB，约 0.78 s/step |
| H3 + rank-4 attention LoRA | 7.88M | flow loss 1.6247；4-step MSE **0.7359** | 21.00 GiB，约 0.88 s/step |
| LoRA + 0.2 video loss | 7.88M | action/video flow MSE 1.5675/0.2546；4-step MSE 0.8426 | 21.09 GiB，约 0.88 s/step |

H3 LoRA 相比 head-only 的 4-step MSE 改善约 5%，说明 H3 可以被机器人动作目标适配；
但它仍明显落后于小基线。联合目标把视频 flow MSE 从 0.2615 降至 0.2546，却使
最终动作采样变差，因此当前默认检查点是纯 LoRA，而不是 joint checkpoint。

纯 LoRA 推理实测：

| H3 模型调用次数 | normalized action MSE | 稳态延迟 | 峰值显存 |
| ---: | ---: | ---: | ---: |
| 2 | 2.3569 | 0.306 s | 20.02 GiB |
| 4 | 0.7359 | 0.609 s | 20.02 GiB |

## LIBERO 闭环 rollout（2026-08-04）

已接入真实 LIBERO Goal/MuJoCo 闭环。模拟器使用已有的 Python 3.10 LIBERO
环境，H3 使用 Comfy 量化算子所在的 Python 3.12 环境；两个进程通过仅监听
localhost 的 IPC 通信，避免修改或污染任一环境。评估协议与原 FastWAM 对齐：
episode 开始等待 30 步、每 10 步重新规划、夹爪动作二值化、最长 400 步。

在线部署链路为：

```text
LIBERO 双相机 + proprio
        ↓ IPC
Qwen 首帧 context（每 episode 一次）+ H3 VAE 首帧（每次 replan）
        ↓
4-step H3 action sampling → 反归一化/夹爪转换 → MuJoCo
```

| 策略 | task 0 pilot success | chunk 往返 | 峰值显存 |
| --- | ---: | ---: | ---: |
| 小型 ActionDiT，cached context | 0/3 | 约 0.004 s | 0.05 GiB |
| H3 LoRA，online episode context | **0/3** | 约 0.86 s | 27.86 GiB |

H3 的在线 Qwen context 首次编码约 6.5～7.1 秒，之后每个 replan 的 H3 推理约
0.64 秒，VAE 约 0.05 秒。额外测试了一个专家动作能够完成的 init-state 45，H3
仍为 0/1。专家数据动作在 init-state 45/25/36 上可以成功回放，因此 action 符号、
gripper 变换和仿真 step 接口已经验证，当前失败主要来自只有 500 个窗口的模型能力，
而不是 rollout 管线。

这些数字只是管线 pilot，不是正式 LIBERO benchmark；0/3 的样本量不能给出可靠的
任务成功率，但足以说明当前 checkpoint 不应继续做大规模 50-trial 评测。

## 失败定位更新（checkpoint ladder）

task 0 已扩展为 42 episodes / 210 windows，并加入显式 state/context action
conditioning。累计约 2k～5k step 的 checkpoint 已逐点保存和比较；5090 峰值显存
约 21.1 GiB、训练约 0.85 s/step。

| 累计训练量 | held-out first-10 MSE（4-step，seed0） |
| ---: | ---: |
| 约 2k | 0.0481 |
| 约 3k | 0.0454 |
| 约 4k | 0.0462 |
| 约 5k | **0.0408** |

已确认并修复三项问题：

1. action/video 使用不同 shift，旧 sampler 错用 video sigma delta 更新 action；修复后
   不再出现约 4 倍过冲。
2. 原 action decoder 几乎忽略视觉、任务和 state；加入显式条件投影后，打乱 context
   会使 first-10 MSE 从 0.149 升到 0.165，条件开始生效。
3. 训练每个窗口都使用当前首帧 context，而 rollout 曾整段复用 episode 初始 context；
   新增 `online_replan`，使机器人从漂向桌面改善为能到达中层抽屉把手附近。

累计约 5k 的 H3 flow checkpoint 在 task0 上仍为 0/1。继续定位后确认，主要问题不是
训练步数，而是动作建模和数据覆盖：

1. 旧 cache 每个 episode 只有 5 个 chunk 起点，无法覆盖频繁闭环重规划遇到的状态。
2. 训练对 32 个动作等权，但部署主要执行 chunk 首动作；首动作精度被稀释。
3. 确定性 MSE 会把多条成功轨迹逐时刻平均，产生一条不存在的失败轨迹。
4. 8D 瞬时 proprio 无法区分接近、接触、拉出和撤离等轨迹相位。
5. horizon=32 的 manifest 起点只覆盖 0～105，而 138 步 episode 的 106～137 没有
   一步动作监督。
6. baseline 训练时使用逐窗口视觉 context，部署却固定使用 episode 首帧 context。

已加入首动作 loss、轨迹相位、chunk 内动作展开、固定部署 context 和单 episode 诊断模式。
最小学习重放基线在真实 LIBERO task0 上得到 **3/3 成功**：trial25/36/45 分别在
128/129/132 步完成。成功 checkpoint：
`data/h3wam_checkpoints/libero_goal_task0_ep0_action_regression_h1_phaseonly_expanded_fixedctx_5000.pt`。

这个结果证明数据、动作坐标、训练、模型服务和 MuJoCo 执行链路有效，但它是单轨迹
phase-only 里程碑，不能作为 H3 视觉泛化成功率。下一步应把同样的全时域监督和相位/历史
条件迁移到 H3 动作头，并用 flow/mixture 保留多模态；之后再用策略偏离状态做 DAgger。

### H3 动作头迁移结果（2026-08-05）

已将 deterministic regression、per-action phase sequence、完整 32-step 监督和直接 action
residual head 迁移进 H3 adapter。两组各 500-step 训练均使用 5090，约 0.85 s/step、峰值
21.10 GiB；部署单次 H3 forward 约 0.155 s、峰值 20.02 GiB。

| H3 动作头 | 最佳 validation loss | task0 trial25 |
| --- | ---: | ---: |
| per-action phase sequence | 0.1288 | 0/1 |
| phase sequence + direct action residual | **0.1241** | 0/1 |

checkpoint ladder 的 100/200/300/400/500 step 以及 final 均未成功，视频仍显示机械臂偏向
桌面圆盘。相同 rollout 代码对成功轻量控制头再次回归得到 success=True（128 steps），所以
失败不在部署接口。当前停止扩大 H3 audio-slot 低层控制实验，采用以下基模结构：

```text
H3：首帧条件视频预测 / 世界表征 / 高层候选轨迹
                     ↓ condition / future representation
轻量 action controller：previous-action + proprio + visual feature → action chunk
```

这保留 H3 首帧生成未来视频的优势，同时避免让音频 latent 承担不擅长的精密接触控制。

### H3 Video Expert + 独立 ActionDiT 成功结果（2026-08-05）

已按 FastWAM 的职责拆分重新实现最小原型：冻结 H3 作为 Video Expert，从第
9/19/29/39/49 层抓取当前双目首帧的 condition token；独立 5.72M 参数 ActionDiT
逐层 cross-attend H3 token，回归 32-step 动作块，不再使用 H3 audio latent。

- 固定任务 context 的 310 个 H3 特征窗口缓存耗时 38.5 秒，H3 峰值 19.81 GiB。
- ActionDiT 训练 3000 steps 耗时 63.3 秒，峰值 0.38 GiB，最佳跨 episode MSE 0.0654。
- 在线 VAE + H3 + ActionDiT 每步重规划约 0.137 秒，峰值 24.69 GiB。
- LIBERO Goal task0 的 trial25/36/45 均成功，分别用 145/153/146 步，当前为 **3/3**。
- trial25 将 H3 token 全部置零后 220 步失败；正确特征成功，证明闭环动作确实依赖
  H3 当前观测表征，而不是只由 phase 复读。

主 checkpoint：
`data/h3wam_checkpoints/libero_goal_task0_ep0_h3_feature_action_fixedctx_3000.pt`。
成功结果位于 `data/h3wam_rollouts/h3_feature_fixedctx_3000_trial25/` 和
`data/h3wam_rollouts/h3_feature_fixedctx_3000_trials36_45/`；置零消融位于
`data/h3wam_rollouts/h3_feature_fixedctx_3000_trial25_zero_features/`。

### 扩大到 42 轨迹后的 FastWAM 对照（2026-08-05）

对本仓库 FastWAM 实现逐行核对后，补齐了 H3 视频分支 `t=0`、shift=5 continuous
flow matching、20-step ODE、当前 proprio 和无人工 phase 的对照路径。42 条 task0
轨迹形成 4479 个 dense windows；同时发现旧固定 Qwen context 含 100 个 episode-0
图像 token，新增了只有 7 个任务文本 token 的 context 对照。两种 flow 对照均未在
闭环成功，100-step 积分也无改善；当前 36.77M 随机初始化动作头不能替代 FastWAM
30 层、1024 hidden 且由 Wan2.2 插值初始化的 ActionDiT。

扩大数据的 mixture3 phase 模型已取得闭环成功。自动 gate 错选 mode2 时失败；mode1
在 task0 的 10 个固定 trial（0/5/10/15/20/25/30/35/40/45）中成功 **8/10**，成功
回合均在 175～183 步完成。原型留一法进一步证明，训练时按动作签名聚类得到的 mode
不是可由初始观测可靠识别的语义变量：首帧 H3 特征仅 12/41、初始 proprio 仅 13/41
能正确预测 episode mode。因此实验室基线采用 task-level rollout validation 选择并锁定
mode，而不再让不可观测的 gate 在线猜测。

原始 checkpoint 为
`data/h3wam_checkpoints/libero_goal_task0_42ep_h3_feature_action_mixture3_phase_5000_best.pt`；
写入 `recommended_action_mode=1` 的部署 checkpoint 为
`data/h3wam_checkpoints/libero_goal_task0_42ep_h3_feature_action_mixture3_phase_5000_promoted_mode1.pt`。
部署参数仍可用 `--h3-action-mode` 显式覆盖推荐值。

### 扩展到 LIBERO Goal task1（2026-08-05）

已将 `open the top drawer and put the bowl inside` 的全部 33 条公开专家轨迹扩成
5646 个 dense windows，并缓存首帧 H3 特征。特征缓存耗时 727 秒、峰值 19.81 GiB；
5000-step、5.73M 参数动作头训练约 104～110 秒、峰值 0.38 GiB，5090 可以稳定完成。

task1 是两阶段长时任务。phase-only、proprio+phase、proprio+no-phase 三组 mixture3
模型均未在 trial0 闭环；FastWAM 官方 LIBERO 的 `replan_steps=10` 也未改变结果。失败
视频和专家动作对比发现一个可复现问题：普通 7 维等权 MSE 会忽略轨迹末尾仅 4～10
步的 gripper release。训练脚本现支持 `--gripper-loss-weight`；设为 10 后策略已经能在
后段重新张开夹爪，但运动轨迹仍未把碗稳定放入抽屉。该模型的 mode0 在额外 5 个固定
初始状态仍为 0/5，去掉 phase 的三个 mode 也为 0/3。

后续定位到评测中存在关键的编号语义错误：LeRobot 数据的 `task_index=1` 不是当前
LIBERO benchmark id。该数据实际任务 `open the top drawer and put the bowl inside`
对应 benchmark id 3；此前 id 1 实际是 `put the bowl on the stove`。模型一直在开抽屉，
却按放到炉子上的条件判失败。rollout 现支持 `--task-languages` 按完整语言解析任务，
新 checkpoint 也记录 `training_tasks` 并在不匹配时 fail-fast。

修正映射后，重建了与部署 token 布局严格一致的 Horizon1 缓存：6669 个窗口，补齐
每条轨迹最后约 31 个状态；H3 特征缓存 821 秒、峰值 19.80 GiB。严格 Horizon1
ActionDiT 在正确 task 的 trial0/4 分别于 190/183 步成功，6 个初态共 **2/6**；另有
两个失败回合成功打开抽屉但未完成放碗。Horizon32 + replan1 + temporal ensemble 在
trial0 失败，抽屉仅移动 0.00168；Horizon1 是当前 task1 主线。

正确 H3 特征在 trial0/4 为 2/2；将特征全部置零后为 0/2，且两个回合抽屉位移均为
0，证明成功依赖 H3 当前视觉表征，而不是只靠 phase/proprio 回放。主 checkpoint：
`data/h3wam_checkpoints/libero_goal_task1_33ep_h3_feature_action_horizon1_strict_proprio_phase_gripper10_5000_best.pt`。

后续严格对照没有替换该基线：Horizon1 event-stage mixture 的最佳 validation loss
为 0.0544，但 trial0 未打开抽屉；去掉 phase 后 validation loss 降到 0.0269，6-trial
仍为 2/6（trial4/5），与 phase 模型的成功集合互补但总成功率相同；两个动作头直接
平均仅 1/6。加入 previous-action 后 validation loss 进一步降到 0.01795，并在抽测的
trial0/4/5 中 **3/3 打开抽屉**，但 0/3 完成放碗，说明动作历史改善了第一子技能，
也造成难以退出的自回归惯性。step100 固定切回 phase 头为 1/3，尚不如原基线。

因此当前推荐模型仍是 phase 单头 2/6。下一轮需要基于可观测抽屉状态的事件 gate，
或者采集开抽屉后抓碗/放置失败状态的人工纠偏数据；不再以更低离线 MSE 作为晋级依据。

### 成功主线上的 H3 LoRA（2026-08-05）

已将可微 H3 层特征接到 Horizon1 ActionDiT，使动作回归损失能直接训练 H3 attention
LoRA；部署端也会从 feature-action checkpoint 自动注入并加载 LoRA。rank4、最后 10 个
H3 block 共 1.58M LoRA 参数，联合动作头训练峰值 20.56 GiB、约 0.31 秒/step；冻结
动作头的保守训练峰值 20.50 GiB、约 0.50 秒/step。5090 因此已经实证可以稳定训练并
部署当前 H3 feature-policy LoRA。

联合训练 300 step 把 32 个留出窗口损失从 0.04465 降到 0.02109，但 task3 trial0
从成功退化为失败且没有打开抽屉。冻结动作头、LoRA 学习率 1e-5 的最佳点仅把离线损失
从 0.04478 降到 0.04393；闭环为 1/6，低于冻结 H3 基线的 2/6。step0 LoRA 对照仍在
190 步成功，排除了部署加载错误；step10 在原成功 trial4 能打开抽屉并移动碗，但没有
完成放置。

这轮实验说明当前瓶颈不是显存或 LoRA 实现，而是纯专家 BC 缺少策略偏离后的抓取/放置
纠偏状态。LoRA checkpoint 暂不晋级；收集 corrective rollout 后再复用该训练链路，且
以固定 6-trial 闭环成功率而不是离线 MSE 选 checkpoint。

### Future-video LoRA 与配套动作头（2026-08-05）

将 LoRA 目标改为纯未来视频 flow matching，不再用 action MSE 直接扭曲 H3。rank4、
最后 10 个 block 的 attention+FFN 共 3.73M 参数；500-step 训练把 20-window 留出视频
loss 从 0.2736 降到 0.2221（约 18.8%），final-layer 首帧 token cosine drift 仅
0.0112，5090 峰值 20.50 GiB。旧 Horizon1 head 直接搭配新特征仍为 2/6，说明动作头
必须随 H3 表征共同适配。

FastWAM/DiT4DiT 风格的 36.77M Horizon8 flow ActionDiT 使用每观测 4 组独立噪声、
Beta time sampling 和 4-step ODE；5000-step 仅需 143 秒、峰值 2.11 GiB，但
previous-action 与 phase 两个版本在原成功 trial0/4 均为 0/2。20-step ODE 也没有
恢复成功，因此该 H8 flow checkpoint 不晋级。

随后按训练—部署一致性重新缓存全部 6669 个 Horizon1 video-LoRA 特征，并重训原已
验证的 5.72M proprio+phase regression head。best validation loss 为 0.02397；正确
task3 固定 trial0～5 中 trial3/4/5 成功，达到 **3/6**，超过冻结 H3 基线的 **2/6**。
保持同一动作头并在部署时关闭 video-LoRA 后，trial0～5 只有 trial4/5 成功，为
**2/6**；唯一变量消融使 trial3 从成功变失败，说明新增成功确实依赖 future-video
目标适配后的 H3 表征，而不是仅由动作头重训产生。扩大到固定 trial0～9 后，两组在
新增 trial6～9 均为 0/4，因此累计为 **video-LoRA 3/10、关闭 LoRA 2/10**。提升仍保持
一个绝对成功 trial，但 10 次样本不足以宣称统计显著或稳定策略；结论限定为“存在正向
闭环信号，值得进入扩大数据/80GB 训练”，而不是已经解决 LIBERO。

进一步在 video-LoRA 原成功的 trial3/4/5 上保持所有条件不变，仅将送入动作头的 H3
feature 置零，结果从 **3/3 降为 0/3**，且三个初始状态的 top-drawer joint delta 均为
0。由此可排除该动作头只依赖 phase/proprio 进行开环轨迹回放；成功动作确实需要 H3
视觉特征。该消融结果位于
`data/h3wam_rollouts/h3_videolora500_retrained_feature_action_h1_phase_task3_trials3_4_5_zero_feature/`。
这是当前 video-objective H3 LoRA 主 checkpoint：
`data/h3wam_checkpoints/libero_goal_task3_h3_videolora500_feature_action_h1_proprio_phase_gripper10_5000_best.pt`。
结果位于 `data/h3wam_rollouts/h3_videolora500_retrained_feature_action_h1_phase_task3_trials0_4/`
、`data/h3wam_rollouts/h3_videolora500_retrained_feature_action_h1_phase_task3_trials1_2_3_5/`
和 `data/h3wam_rollouts/h3_videolora500_retrained_feature_action_h1_phase_task3_trials6_7_8_9/`；
对应关闭 LoRA 的目录名带 `_without_lora`。

### 与官方 FastWAM 的同状态对照（2026-08-05）

官方 `libero_uncond_2cam224.pt` 已完整下载并通过 SHA256
`1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`；本地 5090 可加载
5.00B video expert、1.02B action expert、T5 和 VAE。修复 `eval_official_fastwam.py`
通过 `runpy` 启动时缺少 evaluator sibling import path 后，官方模型在 `libero_goal`
task3、seed42、trial0～9 的标准 400-step 闭环评测为 **7/10**，成功集合为
`{0,1,2,4,7,8,9}`。结果和视频位于
`data/h3wam_rollouts/official_fastwam_task3_trials0_9/`。

将 H3 future-video LoRA 改用相同 seed42、相同 trial0～9 和 400 控制步预算后，成功率为
**4/10**，成功集合 `{0,3,4,7}`；同一个动作头关闭 video-LoRA 后仅 **1/10**，成功集合
`{1}`。因此在该协议中 video-objective H3 适配带来净 **+3/10**，并产生 4 个 LoRA-on
独有成功。10 次配对仍不足以宣称统计显著，但结合成功 seed 上 zero-feature 3/3→0/3，
已经能证明 H3 表征和 future-video LoRA 都对闭环动作有因果贡献，而不是 phase/proprio
动作头单独回放。

H1 H3 当前仍低于官方 FastWAM 的 7/10，不能替换它作为主策略；但两者成功集合并不相同，
H3 的 trial3 是官方模型失败状态，而两者并集覆盖 9/10（仅 trial6 均失败）。这促使我们
优先验证 FastWAM teacher corrective data，而不是继续盲目增加纯专家 BC steps。

已打通 5090 两阶段 corrective pipeline：`rollout_libero.py --save-trajectories` 保存每次
重规划的原始双相机图像、proprio、前一动作与 H3 动作；卸载 H3 后，
`label_fastwam_teacher.py` 单独加载官方 FastWAM，为这些状态离线输出 32-step teacher
action。FastWAM relabel 峰值 23.26 GiB，实测约 5.7 states/s。双方都失败的 seed42/trial6
已采集 400 个状态并全部标注，位于
`data/h3wam_rollouts/h3_videolora500_corrective_trial6_seed42/`。

该失败轨迹上 H3/teacher 首动作 motion L2 均值 0.435；steps100～149 上升到 0.819，
最大分歧集中在 steps103～127。teacher 在 step310 切换夹爪，H3 到 step333 后才开始且
反复抖动，steps300～349 的夹爪分歧率为 70%。`select_corrective_states.py` 以 motion
分歧 top25%、全部夹爪分歧及前后 2 帧邻域筛出 195/400 个高价值状态，完整保留 48 个
夹爪分歧状态。195 个状态已完成 feature cache 和动作头微调，但 trial6 上 teacher 自身也
失败，因此该轨迹只能验证纠偏基础设施，不能提供通向成功的监督。

随后只使用「官方 FastWAM 成功、H1 H3 失败」的 trial8/9，共筛出 255 个 corrective
状态。H1 step25/50/75 的纠偏离线损失从 0.160 依次降到 0.142/0.128/0.115，但三档在
trial8/9 都是 0/2；step75 还出现同时推碰多个物体。点式 teacher 首动作拟合不能解决
闭环恢复，离线纠偏损失继续降低反而会破坏原专家分布。

### Horizon8 动作块突破（2026-08-05）

核对官方 evaluator 后确认 FastWAM 每次预测 32 步、连续执行 10 步再重规划，而先前 H3
成功头仅训练和执行 1 步；旧 corrective cache 也只保留 teacher chunk 的第 1 步。原 H1
checkpoint 实际来自 `manifest_h8.jsonl`（6438 windows）但训练参数为
`action_horizon=1`；用零学习率复核时 `manifest_h8` 验证损失 0.0267 与旧记录一致，
`manifest_h1` 的 0.0922 是错误对照。

因此保持 H3 video-LoRA、proprio、phase 和 gripper-weight10 不变，直接在同一 33 条专家
轨迹上训练 Horizon8 regression head。最佳 step4600 的 H8 验证损失为 0.04399；在 task3、
seed42、trial0～9、400-step 协议下，以 8-step chunk 执行得到 **5/10**，成功集合
`{0,4,7,8,9}`。H1 为 4/10 `{0,3,4,7}`，因此 H8 净增 1 次，并新增 FastWAM 成功而
H1 失败的 trial8/9。结果位于
`data/h3wam_rollouts/h3_videolora500_feature_action_h8_regression_step4600_trials0_9_seed42/`。
按相同 trial0→9 顺序独立复测仍为 5/10，且成功集合、成功步数和物体位移逐项一致；复测
结果位于同名目录加 `_repeat2`。单独抽取某个 trial 运行时可能与整组结果不同，因为官方
LIBERO evaluator 和本实现都复用一个 seeded env，前序 episode 会推进环境 RNG；因此
晋级数字只采用固定顺序的完整 10-trial 评测，单 trial 仅作诊断。

关闭 H3 video-LoRA 的完全配对 H8 对照为 **3/10**，成功集合 `{4,5,9}`；LoRA-on 新增
`{0,7,8}`、丢失 `{5}`，净提升 +2。H8 每次策略往返约 0.15 秒、峰值 24.96 GiB，且每个
400-step episode 只需 50 次 H3 调用。当前最佳 H3 由 4/10 提升到 5/10，仍低于官方
FastWAM 7/10，但已经确认动作块时间一致性是此前主要瓶颈之一。

纠偏缓存现支持 `--action-horizon 1..32`，不再截断 teacher chunk。针对 H8 失败、FastWAM
原始 rollout 成功的 trial1，50 个重规划状态中筛出 26 个高分歧状态并保留完整 8-step
teacher 动作；轻量微调把专家验证损失从 0.0441 降到 0.0425，但 trial1 仍失败。这说明
teacher 在 H3 已偏离状态上的点式动作不保证后续可恢复；下一轮需要 teacher
intervention/roll-in 产生连续可达的恢复轨迹，而不是继续提高单状态 sample weight。

### 连续 teacher roll-in 与阶段化策略（2026-08-05）

`rollout_libero.py --save-trajectories` 现额外保存每个重规划点的完整 79 维 MuJoCo state；
新增 `rollout_fastwam_teacher_intervention.py`，可在独立进程恢复任意 H3 状态并让官方
FastWAM 连续接管，避免两模型同时驻留超过 5090 显存。trial1 从首个持续高分歧区间
index9（控制 step72）接管后，FastWAM 用 14 次重规划在 **step182 成功**，证明该状态
真实可恢复，并产出 14 个连续、可达、带完整 32-step teacher chunk 的状态。

将这 14 个状态缓存为 H8，基于原 H8 head 仅训练 recovery head 25 steps（lr=1e-5）；
全程直接换成 recovery head 会破坏前半段开抽屉，而保留 base head 到 step72、之后切换
recovery head，可在 H3 闭环中同样于 **step182 完成 trial1**。这验证了失败恢复监督必须
配合其有效阶段，不能作为全局 BC 样本无条件混合。

阶段化策略在固定 task3、seed42、trial0→9 上达到 **9/10**，成功集合
`{0,1,2,4,5,6,7,8,9}`，仅 trial3 失败；第二次完整独立复测的成功集合、完成步数与物体
位移逐项相同。它相对基础 H8 的 5/10 新增 `{1,2,5,6}` 且未丢失原成功，超过官方
FastWAM 的 7/10。主结果位于
`data/h3wam_rollouts/h3_videolora500_h8_teacher_rollin_only25_switch72_trials0_9_seed42/`，
复测目录加 `_repeat2`。

严格消融结果为：保持双动作头和 step72 切换不变，关闭 video-LoRA 得 **8/10**；将 H3
feature 全部置零则 **0/10**，而且 10 个 trial 基本都无法打开抽屉。因此 H3 视觉表征是
闭环成功的必要条件，连续 teacher roll-in/H8 提供动作时间一致性，video-LoRA 再提供
净 +1/10。当前 9/10 仍是单任务、人工固定切换点的实验室结果；下一步必须把 step72
替换为视觉/状态触发器，并在更多 LIBERO 任务验证泛化。此外部分成功轨迹会碰动非目标
物体，扩展任务时应增加安全/扰动指标，不能只看 LIBERO success predicate。

### H3 视觉状态 gate（2026-08-05）

固定 step72 能验证恢复头，但不具备状态泛化。先尝试 base/recovery action-chunk 分歧触发：
固定策略遥测显示早期 motion disagreement 为 0.12～0.19，后期升至 0.23～0.30；然而
阈值0.20往往到 step104～128 才触发，完整评测仅 4/10，说明动作明显分歧是滞后信号，
不能代表恢复入口。

随后新增 700,545 参数的 `H3FeatureSwitchGate`，输入为 pooled H3 feature 和归一化 8-D
proprio，明确不输入 step/phase。用 6438 个专家 cache windows 监督“是否进入原 step72
之后的操作阶段”，最佳验证 accuracy 96.97%。部署采用一次触发后锁存 recovery head：
默认阈值0.5得到 8/10，trial6 因 step72 概率0.101未触发而失败；校准阈值为0.10后恢复
到 **9/10**，第二次完整复测仍为 **9/10**。

learned gate 的实际触发步随状态变化：trial0/1/3 为80，trial2/5/6/7为72，trial4/8/9
为64；它不再读取 rollout step。当前结果与固定切换同为9/10，但消除了部署时硬编码阶段
时间。gate 的监督标签仍由单任务 step72 派生，因此下一阶段要在多个任务上用 teacher
intervention 成功区间产生状态标签，验证跨任务泛化。

### 第二任务验证：task0（2026-08-05）

在 `open the middle drawer of the cabinet` 的 42 条专家轨迹、4479 个 base-H3 特征窗口上，
按 task3 的同一 proprio+phase、Horizon8 regression 配方训练 5000 steps。离线验证损失最优
的 step700 闭环只有 **4/10**；最终 step5000 达到稳定 **8/10**，两次完整、固定顺序复测的
成功集合均为 `{0,2,4,5,6,7,8,9}`。这再次证明 checkpoint 必须按闭环 rollout 选择，不能
只按离线 validation loss 选择。

对 step5000 做完全配对的 zero-feature 消融后从 **8/10 降为 0/10**；因此 task0 的成功同样
依赖 H3 视觉表征，而不是 proprio/phase 轨迹回放。官方 FastWAM 在相同 seed42、trial0～9、
400-step 预算下为 **10/10**。H3 已接近 teacher，但基础头仍在 trial1/3 失败。

从两条失败轨迹的 step64 保存 MuJoCo 状态后，官方 FastWAM 连续接管均可成功：trial1 在
step125 完成，trial3 在 step124 完成，各产生 8 个可达 teacher 状态。合并 16 个状态，以
lr=1e-5 微调 recovery head 25 steps，并在 step64 后切换，完整评测提升到稳定 **9/10**，
成功集合 `{0,1,2,4,5,6,7,8,9}`，独立复测逐项一致。trial3-only 继续训练至 50 steps 仍
不能闭环恢复且会损害 trial1，说明剩余问题是 recovery 状态覆盖不足，不是简单欠拟合；
下一轮应缩短 teacher roll-in 的状态间隔，而不是继续压低 8 个状态上的训练损失。

task0 没有使用视频 LoRA，因为训练 cache 本身来自 base H3。corrective 预计算脚本现支持
可选 `--h3-lora-checkpoint`，可在 base-H3 与 LoRA-H3 两条路径中保持训练/部署一致。晋级
配置位于
`data/h3wam_checkpoints/libero_goal_task0_h3wam_h8_staged_teacher_rollin_policy.json`。

### 第三任务最终确认：push plate（2026-08-05）

按完整任务文本选择 `push the plate to the front of the stove`；该文本在当前
LIBERO benchmark 中解析为 task id 5，不能直接使用 LeRobot 数据中的
`task_index=2`。本地 33 条专家轨迹形成 4877 个 dense windows，H3 特征已全量缓存。

复用 task0 已验证的单头配方：Horizon8 regression、proprio+phase、gripper loss
weight 10，训练 5000 steps。训练耗时 106.7 秒，峰值显存 0.32 GiB，最终/最佳
validation loss 分别为 0.02968/0.02840。闭环选择仍不使用离线 loss 排名替代 rollout。

在固定 seed42、trial0→9、400-step、每 8 步重规划协议下，最终 checkpoint 得到
**10/10**；第二次独立完整复测仍为 **10/10**，每个 trial 的完成步数逐项一致：
`[136, 177, 139, 120, 134, 135, 139, 153, 148, 132]`。保持所有条件不变、只将送入
动作头的 H3 feature 置零后为 **0/10**，10 个 episode 均耗尽 400 steps。

主 checkpoint：
`data/h3wam_checkpoints/libero_goal_task2_h3_feature_action_h8_regression_proprio_phase_gripper10_5000.pt`。
结果目录：

- `data/h3wam_rollouts/h3_task2_h8_regression_baseh3_step5000_trials0_9_seed42/`
- `data/h3wam_rollouts/h3_task2_h8_regression_baseh3_step5000_trials0_9_seed42_repeat2/`
- `data/h3wam_rollouts/h3_task2_h8_regression_baseh3_step5000_trials0_9_seed42_zero_feature/`

至此 H3 video-expert + H8 action head 已在三个 LIBERO 任务得到正向闭环结果，其中
第三任务无需 teacher recovery 即达到稳定 10/10，且 zero-feature 0/10。该结果通过
进入 BF16 主干适配阶段的实验室 Go 门槛；它仍不是完整 LIBERO 多任务 benchmark。

### 云端 official-H3 task3 复核与对象中心对齐方向（2026-08-08）

在 33 episodes / 5646 dense windows 的 official BF16 H3 特征上，H8 单头在前四个
初态为 1/4；四阶段 mixture 在两个初态均失败。新增的单调阶段路由证明分类顺序本身
正确，但切分后的第一阶段专家仍不能稳定开抽屉，因此不再扩测 mixture。

同一缓存直接训练 Horizon1 单头 5000 steps，最佳验证损失为 0.01581。固定 trial0
中，final5000 打开顶层抽屉（最大位移 0.160）但未接触碗；best3500 同样打开抽屉并
转向碗（碗位移 0.954），同时误碰酒瓶（0.756），最终均未成功。失败已从阶段路由
收敛到目标对象定位、干扰物区分和精细几何控制，不再通过扫 checkpoint 处理。

下一轮采用 SAM3D 风格的训练期对象中心表示对齐：LIBERO 先使用仿真对象 mask，按
“抽屉/碗”子任务选择目标；冻结 SAM3D 教师，将 mask 内池化后的 FP16 教师向量与 H3
观测 token 投影做归一化 MSE，并与动作损失联合训练 H3 LoRA/尾部。首轮只缓存池化
向量，不保存稠密 3D token；用相同数据、步数和 trial0/4/5 做有/无对齐 A/B，晋级
条件是成功率提高且不能退化已经稳定的 task0/task5。

## 当前产物

- 模型实现：`src/fastwam/models/h3wam/`
- 数据、训练、评估脚本：`scripts/h3wam/`
- 500-window cache：`data/h3wam_cache/libero_goal_500/`
- 推荐检查点：`data/h3wam_checkpoints/libero_goal_500_lora_r4_1000_best.pt`
- 联合训练检查点：`data/h3wam_checkpoints/libero_goal_500_joint_v02_r4_300_best.pt`
- 52 个 H3-WAM 单元测试，当前全部通过
- rollout 策略服务：`scripts/h3wam/serve_rollout_policy.py`
- LIBERO 闭环入口：`scripts/h3wam/rollout_libero.py`
- rollout 结果和视频：`data/h3wam_rollouts/`
- 当前 task3 最佳策略配置：`data/h3wam_checkpoints/libero_goal_task3_h3wam_h8_learned_gate_teacher_rollin_policy.json`
- 当前 task0 最佳策略配置：`data/h3wam_checkpoints/libero_goal_task0_h3wam_h8_staged_teacher_rollin_policy.json`
- 首个成功基线：`data/h3wam_checkpoints/libero_goal_task0_ep0_action_regression_h1_phaseonly_expanded_fixedctx_5000.pt`
- 成功 rollout：`data/h3wam_rollouts/regression_ep0_expanded_fixedctx_final_trial25/` 和
  `data/h3wam_rollouts/regression_ep0_expanded_fixedctx_final_trials36_45/`

## 下一步（按实验价值排序）

1. 固定 task3 9/10、task0 9/10、push-plate 10/10 及三个 zero-feature 0/10 为回归集，
   先在 A800 上复现相同 INT8 H3 特征与检查点；成功率允许最多相差 1/10。
2. 在四套 LIBERO 的 277,713 个 stride-1 训练窗口上训练统一动作模型；先做严格配对的
   H4/H8/H10、previous-action 和 2/4 帧历史筛选，不混入 H3 权重更新。
3. 保留 DoT depth-1/depth-4 一整 epoch 配对实验；只有离线动作与闭环同时改善才继续加深，
   不因 world/video loss 下降而晋级。
4. 用多个任务的连续 teacher roll-in/intervention 训练 recovery 与 learned gate，并记录
   非目标物体位移；禁止单点 relabel 伪装成恢复数据。
5. regression 先学稳定均值，再尝试 FastWAM 风格 flow action 和完整 action-chunk teacher
   distillation。DreamWAM 的结构化 future 作为辅助监督，不让它压过动作目标。
6. BF16 H3 只保留为参考/teacher。除非冻结 H3 的动作路线在 held-out 闭环已经饱和且有
   明确因果证据，否则不再解冻 H3 主干。

快速判定标准：统一策略在三任务回归集至少 26/30，并在至少两个未训练任务各得到
至少 1/10；否则不能宣称多任务泛化。任何分支若连续两个 checkpoint 的闭环 success、
接触谓词和目标推进均不改善，则停止该分支，而不是继续堆训练步数。
