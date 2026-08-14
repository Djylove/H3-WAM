# H3 R1 Candidate F：FastWAM flow + clean-chunk 回归互补预注册

日期：2026-08-14
状态：`COMPLETED / FAIL_CONDITIONING_COLLAPSE / NO_GO_LONG`
分类：`novel_composition`，不是 FastWAM 复现

## 结论先行

官方 FastWAM 动作策略是纯 flow matching：对完整动作 chunk 加高斯噪声，预测
`noise - clean_action`，再做 padding-aware、timestep-weighted MSE。它没有策略
teacher roll-in，也没有额外的 clean-action regression head。FastWAM 仓库中出现的
teacher forcing 位于独立 IDM，不是动作策略训练路径。

Candidate F 只在现有 R1 同一次 forward、同一批噪声上增加一个可选 clean-chunk
重建损失，不增加 head、不改变 H3 特征、数据、模型容量、噪声或采样器。默认权重为
`0.0`，因此历史 R1 checkpoint contract 和基础 loss 均保持不变。当前只完成源码审计、
显式配置、CPU 单测和 canary 预注册；本轮没有启动训练。

## 固定源码身份

| 来源 | 固定 revision | 用途 | 状态 |
|---|---|---|---|
| `yuantianyuan01/FastWAM` | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` | 主要动作目标、scheduler、数据与 rollout 合同 | 官方、可训练 |
| `Mondo-Robotics/DiT4DiT` | `66a6f3a12e2c8740157c1e478795952c040d31dd` | 独立交叉检查动作 flow 目标 | 官方、可训练 |
| `LiQiiiii/BadWAM` | `112630aa482eff43154c09e9af03339e72391d32` | 排除攻击/评测分支作为回归目标依据 | 官方，但不提供新策略目标 |
| 本地 R1 trainer | `scripts/h3wam/train_h3_int8_starwam_action.py` | H3 frozen-feature ActionDiT 实现 | 本地组合 |

本地 FastWAM 与 DiT4DiT vendor checkout 有 staged deletion/局部修改，不能把工作树
状态当作可信证据；本审计使用 `git show HEAD:<path>` 读取固定 commit 内容，未修改
vendor 仓库。

## FastWAM 源码合同

| 字段 | 固定实现证据 | 结论 |
|---|---|---|
| 动作形状 | `fastwam.py:301-323` | `[B,T,D]`，horizon 必须可被视频 transition 数整除；支持 `[B,T]` padding mask |
| proprio | `fastwam.py:352-366` | 使用当前状态 `proprio[:,0,:]` 加入上下文 |
| 加噪与目标 | `fastwam.py:470-477`；`scheduler_continuous.py:31-61` | shifted uniform timestep；`x_t=(1-sigma)x0+sigma*noise`；目标 `v=noise-x0` |
| 动作损失 | `fastwam.py:550-568` | token/action-dim MSE，先按有效 chunk token 聚合，再乘 timestep importance weight |
| 推理 | `fastwam.py:952-958`；scheduler `63-88` | 高斯动作初始化，Euler flow integration |
| LIBERO 数据 | `configs/data/libero_2cam.yaml:1-66` | 四套 LIBERO、dense stride 1、32 actions、9 video frames、7D action、8D state、min/max norm |
| 执行合同 | `configs/sim_libero.yaml:25-32`；`eval_libero_single.py:389-428,490-516` | 预测完整 32-step chunk；官方配置只执行前 10 步再 replan；默认 20 个 action inference steps |
| teacher roll-in | policy 代码无对应路径；`fastwam_idm.py` 才含 teacher-forcing mask | 不存在于 FastWAM policy 训练，不能作为 Candidate F 的上游依据 |

DiT4DiT 的独立实现也只做 flow：
`ActionDiT.py:268-318` 直接构造 `velocity=noise-actions` 并计算 mask 后平方误差；
其 LIBERO 配置使用 horizon 8、4 次 inference steps、100000 train steps。这一交叉检查
不提供 flow+regression 组合证据。

## Candidate F 的唯一变量

基础 R1 保持：

```text
x_t = (1 - sigma) * x0 + sigma * epsilon
v_target = epsilon - x0
L_flow = pinned StarWAM weighted masked MSE(v_pred, v_target)
```

同一次预测可恢复 clean action：

```text
x0_pred = x_t - sigma * v_pred
L_clean = masked MSE(x0_pred, x0)
L_total = L_flow + lambda_clean * L_clean
```

实现参数是 `--clean-action-regression-weight`，默认 `0.0`；预注册 canary 唯一改为
`1.0`。实现位于 trainer 的 `reconstruct_clean_action_from_flow()`、
`masked_chunk_regression_loss()` 和 `optimizer_step()`，没有新参数或第二次 forward。

必须诚实说明：

```text
x0_pred - x0 = -sigma * (v_pred - v_target)
L_clean = sigma^2 * unweighted_masked_MSE(v_pred, v_target)
```

所以它不是独立可辨识的新监督信号，而是在保留官方 weighted flow loss 的同时，额外
强调较高噪声/接近高斯初始化的训练位置。可证伪假设是：R1 的离线动作退化与从高噪声
起步的误差有关，这种重新加权能改善完整 chunk 动作质量；若没有改善就停止该线，不能
归因于“回归 head 不够大”。

## 配对 canary 预注册

父实验固定为 `h3_int8_starwam_split_native_full30_v1`。除
`clean_action_regression_weight: 0.0 -> 1.0` 外全部保持一致：

- source manifest：`/mnt/h3-wam/data/v8_multisuite_frameindexed_candidate/manifest_train_uniform.jsonl`，SHA256 `5a5f605ed1607a38c22a6cdb892d660bf7a8e046eb2e69af62aa8a035ae5f5d8`；
- train manifest：`/mnt/h3-wam/data/v8_multisuite_frameindexed_candidate/episode_disjoint_v1/manifest_train_episode_disjoint.jsonl`，SHA256 `a4ad2a7955f539c2f709912d423bfa892688885ed14144a9988595de14b8e78c`；
- frozen validation manifest：同目录 `manifest_val_episode_disjoint.jsonl`，SHA256 `df0c6ab6efce8a89e5c249548a17a34f386b388b614b417ec531fb10113a4fa6`；
- cache root：`/mnt/h3-wam/data/v8_frameindexed_h3_cache`；feature subdir：`h3_int8_libero40_last32_starwam`；
- seed 42、8 GPU、global batch 8、ActionDiT 30 层、H3 INT8 layer49 frozen features、shift 5、学习率/scheduler/clip 与 R1 相同；
- s1：offset 0、limit 8、1 step；s2-s50：restore s1、offset 8、limit 504、49 steps；s51-s100：restore s50、offset 512、limit 512、50 steps；
- 保存并 fresh-process restore 检查 s1/s50/s100；800 个 sample ID 必须唯一且跨 stage overlap 为 0；
- 总计 800 training samples，即 train split 的 `0.0031995520622672785` effective epoch；这只是配对 canary，不代表收敛。

对应标准 dossier：
`experiments/evidence/h3_int8_starwam_candidate_f_flow_regression_s100_v1.json`。
虽然 dossier 通过 `GO_CANARY` 机械门槛，本次按用户要求保持 `READY_NOT_LAUNCHED`。

## 评测与停止门槛

先在同一 frozen balanced-80（40 task × 2 episode-disjoint windows）、同一 sample ID 和
同一采样噪声上对比 Candidate F s100 与父 R1 s100：

1. normalized action MSE 和 chunk ADE 至少有一项严格改善，另一项不得退化；
2. gripper macro-F1 不得下降；
3. E1 visual-feature shuffle 完成后，Candidate F 的 visual-conditioning sensitivity 不得
   低于父 R1 的 80%，防止用动作先验掩盖 H3 视觉条件；
4. s1/s50/s100 均须 finite、三条 gradient path 为正、fresh restore max-abs 为 0；
5. 任一基础动作指标退化或视觉依赖坍塌即 `NO_GO_LONG`；只有离线机制门槛通过才允许
   固定 LIBERO closed-loop canary。

balanced-80 和 visual-feature shuffle 已完成。s100→s850 的 physical MSE
`0.352986→0.202512`、ADE `1.517437→1.143283`，但 gripper accuracy
`0.555115→0.428571`、macro-F1 `0.550772→0.316026`。visual-shuffle action MSE delta
从 `0.209107` 降至 `9.849e-7`，language mean-abs delta 从 `0.415041` 降至
`0.032469`；step851 随后出现 `feature_projector_gradient_norm=0`，而 expert/proprio
梯度仍为正。因此 Candidate F 判定 `FAIL_CONDITIONING_COLLAPSE`：不能宣称有效或闭环成功，
不得仅增加 steps 或放宽视觉梯度门。完整路径、SHA 和指标见
`docs/H3_R1_A2_F2_CONDITIONING_COLLAPSE_2026-08-14.md` 与
`experiments/evidence/h3_r1_a2_f2_conditioning_collapse_v1.json`。

## CPU 验证范围

`tests/test_h3_int8_starwam_action_trainer.py` 覆盖：

- CLI 默认关闭与显式启用；
- 正确 flow velocity 能精确恢复 clean action；
- `L_clean = sigma^2 * velocity_error^2` 且 padding 被忽略；
- 权重 0 时基础 flow loss 不变；权重 1 时 loss 精确相加且梯度发生变化；
- 负权重被拒绝。

这些测试只证明实现与数学合同成立，不是模型效果证据。
