# H3 R1 A2/F2 条件坍塌证据

日期：2026-08-14  
效果结论：`NOT_EVIDENCE_READY`  
机制门：baseline v2 与 Candidate F v2 均为 `FAIL_CONDITIONING_COLLAPSE`

## 一句话结论

在相同的 balanced-80 episode-disjoint 合同下，baseline s913 与 Candidate F s850
虽然继续降低了物理动作回归误差，却几乎不再响应样本特定 H3 视觉特征和替换语言，同时夹爪指标
退化；紧接着的 step914/step851 又分别出现视觉 projector 梯度精确为零，因此不能把这两条曲线
解释成“多训练有效”，更不能靠增加 steps 或放宽梯度门晋级。

## 固定身份与协议

- 源码 commit：`a57a72bfa4b6c240d07e8a579469153e4e639d4d`。
- trainer SHA256：`f98cf0a424f1e9fbeef94b737d7cac7d4212ceb041db9a999e1d3a4189085a78`。
- evaluator SHA256：`8a7c85fdc8c4b45e534f29b44641b7e0e044aa21002d15822605a9e5d17086e6`。
- 同一 40 tasks × 2 windows，共 80 个 episode-disjoint 样本；selected IDs SHA256：
  `75d888fbb4298bef3517b623c00861ac6fe036495dee3bf4f0c68b5c097c5f54`。
- seed `42`、action shift `5`、10-step Euler、相同噪声；visual shuffle 为无 self-map 的固定置换；
  language sensitivity 只替换文本，保持视觉、proprio 和噪声不变。
- source/train/val manifest SHA256 分别为
  `5a5f605ed1607a38c22a6cdb892d660bf7a8e046eb2e69af62aa8a035ae5f5d8`、
  `a4ad2a7955f539c2f709912d423bfa892688885ed14144a9988595de14b8e78c`、
  `df0c6ab6efce8a89e5c249548a17a34f386b388b614b417ec531fb10113a4fa6`。
- 所有四个被比较 checkpoint 的 fresh restore `max_abs=0`。

## A2：baseline v2 s100 → s913

原始证据目录：
`/mnt/h3-wam/r1-baseline-v2-s1000-a57a72b-20260814/evidence/a2_balanced80_s100_s913_20260814`

- comparison：`comparison_summary.json`，SHA256
  `5358862e005b65ee634f7af440027751adf69c0d85f6bce8d128e4fe03c260bc`。
- s100 result SHA256：`f90f37f89554197290e329304cac678b917ee6f63956ea206bbb67973ad4b1b6`；
  checkpoint SHA256：`6ca9ea9d1713b6024bc07407b84f38a02bfb839de62d1725d7650f967813cfc6`。
- s913 result SHA256：`cf1961a2cc83decc2cc3e2cc6e12d30714daaf6814d1e115de97a78ca5797967`；
  checkpoint SHA256：`30dd7809f8a630b930b8202e73f220ba76e9b85d1aa0e6d21d041f5f76a10fb4`。

| 指标 | s100 | s913 | s913 相对变化 |
|---|---:|---:|---:|
| physical MSE | 0.343551 | 0.124351 | -63.80% |
| physical MAE | 0.447918 | 0.225720 | -49.61% |
| physical ADE | 1.499122 | 0.876306 | -41.55% |
| physical endpoint | 1.591821 | 0.832603 | -47.69% |
| gripper accuracy | 0.582451 | 0.626984 | +7.65% |
| gripper F1 | 0.562990 | 0.394850 | -29.87% |
| gripper macro-F1 | 0.581622 | 0.562626 | -3.27% |
| language mean-abs delta | 0.433882 | 0.019489 | -95.51% |
| language RMS delta | 0.557499 | 0.031465 | -94.36% |
| visual-shuffle action MSE delta | 0.236696 | 4.096e-7 | 只保留 0.000173% |
| visual-shuffle ADE delta | 1.179993 | 0.000477 | 只保留 0.040413% |

s913 的 headline physical error 显著更低，但视觉置换几乎不改变输出，语言替换响应也下降超过
94%；gripper accuracy 上升不能抵消 F1 明显下降，后者与类别不平衡下退化到动作先验相符。

训练在下一个尝试步 step914 fail closed：`expert=6.331165`、`feature_projector=0`、
`proprio=0.017414`。原始日志：
`/mnt/h3-wam/r1-baseline-v2-s1000-a57a72b-20260814/logs/stage_s100_to_s1000.log`，
SHA256 `b004e7b5e42d0f39c948492e5ea778105c45c31d1d7f70da09958745e1dea88a`。

## F2：Candidate F v2 s100 → s850

原始证据目录：
`/mnt/h3-wam/experiments/candidate-f-s1000-a57a72b-v2/evidence/f2_balanced80_lang_visual_shuffle`

- comparison：`F2_COMPARISON.json`，SHA256
  `1ccf51afd6ea17174d4046c49f4b8d811ee252a64282f1122c5a48a20123d9ce`。
- s100 result SHA256：`acd5566a80bef1b83171c6eef4ac04decdf85b1708f1a8fe1c20e1918f49f85c`；
  checkpoint SHA256：`759bdcba3459d094dfb27ae6ec52cf96de19a004994ab40088df24009100b1e2`。
- s850 result SHA256：`bd9ae1410ae6f90bfb066edbf350577703bc38ffe152bab528c58e34b578c331`；
  checkpoint SHA256：`4539e3eeaaea2fcb5fdbbe34306432ffe535635c977b80a02e5f08793f0e2a74`。

| 指标 | s100 | s850 | s850 相对变化 |
|---|---:|---:|---:|
| physical MSE | 0.352986 | 0.202512 | -42.63% |
| physical MAE | 0.455056 | 0.295677 | -35.02% |
| physical ADE | 1.517437 | 1.143283 | -24.66% |
| physical endpoint | 1.628583 | 1.052034 | -35.40% |
| gripper accuracy | 0.555115 | 0.428571 | -22.80% |
| gripper F1 | 0.594942 | 0.593476 | -0.25% |
| gripper macro-F1 | 0.550772 | 0.316026 | -42.62% |
| language mean-abs delta | 0.415041 | 0.032469 | -92.18% |
| language RMS delta | 0.525571 | 0.052523 | -90.01% |
| visual-shuffle action MSE delta | 0.209107 | 9.849e-7 | -99.999529% |
| visual-shuffle ADE delta | 1.101472 | 0.000956 | -99.913202% |

s850 同样以更低的 generic regression error 换取了近乎完全的视觉不变性，且夹爪 macro-F1
严重退化。step851 的 `expert=40.550217`、`feature_projector=0`、`proprio=0.060809` 与离线
趋势互证。失败审计：
`/mnt/h3-wam/experiments/candidate-f-s1000-a57a72b-v2/output/FAILURE_STEP851_ZERO_VISUAL_GRADIENT.json`，
SHA256 `5dba18333321373dde157fd32ec801069ab7d34c2edf1d97eee0c3477a2c0818`。

## 决策与后续门槛

1. baseline v2 与 Candidate F v2 均判定 `FAIL_CONDITIONING_COLLAPSE`；不进入 rollout，不能宣称
   H3 条件动作能力提高。
2. 禁止仅把 s913/s850 继续增加 steps，也禁止把视觉梯度必须 finite/nonzero 的门改成 warning。
   单个 batch 的零梯度本来不足以判死，但这里还有两套独立配方、配对反事实敏感度曲线和 gripper
   退化共同支撑，不能按偶发 batch 处理。
3. 下一候选必须是新的、单变量、可证伪合同；在较早与较晚 checkpoint 同时报 physical、gripper、
   visual shuffle、language replacement，并证明条件敏感度没有被 generic action prior 取代，才可
   重新申请昂贵训练或闭环。
4. 证据优先级固定为：作者/本地可执行代码与 resolved command → 原始 checkpoint/log/evaluator
   JSON → 论文解释。论文可帮助提出修复机制，但不能覆盖上述负实验或单独放宽门。

本地机器可读汇总：
`experiments/evidence/h3_r1_a2_f2_conditioning_collapse_v1.json`。
