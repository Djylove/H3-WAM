# C67/C69 fixed-s20 归因门

## 可证伪问题与结论边界

唯一问题是：在相同C58父节点、20k预算、样本顺序、优化器和评测协议下，C67的FACT
future-representation/state/value监督是否能在闭环中优于C69 action-only。C67是joint-objective候选，C69是
唯一直接父对照；两者都不是仅凭balanced80即可晋级的赛道冠军。

本链路不重新执行模型、不按离线误差挑checkpoint，也不输出离线winner。它只证明固定
`C67-s20000 vs C69-s20000`是否具备进入paired LIBERO归因评测的完整、conditioning-safe证据。

## 候选与来源

| 臂 | 赛道 | 固定来源 | 角色 |
|---|---|---|---|
| C67 FACT joint | consequence/action objective | FACT `618a6c16868699b6d4138941de6a863589ac00dd`；FastWAM `45d8e1458921d83f8ad6cf9ce993d371208dabd0` | 当前joint treatment；尚非该赛道冠军 |
| C69 action-only | action objective control | 与C67相同源码、模型和forward | C67的唯一同预算直接父对照 |

DreamWAM、WLA等项目只提供“world-on/off必须同预算归因”的方法依据，不作为第三臂混入本轮。C58仍是
当前已晋级carrier，历史C60、C55和D0不能替代C69。

## 唯一允许差异

cross-arm aggregator逐字段读取双方20个train report中的真实contract。contract键集合必须完全相同，且
不同值的字段集合必须恰好为：

| 字段 | C67 | C69 |
|---|---|---|
| `objective_mode` | `fact_joint` | `action_only` |
| `loss_weights` | `[10,1,0.4,0.4]` | `[10,0,0,0]` |
| `frozen_auxiliary_parameters` | `[]` | 六类future/value专属encoder/decoder参数 |

真实C67 checkpoint生成于这两个显式字段加入之前，因此只允许C67缺省`objective_mode`和
`frozen_auxiliary_parameters`；聚合器按当时执行语义分别规范化为`fact_joint`和`[]`，同时保留并绑定
未规范化的原始contract SHA。除这两个历史缺省外，键集合仍必须完全一致。

seed、4/2/1/1 rank混合、failure mask、初始化、数据七SHA、causal failure SHA、LR、warmup、cosine20k、
action horizon/shift、H3层、INT8 H3执行、normalization等任一额外差异均直接失败。

## 证据链

1. 双方分别完成20/20里程碑、160000训练样本和20次strict restore；每个embedded audit的所有gate必须为真。
2. C69 preview必须先有20/20 `PREVIEWS_COMPLETE.json`。`seal_c69_milestone_previews.py`逐个重算checkpoint
   SHA、比较preview audit与最终embedded audit，并把后者物化到sealed目录；`model_reevaluations_during_seal=0`。
3. C67继续复用已有sealer。双方sealed manifest、training-complete SHA、report SHA、restore-audit SHA和固定
   s20 checkpoint实际字节SHA必须一致。
4. 全部40份报告必须使用相同80 IDs、data对象、execution对象、seed42、shift5、10-step solver、
   normalization、visual shuffle映射和per-sample ID集合。
5. 输出只包含固定s20的normalized/physical paired error和双方condition safety。20个中间点只用于完整性与
   conditioning-collapse检查，不参与选择。
6. 只有双方20/20 conditioning gate全部通过、s20指标有限且上述身份完整，才输出
   `GO_C67_VS_C69_FIXED_S20_PAIRED_LIBERO_ATTRIBUTION`。该许可不启动rollout，也不说明C67或C69获胜。

训练许可已经由各自dossier独立给出：每臂global batch 8、20000 steps、160000 samples；C67约
`0.733522` effective epoch，C69同数据预算。C69预注册估计约8.82小时/8×A800；本归因seal/aggregate为
CPU/存储审计，不增加模型forward。

## 实现与执行

- C69零重评sealer：`scripts/h3wam/seal_c69_milestone_previews.py`
- 固定cross-arm gate：`scripts/h3wam/aggregate_c67_c69_fixed_s20_attribution.py`
- 一次性只读入口：`scripts/h3wam/launch_c67_c69_fixed_s20_attribution_gate.sh`
- fixture测试：`tests/test_c67_c69_fixed_s20_attribution.py`

训练和preview结束后，从新生成并独立复审的只读snapshot运行：

```bash
C67_C69_ATTRIBUTION_SOURCE_SNAPSHOT=/readonly/snapshot \
C67_C69_ATTRIBUTION_SOURCE_FREEZE_SHA256=<SOURCE_FREEZE_SHA256> \
C67_TRAIN_ROOT=/mnt/h3-wam/outputs/c67-c60-budget-ablation-v1/online-long20000-v1 \
C67_SEALED_ROOT=/mnt/h3-wam/outputs/<c67-preview>/sealed \
C69_TRAIN_ROOT=/mnt/h3-wam/outputs/c69-matched-action-only-v1/online-long20000-v1 \
C69_PREVIEW_ROOT=/mnt/h3-wam/outputs/c69-matched-action-only-v1/milestone-preview-55b622f-v1 \
C67_C69_ATTRIBUTION_ROOT=/mnt/h3-wam/eval/c67-vs-c69-fixed-s20-attribution-v1 \
bash /readonly/snapshot/scripts/h3wam/launch_c67_c69_fixed_s20_attribution_gate.sh
```

该入口不含trainer、`torch.distributed.run`、LIBERO或rollout调用。若结果给出GO，仍需另行生成固定paired
LIBERO manifest与独立授权；闭环结果之前效果状态始终是`NOT_EVIDENCE_READY`。
