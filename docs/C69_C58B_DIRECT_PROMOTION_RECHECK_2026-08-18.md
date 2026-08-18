# C69 与 C58b 直接晋级复核（2026-08-18）

## 结论快照

严格配对 LIBERO 直接复跑已经完成并通过全部晋级门：C69 action-only s20000 为 338/680（49.71%），C58b FastWAM s10000 为 295/680（43.38%），绝对提升 6.32 个百分点；C69 独赢 79 对，C58b 独赢 36 对，单侧精确 McNemar p=3.758e-5。聚合器逐项核验 1360 个结果合同、680 对轨迹首状态、对象关节、checkpoint/授权身份和证据 SHA 后发布 `PROMOTE_C69`。因此 C69 现在正式替代 C58b，成为本项目当前闭环 action/carrier endpoint 擂主。

此前历史审计的四项方向门虽已通过，但属于结果产生后的事后审计。现已用同一批次、同一不可变源码、同一任务清单和 fresh process 完整重跑 680 对，并逐项复现相同的成功标签与总结果，消除了跨批次执行环境疑问。由于 trials 33..49 已被用于模型开发与确认，本结论仍是直接端点晋级证据，不能宣称未见初始状态泛化。

直接复跑的第一个完整 block（trial 33，40 对）为 C69 22/40、C58b 18/40，C69 独赢 4 对且 C58b 独赢 0 对；它只承担机械 canary。最终结论来自全部 680 对，而不是该 40 对子集。

## 固定端点

- C69 action-only s20000：`/mnt/h3-wam/outputs/c69-matched-action-only-v1/online-long20000-v1/checkpoints/c69_action_only_s20000.pt`
  - SHA256：`20914729d340b05768ec99e152cc026313d5a0dab064c963df90ac8184d8a12a`
- C58b FastWAM s10000：`/mnt/h3-wam/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt`
  - SHA256：`2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541`
- 不可变 rollout 源码：`/mnt/h3-wam/code-snapshots/h3-wam-8518821-rollout-v1`
- 控制与聚合代码 commit：`a8667ab`

## 历史 680 对审计

| Suite | C69 | C58b | C69-C58b | C69 独赢 / C58b 独赢 |
|---|---:|---:|---:|---:|
| LIBERO Spatial | 114/170 | 92/170 | +12.94pp | 34 / 12 |
| LIBERO Object | 149/170 | 131/170 | +10.59pp | 25 / 7 |
| LIBERO Goal | 40/170 | 44/170 | -2.35pp | 6 / 10 |
| LIBERO 10 | 35/170 | 28/170 | +4.12pp | 14 / 7 |
| 总计 | 338/680 | 295/680 | +6.32pp | 79 / 36 |

配对身份中，640 对通过完整轨迹首状态摘要逐字节一致；trial 33 早期 C58b 未保存轨迹，剩余 40 对通过 episode seed、trial 与完整 `initial_object_joints` 一致确认。审计输出位于：

`/mnt/h3-wam/eval/c69-vs-c58b-direct-paired680-trials33-49-d8e1bdb-v1/RETROSPECTIVE_RESULTS.json`

## 已完成的直接复跑

- 根目录：`/mnt/h3-wam/eval/c69-vs-c58b-direct-paired680-trials33-49-d8e1bdb-v1`
- 授权：`AUTHORIZATION.json`
- 冻结任务：`jobs.jsonl`，1360 个单 episode 进程，组成 680 个严格配对状态
- 分片：5 台服务器各 8 张 A800；端口 32611、30907、32409、30234、30137
- 协议：LIBERO 四个 suite，各 10 个任务，trials 33..49；400 max steps，replan 8，action horizon 32，10 次模型求值
- 每个 pair 在同一 GPU 上依次执行 C69 与 C58b；每个 episode 使用独立策略进程，保存完整轨迹
- 自动聚合：五个 `SHARD_*_COMPLETE.json` 全部通过；已生成 `PAIR_EVIDENCE_DIRECT.jsonl`、`RESULTS_DIRECT.json` 与 `COMPLETED.json`

晋级门固定为：整体至少 +3pp、净独赢至少 20、单侧配对 p 不大于 0.05、所有 suite 的回退不低于 -3pp。C69 分别得到 +6.32pp、净独赢 43、p=3.758e-5、最差 suite 为 Goal -2.35pp，四门全部通过，正式晋级。Goal 回退仍是下一轮必须单独解决的短板。

本地冻结证据位于 `experiments/evidence/c69_vs_c58b_direct_paired680_20260818/`：

- `RESULTS_DIRECT.json` SHA256 `ab78350825ff9de66e46154b6a6853695dd15bf0c4ebe5353041ef6ed79bf0f8`
- 680 行 `PAIR_EVIDENCE_DIRECT.jsonl` SHA256 `2eb062c781341d99e62f0c36b6aee11043cfe15758253092a93a27e1ed3fbba2`
- `AUTHORIZATION.json` SHA256 `59a3f079e971af642b67e665f2bf01f9c8abca7216cbf700f2a435e57f7d3e31`

## 中断与恢复

本轮 launcher 会拒绝覆盖已有输出，不能直接在原 root 重启某个分片。若云服务器被回收，保留 root 中所有已完成的 `results.json` 与轨迹作为部分证据；恢复时应生成只包含缺失 job 的新补跑清单，继续绑定同一个授权、端点哈希和源码快照，完成后再做统一聚合。不得把部分配对结果当作 680 对最终晋级结论。

相关脚本：

- `scripts/h3wam/prepare_c69_c58b_direct_paired680.py`
- `scripts/h3wam/launch_c69_c58b_direct_paired680_shard.sh`
- `scripts/h3wam/audit_c69_c58b_retrospective_paired680.py`
- `scripts/h3wam/aggregate_c69_c58b_direct_paired680.py`
- `scripts/h3wam/watch_c69_c58b_direct_paired680.sh`
