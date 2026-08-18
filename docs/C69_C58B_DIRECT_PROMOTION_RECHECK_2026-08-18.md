# C69 与 C58b 直接晋级复核（2026-08-18）

## 结论快照

服务器回收前启动了 C69 action-only s20000 与 C58b FastWAM s10000 的严格配对 LIBERO 复跑。历史结果的跨实验字节审计已经闭合 680 对：C69 为 338/680（49.71%），C58b 为 295/680（43.38%），绝对提升 6.32 个百分点；C69 独赢 79 对，C58b 独赢 36 对，单侧精确 McNemar p=3.76e-5。

历史审计的四项方向门均通过，但它属于结果产生后的事后审计，且 trials 33..49 已被两个模型分别使用过。因此它是很强的直接配对证据，不是预注册证据，也不能宣称未见初始状态泛化。为消除跨批次执行环境疑问，已启动同一批次、同一不可变源码、同一任务清单的 680 对直接复跑。

直接复跑的第一个完整 block（trial 33，40 对）已经结束：C69 为 22/40，C58b 为 18/40，C69 独赢 4 对且 C58b 独赢 0 对。40 对的成功标签与两个模型各自的历史结果全部一致；已完成检查的首动作也逐元素一致，证明端点恢复和仿真协议可复现。该 block 仅作为机械 canary，不单独用于模型晋级。

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

## 正在运行的直接复跑

- 根目录：`/mnt/h3-wam/eval/c69-vs-c58b-direct-paired680-trials33-49-d8e1bdb-v1`
- 授权：`AUTHORIZATION.json`
- 冻结任务：`jobs.jsonl`，1360 个单 episode 进程，组成 680 个严格配对状态
- 分片：5 台服务器各 8 张 A800；端口 32611、30907、32409、30234、30137
- 协议：LIBERO 四个 suite，各 10 个任务，trials 33..49；400 max steps，replan 8，action horizon 32，10 次模型求值
- 每个 pair 在同一 GPU 上依次执行 C69 与 C58b；每个 episode 使用独立策略进程，保存完整轨迹
- 自动聚合：32611 上的 watcher 等待五个 `SHARD_*_COMPLETE.json`，随后生成 `PAIR_EVIDENCE_DIRECT.jsonl`、`RESULTS_DIRECT.json` 与 `COMPLETED.json`

晋级门固定为：整体至少 +3pp、净独赢至少 20、单侧配对 p 不大于 0.05、所有 suite 的回退不低于 -3pp。四门同时通过才允许把 C69 晋级为新的闭环冠军；否则保留 C58b，并针对 Goal 回退做下一轮单变量改进。

## 中断与恢复

本轮 launcher 会拒绝覆盖已有输出，不能直接在原 root 重启某个分片。若云服务器被回收，保留 root 中所有已完成的 `results.json` 与轨迹作为部分证据；恢复时应生成只包含缺失 job 的新补跑清单，继续绑定同一个授权、端点哈希和源码快照，完成后再做统一聚合。不得把部分配对结果当作 680 对最终晋级结论。

相关脚本：

- `scripts/h3wam/prepare_c69_c58b_direct_paired680.py`
- `scripts/h3wam/launch_c69_c58b_direct_paired680_shard.sh`
- `scripts/h3wam/audit_c69_c58b_retrospective_paired680.py`
- `scripts/h3wam/aggregate_c69_c58b_direct_paired680.py`
- `scripts/h3wam/watch_c69_c58b_direct_paired680.sh`
