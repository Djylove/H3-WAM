# C58 full50 rollout 断点恢复记录（2026-08-17）

## 边界

本记录只描述 LIBERO full50 descriptive rollout 的基础设施恢复，不读取或解释未完成评测的成功率，
不改变 C58/D0 checkpoint、任务、trial、随机种子或晋级结论。最终汇总仍必须等待两个 arm 的
`COMPLETED.json`，且该 full50 报告保持 `DESCRIPTIVE_ONLY / NO_NEW_PROMOTION_CLAIM`。

## 故障与根因

- 原始 candidate 在 1258/1320、control 在 1212/1320 后退出。
- 两个首个失败 episode 的 `policy_server.log` 均为 `OSError: [Errno 98] Address already in use`。
  `rollout_libero.py` 先查询空闲端口、再启动 server，中间存在并发占用窗口。
- 原 launcher 遇到已有输出目录会全部拒绝，不能跳过已经完成且带 `results.json` 的 episode；
  单个瞬时启动失败因此阻塞整个 arm 的安全续跑。

## 修复

- `fcc5212`：增加显式 `C58_FULL50_RESUME=1`、单 episode 最多三次启动尝试；失败尝试移动到
  `quarantine/`，不覆盖也不删除。
- `26c4ea3`：增加 `C58_FULL50_RESUME_REBALANCE=1`，在启动瞬间冻结缺失 episode 清单并重新
  round-robin 到 8 GPU。科学身份仍来自原始只读 `jobs.jsonl`，只改变运行时设备分配。
- `6900984`：launcher 在创建任务前检查 DreamWAM `layers.py / experts.py / mot.py` 存在且
  SHA256 与固定 commit 合同一致，防止不完整 runtime snapshot 进入重试循环。
- 本地 `bash -n` 与 `tests/test_c58b_full50_descriptive_eval.py` 均通过（4 passed）。

## 失败恢复尝试的处理

首个 `26c4ea3` runtime snapshot 漏挂 `third_party/DreamWAM`，8 个 control 启动均在模型构造前失败，
没有生成 `results.json`。进程被停止，所有半成品目录保留在
`quarantine/control_d0/`；它们不进入 completion 或最终 aggregator。

正式恢复使用只读快照：

`/mnt/h3-wam/runtime-snapshots/h3-wam-6900984-c58-resume-v3`

该快照已通过固定 DreamWAM source import/hash 检查。恢复时：

- control：108 个缺失 episode，8 GPU 重分配；
- candidate：62 个缺失 episode，8 GPU 重分配；
- 已完成的 2470 个 episode 保持原路径且不重新执行；
- finalizer 继续等待两个 arm 的完整 1320-episode 审计后才汇总。

截至恢复启动后的首轮检查，两个 arm 都有 8 个活跃 rollout，未触发新的 retry quarantine；
control 已从 1212 增至 1215。该计数仅表示文件完成进度，不是效果结论。
