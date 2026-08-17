# C67 最终证据独立复核

日期：2026-08-17。状态：`PASS_C67_FINAL_EVIDENCE_INDEPENDENTLY_REPRODUCED`。

本复核不训练、不运行模型推理、不选择 checkpoint、不修改正式输出，也不启动 LIBERO rollout。它只在正式
`TRAINING_COMPLETE.json`、preview `SEALED.json` 和固定20点 `RESULTS.json` 都原子落盘后，从另一份完整只读
SOURCE_FREEZE snapshot 执行。

## 固定输入与检查

- 训练根：逐段重新调用冻结 finalizer 的纯验证路径，检查 s1k..s20k 的20个训练报告、1000步连续 history、
  scheduler、前驱 checkpoint、`restore_at_load_max_abs=0`、30层梯度、future leak=0，以及20份独立
  `PASS_C56B_STRICT_RESTORE / restore_max_abs=0`。
- checkpoint：20个 full-state checkpoint 均重新读取并计算 SHA256；checkpoint 内容只通过 CPU mmap 审计，
  不实例化模型，不执行 forward。每个 SHA 必须与对应 preview audit、raw preview和sealed report一致。
- preview/seal：逐字节复核20份 preview audit、20份 raw preview、20份 sealed report及其 SHA manifest；sealed
  report必须能由raw report加固定provenance字段纯重构，`model_reevaluations_during_seal=0`。
- aggregate：使用冻结的固定20点 aggregator 纯函数重新计算 `RESULTS`，要求与正式文件逐字段相同；端点只能是
  s10000 matched control与s20000 treatment，其checkpoint和milestone-audit SHA必须与训练链一致。

## 执行边界

watcher必须提供`C67_FINAL_AUDIT_SOURCE_SNAPSHOT`和其独立复核过的
`C67_FINAL_AUDIT_SOURCE_FREEZE_SHA256`。它先全树验证只读snapshot，再等待三个正式marker，最后仅在新的
`C67_FINAL_AUDIT_ROOT`写入`AUDIT.json`和审计日志。输入缺失、身份漂移、已有输出目录或任何复算不一致均
fail closed；失败输出不能作为新一次审计复用。

成功状态固定为`PASS_C67_FINAL_EVIDENCE_INDEPENDENTLY_REPRODUCED`，权限固定为
`READ_ONLY_AUDIT_COMPLETE_NO_ROLLOUT_AUTHORIZATION`。即使正式`RESULTS`自身通过offline门，本审计也不代替
独立的rollout authorization，更不会自动启动rollout。

## 已部署 watcher

- 只读源码快照：`/mnt/h3-wam/code-snapshots/h3-wam-1561552-c67-final-audit-v1`
- git commit/tree：`15615527b5dcfd2ee0f4e2fa4347b5beefa25447` / `0c9dfdf45bb3691bc4a2432a66f2a631dc19e298`
- `SOURCE_FREEZE.json` SHA256：`7eaa799f3124fb7253cd9ae96f55e15126b6bb5ee5eb5d47c6f9d48ee2ff7fad`
- 全树验证：`PASS_COMPLETE_COMMIT_TREE_DYNAMIC_SOURCE_FREEZE`，`5478`个文件。
- watcher PID/log：`686587` / `/mnt/h3-wam/logs/c67-final-evidence-audit-1561552-watcher.log`
- 新输出根：`/mnt/h3-wam/outputs/c67-final-evidence-audit-1561552-v1`

部署后已复核watcher存活且全树验证通过；在正式三个marker到齐前输出根不存在，说明它仍处于纯等待状态，
没有提前写入审计结论。

## 最终审计结果

- `AUDIT.json` SHA256：`21a3c28567d04116770c01c2f092b15276b8bcc47b183da4a6a97c8bbe2a7b58`。
- 正式结果逐字段复算一致，仍为`FAIL_C67_BUDGET_BALANCED80_GATE / NO_C67_PAIRED_680_ROLLOUT`；正式
  `RESULTS.json` SHA256为`9a1c1bf4614a2ade58489465d6f0b24a2518d2f3f04f47aef998d477dde7dc3b`。
- 审计权限为`READ_ONLY_AUDIT_COMPLETE_NO_ROLLOUT_AUTHORIZATION`，没有运行模型forward、没有重评
  checkpoint、没有修改正式artifact，也没有启动LIBERO。
