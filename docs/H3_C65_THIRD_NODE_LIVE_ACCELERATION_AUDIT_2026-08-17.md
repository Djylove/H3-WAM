# C65 第三节点在线加速审计

日期：2026-08-17。结论：
`NO_GO_C65_THIRD_NODE_LIVE_ACCELERATION_LEGACY_UNCLAIMED_QUEUE`。

新节点 `30137` 的 8 张 A800 均空闲，且可见同一 `/mnt/h3-wam`。阻碍不是算力，而是已经运行的
C65 launcher 没有共享 job ownership 协议。

审计只读取 jobs/launcher 字节和文件系统元数据，没有打开任何 branch `results.json`、trajectory、log、
success 或 score。快照时冻结身份仍为：PREPARED SHA
`a883db26...e1158a`、jobs SHA `c9a13ede...b52cf`、launcher SHA
`8d36e280...54970`；342/3072 个 canonical output 完整，15 个目录正在写，2715 个尚未创建。

现有 launcher 的真实控制流是：

```text
results.json 非空且 trajectory 存在 -> skip
否则 output 目录已存在              -> worker return 1
否则                                -> mkdir -p output 后运行
```

因此不能无扰动热加第三节点：

- 直接写 canonical root：第三节点的 partial 目录会令旧 worker 失败；
- 只让第三节点拿 lock：n1/n2 从不读这个 lock，没有互斥效果；
- staging 后发布：若发布发生在旧 worker 的 absence check 与 `mkdir -p` 之间，旧 worker 仍会写同一目录；
  若旧 worker 已建目录，no-replace 只能放弃，不能加速；
- 取一个“足够远”的 tail：只能依赖预估墙钟，而 branch 时长与八个 cursor 都异步且没有冻结的硬下界，
  不能构成不碰撞证明。

要安全迁移，所有 active worker 必须从一开始共同执行原子 `claim/<ordinal>`，owner 写独立 staging，完整
审计后用 `renameat2(RENAME_NOREPLACE)` 发布，并重做 marker 计数。把该合同热加到第三节点不能保护两个
已经运行的 legacy launcher；迁移当前 root 必须协调静默点并重启 n1/n2，超出本次授权。

所以保持 n1/n2 不变，新节点未启动任何 C65 进程。机器可读证据位于
`experiments/evidence/h3_c65_third_node_live_acceleration_audit_v1.json`；审计器和三个 fail-closed 测试分别为
`scripts/h3wam/audit_c65_third_node_acceleration.py`、
`tests/test_audit_c65_third_node_acceleration.py`。
