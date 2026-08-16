# C60 FACT 扩展评测运行记录

日期：2026-08-16。目标是在不再修改模型或执行参数的前提下，将 C60 完整 FACT port 与不可变
C58b 父模型扩展至 LIBERO `trials33..49`，检验 trial33 的正向信号是否可复现。

## 放大依据与停止项

trial33 的严格结果位于
`/mnt/h3-wam/outputs/c56b-fact-online-v1/paired-final-eval-v2/fresh-execution-libero-trial33/RESULTS.json`
（SHA256 `fe4c7c49...17ebe2`）。C60/C61/C58b 分别为 `20/17/18` 成功；C60 对 C58b
为 3 胜 1 负且无 suite 下降 3 个成功，因此只放大 C60。C61 对 C60 为 0 胜 3 负，按预注册
门禁停止，不用更大评测为其补救。

## 不可变执行合同

- trials34..49、四个 suite、每 suite 10 task，共 640 个新 episode；加 trial33 后为 680 对；
- 每个 episode 启动全新的 simulator 与 policy 进程，八卡各顺序执行 80 个独立进程；
- `wait30/replan8/horizon32/eval10/max400`、seed `42+task*100000+trial*1000`；
- 不显式传 environment seed 或 policy-noise base；normalized pre-clamp，保存 trajectory；
- 640 个 episode 全部结束前，launcher 和 finalizer 不读取 success，不按中间结果早停；
- checkpoint 固定为 C60 s10000 SHA256 `d6659c6b...75a36`。

## 基础设施门

Git-only archive 会遗漏嵌套源码，不能作为正式运行快照。正式快照必须由已验证的完整运行快照复制，
并显式检查 FastWAM 三文件、StarWAM 两文件、DreamWAM 三文件的 SHA256。640 条之前，在独立目录
执行一个真实 `libero_spatial/task0/trial34` 机械 canary；要求 policy server 完整加载 H3、严格恢复
C60 checkpoint、到达 ready、产生有限 action 与 trajectory。canary 报告不包含 success，也不参与效果统计。

## 终局门

完整 680 对要求：成功率至少 `+3pp`、paired net wins 至少 20、单侧 exact McNemar
`p<=0.05`、任一 suite 下降不超过 `3pp`。只要任一效果门失败就保留 C58b；机械合同失败则整批
无效，不能人工汇总成功数。C58b trials34..49 必须来自同样 one-episode-per-process 的新只读快照，
聚合时逐对核验初始 trajectory digest 与 object joints。
