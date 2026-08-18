# C67/C69 固定 s20000 配对闭环归因结果

日期：2026-08-18

分类：`controlled_ablation`

最终状态：`EVIDENCE_READY_ATTRIBUTION_ONLY`

最终决定：`NO_DETECTABLE_INCREMENTAL_CONSEQUENCE_EFFECT`

## 可证伪问题

在完全相同的 C58b 父模型、20k 优化预算、样本顺序、动作合同和闭环协议下，C67 的 FACT
future-representation/state/value 辅助监督能否相对 C69 action-only 带来至少 3pp、净胜至少 20 对、
单侧 exact McNemar `p<=0.05` 且任一 suite 不退化超过 3pp 的成功率提升。

结果否定了该增量假设。它不证明世界模型特征无用：两臂都继续使用冻结 INT8 H3 的 30 层 K/V；它只说明
当前 FACT consequence objective 没有在这个训练和部署合同下产生可检测的额外闭环价值。

## 来源、父模型与唯一变量

- FACT 官方代码：`Bariona/FACT@618a6c16868699b6d4138941de6a863589ac00dd`。
- FastWAM 官方 ActionDiT：`YuanTianYuan01/FastWAM@45d8e1458921d83f8ad6cf9ce993d371208dabd0`。
- 闭环执行源码：commit `851882139cc676776a7b21d1ed6d354b0995150d`，tree
  `ec3614912a19e185da14a7dbedd8f26074710546`，完整 SOURCE_FREEZE SHA256
  `5f580d76066f0a53963a16d8c6e3f02fb1073a80a6eaba0c0f7704a52480fcba`。
- 共同父模型：C58b s10000，SHA256
  `2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541`。
- 唯一变量：C67 权重为 `[10,1,0.4,0.4]` 并训练 consequence 专属模块；C69 权重为
  `[10,0,0,0]` 并冻结六类 consequence-only encoder/decoder。30 层 joint-token forward、冻结 INT8 H3、
  数据、seed、optimizer、LR、schedule、动作 normalization、horizon32 和部署求解器不变。
- DreamWAM、MiniWorld、LingBot、Light-WAM 和 ranker/best-of-N 没有进入本轮，因为它们会引入第二主变量，
  不能回答 consequence objective 的增量归因问题。

## 实际预算与执行

- 每个训练臂：8×A800、global batch 8、20,000 optimizer steps、160,000 samples、`0.733522`
  effective epoch、每 1,000 step 保存并严格恢复。C69 的 20 段 trainer invocation 实测
  `25,102.055s = 6.973h`；本轮不重训任一 endpoint。
- 闭环：四个 LIBERO suite × 10 task × trials33..49 × 2 arms，共 680 个完全配对初态、1,360 个隔离
  simulator+policy 进程，40×A800 并行；wait30、max400、replan8、horizon32、10 次动作去噪求解。
- 两臂合计执行 371,027 个环境动作、46,664 次 replan、466,640 次 action-denoiser forward；episode
  duration 累计 `216,397.323s`。从授权生成到 RESULTS 发布的实际墙钟为 `7,506s = 2.085h`。

## 结果

| 指标 | C67 FACT joint | C69 action-only | C67-C69 |
|---|---:|---:|---:|
| 全部成功 | 324/680 = 47.647% | 338/680 = 49.706% | -2.059pp |
| 配对独胜 | 23 | 37 | 净胜 -14 |
| one-sided p（C67更好） | 0.97405 | — | FAIL |
| two-sided p | — | — | 0.09246 |

按 suite 的 C67-C69：LIBERO-10 `+2.353pp`、goal `-1.765pp`、object `-2.353pp`、spatial
`-6.471pp`。C67 的四个正向门全部失败；C69 的单侧显著性和 suite 安全门通过，但绝对提升只有
2.059pp、净胜只有 14，对应的 3pp/20 对门失败。因此不能宣称 C69 统计晋级，只能得出“没有检测到
consequence objective 的增量价值”。

## 身份与证据

- authorization SHA256：`9241a9d394857f39dc3f024b594010aa3c567b578db660e2635fbb21013ae9cb`
- manifest SHA256：`409d5d2d9395ad163069a95b8701ecfa1003ae2c1e8cfc213d7a80e5a8b307ae`
- completion SHA256：`5235fc11c817843cfea3297dd53e39560b93e2a419f83fa908f0e37d5784fb9e`
- pair evidence SHA256：`7238e82b3956b1abf53e2cf3a52bc7831dc5fa6bed220957cd896a9d4fa2acbe`
- RESULTS SHA256：`473a499435a473068c489f41587d89c44be996eefa9a17e58eaea8af0a47527a`
- 680/680 对的 authorization、冻结源码、checkpoint、trial、任务、初始 MuJoCo state 和轨迹初态均通过
  exact identity gate；没有中间 checkpoint 选择。

## 研发决定

- 对“不改合同、只继续增加 C67 consequence-objective 训练”的许可为 `NO_GO`；paired gate 已经给出
  完整负归因，增加 steps 不能修复这个假设。
- C69 仅是 action-objective 直接对照，不自动晋升冠军；C58b 仍是唯一 carrier/project champion：
  `295/680=43.382%` 对 D0 `270/680=39.706%`，并已通过其独立预注册晋级门。
- fusion lineage 仍为 `C58b carrier champion`。FACT consequence、MiniWorld/LingBot context 和 C71
  Light-WAM state fusion 均未取得独立赛道胜者资格，不进入融合。
- 云资源到期前不再启动新的长训。C71 保持 `PROBE_ONLY / NOT_EVIDENCE_READY`；后续有新资源时，应先完成
  direct-action/state-fusion trainer 和独立父对照，而不是恢复 C67 consequence loss。

原始结果位于
`/mnt/h3-wam/eval/c67-vs-c69-fixed-s20-paired680-trials33-49-8518821-v1/RESULTS.json`；关键 JSON 已按
相同 SHA256 转存到 `/home/ubuntu/h3-wam-critical-backup-20260818/eval-summary/`。
