# C58：FastWAM 官方 30 层 ActionDiT × H3 冻结世界特征

日期：2026-08-16

分类：`backbone_port`
状态：长训已启动；机械证据通过；效果证据尚未产生

## 结论边界

C58 不是在旧动作头上继续加辅助损失，而是把 FastWAM 官方完整 30 层
ActionDiT 执行路径接到已验证的 H3 世界特征上。唯一主变量是动作塔容量从
D0 的 5 层扩展为 FastWAM 的 30 层；H3 缓存、训练窗口、动作表示、flow
目标和最终 LIBERO 评测协议不变。

当前只能确认该端口“能正确训练”：step 0 与 D0 输出逐 bit 相同、30/30 层
均有非零梯度、10 step checkpoint 可严格恢复。只有相同样本/优化预算下的
held-out 指标和新 LIBERO 闭环成功率胜出后，才能声称 H3 的世界预测能力被
转化为动作成功率。

可证伪假设：在相同冻结 H3 layer-49 K/V、相同 dense LIBERO 数据、相同
shift-5 flow loss 和相同评测条件下，完整 30 层动作塔将优于匹配的新优化器
D0-5 层控制组，并最终提高 fresh-trial LIBERO 成功率。若离线动作误差、视觉
打乱敏感度和闭环成功率没有共同改善，则否定该假设，而不是用更多训练步数
解释失败。

## 官方实现审计

官方仓库固定为 `FastWAM@45d8e1458921d83f8ad6cf9ce993d371208dabd0`，
本地 vendor 工作树有既有脏改动，因此 C58 从该 commit 的 git object 建立只读
镜像，并在运行时逐文件校验：

| 文件 | SHA256 |
|---|---|
| `src/fastwam/models/wan22/action_dit.py` | `1301d9224149de43bb701f620a5d41858ecc63c6b19a573ec32edd45a3bdb0a2` |
| `src/fastwam/models/wan22/wan_video_dit.py` | `d098ad77665feeefa81634f31f5bb1d5771c4556d1a67859135f0ed35f9eb6c2` |
| `src/fastwam/models/wan22/helpers/gradient.py` | `ba5d8f7272eb029dc6cd2849ca99b70f6ad5abb838d21c818beb0590620dc793` |
| `src/fastwam/models/wan22/mot.py` | `9f3f09cd73bf6e4547955336cb8a9055c5345f9dbabf06361ebf139b8d8accb5` |

官方关键合同：ActionDiT 为 hidden 1024、FFN 4096、30 层、head dim 128；
LIBERO 使用四个 suite、双相机、32 action/9 video frame、stride 1、7D action、
8D state、min/max normalization、AdamW `(0.9, 0.95)`、LR `1e-4`、WD
`1e-2`、5% warmup + cosine。官方模型同时训练 video/action MoT；C58 有意
只训练动作分支并冻结预计算 H3 特征，这是 backbone port，不是官方复现。

## 本地端口

- [fastwam_full_tower.py](../src/fastwam/models/h3wam/fastwam_full_tower.py) 动态加载并校验官方 ActionDiT 源码，不复制或改写 vendor；使用 H3 兼容的 56 heads × 128 head dim。
- 5 个 D0 block 放到 30 层的 `(0, 7, 14, 22, 29)`；其余 25 层从最近 block 克隆后，将 self-attention、cross-attention 和 FFN 的残差输出投影置零，因此新增层在 step 0 是严格 identity。
- 30 个 action block 都读取同一个因果 H3 layer-49 K/V；完整五层 cache 仍被严格校验且禁止 storage alias。这保留 D0 的因果部署合同，同时只改变动作塔深度。
- [train_h3_fastwam_full_tower.py](../scripts/h3wam/train_h3_fastwam_full_tower.py) 复用已审计 D0 dense loader、min/max stats、padding mask、continuous-fp32 timestep 和 shift-5 flow velocity MSE；每步检查 30 个 block 与 proprio 路径梯度。
- [evaluate_h3_fastwam_full_tower.py](../scripts/h3wam/evaluate_h3_fastwam_full_tower.py) 复用 frozen balanced-80 的样本选择、flow sampler、语言替换、视觉 K/V 打乱和指标代码，只替换 C58 checkpoint 合同及 model constructor。

## 与官方对齐矩阵

| 项目 | 状态 | C58 实际做法 |
|---|---|---|
| 官方 30 层 ActionDiT block 执行 | EXACT | 直接加载 byte-pinned 官方 `action_dit.py` |
| hidden/FFN/head-dim/depth | EXACT | 1024/4096/128/30 |
| attention head 数 | INTENTIONAL | 24 改为 56，以无投影消费 H3 7168D K/V |
| 初始化 | INTENTIONAL | 官方从 Wan video DiT 线性插值并 alpha scale；C58 以 D0 function-preserving depth expansion，避免 step-0 退化 |
| LIBERO suite、horizon、action/state 维度 | EQUIVALENT | 四 suite、32、7D/8D；沿用现有 dense manifest 与部署动作合同 |
| 训练 objective | INTENTIONAL | 保留官方 action shift-5 flow 分支；不重建 RGB video loss，H3 特征冻结 |
| optimizer/schedule | EQUIVALENT | AdamW `.9/.95`、LR `1e-4`、WD `.01`、warmup + cosine |
| 官方 epoch/batch 预算 | INTENTIONAL | 8 GPU、global batch 8、10k step/80k samples；匹配控制组必须用同预算 |
| 评测 | INTENTIONAL | 目标为项目 frozen balanced-80 + LIBERO replan8；不把官方 replan5 结果混为同协议 |

## 已通过机械门

云端：`ssh -p 32409 dev@117.50.181.177`

探针目录：`/mnt/h3-wam/outputs/c58-fastwam-full30-v1/probe10`

- 8×A800 BF16、global batch 8、10 个真实 optimizer step、80 samples。
- 训练参数 `2,029,842,439`；峰值 allocated `22.7113 GiB/GPU`，reserved
  `24.2715 GiB/GPU`。
- step-0 D0 parity `max_abs=0.0`。
- 30 层梯度全部有限且非零，10 step 范围 `0.0348871..2.00096`；loss
  `0.00842285..0.359375`。
- checkpoint `c58_s10.pt` 为 `12,180,221,635` bytes，SHA256
  `4c47305c378567fcb6ce543e6924fdb7b316c39fee281c6aab80106de245c2b1`。
- 独立进程 strict restore 的固定样本预测 `max_abs=0.0`。
- 本地 focused tests：模型/训练合同 7 项，加评测合同 5 项，共 `12 passed`。

这些结果仅允许诊断性长训，不是动作效果证据。

## 长训与预注册判定

长训目录：`/mnt/h3-wam/outputs/c58-fastwam-full30-v1/long10000`。当前在 8 张
A800 上运行，共 10,000 step，每 1,000 step 保存原子 checkpoint；训练从
offset 112000 开始，每段 8000 个窗口，十段共 80,000 个互不重叠窗口。预算
为 `80,000 / 200,779 = 0.398448` effective epoch，约占 120 GB checkpoint
空间。首段 s1000 已在 513.20 秒完成，实测稳态约
0.513 秒/step；因此总 wall time 修正为约 2.5–4 小时，不能用 10-step probe
中被初始化与 12 GB 保存主导的 8.7 秒/step 外推。2026-08-16 最近复核已到
step 1487，8 卡各约 25.8 GiB，30 层梯度持续非零。

每个 1000-step milestone：

1. strict restore 必须为 `max_abs=0`，且 checkpoint/manifest/H3/FastWAM SHA 全匹配；
2. 运行固定 v7 balanced-80，不能改样本 ID、seed、noise、shift 或 inference steps；
3. 与同 checkpoint 阶段的匹配 D0-5 层 fresh-optimizer 控制比较；
4. 只有离线 action MSE、gripper 与视觉打乱机制信号共同改善的 milestone 才进入 fresh LIBERO；
5. 最终晋级必须依赖相同初态/seed/replan8 的新 trial 闭环成功率，不允许用训练 loss 代替。

提前停止条件：NaN/OOM、任一 block 梯度为零、严格恢复不一致、身份漂移，或
连续多个 milestone 相对匹配控制无改善。磁盘只保留预注册 milestone；无效
临时/重复 checkpoint 在结果审计后清理。

## 未解决项

- 匹配的 5 层 D0 fresh-optimizer 10k 控制尚未与 C58 同时启动；没有该控制，不能把“继续训练”与“30 层结构”拆开。
- balanced-80 adapter 已实现并通过合同测试，但尚未对 C58 milestone 跑完真实 80 样本。
- 还没有新 LIBERO 闭环成功率，因此 effect status 必须保持 `NOT_EVIDENCE_READY`。
- H3 layer49 重复是为保持 D0 因果合同的主实验；多层对齐 carrier 应作为另一条单变量支线，不能混入 C58 主臂。

## C58b：layer-wise H3-50 → ActionDiT-30 支线

架构审计后的重要修正：C58 只能叫“官方完整 30 层动作塔”，不能叫“完整
FastWAM 世界—动作联合训练”，因为它把同一 layer49 K/V 重复给 30 个 block，
并且 H3 冻结、没有 video flow loss。C58b 不打断 C58，而是隔离验证 carrier
深度对齐这个变量。

C58b 将 H3 的 50 层按单调均匀深度映射到 30 个 ActionDiT block：

`(0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24, 25, 27, 29, 30, 32, 34, 35, 37, 39, 41, 42, 44, 46, 47, 49)`。

每个 block 的 mixed self-attention 仍是官方语义：action query 同时 attend
对应 H3 K/V 和当前 action K/V。初始化仍使用 D0 function-preserving 5→30
扩深。机械检查先把上述 30 份 K/V 全部替换成 layer49，此时 C58b 必须与 D0
`max_abs=0`；再换回真实 layer-wise carrier，输出必须发生非零变化。这能把
“初始化差异”和“世界层级 carrier 差异”拆开。

缓存成本是主要约束：BF16 `30×2×32×56×128` 为每样本 `27,525,120`
bytes；训练所需 80,000-window slice 的纯 tensor 约 `2.202 TB`，全 200,779
窗口约 `5.527 TB`。因此先用 80 样本 cache canary，不盲目生成全量缓存。
训练可复用缓存以避免每 step 重跑冻结 H3；闭环部署必须走在线 INT8 H3
抽取 30 层 K/V，这条部署适配将在 C58b 离线机制信号通过后再放行。

已实现：

- 模型 layer-wise 路由、严格 30 层 mapping/no-alias 合同；
- C58/C58b 共用 trainer 的隔离 `--carrier-mode`，默认行为和已有 C58 checkpoint 合同不变；
- 8-GPU cache-canary80 与 10-step BF16 probe launcher；
- C58 + C58b + evaluator 共 16 项本地测试通过。

尚未执行真实 cache/probe，因为现有四台服务器分别被 C58、C57、C60 cache
和 rollout 占用；需要等待一台 8×A800 释放，或增加一台同规格资源。
