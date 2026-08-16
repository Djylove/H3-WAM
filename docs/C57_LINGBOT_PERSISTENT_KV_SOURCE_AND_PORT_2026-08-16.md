# C57 LingBot-VA persistent K/V：源码对齐与 H3-D0 移植合同

## 可证伪问题

在 `D0-H32-s14000 / replan8 / no-ensemble` 不变的前提下，把每次重规划冷启动改成 LingBot-VA
训练、server、client 一致的真实 observation/action rolling K/V，能否让 Goal 与 LIBERO-10 至少新增
3 个配对成功，同时 Object 回归不超过 1 个任务；若进入完整 fresh benchmark，则总成功率至少提升
3pp、净胜至少 20、单侧 exact McNemar `p<=0.05`，任一 suite 退化不超过 3pp。

实验分类为 `backbone_port`，不是 LingBot 官方复现。父模型和动作解码不变，唯一主变量是跨 replan
持续存在的真实 observation/action K/V 生命周期。

边界必须明确：C57 完整移植的是 persistent predicted/real observation-action K/V 生命周期，承载体仍是
D0 的 5 个 ActionDiT block；它没有复现官方 30 层 shared world/action backbone。若 C57 胜出，胜者融合
优先把该生命周期接到 C58b 的 30 层 layer-wise carrier，禁止把本支线单独称为“完整 LingBot”。

## 来源身份

- 官方仓库：`https://github.com/Robbyant/lingbot-va.git`
- 固定 commit：`7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`
- 2026-08-16 `git ls-remote ... HEAD`：仍为相同 commit。
- 本地只读镜像：`third_party/code_audit/lingbot-va`，`git status --short` 为空；未修改 vendor。
- 论文：*Causal World Modeling for Robot Control*，arXiv `2601.21998`。
- 开源级别：`TRAINABLE`。仓库包含 LeRobot dataloader、联合 forward/loss、AdamW、FSDP launcher、
  checkpoint save/load、server/client 和 LIBERO evaluator。

## 官方执行代码，而非 README 摘要

### 数据与训练

1. `lerobot_latent_dataset.py:256-285` 按 latent frame stride 对齐动作，在最前面补
   `frame_stride*4` 个零动作；动作映射进 30 维 channel mask，使用 q01/q99 映射至 `[-1,1]`，再 clip
   到 `[-1.5,1.5]`。
2. `train.py:168-247` 对每个 latent/action frame 独立采样 flow timestep；clean video 有 0.5
   概率加噪，action clean stream 不加噪；训练时随机 `chunk_size=1..4`、`window_size=4..64`。
3. `modules/model.py:93-201,702-798` 将 noisy-video、clean-video、noisy-action、clean-action 四路 token
   一次性送入 30 个共享 Wan block。mask 保证 clean 路径 block-causal、noisy 路径只能看严格过去 clean
   与自身 noisy block；这不是一个动作 history MLP。
4. `train.py:84-118,297-322` 整个 transformer `requires_grad_(True)`；AdamW、fused，梯度裁剪 2.0，
   warmup 后 constant LR。`va_libero_train_cfg.py:13-25` 的 resolved LIBERO 配方为：8-GPU launcher，
   microbatch1、accumulation10、LR `1e-5`、betas `(0.9,0.95)`、weight decay `0.1`、warmup10、5000
   optimizer steps、每200步保存。有效 global batch 为 `8*1*10=80`，共见 `400000` 个样本。

### Server/client 的状态机

1. reset：`wan_va_server.py:377-411` 将 `frame_st_id=0`，清空 transformer 与 streaming-VAE cache；
   `attn_window=30` 时为 15 个 video chunk 与 15 个 action chunk 分配同一 layer-local token pool。
2. predict：`wan_va_server.py:443-561` 只在 `frame_st_id==0` 固定首视频帧和零动作帧；video 与 action
   各自仅在最后一个 denoise step 用 `update_cache=1` 写入 predicted K/V，其他 denoise step 临时写入后
   立即回滚。
3. feedback：`wan_va_server.py:572-604` 首先 `clear_pred_cache`，再把真实 observation K/V 以
   `update_cache=2` 提交，然后提交真实 executed-action K/V；只有真实 observation 提交成功后才增加
   `frame_st_id`。
4. client：`evaluation/libero/client.py:93-125` 第一次跳过零动作 frame；每执行
   `action_per_frame=4` 个动作记录一帧真实 observation，执行完一个 chunk 后把 key frames 与完整实际
   action tensor 回传 server。下一次 predict 因此不是冷启动。
5. cache：`modules/model.py:333-459` 每个 update 有递增 id；池满时按最老 id 淘汰；predicted bit 与
   committed bit 分开，`update_cache=0` 必须恢复临时 slots。

## H3-D0 端口及逐字段差异

| 字段 | LingBot 官方 | C57 本地 | 状态 |
|---|---|---|---|
| backbone | Wan2.2，30个共享 video/action block，全参数后训练 | frozen INT8 H3 layer49 K/V 重复到 D0 的5个 ActionDiT block | `INTENTIONAL_DEVIATION`：H3 backbone port |
| history 表示 | 每层 raw observation/action K/V | 每个 D0 action block 的 observation/action K/V | `EQUIVALENT` 生命周期，维度端口 |
| predicted cache | last denoise video 后 action，均标 predicted | H3 visual carrier 后 action，依次标 predicted | `EQUIVALENT` 状态顺序；visual carrier 不是 Wan future latent |
| real feedback | 清 predicted；先 obs、后 executed action | 原子 transaction 清 predicted；先 obs、后 executed action | `EXACT` |
| frame identity | `frame_st_id` 注入 video/action 3D RoPE | H3 video/audio temporal position 加 frame offset；ActionDiT 使用 absolute executed-action offset | `INTENTIONAL_DEVIATION`：H3/DreamWAM RoPE 坐标系 |
| rolling eviction | unified layer pool，最老 update id 淘汰 | layer-local chronological entries，达到相同 token capacity 后淘汰最老 entry | `EQUIVALENT`：RoPE 后 K/V 的物理 slot 顺序不影响 attention 集合 |
| cache capacity | `(30/2)*video-frame tokens + (30/2)*action-per-frame tokens` | `15*(32 pooled observation tokens + 4 actions/frame)=540 tokens/layer`；replan8 每次反馈2帧与8动作 | `EQUIVALENT` 对应 D0 replan8 |
| training/inference | 四流 block-causal训练；server真实反馈 | trainer 重放同一 commit API；rollout session 使用同一 state/commit API | `EQUIVALENT` 生命周期接口；非30层共享骨干 |
| action合同 | LIBERO 7D 映射到 30D，16步执行 | D0 7D normalized H32，replan8，原有解码 | `INTENTIONAL_DEVIATION`：保持父策略动作合同 |
| optimizer | 8 GPU、global80、AdamW `1e-5`、5000 steps | 配对 control/C57 都采用这套 optimizer，均从同一 D0 初始化 | `INTENTIONAL_DEVIATION`：模型较小但预算和顺序对齐 |

本地实现：

- `src/fastwam/models/h3wam/lingbot_persistent_kv.py`
  - 预测 observation/action K/V 与 committed K/V 分离；
  - 真实反馈原子替换，禁止遗漏 feedback 后再次预测；
  - runtime state 可独立 snapshot/strict restore；
  - 开关关闭时直接调用父 D0，参数名与 state dict 不变。
- `src/fastwam/models/h3wam/c57_lingbot_interfaces.py`
  - teacher-forced trainer 与 rollout 复用同一个 `commit_executed_feedback`；
  - 首次 feedback 自动包含 initial observation；
  - H3 packed video/audio rows 使用绝对 `frame_st_id`，text position 不移动。
- `scripts/h3wam/probe_c57_lingbot_persistent_kv.py`
  - 真实 D0 checkpoint、真实 K/V cache 两样本机械 probe；
  - 关闭开关 bit-exact、persistent finite、history gradient、模型+runtime restore；
  - 可选最多一个不保留权重的 optimizer step；不会写 candidate checkpoint。

## 已通过的本地机械证据

使用本机 PyTorch 2.7 环境执行：

```bash
PYTHONPATH=src:. /home/ubuntu/miniconda3/envs/RoboDojo/bin/python \
  -m unittest -v tests.test_c57_lingbot_persistent_kv
```

6/6 PASS：

- C57 关闭时与原 D0 `torch.equal`，state-dict keys 完全相同；
- predicted observation/action 均在真实 feedback 时删除；
- 真实 observation/action 提交顺序、frame/action id 前进正确；
- teacher-forced current action loss 对 clean history action K projection 梯度 nonzero、finite；
- 模型参数与 runtime K/V snapshot 恢复后 `max_abs=0`；
- rollout 在未提交真实 feedback 时 fail closed；absolute H3 temporal offset 不污染 text token。

这只是 synthetic `MECHANICAL_GATE`，不是效果证据。

## 云端真实机械 probe

在共享项目同步后，单张 A800、现有 INT8 runtime 执行，不保留 checkpoint：

```bash
cd /mnt/h3-wam/candidate-d0-rollout-96976ce/project
py=/mnt/h3-wam/runtime/h3-int8-native/bin/python
lib=$($py -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["purelib"])/"nvidia"/"cu13"/"lib")')
export LD_LIBRARY_PATH=$lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export PYTHONPATH=$PWD/third_party/diffusers_h3/src:$PWD/src:$PWD
$py scripts/h3wam/probe_c57_lingbot_persistent_kv.py \
  --checkpoint /mnt/h3-wam/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt \
  --manifest /mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl \
  --source-manifest /mnt/h3-wam/data/v7_multisuite_dense_candidate/manifest_all.jsonl \
  --cache-root /mnt/h3-wam/data/v7_dense_h3_cache \
  --kv-subdir h3_int8_dreamwam_kv_5x32_dense_v1 \
  --executed-action-steps 8 --actions-per-observation-frame 4 \
  --persistent-window-chunks 15 \
  --output /mnt/h3-wam/outputs/c57-lingbot-persistent-kv/probe/real_mechanical.json
```

真实机械通过条件：disabled max_abs `0`、persistent 全 finite、history K gradient norm `>0`、
model+runtime restore max_abs `0`、没有 candidate checkpoint。

## 正式训练预算与当前阻塞

候选与 control 必须从相同 D0 checkpoint 初始化，使用完全相同的 episode order/noise/optimizer：

- 8 A800/arm；microbatch1、gradient accumulation10、global batch80；
- AdamW LR `1e-5`、betas `(0.9,0.95)`、weight decay `0.1`、warmup10 后 constant；
- 5000 optimizer steps，`400000 samples`；对 200779 个 unique dense windows 是 `1.9922 epochs`；
- 每200步保存，500/1000/2000/3000/4000/5000 做异步 paired offline；
- 墙钟必须由真实 10-step probe 测量：`hours=5000*seconds_per_step/3600`，当前为 `UNKNOWN`；
- control 唯一不启用 persistent cache，其他全部一致。

## 2026-08-16 sequence/canary 更新

- sequence manifest：200779 rows / 1542 episodes，SHA256
  `8f95005ac66fd89ca3a22a80d75480e9792b09f976e928f2eb70d4f08680049`；missing=0、future leakage=0；
  最大 15 observation + 56 executed action = 536 token/layer，小于容量540。
- 专用 rollout feedback wire 只接受实际执行动作，每4动作强制一帧 post-action observation，第8动作后原子
  commit；本地与云端9项测试通过。真实 LIBERO process trace 仍是 evaluation gate。
- 真实 A800 机械 probe PASS：disabled/restore max_abs=0，history K grad norm `0.56703049`，峰值显存
  `3.3634 GiB`。
- canary 许可已升为 `GO_CANARY`；8卡10-step通过后才可启动5000-step长训。

尚未完成的效果门：

1. 真实 LIBERO runner 需要产出 reset/predict/每4动作 observation/第8动作 commit 的完整 trace；
2. H3 visual predicted entry 当前复用 D0 的 frozen visual carrier接口；如果要求复现 LingBot 的显式未来
   video denoise，必须把 H3 I2V future K/V 作为独立 carrier 输入并单独做延迟/效果消融，不能静默宣称
   与官方 Wan future cache 完全相同。

通过真实机械 probe、sequence manifest/data audit 和 rollout wire canary 后，dossier 才能改为
`GO_CANARY`；在此之前禁止启动5000步正式训练。
