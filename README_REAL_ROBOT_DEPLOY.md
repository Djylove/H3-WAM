# FastWAM AnyGrasp 真机部署说明

本文档用于部署 AnyGrasp 的两类 checkpoint：FastWAM 原生模型 `real_anygrasp_v2_uncond_1cam_384_1e-4` 和 hierarchical 模型 `real_anygrasp_v2_hierarchical_1cam_384_1e-4`。当前部署方式参考 GR00T 的 server/client 范式：GPU 机器常驻 policy server，机器人控制端通过 ZMQ client 发送观测并接收 action chunk。

## 1. Conda 环境

按照主 README 配置 FastWAM 环境：

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

`pip install -e .` 会安装 server/client 需要的 `pyzmq`、`msgpack`、`msgpack-numpy` 等依赖。

如果机器人控制端只运行轻量 client，可以只安装：

```bash
pip install numpy msgpack==1.1.0 msgpack-numpy==0.4.8 pyzmq==27.0.1
```

并确保控制端能 import 或复制 `experiments/anygrasp/server_client.py`。

## 2. Model Preparation

推理环境配置完 conda 后，按照主 README 的 Model Preparation 准备模型组件即可。这一步在训练和推理前都需要执行。

Step 1: 设置 Wan 模型目录，默认是 `./checkpoints`：

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

Step 2: 预生成 ActionDiT backbone：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

真机推理不需要执行 `scripts/precompute_text_embeds.py`，该步骤只用于训练前缓存文本 embedding。

## 3. Checkpoint 文件组织

根据从移动云下载得到的 `.pt` 权重文件和对应的 `dataset_stats.json`，新建一个 `ckpt` 文件夹并把两者放进去：

```bash
mkdir -p ckpt/real_anygrasp_v2_hierarchical ckpt/real_anygrasp_v2_uncond

# 将移动云下载的文件放入该目录，例如：
cp /path/to/downloaded/step_xxxxxx.pt ckpt/real_anygrasp_v2_hierarchical/
cp /path/to/downloaded/dataset_stats.json ckpt/real_anygrasp_v2_hierarchical/
```

原生模型同样放到 `ckpt/real_anygrasp_v2_uncond/`。两种模型的权重不能混用，启动 server 时的 `--task` 必须和 checkpoint 架构一致。

推荐目录结构：

```text
ckpt/real_anygrasp_v2_hierarchical/
├── step_xxxxxx.pt
└── dataset_stats.json
```

启动推理时在命令中显式指定这两个路径：

```bash
--ckpt ckpt/real_anygrasp_v2_hierarchical/step_xxxxxx.pt
--dataset-stats-path ckpt/real_anygrasp_v2_hierarchical/dataset_stats.json
```

`dataset_stats.json` 必须和 `.pt` 权重来自同一套训练配置，用于 state/action 归一化和 action 反归一化。不要用 LeRobot 原始 `meta/stats.json` 或其他 `norm_stats.json` 替代。

## 4. 启动 AnyGrasp Policy Server

### 4.1 FastWAM 原生模型

原生模型只使用当前观测帧预测 action chunk，不需要 high/low video、joint denoise 或 reuse 参数：

```bash
python scripts/run_anygrasp_server.py \
  --ckpt ckpt/real_anygrasp_v2_uncond/step_xxxxxx.pt \
  --task real_anygrasp_v2_uncond_1cam_384_1e-4 \
  --dataset-stats-path ckpt/real_anygrasp_v2_uncond/dataset_stats.json \
  --host 0.0.0.0 \
  --port 5555 \
  --device cuda:0 \
  --mixed-precision bf16 \
  --action-horizon 32 \
  --replan-steps 24 \
  --num-inference-steps 20
```

训练原生模型及预计算文本 embedding：

```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py \
  task=real_anygrasp_v2_uncond_1cam_384_1e-4

bash scripts/train_zero1.sh 8 \
  task=real_anygrasp_v2_uncond_1cam_384_1e-4
```

### 4.2 Hierarchical 模型

基础启动命令：

```bash
python scripts/run_anygrasp_server.py \
  --ckpt ckpt/real_anygrasp_v2_hierarchical/step_xxxxxx.pt \
  --task real_anygrasp_v2_hierarchical_1cam_384_1e-4 \
  --dataset-stats-path ckpt/real_anygrasp_v2_hierarchical/dataset_stats.json \
  --host 0.0.0.0 \
  --port 5555 \
  --device cuda:0 \
  --mixed-precision bf16
```

推荐的初始真机参数：

```bash
python scripts/run_anygrasp_server.py \
  --ckpt ckpt/real_anygrasp_v2_hierarchical/step_xxxxxx.pt \
  --task real_anygrasp_v2_hierarchical_1cam_384_1e-4 \
  --dataset-stats-path ckpt/real_anygrasp_v2_hierarchical/dataset_stats.json \
  --host 0.0.0.0 \
  --port 5555 \
  --device cuda:0 \
  --mixed-precision bf16 \
  --high-denoise-step 5 \
  --low-denoise-step 5 \
  --joint-denoise
```

如果需要进一步加速，可以在上述基础上增加 reuse：

```bash
python scripts/run_anygrasp_server.py \
  --ckpt ckpt/real_anygrasp_v2_hierarchical/step_xxxxxx.pt \
  --task real_anygrasp_v2_hierarchical_1cam_384_1e-4 \
  --dataset-stats-path ckpt/real_anygrasp_v2_hierarchical/dataset_stats.json \
  --host 0.0.0.0 \
  --port 5555 \
  --device cuda:0 \
  --mixed-precision bf16 \
  --high-denoise-step 5 \
  --low-denoise-step 5 \
  --high-reuse-step 2 \
  --low-reuse-step 2 \
  --joint-denoise
```

对于 hierarchical checkpoint，当前建议一直保持 `joint_denoise=true`，也就是启动时使用 `--joint-denoise`，不要使用 `--no-joint-denoise`。这些 hierarchical 专属参数不会传给 FastWAM 原生模型。

## 5. GR00T-style Server/Client 逻辑

当前 AnyGrasp 部署接口和 GR00T 类似：

```text
Robot control process
  -> PolicyClient 构造 observation
  -> ZMQ REQ + msgpack_numpy 发送 endpoint=get_action
  -> PolicyServer 常驻 GPU 机器
  -> FastWAMAnyGraspPolicy.get_action()
  -> 返回 denormalized action chunk
  -> Robot 执行动作，并按 replan 频率再次请求
```

内置 endpoint：

```text
ping
get_action
reset
get_modality_config
kill
```

最小 client 示例：

```python
import numpy as np
from experiments.anygrasp.server_client import PolicyClient

client = PolicyClient(host="127.0.0.1", port=5555, timeout_ms=30000)

if not client.ping():
    raise RuntimeError("FastWAM AnyGrasp server is not reachable.")

print(client.get_modality_config())

image = np.zeros((480, 832, 3), dtype=np.uint8)
state = np.zeros((33,), dtype=np.float32)

observation = {
    "video": {
        "top": image,
    },
    "state": {
        "default": state,
    },
    "language": {
        "task": [["put the object into the target area"]],
    },
}

action, info = client.get_action(
    observation,
    options={
        "action_space": "selected",
    },
)

action_chunk = action["default"][0]  # [T, 31]
execute_steps = int(info["replan_steps"])
actions_to_execute = action_chunk[:execute_steps]
```

每个 episode 或一次完整任务开始前建议调用：

```python
client.reset()
```

`reset()` 会清空服务端历史帧缓存；使用 hierarchical 模型时还会清空 reuse cache，避免上一次任务的预测 latent 影响当前任务。

## 6. Observation 与 Action 格式

输入 observation：

```python
observation = {
    "video": {
        "top": image,       # uint8, HWC / THWC / BTHWC / CHW / TCHW / BTCHW
    },
    "state": {
        "default": state,   # float32, D / TD / BTD
    },
    "language": {
        "task": [[instruction]],
    },
}
```

当前 deploy wrapper 只支持 batch size 1。图片会在 server 端 resize 到训练配置的输入分辨率，AnyGrasp 当前为 `256 x 384`。state 可以传原始 33 维，也可以传训练选择后的 31 维；如果传 33 维，server 会按训练时的 `SelectDimensions` 选择 0-30 维。

### 6.1 原始 37 维 action 含义

AnyGrasp LeRobot v3 原始 `action` 是 37 维，维度顺序来自数据集 `meta/info.json`：

| index | name |
|---:|---|
| 0 | `head_yaw_joint` |
| 1 | `head_pitch_joint` |
| 2 | `waist_yaw_joint` |
| 3 | `waist_pitch_joint` |
| 4 | `waist_roll_joint` |
| 5 | `left_shoulder_pitch_joint` |
| 6 | `left_shoulder_roll_joint` |
| 7 | `left_shoulder_yaw_joint` |
| 8 | `left_elbow_pitch_joint` |
| 9 | `left_wrist_yaw_joint` |
| 10 | `left_wrist_roll_joint` |
| 11 | `left_wrist_pitch_joint` |
| 12 | `right_shoulder_pitch_joint` |
| 13 | `right_shoulder_roll_joint` |
| 14 | `right_shoulder_yaw_joint` |
| 15 | `right_elbow_pitch_joint` |
| 16 | `right_wrist_yaw_joint` |
| 17 | `right_wrist_roll_joint` |
| 18 | `right_wrist_pitch_joint` |
| 19 | `L_pinky_proximal_joint` |
| 20 | `L_ring_proximal_joint` |
| 21 | `L_middle_proximal_joint` |
| 22 | `L_index_proximal_joint` |
| 23 | `L_thumb_proximal_pitch_joint` |
| 24 | `L_thumb_proximal_yaw_joint` |
| 25 | `R_pinky_proximal_joint` |
| 26 | `R_ring_proximal_joint` |
| 27 | `R_middle_proximal_joint` |
| 28 | `R_index_proximal_joint` |
| 29 | `R_thumb_proximal_pitch_joint` |
| 30 | `R_thumb_proximal_yaw_joint` |
| 31 | `vel_height` |
| 32 | `vel_pitch` |
| 33 | `base_yaw` |
| 34 | `vel_x` |
| 35 | `vel_y` |
| 36 | `vel_yaw` |

### 6.2 当前训练和推理默认 action 维度

当前 `configs/data/real_anygrasp_v2.yaml` 中配置为：

```yaml
action:
  raw_shape: 37
  shape: 31

action_state_transforms:
  - _target_: fastwam.datasets.lerobot.transforms.misc.SelectDimensions
    action_indices: [0, 1, ..., 30]
    original_action_dim: 37
```

因此当前模型实际训练和默认推理接口使用的是 **31 维 selected action**：

```text
0-30: 头部、腰部、双臂、双手关节
31-36: base/lower-body 相关控制被丢弃，没有参与训练
```

默认 client 请求：

```python
action, info = client.get_action(observation)
selected_action = action["default"]  # [1, T, 31]
```

如果控制器确实需要 37 维容器，可以请求 raw action：

```python
action, info = client.get_action(
    observation,
    options={
        "action_space": "raw",
    },
)
raw_action = action["default"]  # [1, T, 37]
```

注意：`raw` 模式只是把模型预测的 31 维 scatter 回原始 37 维，未训练的 31-36 维会填 0。也就是说，当前模型不会有效预测 `vel_height`、`vel_pitch`、`base_yaw`、`vel_x`、`vel_y`、`vel_yaw`。

## 7. 推荐推理参数

FastWAM 原生模型主要调整 `action_horizon`、`replan_steps` 和 `num_inference_steps`。下面的 high/low denoise、reuse 和 `joint_denoise` 参数仅适用于 hierarchical 模型。

### `high_denoise_step`

high-level keyframe imagination 实际执行的去噪步数。推荐初始值：

```text
high_denoise_step = 5
```

值越大，高层关键帧去噪越充分，但推理更慢；值越小速度更快，但 imagination 更粗。

### `low_denoise_step`

low-level video imagination 实际执行的去噪步数。推荐初始值：

```text
low_denoise_step = 5
```

在当前建议的 `joint_denoise=true` 下，保持：

```text
high_denoise_step <= low_denoise_step <= action_inference_steps
```

当前推荐 `high_denoise_step=5`、`low_denoise_step=5`，满足这个约束。

### `high_reuse_step`

跨 replan 复用上一次 high-level keyframe latent 的位置。默认建议先不打开；如果需要加速，推荐：

```text
high_reuse_step = 2
```

它会减少每个 chunk 重新去噪 high-level latent 的计算量。打开后一定要在每个 episode/任务开始时调用 `client.reset()`。

### `low_reuse_step`

跨 replan 复用上一次 low-level video latent 的位置。默认建议先不打开；如果需要加速，推荐：

```text
low_reuse_step = 2
```

如果机器人或物体状态变化很大，reuse 可能带来滞后感；可以先用无 reuse 的配置跑通，再开启 `high_reuse_step=2`、`low_reuse_step=2` 对比速度和效果。

### `joint_denoise`

建议一直保持：

```text
joint_denoise = true
```

也就是 server 启动时使用：

```bash
--joint-denoise
```

当前 hierarchical AnyGrasp task 默认也是 `joint_denoise: true`。真机部署建议优先保持和训练/evaluate 逻辑一致，只调 `high_denoise_step`、`low_denoise_step` 和可选 reuse。

## 8. 推荐排查顺序

1. 确认 server 能启动并打印 `FastWAM AnyGrasp server is ready`。
2. client 调用 `ping()` 和 `get_modality_config()`，确认网络和接口可用。
3. 用一帧真实相机图像、当前 state 和固定 instruction 调 `get_action()`，确认返回 action shape。
4. 先执行很短的 action 前缀，观察机器人方向和尺度是否正确。
5. 如果 action 尺度异常，优先检查 `.pt` 和 `dataset_stats.json` 是否来自同一次训练配置。
6. 每个 episode 或任务开始前调用 `client.reset()`。
