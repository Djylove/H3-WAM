# FastWAM AnyGrasp 真机部署说明

本文档用于部署 AnyGrasp 的两类 checkpoint：FastWAM 原生模型 `real_anygrasp_v2_uncond_1cam_384_1e-4` 和 hierarchical 模型 `real_anygrasp_v2_hierarchical_1cam_384_1e-4`。当前部署方式参考 GR00T 的 server/client 范式：GPU 机器常驻 policy server，机器人控制端通过 ZMQ client 发送观测并接收 action chunk。

## 0. 当前 GR3 一体化部署启动流程

以下流程是当前实验室 GR3 + FastWAM + Dagger + QNexo 外骨骼链路的日常启动方式。模型工程位于：

```text
/home/fourier/xingyu/Hierarchical_WAM
```

真机控制工程位于：

```text
/home/fourier/dagger-gr3
```

### 0.1 启动前检查

1. 机器人已经上电。
2. 已经 SSH 进入机器人并启动机器人侧 Aurora/控制服务。
3. 主机能够连接 `ROBOT_ID=115`。
4. OAK 相机、QNexo 外骨骼和脚踏已经连接。
5. GPU 上没有另一个进程占用 FastWAM 服务端口 `5555`。

机器人侧服务不属于这两个仓库，启动命令以机器人系统当前配置为准，不在本文档中虚构命令。

主机设备检查：

```bash
ls -l /dev/qnbot
ls -l /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
ss -ltnp | grep ':5555' || true
nvidia-smi
```

如果 `5555` 已经由正确的 FastWAM 进程监听，不要重复启动模型服务。

### 0.2 首次安装或依赖更新

FastWAM 模型服务使用 Python 3.10 Conda 环境 `fastwam`：

```bash
source /home/fourier/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
cd /home/fourier/xingyu/Hierarchical_WAM
python --version
which python
```

期望解释器为：

```text
/home/fourier/miniconda3/envs/fastwam/bin/python
```

如果环境尚未安装，按本文档第 1 节完成安装。

Dora 真机控制链路使用 `/home/fourier/dagger-gr3/.venv`，首次安装或代码依赖更新后执行：

```bash
cd /home/fourier/dagger-gr3
./scripts/setup_local.sh
./scripts/check.sh
```

日常上电测试不需要重复运行 `setup_local.sh` 和 `check.sh`。

### 0.3 终端一：启动 FastWAM 模型服务

先激活模型 Conda 环境：

```bash
source /home/fourier/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
cd /home/fourier/xingyu/Hierarchical_WAM
```

推荐通过真机工程中固化的启动脚本启动。脚本内部使用 `fastwam` 环境的绝对 Python 路径，因此不会误用 Dora 的 Python 3.13 环境：

```bash
cd /home/fourier/dagger-gr3
./scripts/start_fastwam_server.sh
```

默认配置：

```text
checkpoint: /home/fourier/xingyu/checkpoints/checkpoints_wan_710/step_020650.pt
dataset stats: /home/fourier/xingyu/checkpoints/checkpoints_wan_710/dataset_stats.json
task config: real_anygrasp_v2_hierarchical_1cam_384_1e-4
endpoint: tcp://127.0.0.1:5555
device: cuda:0
precision: bf16
high denoise step: 5
low denoise step: 5
joint denoise: enabled
torch.compile: enabled
CUDA Graph: disabled (RTC 需要 autograd)
inference backend: inductor
text encoder offload: enabled
RTC warmup: enabled
```

第一次启动需要完成模型加载、`torch.compile`、普通推理和 RTC VJP 预热。必须等待：

```text
FastWAM AnyGrasp server is ready on tcp://127.0.0.1:5555
```

后台启动方式：

```bash
cd /home/fourier/dagger-gr3
./scripts/start_fastwam_server.sh --background
tail -f logs/fastwam-server.log
```

如需直接从模型仓库调试服务端，等价命令为：

```bash
source /home/fourier/miniconda3/etc/profile.d/conda.sh
conda activate fastwam
cd /home/fourier/xingyu/Hierarchical_WAM

python scripts/run_anygrasp_server.py \
  --ckpt /home/fourier/xingyu/checkpoints/checkpoints_wan_710/step_020650.pt \
  --task real_anygrasp_v2_hierarchical_1cam_384_1e-4 \
  --dataset-stats-path /home/fourier/xingyu/checkpoints/checkpoints_wan_710/dataset_stats.json \
  --host 127.0.0.1 \
  --port 5555 \
  --device cuda:0 \
  --mixed-precision bf16 \
  --high-denoise-step 5 \
  --low-denoise-step 5 \
  --joint-denoise \
  --compile-hierarchical \
  --no-compile-cudagraphs \
  --optimize-denoise-static \
  --inference-backend inductor \
  --rtc-warmup \
  --config-override +model.offload_text_encoder=true
```

三种编译模式：

```bash
# RTC 默认：torch.compile + Inductor，保留 autograd
./scripts/start_fastwam_server.sh

# 关闭 RTC 后使用 torch.compile + CUDA Graph
FASTWAM_RTC_ENABLED=0 FASTWAM_COMPILE_CUDAGRAPHS=1 \
./scripts/start_fastwam_server.sh

# 完全使用 eager
./scripts/start_fastwam_server.sh \
  --no-compile-hierarchical \
  --no-compile-cudagraphs

# 实验性 TensorRT：joint/video 阶段仍使用 Inductor，固定 video cache 后的 action-only 阶段使用 TensorRT
FASTWAM_INFERENCE_BACKEND=tensorrt \
FASTWAM_RTC_ENABLED=0 \
./scripts/start_fastwam_server.sh
```

TensorRT 模式不会改变 checkpoint、采样步数或 action scheduler。它只替换 action-only 热路径；编译失败时自动回退到 Inductor。第一次启动会在 warmup 中构建引擎，并用 5 个不同去噪步逐步比对 eager 输出，误差超限会禁用 TensorRT，因此该模式不允许关闭 warmup。必须等到 server ready 后再启动 Dora；真机启用前仍应完成限速单步测试。

RTC 的 VJP 需要 autograd，因此不能同时启用 CUDA Graph 或 TensorRT。当前 24 GB GPU 部署只在 action-only refinement（默认去噪第 6-10 步）执行精确 VJP；前 5 个 joint video/action 步保持原模型前向，避免为 5B 视频分支保留反向激活导致显存不足。

RTC 的 prefix schedule、VJP correction 和异步队列替换语义分别对齐 [Physical Intelligence 官方实现](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) 与 [LeRobot RTC](https://huggingface.co/docs/lerobot/en/rtc)。FastWAM 适配在模型归一化后的 31 维 selected action 空间内计算 guidance，client 最终仍下发反归一化后的 GR3 绝对关节角。

### 0.4 终端二：启动完整 Dora 真机链路

模型服务显示 ready 后再启动：

```bash
cd /home/fourier/dagger-gr3

ROBOT_ID=115 \
ACTION_CHUNK_SIZE=32 \
ENABLE_DAGGER_RECORDING=0 \
TASK='Pick the tomato from the table and place it inside the basket on the left.' \
./scripts/run_fastwam_dagger.sh
```

`run_fastwam_dagger.sh` 会检查模型端口、脚踏、Python 节点和 recorder，然后一次性启动：

- OAK 相机，原始画面旋转 180 度；
- 相机可视化窗口；
- FastWAM policy client；
- Dagger router；
- QNexo 外骨骼；
- 脚踏接管；
- Teleop 真机控制；
- 数据记录器。

这些节点不需要分别启动。Dora 使用项目自己的 `.venv`，不依赖终端当前激活的 Conda 环境。

常用参数：

```text
ROBOT_ID=115                    机器人 Aurora domain ID
ACTION_CHUNK_SIZE=32            每次实际执行的 action chunk 长度，必须为正整数
FASTWAM_RTC_ENABLED=1           1 启用异步 RTC，0 使用原队列耗尽后再推理
FASTWAM_RTC_PREFIX_STEPS=16     剩余 16 步时请求；也是 prefix/keyframe 周期
FASTWAM_RTC_INFERENCE_DELAY_STEPS=10  首次 guidance 的预计推理延迟
FASTWAM_RTC_MAX_GUIDANCE_WEIGHT=5.0   官方 RTC guidance 权重上限
ENABLE_DAGGER_RECORDING=0       0 不保存，1 在首次脚踏接管后开始保存
DAGGER_RECORD_DIR=...           数据保存根目录
TASK=...                        模型 prompt
FOOTSWITCH_DEVICE=...           脚踏 event 设备
MAX_JOINT_STEP_RAD=0.15         每周期关节变化保护阈值
```

需要保存接管到 episode 结尾的数据时：

```bash
cd /home/fourier/dagger-gr3

ROBOT_ID=115 \
ACTION_CHUNK_SIZE=32 \
ENABLE_DAGGER_RECORDING=1 \
DAGGER_RECORD_DIR=/home/fourier/dagger-gr3/data \
TASK='Pick the tomato from the table and place it inside the basket on the left.' \
./scripts/run_fastwam_dagger.sh
```

模型启动不会立即保存。当前 episode 第一次踩下脚踏时才开始记录，松开脚踏后继续保存模型恢复阶段，结束 episode 时完成落盘。

### 0.5 真机操作顺序

1. 启动 Dora 后，机器人不会自动开始模型控制。
2. 单击右手白键：机器人回到初始位置并进入准备状态。
3. 单击右手蓝键：开始连续模型控制。
4. 踩下脚踏：QNexo 外骨骼接管。
5. 松开脚踏：恢复模型控制。
6. 单击左手蓝键：执行一个 `ACTION_CHUNK_SIZE` 长度的单步 chunk，完成后进入 HOLD。
7. 单击右手红键：停止 episode，进入 `disengaged / PdStand`。
8. 下一次测试必须再次单击右手白键准备，然后再按蓝键。
9. 左手红键用于正常结束：平滑回到初始位置后进入 `disengaged / PdStand`。

按钮均为单击，不需要长按。

### 0.6 正确停止顺序

1. 先按右手红键或左手红键结束 episode。
2. 确认机器人已经进入 `disengaged / PdStand`。
3. 在 Dora 终端按 `Ctrl-C`，等待所有节点退出。
4. 最后在前台模型服务终端按 `Ctrl-C`。

后台模型服务停止前先核对 PID：

```bash
fastwam_server_pid="$(cat /home/fourier/dagger-gr3/logs/fastwam-server.pid)"
ps -fp "${fastwam_server_pid}"
kill -TERM "${fastwam_server_pid}"
```

不要在机器人仍由模型或外骨骼控制时直接杀死 Dora。

### 0.7 最短日常启动清单

```text
1. 机器人上电，SSH 启动机器人侧服务。
2. 主机终端一激活 conda fastwam。
3. 运行 dagger-gr3/scripts/start_fastwam_server.sh。
4. 等待 FastWAM server ready。
5. 主机终端二运行 dagger-gr3/scripts/run_fastwam_dagger.sh。
6. 右白准备，右蓝连续模型控制，脚踏接管，红键结束。
```

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

需要实验性 TensorRT action 后端时，在已有 PyTorch 2.7.1 环境中额外执行：

```bash
pip install --no-cache-dir -e '.[tensorrt]'
```

当前固定组合为 Torch-TensorRT 2.7.0 + TensorRT 10.9.0.34；不要单独升级 TensorRT 或 PyTorch 后继续沿用旧引擎。

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
