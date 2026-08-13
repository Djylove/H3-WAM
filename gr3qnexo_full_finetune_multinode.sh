#!/bin/bash
# DreamZero GR3-QNEXO Full Fine-Tuning — Multi-Node (32 GPUs / 4 nodes × 8 GPUs)
#
# For DLC (PAI-DLC) cluster. DLC sets these env vars automatically:
#   WORLD_SIZE — total number of nodes
#   RANK       — this node's rank (0-based)
#   MASTER_ADDR — master node address
#   MASTER_PORT — master node port
#
# Usage on DLC:
#   bash scripts/train/gr3qnexo_full_finetune_multinode.sh
#
# Or manually for 4-node setup:
#   # On node 0:
#   WORLD_SIZE=4 RANK=0 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 bash scripts/train/gr3qnexo_full_finetune_multinode.sh
#   # On node 1:
#   WORLD_SIZE=4 RANK=1 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 bash scripts/train/gr3qnexo_full_finetune_multinode.sh
#   # etc.
#
# Architecture: full fine-tune (all 14B parameters trainable)
# DeepSpeed: ZeRO-3 + CPU offload (param + optimizer)
# Data: 126 tasks (26 lerobotv3.0 + 100 fourier-data)

set -e

# ============ CONDA ============
# Directly activate dreamzero env — bypasses conda shebang issues on DLC
export PATH="/mnt/cpfs/ben/miniconda3/envs/dreamzero/bin:$PATH"
export CONDA_DEFAULT_ENV=dreamzero
export CONDA_PREFIX="/mnt/cpfs/ben/miniconda3/envs/dreamzero"
cd /mnt/cpfs/ben/dreamzero

# Install package in dev mode (pip shebang broken, use python -m pip)
python -m pip install -e . --no-deps --quiet 2>/dev/null || true

# Fix broken symlinks: ensure /mnt/workspace/ points to /mnt/oss/home/ on DLC
if [ ! -e /mnt/workspace ] && [ -d /mnt/oss/home ]; then
    ln -sf /mnt/oss/home /mnt/workspace 2>/dev/null || true
fi

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
# Prevent HF downloads on offline clusters — files must be local
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1

# ============ WANDB ============
export WANDB_API_KEY=${WANDB_API_KEY:?Set WANDB_API_KEY in the environment before launching}
export WANDB_ENTITY=${WANDB_ENTITY:-"elgceben"}

# ============ PROXY (for DLC clusters behind firewall) ============
# Uncomment if needed:
# export http_proxy="https://benqingwei:M1u1Nzw0MrAzBSD4RSGv6uFTqzPCTpYfDzOSip3tmEEPGU00HhKErL9JpJHH@aliyun-proxy.pjlab.org.cn:13128"
# export https_proxy="$http_proxy"
# export HTTP_PROXY="$http_proxy"
# export HTTPS_PROXY="$http_proxy"

# ============ USER CONFIGURATION ============
# Data paths (must be accessible from all nodes via shared filesystem)
GR3QNEXO_DATA_ROOT=${GR3QNEXO_DATA_ROOT:-"/mnt/oss/Data/lerobotv3.0"}
FOURIER_DATA_ROOT=${FOURIER_DATA_ROOT:-"/mnt/oss/Data/fourier-data"}

# Output directory (shared filesystem)
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/cpfs/ben/dreamzero_gr3_ckpt"}

# GPUs per node
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Model weight paths
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
PRETRAINED_DIR=${PRETRAINED_DIR:-"./checkpoints/DreamZero-AgiBot"}

# Training hyperparameters
MAX_STEPS=${MAX_STEPS:-10000}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-1}
SAVE_STEPS=${SAVE_STEPS:-500}
# =============================================

# ============ DLC MULTI-NODE SETUP ============
# DLC provides WORLD_SIZE (num nodes), RANK, MASTER_ADDR, MASTER_PORT
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER=${MASTER_ADDR:-"127.0.0.1"}
PORT=${MASTER_PORT:-29500}

echo "============================================"
echo "  DreamZero Full Fine-Tune (Multi-Node)"
echo "  Nodes: $NNODES × $GPUS_PER_NODE GPUs = $((NNODES * GPUS_PER_NODE)) total"
echo "  This node: rank $NODE_RANK"
echo "  Master: $MASTER:$PORT"
echo "  Data: $GR3QNEXO_DATA_ROOT + $FOURIER_DATA_ROOT"
echo "  Output: $OUTPUT_DIR"
echo "  Architecture: full (all params trainable)"
echo "  DeepSpeed: ZeRO-3 + CPU offload"
echo "  LR: $LEARNING_RATE, BS/GPU: $BATCH_SIZE, Steps: $MAX_STEPS"
echo "============================================"

# Validate paths
for p in "$WAN_CKPT_DIR" "$TOKENIZER_DIR" "$PRETRAINED_DIR"; do
    if [ ! -d "$p" ]; then
        echo "ERROR: Required directory not found: $p"
        exit 1
    fi
done

# Validate data paths
echo "Checking data paths..."
for ds_root in "$GR3QNEXO_DATA_ROOT" "$FOURIER_DATA_ROOT"; do
    if [ ! -d "$ds_root" ]; then
        echo "ERROR: Data root not found: $ds_root"
        echo "Available /mnt/ mounts:"
        ls /mnt/ 2>/dev/null
        exit 1
    fi
    # Check first dataset has metadata
    first_ds=$(ls -d "$ds_root"/*/ 2>/dev/null | head -1)
    if [ -n "$first_ds" ] && [ ! -f "${first_ds}meta/modality.json" ]; then
        echo "WARNING: No modality.json in $first_ds — metadata may be missing"
    fi
done
echo "Data paths OK"

# Run metadata generation on this node (idempotent, only creates missing files)
echo "Ensuring metadata exists for all datasets..."
for ds_path in "$GR3QNEXO_DATA_ROOT"/*/ "$FOURIER_DATA_ROOT"/*/; do
    [ ! -d "$ds_path/data" ] && continue
    if [ ! -f "$ds_path/meta/modality.json" ]; then
        echo "Generating metadata for $(basename $ds_path)..."
        python scripts/data/convert_lerobot_to_gear.py \
            --dataset-path "$ds_path" \
            --embodiment-tag gr3qnexo \
            --state-keys '{"joints": [0, 31, "observation.state"], "base": [31, 33, "observation.state"]}' \
            --action-keys '{"joints": [0, 31, "action"], "base": [31, 37, "action"]}' \
            --relative-action-keys joints \
            --action-horizon 24 --force 2>&1 | tail -1
    fi
done
echo "Metadata check complete"

python -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER \
    --master_port=$PORT \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/gr3qnexo_all_relative \
    wandb_project=dreamzero \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=1 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=$LEARNING_RATE \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2_offload.json" \
    training_args.gradient_accumulation_steps=$GRADIENT_ACCUMULATION \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$BATCH_SIZE \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=1 \
    logging_steps=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=2 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=false \
    max_chunk_size=9 \
    frame_seqlen=220 \
    save_strategy=steps \
    gr3qnexo_data_root=$GR3QNEXO_DATA_ROOT \
    fourier_data_root=$FOURIER_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_DIR
