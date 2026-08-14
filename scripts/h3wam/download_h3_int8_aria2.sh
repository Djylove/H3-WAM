#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 OUTPUT_ROOT ARIA2_ROOT" >&2
    exit 2
fi

output_root=$1
aria2_root=$2
aria2_bin="$aria2_root/usr/bin/aria2c"
aria2_lib="$aria2_root/usr/lib/x86_64-linux-gnu"
base_url="https://www.modelscope.cn/models/Comfy-Org/MiniMax-H3/resolve/master"

if [[ ! -x "$aria2_bin" ]]; then
    echo "aria2c not found at $aria2_bin" >&2
    exit 1
fi

download_one() {
    local relative_path=$1
    local expected_sha256=$2
    local target_dir="$output_root/${relative_path%/*}"
    local target_name="${relative_path##*/}"

    mkdir -p "$target_dir"
    LD_LIBRARY_PATH="$aria2_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$aria2_bin" \
        --continue=true \
        --max-connection-per-server=16 \
        --split=16 \
        --min-split-size=32M \
        --file-allocation=none \
        --max-tries=0 \
        --retry-wait=5 \
        --timeout=60 \
        --summary-interval=30 \
        --console-log-level=notice \
        --dir="$target_dir" \
        --out="$target_name" \
        "$base_url/$relative_path"

    printf '%s  %s\n' "$expected_sha256" "$target_dir/$target_name" | sha256sum -c -
}

download_one \
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
download_one \
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
    "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6"
download_one \
    "vae/minimax_h3_video_vae_fp16.safetensors" \
    "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"
