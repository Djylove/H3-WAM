#!/usr/bin/env python3
"""Cache MiniMax H3 Qwen text/first-frame conditioning for a WAM window."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("window", type=Path)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Encode only the task text; the current image stays in H3's first-frame branch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    window_path = args.window.resolve()
    comfy_root = args.comfy_root.resolve()
    encoder_path = args.text_encoder.resolve()
    output = args.output.resolve()
    sys.path.insert(0, str(comfy_root))

    import comfy.model_management as model_management
    import comfy.sd

    window = torch.load(window_path, map_location="cpu", weights_only=False)
    load_started = time.perf_counter()
    clip = comfy.sd.load_clip(
        ckpt_paths=[str(encoder_path)],
        clip_type=comfy.sd.CLIPType.MINIMAX,
    )
    load_seconds = time.perf_counter() - load_started

    encode_started = time.perf_counter()
    if args.text_only:
        tokens = clip.tokenize(window["task"])
    else:
        tokens = clip.tokenize(window["task"], images=[window["first_frame_pixels"]])
    conditioning = clip.encode_from_tokens_scheduled(tokens, show_pbar=False)
    if model_management.get_torch_device().type == "cuda":
        torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - encode_started
    context, metadata = conditioning[0]
    token_tags = metadata.get("minimax_token_tags")
    if token_tags is None:
        raise RuntimeError("MiniMax text encoder did not return minimax_token_tags")
    if context.ndim != 3 or context.shape[0] != 1 or context.shape[-1] != 5120:
        raise RuntimeError(f"unexpected H3 context shape {tuple(context.shape)}")

    artifact = {
        "context": context.cpu(),
        "token_tags": token_tags.cpu(),
        "task": window["task"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "context": list(context.shape),
                "token_tags": list(token_tags.shape),
                "video_tag_tokens": int((token_tags == 0).sum().item()),
                "text_tag_tokens": int((token_tags == 1).sum().item()),
                "text_only": args.text_only,
                "text_encoder_load_seconds": load_seconds,
                "text_encoder_encode_seconds": encode_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
