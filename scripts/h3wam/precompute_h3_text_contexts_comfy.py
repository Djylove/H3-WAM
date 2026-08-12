#!/usr/bin/env python3
"""Batch-cache text-only H3 contexts with a local Comfy Qwen checkpoint.

This is an offline preprocessing shortcut, not a runtime Comfy dependency.
It is useful when the official 32B Qwen conditioner is still being staged on
the training host: the 40 LIBERO task embeddings can be produced once on the
development workstation and copied with the dataset cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_contexts", type=Path)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comfy_root = args.comfy_root.resolve()
    encoder_path = args.text_encoder.resolve()
    if not encoder_path.is_file():
        raise FileNotFoundError(encoder_path)
    contexts: dict[str, str] = json.loads(
        args.task_contexts.resolve().read_text(encoding="utf-8")
    )
    if not contexts:
        raise ValueError("task context mapping is empty")
    sys.path.insert(0, str(comfy_root))

    import comfy.model_management as model_management
    import comfy.sd

    started = time.perf_counter()
    clip = comfy.sd.load_clip(
        ckpt_paths=[str(encoder_path)],
        clip_type=comfy.sd.CLIPType.MINIMAX,
    )
    load_seconds = time.perf_counter() - started
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    encoded = 0
    for context_id, task in sorted(contexts.items()):
        output = output_root / f"{context_id}.pt"
        if output.exists() and not args.overwrite:
            continue
        tokens = clip.tokenize(task)
        conditioning = clip.encode_from_tokens_scheduled(tokens, show_pbar=False)
        if model_management.get_torch_device().type == "cuda":
            torch.cuda.synchronize()
        context, metadata = conditioning[0]
        token_tags = metadata.get("minimax_token_tags")
        if token_tags is None:
            raise RuntimeError("MiniMax text encoder did not return token tags")
        context = context.detach().cpu()
        token_tags = token_tags.detach().cpu().reshape(-1).long()
        if context.ndim != 3 or tuple(context.shape[:1]) != (1,) or context.shape[-1] != 5120:
            raise RuntimeError(f"unexpected H3 context shape {tuple(context.shape)}")
        if token_tags.numel() != context.shape[1] or not bool((token_tags == 1).all()):
            raise RuntimeError(
                f"text-only tags {tuple(token_tags.shape)} do not match {tuple(context.shape)}"
            )
        artifact = {
            "context": context,
            "token_tags": token_tags,
            "task": task,
            "text_only": True,
            "text_encoder": encoder_path.name,
        }
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        torch.save(artifact, temporary)
        os.replace(temporary, output)
        encoded += 1
        print(
            json.dumps(
                {
                    "event": "context",
                    "context_id": context_id,
                    "shape": list(context.shape),
                    "tokens": int(token_tags.numel()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "event": "complete",
                "contexts": len(contexts),
                "encoded": encoded,
                "load_seconds": load_seconds,
                "seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
