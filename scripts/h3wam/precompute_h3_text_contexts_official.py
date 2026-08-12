#!/usr/bin/env python3
"""Encode the 40 LIBERO task prompts with H3's official BF16 Qwen3-VL conditioner."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_contexts", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-encoder-layer", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_root = args.model.resolve()
    encoder_root = model_root / "text_encoder"
    if not (encoder_root / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"official H3 text encoder is incomplete: {encoder_root}")
    contexts: dict[str, str] = json.loads(
        args.task_contexts.resolve().read_text(encoding="utf-8")
    )
    if not contexts:
        raise ValueError("task context mapping is empty")
    from diffusers.modular_pipelines.minimax_h3.encoders import (
        get_qwen3vl_prompt_embeds,
    )
    from transformers import (
        Qwen2TokenizerFast,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    started = time.perf_counter()
    tokenizer = Qwen2TokenizerFast.from_pretrained(encoder_root)
    processor = Qwen3VLProcessor.from_pretrained(encoder_root)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        encoder_root,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    text_encoder.eval()
    load_seconds = time.perf_counter() - started
    output_root = args.cache_root.resolve() / "contexts"
    output_root.mkdir(parents=True, exist_ok=True)
    encoded = 0
    for context_id, task in sorted(contexts.items()):
        output = output_root / f"{context_id}.pt"
        if output.exists() and not args.overwrite:
            continue
        token_ids = tokenizer(task, add_special_tokens=False)["input_ids"]
        with torch.inference_mode():
            prompt_embeds = get_qwen3vl_prompt_embeds(
                text_encoder,
                processor,
                token_ids,
                {},
                text_encoder_layer=args.text_encoder_layer,
                device=device,
                dtype=torch.bfloat16,
            )
        artifact = {
            "context": prompt_embeds.cpu(),
            "token_tags": torch.ones(len(token_ids), dtype=torch.long),
            "task": task,
            "text_only": True,
            "text_encoder": "official MiniMax-H3 BF16 Qwen3-VL",
            "text_encoder_layer": args.text_encoder_layer,
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
                    "tokens": len(token_ids),
                    "shape": list(prompt_embeds.shape),
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
