#!/usr/bin/env python3
"""Inspect the Diffusers MiniMax-H3 module contract without loading weights."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else [int(size) for size in shape]


def _linear_contract(module: Any) -> dict[str, Any]:
    return {
        "type": type(module).__name__,
        "weight": _shape(getattr(module, "weight", None)),
        "bias": _shape(getattr(module, "bias", None)),
    }


def _require_shape(name: str, actual: list[int] | None, expected: list[int]) -> None:
    if actual != expected:
        raise ValueError(f"{name} shape {actual} does not match expected {expected}")


def build_report(model_root: Path) -> dict[str, Any]:
    import diffusers
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3Transformer3DModel

    root = model_root.resolve()
    config = MiniMaxH3Transformer3DModel.load_config(root, subfolder="transformer")
    with init_empty_weights():
        model = MiniMaxH3Transformer3DModel.from_config(config)

    hidden_size = int(config["hidden_size"])
    in_channels = int(config["in_channels"])
    patch_size = tuple(int(value) for value in config["patch_size"])
    patch_volume = math.prod(patch_size)
    patch_channels = in_channels * patch_volume
    num_heads = int(config["num_attention_heads"])
    head_dim = int(config["attention_head_dim"])
    qkv_width = num_heads * head_dim
    num_layers = int(config["num_layers"])
    ffn_dim = int(config["ffn_dim"])
    time_embed_dim = int(config["time_embed_dim"])

    if len(model.transformer_blocks) != num_layers:
        raise ValueError(
            f"block count {len(model.transformer_blocks)} does not match {num_layers}"
        )
    block = model.transformer_blocks[0]
    projections = {
        "proj_in": _linear_contract(model.proj_in),
        "proj_out": _linear_contract(model.proj_out),
        "audio_proj_in": _linear_contract(model.audio_proj_in),
        "audio_proj_out": _linear_contract(model.audio_proj_out),
        "context_embedder": _linear_contract(model.context_embedder),
    }
    first_block = {
        "attention_q": _linear_contract(block.attn.to_q),
        "attention_k": _linear_contract(block.attn.to_k),
        "attention_v": _linear_contract(block.attn.to_v),
        "attention_out": _linear_contract(block.attn.to_out[0]),
        "ffn_in": _linear_contract(block.ff.net[0].proj),
        "ffn_out": _linear_contract(block.ff.net[2]),
        "adaln": _linear_contract(block.adaln_proj.linear),
        "norm1": type(block.norm1).__name__,
        "norm2": type(block.norm2).__name__,
    }

    _require_shape("proj_in", projections["proj_in"]["weight"], [hidden_size, patch_channels])
    _require_shape("proj_out", projections["proj_out"]["weight"], [patch_channels, hidden_size])
    for name in ("attention_q", "attention_k", "attention_v"):
        _require_shape(name, first_block[name]["weight"], [qkv_width, hidden_size])
    _require_shape(
        "attention_out", first_block["attention_out"]["weight"], [hidden_size, qkv_width]
    )
    _require_shape("ffn_in", first_block["ffn_in"]["weight"], [2 * ffn_dim, hidden_size])
    _require_shape("ffn_out", first_block["ffn_out"]["weight"], [hidden_size, ffn_dim])
    _require_shape(
        "adaln",
        first_block["adaln"]["weight"],
        [18 * hidden_size, time_embed_dim],
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "format": "h3dreamwam-contract-v1",
        "model_root": str(root),
        "diffusers_version": diffusers.__version__,
        "model_class": type(model).__name__,
        "parameter_count": int(parameter_count),
        "config": {
            key: value
            for key, value in sorted(config.items())
            if not str(key).startswith("_")
        },
        "derived": {
            "patch_volume": patch_volume,
            "patch_channels": patch_channels,
            "qkv_width": qkv_width,
            "rgb_flow_patch_channels": 2 * patch_channels,
        },
        "projections": projections,
        "first_block": first_block,
        "top_level_modules": [
            {"name": name, "type": type(module).__name__}
            for name, module in model.named_children()
        ],
    }


def main() -> None:
    args = parse_args()
    report = build_report(args.model)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"event": "contract", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
