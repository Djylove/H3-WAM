#!/usr/bin/env python3
"""Lossily encode an H3 Comfy feature delta as per-tensor INT8 for transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    state = {}
    errors = {}
    with safe_open(args.input.resolve(), framework="pt", device="cpu") as handle:
        source_metadata = handle.metadata() or {}
        if source_metadata.get("format") != "h3wam_comfy_feature_delta_v1":
            raise ValueError("input is not a dense H3 Comfy feature delta")
        for key in handle.keys():
            value = handle.get_tensor(key).float()
            max_abs = float(value.abs().max().item())
            scale = max_abs / 127.0 if max_abs else 1.0
            quantized = (value / scale).round().clamp(-127, 127).to(torch.int8)
            reconstructed = quantized.float() * scale
            error = reconstructed - value
            rms = float(error.square().mean().sqrt().item())
            signal_rms = float(value.square().mean().sqrt().item())
            errors[key] = {
                "max_abs": float(error.abs().max().item()),
                "relative_rms": rms / signal_rms if signal_rms else 0.0,
            }
            state[key] = quantized.contiguous()
            state[key + ".scale"] = torch.tensor(scale, dtype=torch.float32)
        metadata = dict(source_metadata)
    metadata["format"] = "h3wam_comfy_feature_delta_int8_v1"
    metadata["dense_source"] = str(args.input.resolve())
    metadata["quantization"] = "symmetric_per_tensor_int8"
    metadata["quantization_errors"] = json.dumps(errors, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, output, metadata=metadata)
    worst_relative = max(value["relative_rms"] for value in errors.values())
    worst_max_abs = max(value["max_abs"] for value in errors.values())
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(output),
                "bytes": output.stat().st_size,
                "tensor_count": len(errors),
                "worst_relative_rms": worst_relative,
                "worst_max_abs_error": worst_max_abs,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
