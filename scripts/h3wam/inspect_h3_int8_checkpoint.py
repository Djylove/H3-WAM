#!/usr/bin/env python3
"""Audit and smoke-test the standalone MiniMax-H3 INT8 checkpoint.

This command imports no ComfyUI modules. It validates the tensorwise ConvRot
contract and can execute one real quantized projection with the public
``comfy-kitchen`` kernel on CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from fastwam.models.h3wam import ConvRotInt8Linear, parse_int8_marker


EXPECTED_SHA256 = "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
DEFAULT_SMOKE_PREFIX = "blocks.0.attn.qkv_proj"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_checkpoint(path: Path, *, verify_hash: bool) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        marker_keys = sorted(key for key in keys if key.endswith(".comfy_quant"))
        quantized = []
        for marker_key in marker_keys:
            prefix = marker_key.removesuffix(".comfy_quant")
            required = {f"{prefix}.weight", f"{prefix}.weight_scale"}
            missing = sorted(required - keys)
            if missing:
                raise ValueError(f"{prefix} is missing {missing}")
            weight = checkpoint.get_tensor(f"{prefix}.weight")
            scale = checkpoint.get_tensor(f"{prefix}.weight_scale")
            metadata = parse_int8_marker(checkpoint.get_tensor(marker_key))
            if weight.dtype != torch.int8 or weight.ndim != 2:
                raise ValueError(f"{prefix} has invalid INT8 weight")
            if scale.dtype != torch.float32 or tuple(scale.shape) != (weight.shape[0], 1):
                raise ValueError(f"{prefix} has invalid weight scale")
            if metadata["convrot"] and weight.shape[1] % metadata["convrot_groupsize"]:
                raise ValueError(f"{prefix} violates its ConvRot group size")
            quantized.append(prefix)

        block_ids = sorted(
            {
                int(key.split(".")[1])
                for key in keys
                if key.startswith("blocks.") and key.split(".")[1].isdigit()
            }
        )
        result: dict[str, object] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "tensor_count": len(keys),
            "quantized_linear_count": len(quantized),
            "transformer_block_count": len(block_ids),
            "block_ids_contiguous": block_ids == list(range(50)),
            "quantized_linears_per_block": {
                str(index): sum(prefix.startswith(f"blocks.{index}.") for prefix in quantized)
                for index in block_ids
            },
            "has_curve_adaln": "adaln_t_table" in keys,
            "has_token_refiner": any(key.startswith("token_refiner.") for key in keys),
            "format": "int8_tensorwise_convrot",
        }
    if verify_hash:
        result["sha256"] = _sha256(path)
        result["sha256_matches_expected"] = result["sha256"] == EXPECTED_SHA256
        if not result["sha256_matches_expected"]:
            raise ValueError("checkpoint SHA256 does not match the pinned H3 INT8 artifact")
    return result


def smoke_kernel(path: Path, *, prefix: str, device: str) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real comfy-kitchen kernel smoke test")
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        bias_key = f"{prefix}.bias"
        layer = ConvRotInt8Linear.from_checkpoint_tensors(
            weight=checkpoint.get_tensor(f"{prefix}.weight"),
            weight_scale=checkpoint.get_tensor(f"{prefix}.weight_scale"),
            marker=checkpoint.get_tensor(f"{prefix}.comfy_quant"),
            bias=checkpoint.get_tensor(bias_key) if bias_key in keys else None,
        ).to(device)
    torch.manual_seed(42)
    sample = torch.randn(2, layer.in_features, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        output = layer(sample)
    if output.shape != (2, layer.out_features) or not torch.isfinite(output).all():
        raise RuntimeError("INT8 kernel returned an invalid output")
    return {
        "prefix": prefix,
        "device": str(torch.device(device)),
        "input_shape": list(sample.shape),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "output_abs_mean": float(output.float().abs().mean()),
        "output_abs_max": float(output.float().abs().max()),
        "finite": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--verify-hash", action="store_true")
    parser.add_argument("--smoke-kernel", action="store_true")
    parser.add_argument("--smoke-prefix", default=DEFAULT_SMOKE_PREFIX)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_checkpoint(args.checkpoint, verify_hash=args.verify_hash)
    if args.smoke_kernel:
        report["kernel_smoke"] = smoke_kernel(
            args.checkpoint, prefix=args.smoke_prefix, device=args.device
        )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
