#!/usr/bin/env python3
"""Compare the standalone H3 VAE helper with the extracted official function."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from fastwam.models.h3wam import encode_h3_vae_condition_standalone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_encoders", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


class _Posterior:
    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean

    def sample(self, generator: torch.Generator) -> torch.Tensor:
        return self.mean + torch.randn(
            self.mean.shape, generator=generator, dtype=self.mean.dtype
        )


class _VAE:
    config = SimpleNamespace(
        latents_mean=[0.25, -0.75], latents_std=[1.5, 2.5]
    )

    def encode(self, pixels: torch.Tensor, return_dict: bool = False):
        if return_dict:
            raise AssertionError("official helper must request tuple output")
        mean = pixels.mean(dim=1, keepdim=True).expand(-1, 2, -1, -1, -1)
        return (_Posterior(mean),)


def main() -> None:
    args = parse_args()
    source = args.official_encoders.resolve().read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "encode_vae_condition"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module([function], []), str(args.official_encoders), "exec"), namespace)
    official = namespace["encode_vae_condition"]

    pixels = torch.arange(1 * 3 * 1 * 5 * 7, dtype=torch.uint8).reshape(1, 3, 1, 5, 7)
    kwargs = {
        "pixel_mean": (0.485, 0.456, 0.406),
        "pixel_std": (0.229, 0.224, 0.225),
        "encode_seed": 42,
    }
    expected = official(_VAE(), pixels, **kwargs)
    actual = encode_h3_vae_condition_standalone(_VAE(), pixels, **kwargs)
    report = {
        "official_source": str(args.official_encoders.resolve()),
        "official_file_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "official_function_sha256": hashlib.sha256(
            ast.unparse(function).encode()
        ).hexdigest(),
        "standalone_function_sha256": hashlib.sha256(
            inspect.getsource(encode_h3_vae_condition_standalone).encode()
        ).hexdigest(),
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "device": actual.device.type,
        "exact_equal": bool(torch.equal(actual, expected)),
        "max_abs": float((actual - expected).abs().max()),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not report["exact_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
