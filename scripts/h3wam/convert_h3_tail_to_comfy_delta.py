#!/usr/bin/env python3
"""Convert a Diffusers H3 BF16 tail change into a Comfy feature-only delta."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


FEATURE_TIMESTEPS = (0.0, 0.999)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-tail", type=Path, required=True)
    parser.add_argument("--finetuned-tail", type=Path, required=True)
    parser.add_argument("--comfy-support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sinusoidal_timesteps(timesteps: torch.Tensor, embedding_dim: int = 256) -> torch.Tensor:
    half = embedding_dim // 2
    exponent = -math.log(10000.0) * torch.arange(
        half, dtype=torch.float32, device=timesteps.device
    ) / half
    frequencies = torch.exp(exponent)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)


def linear(x: torch.Tensor, handle, prefix: str) -> torch.Tensor:
    return F.linear(
        x,
        handle.get_tensor(prefix + ".weight").float(),
        handle.get_tensor(prefix + ".bias").float(),
    )


def official_adaln_input(base_handle, timesteps: torch.Tensor) -> torch.Tensor:
    embedded = sinusoidal_timesteps(timesteps)
    hidden = F.silu(linear(embedded, base_handle, "time_embedder.linear_1"))
    temb = linear(hidden, base_handle, "time_embedder.linear_2")
    return F.silu(temb)


def interpolate_table(table: torch.Tensor, timestep: float) -> torch.Tensor:
    position = min(max(float(timestep), 0.0), 1.0) * (table.shape[0] - 1)
    lower = min(math.floor(position), table.shape[0] - 2)
    fraction = position - lower
    return torch.lerp(table[lower], table[lower + 1], fraction)


def mapped_linear_deltas(base_handle, fine_handle, block: int) -> dict[str, torch.Tensor]:
    source = f"transformer_blocks.{block}."

    def delta(suffix: str) -> torch.Tensor:
        return (
            fine_handle.get_tensor(source + suffix).float()
            - base_handle.get_tensor(source + suffix).float()
        ).to(torch.bfloat16)

    return {
        f"blocks.{block}.attn.qkv_proj.weight": torch.cat(
            (delta("attn.to_q.weight"), delta("attn.to_k.weight"), delta("attn.to_v.weight")),
            dim=0,
        ).contiguous(),
        f"blocks.{block}.attn.out_proj.weight": delta("attn.to_out.0.weight").contiguous(),
        f"blocks.{block}.mlp.fc1.weight": delta("ff.net.0.proj.weight").contiguous(),
        f"blocks.{block}.mlp.fc2.weight": delta("ff.net.2.weight").contiguous(),
        f"blocks.{block}.norm1.weight": delta("norm1.weight").contiguous(),
        f"blocks.{block}.norm2.weight": delta("norm2.weight").contiguous(),
        f"blocks.{block}.attn.q_norm.weight": delta("attn.norm_q.weight").contiguous(),
        f"blocks.{block}.attn.k_norm.weight": delta("attn.norm_k.weight").contiguous(),
    }


def mapped_adaln_delta(
    base_handle,
    fine_handle,
    table: torch.Tensor,
    adaln_input: torch.Tensor,
    block: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    source = f"transformer_blocks.{block}.adaln_proj.linear"
    delta_weight = (
        fine_handle.get_tensor(source + ".weight").float()
        - base_handle.get_tensor(source + ".weight").float()
    )
    delta_bias = (
        fine_handle.get_tensor(source + ".bias").float()
        - base_handle.get_tensor(source + ".bias").float()
    )
    basis = torch.stack(
        [interpolate_table(table, timestep) for timestep in FEATURE_TIMESTEPS]
    ).float()
    augmented = torch.cat((basis, torch.ones((basis.shape[0], 1))), dim=1).to(device)
    target = F.linear(
        adaln_input.to(device),
        delta_weight.to(device),
        delta_bias.to(device),
    )
    coefficients = torch.linalg.pinv(augmented) @ target
    reconstructed = augmented @ coefficients
    max_error = float((reconstructed - target).abs().max().item())
    curve_weight = coefficients[:-1].T.to(device="cpu", dtype=torch.bfloat16).contiguous()
    curve_bias = coefficients[-1].to(device="cpu", dtype=torch.bfloat16).contiguous()
    return curve_weight, curve_bias, max_error


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (
        safe_open(args.base_tail.resolve(), framework="pt", device="cpu") as base,
        safe_open(args.finetuned_tail.resolve(), framework="pt", device="cpu") as fine,
        safe_open(args.comfy_support.resolve(), framework="pt", device="cpu") as support,
    ):
        base_metadata = base.metadata() or {}
        fine_metadata = fine.metadata() or {}
        if base_metadata.get("format") != "h3wam_diffusers_tail_bf16_v1":
            raise ValueError("base tail has the wrong format")
        if fine_metadata.get("format") != "h3wam_diffusers_tail_bf16_v1":
            raise ValueError("finetuned tail has the wrong format")
        table = support.get_tensor("adaln_t_table").float()
        timesteps = torch.tensor(FEATURE_TIMESTEPS, dtype=torch.float32)
        adaln_input = official_adaln_input(base, timesteps)
        blocks = sorted(
            {
                int(key.split(".")[1])
                for key in fine.keys()
                if key.startswith("transformer_blocks.")
            }
        )
        state: dict[str, torch.Tensor] = {}
        errors = {}
        for block in blocks:
            state.update(mapped_linear_deltas(base, fine, block))
            weight, bias, error = mapped_adaln_delta(
                base, fine, table, adaln_input, block, device
            )
            state[f"blocks.{block}.adaln_proj.linear.weight"] = weight
            state[f"blocks.{block}.adaln_proj.linear.bias"] = bias
            errors[str(block)] = error

    nonzero = sum(int(torch.count_nonzero(value)) for value in state.values())
    numel = sum(value.numel() for value in state.values())
    metadata = {
        "format": "h3wam_comfy_feature_delta_v1",
        "base_tail": str(args.base_tail.resolve()),
        "finetuned_tail": str(args.finetuned_tail.resolve()),
        "source_step": fine_metadata.get("step", "unknown"),
        "feature_timesteps": json.dumps(FEATURE_TIMESTEPS),
        "blocks": json.dumps(blocks),
        "adaln_max_errors": json.dumps(errors, sort_keys=True),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, output, metadata=metadata)
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(output),
                "bytes": output.stat().st_size,
                "tensor_count": len(state),
                "numel": numel,
                "nonzero": nonzero,
                "nonzero_fraction": nonzero / numel,
                "metadata": metadata,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
