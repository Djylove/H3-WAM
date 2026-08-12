#!/usr/bin/env python3
"""Export full BF16 H3 tail weights from base or rank-sharded FSDP checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FORMAT = "h3wam_fsdp_same_world_size_v2"
JOINT_FORMAT = "h3wam-fsdp-local-bf16-v1"
TIME_EMBEDDER_KEYS = (
    "time_embedder.linear_1.weight",
    "time_embedder.linear_1.bias",
    "time_embedder.linear_2.weight",
    "time_embedder.linear_2.bias",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--last-blocks", type=int, default=2)
    parser.add_argument(
        "--include-time-embedder",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def clean_fsdp_name(name: str) -> str:
    return name.replace("_fsdp_wrapped_module.", "")


class BaseWeights:
    def __init__(self, transformer_root: Path) -> None:
        self.root = transformer_root
        index_path = self.root / "diffusion_pytorch_model.safetensors.index.json"
        self.index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]

    def tensor(self, key: str) -> torch.Tensor:
        filename = self.index[key]
        with safe_open(self.root / filename, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def shape(self, key: str) -> tuple[int, ...]:
        filename = self.index[key]
        with safe_open(self.root / filename, framework="pt", device="cpu") as handle:
            return tuple(handle.get_slice(key).get_shape())


def tail_keys(base: BaseWeights, last_blocks: int) -> list[str]:
    block_indices = {
        int(key.split(".")[1])
        for key in base.index
        if key.startswith("transformer_blocks.")
        and len(key.split(".")) > 2
        and key.split(".")[1].isdigit()
    }
    if not block_indices:
        raise RuntimeError("official H3 index contains no transformer blocks")
    depth = max(block_indices) + 1
    if not 1 <= last_blocks <= depth:
        raise ValueError(f"last-blocks must be in [1,{depth}]")
    selected = set(range(depth - last_blocks, depth))
    return sorted(
        key
        for key in base.index
        if key.startswith("transformer_blocks.")
        and int(key.split(".")[1]) in selected
    )


def checkpoint_weights(
    checkpoint_dir: Path, base: BaseWeights, expected_keys: list[str]
) -> tuple[dict[str, torch.Tensor], dict]:
    legacy_manifest = checkpoint_dir / "checkpoint.json"
    joint_manifest = checkpoint_dir / "manifest.json"
    if joint_manifest.is_file():
        manifest = json.loads(joint_manifest.read_text(encoding="utf-8"))
        if manifest.get("format") != JOINT_FORMAT:
            raise ValueError("unsupported joint H3-WAM checkpoint format")
        rank_files = [
            checkpoint_dir / f"h3_rank{rank:05d}.pt"
            for rank in range(int(manifest["world_size"]))
        ]
        state_key = "parameters"
    elif legacy_manifest.is_file():
        manifest = json.loads(legacy_manifest.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT:
            raise ValueError("unsupported H3-WAM checkpoint format")
        rank_files = [checkpoint_dir / name for name in manifest["rank_files"]]
        state_key = "trainable_state"
    else:
        raise FileNotFoundError(f"checkpoint manifest is missing below {checkpoint_dir}")
    shards: dict[str, list[torch.Tensor]] = {key: [] for key in expected_keys}
    seen_names: set[str] | None = None
    for rank, path in enumerate(rank_files):
        artifact = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
        if artifact.get("format") not in (FORMAT, JOINT_FORMAT):
            raise ValueError(f"rank {rank} has an unsupported checkpoint format")
        if int(artifact["rank"]) != rank or int(artifact["world_size"]) != len(rank_files):
            raise ValueError(f"rank metadata mismatch in {path}")
        named = {
            clean_fsdp_name(name): value
            for name, value in artifact[state_key].items()
        }
        if seen_names is None:
            seen_names = set(named)
        elif set(named) != seen_names:
            raise ValueError("rank trainable parameter names differ")
        for key in expected_keys:
            value = named.get(key)
            if value is None:
                raise KeyError(f"checkpoint does not contain {key}")
            if value.numel():
                shards[key].append(value.detach().reshape(-1).clone())
        del artifact, named

    state = {}
    for key in expected_keys:
        shape = base.shape(key)
        expected_numel = 1
        for dimension in shape:
            expected_numel *= dimension
        value = torch.cat(shards[key])
        if value.numel() != expected_numel:
            raise ValueError(
                f"reconstructed {key} has {value.numel()} values, expected {expected_numel}"
            )
        state[key] = value.view(shape).to(torch.bfloat16).contiguous()
    return state, manifest


def main() -> None:
    args = parse_args()
    transformer_root = args.model.resolve() / "transformer"
    base = BaseWeights(transformer_root)
    keys = tail_keys(base, args.last_blocks)
    if args.checkpoint_dir is None:
        state = {key: base.tensor(key).to(torch.bfloat16).contiguous() for key in keys}
        source = "official_base"
        step = 0
        world_size = 0
    else:
        checkpoint_dir = args.checkpoint_dir.resolve()
        state, manifest = checkpoint_weights(checkpoint_dir, base, keys)
        source = str(checkpoint_dir)
        step = int(manifest["step"])
        world_size = int(manifest["world_size"])
    if args.include_time_embedder:
        for key in TIME_EMBEDDER_KEYS:
            state[key] = base.tensor(key).to(torch.float32).contiguous()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    metadata = {
        "format": "h3wam_diffusers_tail_bf16_v1",
        "source": source,
        "step": str(step),
        "world_size": str(world_size),
        "last_blocks": str(args.last_blocks),
        "include_time_embedder": str(args.include_time_embedder).lower(),
    }
    save_file(state, output, metadata=metadata)
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(output),
                "metadata": metadata,
                "tensor_count": len(state),
                "bytes": output.stat().st_size,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
