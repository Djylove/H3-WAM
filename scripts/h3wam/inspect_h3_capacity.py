#!/usr/bin/env python3
"""Compare an H3 checkpoint's logical BF16 size with local GPU capacity."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--target-gpu-gib",
        type=float,
        help="Capacity to evaluate instead of the currently visible GPU.",
    )
    parser.add_argument(
        "--trainable-last-blocks",
        type=int,
        nargs="+",
        default=[2, 4, 8, 10, 50],
        help="Partial-unfreeze sizes to estimate.",
    )
    parser.add_argument(
        "--activation-headroom-gib",
        type=float,
        default=8.0,
        help="Reserved activation/runtime memory in each training estimate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    counts: dict[str, int] = defaultdict(int)
    block_elements: dict[int, int] = defaultdict(int)
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor_slice = handle.get_slice(key)
            elements = 1
            for dimension in tensor_slice.get_shape():
                elements *= dimension
            counts[tensor_slice.get_dtype()] += elements
            parts = key.split(".")
            if len(parts) > 1 and parts[0] == "blocks" and parts[1].isdigit():
                block_elements[int(parts[1])] += elements
    storage_bytes = sum(count * DTYPE_BYTES[dtype] for dtype, count in counts.items())
    logical_elements = sum(counts.values())
    bf16_bytes = logical_elements * 2
    visible_gpu_total = (
        torch.cuda.get_device_properties(0).total_memory
        if torch.cuda.is_available()
        else None
    )
    target_gpu_gib = args.target_gpu_gib
    if target_gpu_gib is None and visible_gpu_total is not None:
        target_gpu_gib = visible_gpu_total / 2**30
    block_indices = sorted(block_elements)
    estimates = []
    for requested in args.trainable_last_blocks:
        if requested <= 0 or requested > len(block_indices):
            raise ValueError(
                f"trainable-last-blocks must be within 1..{len(block_indices)}"
            )
        selected = block_indices[-requested:]
        trainable = sum(block_elements[index] for index in selected)
        # Conservative BF16 training estimate: all BF16 model weights, BF16
        # gradients for trainable tensors, FP32 Adam first/second moments, plus
        # explicit runtime headroom. No optimizer master-weight copy is assumed.
        weights_gib = bf16_bytes / 2**30
        gradients_gib = trainable * 2 / 2**30
        adam_states_gib = trainable * 8 / 2**30
        estimated_gib = (
            weights_gib
            + gradients_gib
            + adam_states_gib
            + args.activation_headroom_gib
        )
        estimates.append(
            {
                "last_blocks": requested,
                "trainable_parameters_billions": trainable / 1e9,
                "bf16_gradients_gib": gradients_gib,
                "fp32_adam_states_gib": adam_states_gib,
                "activation_headroom_gib": args.activation_headroom_gib,
                "estimated_total_gib": estimated_gib,
                "fits_target": (
                    None if target_gpu_gib is None else estimated_gib < target_gpu_gib
                ),
            }
        )
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_file_gib": checkpoint.stat().st_size / 2**30,
        "tensor_storage_gib": storage_bytes / 2**30,
        "logical_tensor_elements_billions": logical_elements / 1e9,
        "bf16_weights_only_gib": bf16_bytes / 2**30,
        "visible_gpu_total_gib": (
            None if visible_gpu_total is None else visible_gpu_total / 2**30
        ),
        "target_gpu_gib": target_gpu_gib,
        "bf16_weights_only_fit": (
            None if target_gpu_gib is None else bf16_bytes / 2**30 < target_gpu_gib
        ),
        "full_adamw_estimated_gib": (
            bf16_bytes / 2**30
            + logical_elements * 2 / 2**30
            + logical_elements * 8 / 2**30
            + args.activation_headroom_gib
        ),
        "partial_unfreeze_estimates": estimates,
        "dtype_elements": dict(counts),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
