#!/usr/bin/env python3
"""Export normalized action targets predicted by a trained H3 feature policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fastwam.models.h3wam import H3FeatureActionTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    rows = [
        json.loads(line)
        for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    if checkpoint.get("policy_type") != "h3_feature_action":
        raise ValueError("checkpoint is not an H3 feature-action policy")
    if checkpoint.get("objective") != "regression":
        raise ValueError("teacher target export requires a regression checkpoint")
    horizon = int(checkpoint["action_horizon"])
    model = H3FeatureActionTransformer(
        action_dim=int(checkpoint["action_dim"]),
        state_dim=int(checkpoint["state_dim"]),
        h3_feature_dim=int(checkpoint["feature_shape"][-1]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        num_heads=int(checkpoint["num_heads"]),
        ffn_dim=int(checkpoint["ffn_dim"]),
        num_action_modes=int(checkpoint["num_action_modes"]),
    ).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    cache_root = args.cache_root.resolve()
    feature_root = cache_root / str(checkpoint["feature_subdir"])
    stats = checkpoint["normalization"]
    state_min = stats["state_min"].float()
    state_max = stats["state_max"].float()
    phase_length = int(checkpoint["phase_length"])
    targets: dict[str, torch.Tensor] = {}

    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        features, states = [], []
        for row in batch:
            item_id = str(row["id"])
            window = torch.load(
                cache_root / "windows" / f"{item_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            feature = torch.load(
                feature_root / f"{item_id}.pt",
                map_location="cpu",
                weights_only=False,
            )["features"]
            state_parts = []
            if bool(checkpoint.get("use_proprio", False)):
                proprio = 2.0 * (window["state"].float() - state_min) / (
                    state_max - state_min
                ).clamp_min(1e-6) - 1.0
                state_parts.append(proprio.reshape(1, 8).expand(horizon, -1))
            if bool(checkpoint.get("use_previous_action", False)):
                raise NotImplementedError("previous-action teacher export is unsupported")
            if bool(checkpoint.get("include_phase", True)):
                steps = torch.arange(horizon, dtype=torch.float32)
                steps.add_(int(row["start"])).clamp_max_(phase_length - 1)
                state_parts.append(
                    (2.0 * steps / max(phase_length - 1, 1) - 1.0).reshape(-1, 1)
                )
            states.append(
                torch.cat(state_parts, dim=-1)
                if state_parts
                else torch.zeros(horizon, 1)
            )
            features.append(feature)
        feature_batch = torch.stack(features).cuda(non_blocking=True)
        state_batch = torch.stack(states).cuda(non_blocking=True)
        with torch.inference_mode():
            prediction = model(
                torch.zeros(len(batch), horizon, int(checkpoint["action_dim"]), device="cuda"),
                state=state_batch,
                h3_features=feature_batch,
                video_sigma=torch.zeros(len(batch), device="cuda"),
            )
        if not isinstance(prediction, torch.Tensor):
            raise ValueError("mixture teachers require explicit mode selection")
        for index, row in enumerate(batch):
            targets[str(row["id"])] = prediction[index].cpu().to(torch.float16)
        if len(targets) % 500 < len(batch) or len(targets) == len(rows):
            print(json.dumps({"complete": len(targets), "total": len(rows)}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "h3-feature-action-teacher-targets-v1",
            "teacher_checkpoint": str(args.checkpoint.resolve()),
            "action_horizon": horizon,
            "normalization": checkpoint["normalization"],
            "training_tasks": checkpoint.get("training_tasks", []),
            "targets": targets,
        },
        args.output,
    )
    print(json.dumps({"output": str(args.output.resolve()), "targets": len(targets)}))


if __name__ == "__main__":
    main()
