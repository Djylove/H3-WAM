#!/usr/bin/env python3
"""Rank fixed action-flow noise seeds against cached demonstration starts."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-subdir", required=True)
    parser.add_argument("--max-seed", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def load_helpers():
    path = Path(__file__).with_name("train_libero_h3_action.py")
    spec = importlib.util.spec_from_file_location("train_libero_h3_action", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    helpers = load_helpers()
    from fastwam.models.h3wam import H3FeatureActionTransformer
    from fastwam.models.wan22.schedulers.scheduler_continuous import (
        WanContinuousFlowMatchScheduler,
    )

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("objective") != "flow":
        raise ValueError("checkpoint is not a flow policy")
    model = H3FeatureActionTransformer(
        action_dim=int(checkpoint["action_dim"]),
        state_dim=int(checkpoint["state_dim"]),
        h3_feature_dim=int(checkpoint["feature_shape"][-1]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        num_heads=int(checkpoint["num_heads"]),
        ffn_dim=int(checkpoint["ffn_dim"]),
        num_action_modes=1,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scheduler = WanContinuousFlowMatchScheduler(
        shift=float(checkpoint.get("flow_shift", 5.0))
    )
    timesteps, deltas = scheduler.build_inference_schedule(
        int(checkpoint.get("flow_inference_steps", 20)), device, torch.float32
    )

    items = helpers.read_manifest(args.manifest)
    starts: dict[int, dict] = {}
    for item in items:
        episode = int(item["episode"])
        if episode not in starts or int(item["start"]) < int(starts[episode]["start"]):
            starts[episode] = item
    stats = checkpoint["normalization"]
    targets, states, features = [], [], []
    for item in starts.values():
        window = torch.load(
            args.cache_root / "windows" / f"{item['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        target = helpers.minmax_normalize(
            window["actions"].float(), stats["action_min"], stats["action_max"]
        )
        if checkpoint.get("use_proprio", False):
            proprio = helpers.minmax_normalize(
                window["state"].float(), stats["state_min"], stats["state_max"]
            )
            state = proprio.reshape(1, 8).expand(target.shape[0], -1).clone()
            if checkpoint.get("include_phase", True):
                state = torch.cat((state, torch.zeros(target.shape[0], 1)), dim=-1)
        else:
            state = torch.zeros(target.shape[0], int(checkpoint["state_dim"]))
        if checkpoint.get("include_phase", True):
            steps = torch.arange(target.shape[0], dtype=torch.float32)
            steps.clamp_max_(int(item["length"]) - 1)
            state[:, -1] = 2.0 * steps / max(int(item["length"]) - 1, 1) - 1.0
        feature = torch.load(
            args.cache_root / args.feature_subdir / f"{item['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )["features"]
        targets.append(target)
        states.append(state)
        features.append(feature)
    target = torch.stack(targets).to(device)
    state = torch.stack(states).to(device)
    feature = torch.stack(features).to(device)

    scores = []
    for seed in range(args.max_seed + 1):
        generator = torch.Generator(device=device).manual_seed(seed)
        base = torch.randn(
            (1, target.shape[1], target.shape[2]), generator=generator, device=device
        )
        sample = base.expand(target.shape[0], -1, -1).clone()
        for timestep, delta in zip(timesteps, deltas):
            velocity = model(
                sample,
                state=state,
                h3_features=feature,
                video_sigma=timestep.expand(target.shape[0]),
            )
            sample = scheduler.step(velocity, delta, sample)
        scores.append(
            {
                "seed": seed,
                "chunk_mse": float((sample - target).square().mean().item()),
                "first_action_mse": float(
                    (sample[:, 0] - target[:, 0]).square().mean().item()
                ),
            }
        )
    print(json.dumps(sorted(scores, key=lambda row: row["chunk_mse"])[: args.top_k]))


if __name__ == "__main__":
    main()
