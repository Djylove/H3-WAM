#!/usr/bin/env python3
"""Evaluate few-step sampling for the small action-only control model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-evaluations", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--max-windows", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    return parser.parse_args()


def helpers_module():
    path = Path(__file__).with_name("train_libero_h3_action.py")
    spec = importlib.util.spec_from_file_location("train_libero_h3_action", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    helpers = helpers_module()
    from fastwam.models.h3wam import H3ActionFlowScheduler, SmallActionFlowTransformer

    items = helpers.read_manifest(args.manifest.resolve())
    train, held_out = helpers.split_by_episode(items, 2)
    validation = (train if args.split == "train" else held_out)[: args.max_windows]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = checkpoint["normalization"]
    device = torch.device("cuda")
    model = SmallActionFlowTransformer(action_dim=7, state_dim=8).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    scheduler = H3ActionFlowScheduler()
    objective = checkpoint.get("objective", "flow")

    results = []
    for evaluations in args.model_evaluations:
        squared_errors, absolute_errors, durations = [], [], []
        timestep_squared_errors, dimension_squared_errors = [], []
        for index, item in enumerate(validation):
            window = torch.load(
                args.cache_root / "windows" / f"{item['id']}.pt",
                map_location="cpu",
                weights_only=False,
            )
            conditioning = torch.load(
                args.cache_root / "refined_contexts" / f"{item['id']}.pt",
                map_location="cpu",
                weights_only=False,
            )
            target = helpers.minmax_normalize(
                window["actions"].float(), stats["action_min"], stats["action_max"]
            ).unsqueeze(0).to(device)
            state = helpers.minmax_normalize(
                window["state"].float(), stats["state_min"], stats["state_max"]
            ).unsqueeze(0).to(device)
            context = conditioning["context"].float().to(device)
            started = time.perf_counter()
            with torch.inference_mode():
                if objective == "regression":
                    actions = model(
                        torch.zeros_like(target),
                        state=state,
                        context=context,
                        video_sigma=torch.zeros(1, device=device),
                    )
                else:
                    generator = torch.Generator(device=device).manual_seed(args.seed + index)
                    actions = torch.randn(
                        target.shape, generator=generator, device=device, dtype=torch.float32
                    )
                    sigmas, deltas = scheduler.inference_schedule(evaluations, device=device)
                    for sigma, delta in zip(sigmas, deltas):
                        prediction = model(
                            actions,
                            state=state,
                            context=context,
                            video_sigma=sigma.reshape(1),
                        )
                        action_delta = scheduler.action_inference_delta(sigma, delta)
                        actions = actions + prediction / scheduler.action_slope(
                            sigma
                        ) * action_delta
            torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - started)
            error = actions - target
            squared_errors.append(float(error.square().mean().item()))
            absolute_errors.append(float(error.abs().mean().item()))
            timestep_squared_errors.append(error.square().mean(dim=(0, 2)).cpu())
            dimension_squared_errors.append(error.square().mean(dim=(0, 1)).cpu())
        steady_durations = durations[1:] if len(durations) > 1 else durations
        per_timestep = torch.stack(timestep_squared_errors).mean(dim=0)
        per_dimension = torch.stack(dimension_squared_errors).mean(dim=0)
        results.append(
            {
                "model_evaluations": evaluations,
                "windows": len(validation),
                "normalized_action_mse": sum(squared_errors) / len(squared_errors),
                "normalized_action_mae": sum(absolute_errors) / len(absolute_errors),
                "first_10_action_mse": float(per_timestep[:10].mean().item()),
                "remaining_action_mse": float(per_timestep[10:].mean().item()),
                "per_action_dimension_mse": per_dimension.tolist(),
                "per_timestep_mse": per_timestep.tolist(),
                "cold_start_latency_seconds": durations[0],
                "mean_steady_latency_seconds": sum(steady_durations) / len(steady_durations),
            }
        )
    print(json.dumps({"split": args.split, "objective": objective, "results": results}, indent=2))


if __name__ == "__main__":
    main()
