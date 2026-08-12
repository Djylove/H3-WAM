#!/usr/bin/env python3
"""Train the small action-only control model on the same cached LIBERO split."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--val-episodes-per-task", type=int, default=2)
    parser.add_argument(
        "--train-episode",
        type=int,
        action="append",
        help="Restrict optimization to selected episode indices (diagnostic).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--objective",
        choices=("flow", "regression"),
        default="flow",
        help="Train flow velocity or deterministic normalized action chunks.",
    )
    parser.add_argument(
        "--action-loss-horizon",
        type=int,
        default=0,
        help="Only supervise the first N actions (0 supervises the full chunk).",
    )
    parser.add_argument(
        "--include-phase",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append normalized episode progress to the proprioceptive state.",
    )
    parser.add_argument(
        "--phase-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Zero proprioception and condition only on trajectory phase (diagnostic).",
    )
    parser.add_argument(
        "--random-action-offset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move a random action inside each cached chunk to target position zero.",
    )
    parser.add_argument(
        "--fixed-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the first cached task context for every training window.",
    )
    return parser.parse_args()


def load_helpers():
    path = Path(__file__).with_name("train_libero_h3_action.py")
    spec = importlib.util.spec_from_file_location("train_libero_h3_action", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if args.phase_only and not args.include_phase:
        raise ValueError("--phase-only requires --include-phase")
    if args.random_action_offset and not args.phase_only:
        raise ValueError("--random-action-offset currently requires --phase-only")
    helpers = load_helpers()
    from fastwam.models.h3wam import H3ActionFlowScheduler, SmallActionFlowTransformer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    items = helpers.read_manifest(args.manifest.resolve())
    train_items, validation_items = helpers.split_by_episode(items, args.val_episodes_per_task)
    if args.train_episode is not None:
        selected_episodes = set(args.train_episode)
        train_items = [
            item for item in train_items if int(item["episode"]) in selected_episodes
        ]
        if not train_items:
            raise ValueError("--train-episode filters removed every training window")
    stats = torch.load(args.cache_root / "stats.pt", map_location="cpu", weights_only=False)
    state_dim = int(stats["state_min"].numel()) + int(args.include_phase)
    phase_length = int(round(statistics.median(int(item["length"]) for item in items)))
    model = SmallActionFlowTransformer(action_dim=7, state_dim=state_dim).to(device)
    scheduler = H3ActionFlowScheduler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    fixed_condition = None
    if args.fixed_context:
        fixed_condition = torch.load(
            args.cache_root / "refined_contexts" / f"{items[0]['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )

    def action_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if args.action_loss_horizon < 0:
            raise ValueError("--action-loss-horizon must be non-negative")
        horizon = args.action_loss_horizon or prediction.shape[1]
        if horizon > prediction.shape[1]:
            raise ValueError(
                f"--action-loss-horizon={horizon} exceeds chunk length {prediction.shape[1]}"
            )
        return (prediction[:, :horizon] - target[:, :horizon]).square().mean()

    def load_batch(batch_items: list[dict], *, validation_seed: int | None = None):
        actions, states, contexts = [], [], []
        for item in batch_items:
            window = torch.load(
                args.cache_root / "windows" / f"{item['id']}.pt",
                map_location="cpu",
                weights_only=False,
            )
            condition = fixed_condition
            if condition is None:
                condition = torch.load(
                    args.cache_root / "refined_contexts" / f"{item['id']}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            normalized_actions = helpers.minmax_normalize(
                window["actions"].float(), stats["action_min"], stats["action_max"]
            )
            action_offset = 0
            if args.random_action_offset:
                if validation_seed is None:
                    action_offset = random.randrange(normalized_actions.shape[0])
                else:
                    action_offset = (
                        int(item["start"]) + int(validation_seed)
                    ) % normalized_actions.shape[0]
                normalized_actions = torch.roll(
                    normalized_actions, shifts=-action_offset, dims=0
                )
            actions.append(normalized_actions)
            normalized_state = helpers.minmax_normalize(
                window["state"].float(), stats["state_min"], stats["state_max"]
            )
            if args.phase_only:
                normalized_state = torch.zeros_like(normalized_state)
            if args.include_phase:
                phase_step = int(item["start"]) + action_offset
                phase = 2.0 * float(phase_step) / max(int(item["length"]) - 1, 1) - 1.0
                normalized_state = torch.cat(
                    (normalized_state, torch.tensor([phase], dtype=torch.float32))
                )
            states.append(normalized_state)
            contexts.append(condition["context"].float().mean(dim=1).squeeze(0))
        action = torch.stack(actions).to(device)
        state = torch.stack(states).to(device)
        context = torch.stack(contexts).unsqueeze(1).to(device)
        active_generator = generator
        if validation_seed is not None:
            active_generator = torch.Generator(device=device).manual_seed(validation_seed)
        if args.objective == "flow":
            base = torch.rand(action.shape[0], generator=active_generator, device=device)
            video_sigma = scheduler.shift(base, scheduler.video_shift)
            noise = torch.randn(
                action.shape, generator=active_generator, device=device, dtype=action.dtype
            )
            noisy = scheduler.add_action_noise(action, noise, video_sigma)
            target = scheduler.training_target(action, noise, video_sigma)
        else:
            noisy = torch.zeros_like(action)
            video_sigma = torch.zeros(action.shape[0], device=device)
            target = action
        return noisy, action, state, context, video_sigma, target

    def validation_loss() -> float:
        model.eval()
        losses = []
        with torch.inference_mode():
            for offset in range(0, len(validation_items), args.batch_size):
                batch = validation_items[offset : offset + args.batch_size]
                noisy, _, state, context, sigma, target = load_batch(
                    batch, validation_seed=args.seed + offset + 1
                )
                prediction = model(noisy, state=state, context=context, video_sigma=sigma)
                losses.append(float(action_loss(prediction, target).item()))
        model.train()
        return sum(losses) / len(losses)

    started = time.perf_counter()
    best_validation = float("inf")
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        batch = random.choices(train_items, k=args.batch_size)
        noisy, _, state, context, sigma, target = load_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy, state=state, context=context, video_sigma=sigma)
        loss = action_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.validation_every == 0 or step == args.steps:
            val_loss = validation_loss()
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_loss": float(loss.detach().item()),
                        "validation_loss": val_loss,
                        "objective": args.objective,
                        "action_loss_horizon": args.action_loss_horizon,
                        "include_phase": args.include_phase,
                        "phase_only": args.phase_only,
                        "train_episode": args.train_episode,
                        "random_action_offset": args.random_action_offset,
                        "fixed_context": args.fixed_context,
                        "elapsed_seconds": elapsed,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    }
                ),
                flush=True,
            )
            if val_loss < best_validation:
                best_validation = val_loss
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "normalization": stats,
                        "step": step,
                        "validation_loss": val_loss,
                        "objective": args.objective,
                        "action_loss_horizon": args.action_loss_horizon,
                        "include_phase": args.include_phase,
                        "phase_only": args.phase_only,
                        "train_episode": args.train_episode,
                        "random_action_offset": args.random_action_offset,
                        "fixed_context": args.fixed_context,
                        "phase_length": phase_length,
                        "state_dim": state_dim,
                    },
                    args.output.with_name(args.output.stem + "_best" + args.output.suffix),
                )
    torch.save(
        {
            "model": model.state_dict(),
            "normalization": stats,
            "step": args.steps,
            "validation_loss": best_validation,
            "objective": args.objective,
            "action_loss_horizon": args.action_loss_horizon,
            "include_phase": args.include_phase,
            "phase_only": args.phase_only,
            "train_episode": args.train_episode,
            "random_action_offset": args.random_action_offset,
            "fixed_context": args.fixed_context,
            "phase_length": phase_length,
            "state_dim": state_dim,
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "best_validation_loss": best_validation,
                "total_seconds": time.perf_counter() - started,
            }
        )
    )


if __name__ == "__main__":
    main()
