#!/usr/bin/env python3
"""Train and validate the frozen-H3 action adapter on cached LIBERO windows."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-windows", type=int, default=20)
    parser.add_argument(
        "--save-validation-checkpoints",
        action="store_true",
        help="Save a numbered checkpoint at every validation boundary.",
    )
    parser.add_argument("--val-episodes-per-task", type=int, default=2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-last-blocks", type=int, default=50)
    parser.add_argument("--video-loss-weight", type=float, default=0.0)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--direct-action-conditioning", action="store_true")
    parser.add_argument("--direct-action-residual", action="store_true")
    parser.add_argument("--task-groups", type=int, nargs="+")
    parser.add_argument(
        "--objective",
        choices=("flow", "regression"),
        default="flow",
        help="Train H3 flow velocity or a deterministic normalized action prediction.",
    )
    parser.add_argument(
        "--action-loss-horizon",
        type=int,
        default=0,
        help="Only supervise the first N actions; 0 uses the full chunk.",
    )
    parser.add_argument("--train-episode", type=int, action="append")
    parser.add_argument("--phase-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--phase-sequence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Condition every chunk position on its own episode phase.",
    )
    parser.add_argument(
        "--random-action-offset", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--fixed-context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--fixed-first-frame", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_by_episode(
    items: list[dict],
    val_episodes_per_task: int,
) -> tuple[list[dict], list[dict]]:
    if val_episodes_per_task <= 0:
        raise ValueError("val-episodes-per-task must be positive")
    episodes_by_task: dict[int, set[int]] = defaultdict(set)
    for item in items:
        episodes_by_task[int(item["task_group"])].add(int(item["episode"]))
    validation_episodes = set()
    for episodes in episodes_by_task.values():
        ordered = sorted(episodes)
        if len(ordered) <= val_episodes_per_task:
            raise ValueError("not enough episodes to create disjoint train/validation splits")
        validation_episodes.update(ordered[-val_episodes_per_task:])
    train = [item for item in items if int(item["episode"]) not in validation_episodes]
    validation = [item for item in items if int(item["episode"]) in validation_episodes]
    return train, validation


def minmax_normalize(value: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
    scale = (maximum - minimum).clamp_min(1e-6)
    return 2.0 * (value - minimum) / scale - 1.0


class CachedExperiment:
    def __init__(self, args: argparse.Namespace, items: list[dict]) -> None:
        import comfy.model_management as model_management
        import comfy.sd

        from fastwam.models.h3wam import (
            H3ActionAdapter,
            H3ActionBridge,
            H3ActionFlowScheduler,
            enable_comfy_h3_autograd,
            h3_lora_parameters,
            inject_h3_attention_lora,
        )

        self.args = args
        self.items = items
        self.cache_root = args.cache_root.resolve()
        self.model_management = model_management
        model_management.in_training = True
        enable_comfy_h3_autograd(checkpoint_blocks=True)
        self.device = model_management.get_torch_device()
        if self.device.type != "cuda":
            raise RuntimeError(f"CUDA is required, got {self.device}")

        patcher = comfy.sd.load_diffusion_model(str(args.checkpoint.resolve()))
        model_management.load_models_gpu([patcher])
        self.h3_model = patcher.model.diffusion_model
        self.stats = torch.load(
            self.cache_root / "stats.pt", map_location="cpu", weights_only=False
        )
        self.phase_length = int(
            round(statistics.median(int(item["length"]) for item in items))
        )
        first = torch.load(
            self.cache_root / "windows" / f"{items[0]['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.adapter = H3ActionAdapter(
            action_dim=first["actions"].shape[-1],
            state_dim=first["state"].shape[-1],
            direct_conditioning=args.direct_action_conditioning,
            direct_action_residual=args.direct_action_residual,
        ).to(device=self.device, dtype=torch.float32)
        self.bridge = H3ActionBridge(self.h3_model, self.adapter, freeze_h3=True)
        self.lora_report = None
        if args.lora_rank > 0:
            self.lora_report = inject_h3_attention_lora(
                self.h3_model,
                rank=args.lora_rank,
                last_n_blocks=args.lora_last_blocks,
            )
        if args.init_checkpoint is not None:
            initial = torch.load(
                args.init_checkpoint.resolve(), map_location="cpu", weights_only=False
            )
            incompatible = self.adapter.load_state_dict(
                initial["adapter"], strict=not args.direct_action_conditioning
            )
            if incompatible.unexpected_keys:
                raise ValueError(
                    f"unexpected adapter keys: {incompatible.unexpected_keys}"
                )
            allowed_missing = (
                "decoder_state_projection",
                "decoder_context_projection",
                "decoder_action_residual",
            )
            if args.direct_action_conditioning and any(
                not key.startswith(allowed_missing)
                for key in incompatible.missing_keys
            ):
                raise ValueError(
                    f"unexpected missing adapter keys: {incompatible.missing_keys}"
                )
            if args.lora_rank > 0 and initial.get("h3_lora"):
                from fastwam.models.h3wam import load_h3_lora_state_dict

                load_h3_lora_state_dict(self.h3_model, initial["h3_lora"])
        self.scheduler = H3ActionFlowScheduler(
            video_shift=float(self.h3_model.sigma_shift_video),
            action_shift=float(self.h3_model.sigma_shift_audio),
        )
        self.trainable_parameters = list(self.adapter.parameters()) + h3_lora_parameters(
            self.h3_model
        )
        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.generator = torch.Generator(device=self.device).manual_seed(args.seed)
        self.fixed_window = None
        self.fixed_conditioning = None
        if args.fixed_context or args.fixed_first_frame:
            self.fixed_window, self.fixed_conditioning = self._load_sample(items[0])

    def refine_contexts(self) -> None:
        refined_root = self.cache_root / "refined_contexts"
        started = time.perf_counter()
        completed = 0
        for item in self.items:
            output = refined_root / f"{item['id']}.pt"
            if output.exists():
                completed += 1
                continue
            source = torch.load(
                self.cache_root / "contexts" / f"{item['id']}.pt",
                map_location="cpu",
                weights_only=False,
            )
            context = source["context"].to(device=self.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                refined = self.h3_model.preprocess_text_embeds(context)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "context": refined.cpu(),
                    "token_tags": source["token_tags"],
                },
                output,
            )
            completed += 1
            if completed % 50 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "refine_context",
                            "completed": completed,
                            "total": len(self.items),
                            "elapsed_seconds": round(time.perf_counter() - started, 2),
                        }
                    ),
                    flush=True,
                )

    def _load_sample(self, item: dict) -> tuple[dict, dict]:
        window = torch.load(
            self.cache_root / "windows" / f"{item['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        conditioning = torch.load(
            self.cache_root / "refined_contexts" / f"{item['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        return window, conditioning

    def loss_for_item(self, item: dict, *, validation_seed: int | None = None) -> torch.Tensor:
        from fastwam.models.h3wam import make_first_frame_payload, prepare_h3wam_flow_batch

        window, conditioning = self._load_sample(item)
        if self.args.fixed_context:
            conditioning = self.fixed_conditioning
        actions = minmax_normalize(
            window["actions"].float(),
            self.stats["action_min"],
            self.stats["action_max"],
        )
        action_offset = 0
        if self.args.random_action_offset:
            if validation_seed is None:
                action_offset = random.randrange(actions.shape[0])
            else:
                action_offset = (int(item["start"]) + validation_seed) % actions.shape[0]
            actions = torch.roll(actions, shifts=-action_offset, dims=0)
        actions = actions.unsqueeze(0).to(self.device)
        if self.args.phase_only:
            if self.args.phase_sequence:
                state = torch.zeros(
                    actions.shape[1], window["state"].numel(), dtype=torch.float32
                )
                phase_steps = torch.arange(actions.shape[1], dtype=torch.float32)
                phase_steps.add_(int(item["start"])).clamp_max_(int(item["length"]) - 1)
                state[:, -1] = (
                    2.0 * phase_steps / max(int(item["length"]) - 1, 1) - 1.0
                )
            else:
                state = torch.zeros_like(window["state"].float())
                phase_step = int(item["start"]) + action_offset
                state[-1] = (
                    2.0 * float(phase_step) / max(int(item["length"]) - 1, 1) - 1.0
                )
        else:
            state = minmax_normalize(
                window["state"].float(),
                self.stats["state_min"],
                self.stats["state_max"],
            )
        state = state.unsqueeze(0).to(self.device)
        visual_window = self.fixed_window if self.args.fixed_first_frame else window
        video = visual_window["video_latents"].to(self.device, dtype=torch.bfloat16)
        first_frame = visual_window["first_frame_latents"].to(
            self.device, dtype=torch.bfloat16
        )
        context = conditioning["context"].to(self.device, dtype=torch.bfloat16)

        generator = self.generator
        if validation_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(validation_seed)
        flow = None
        if self.args.objective == "flow":
            base = torch.rand(1, generator=generator, device=self.device)
            video_sigma = self.scheduler.shift(base, self.scheduler.video_shift)
            video_noise = torch.randn(
                video.shape, generator=generator, device=self.device, dtype=video.dtype
            )
            action_noise = torch.randn(
                actions.shape, generator=generator, device=self.device, dtype=actions.dtype
            )
            flow = prepare_h3wam_flow_batch(
                video_latents=video,
                actions=actions,
                scheduler=self.scheduler,
                video_sigma=video_sigma,
                video_noise=video_noise,
                action_noise=action_noise,
            )
            video_input = flow.noisy_video_latents
            action_input = flow.noisy_actions
            timestep = flow.timestep
            action_target = flow.action_target
        else:
            video_input = torch.zeros_like(video)
            action_input = torch.zeros_like(actions)
            timestep = torch.zeros(1, device=self.device)
            action_target = actions
        payload = make_first_frame_payload(
            first_frame,
            frame_count=int(window["h3_frame_count"]),
            seed=self.args.seed,
        )
        payload["text_token_tags"] = conditioning["token_tags"]
        output = self.bridge(
            video_latents=video_input,
            noisy_actions=action_input,
            timestep=timestep,
            context=context,
            state=state,
            minimax_payload=payload,
        )
        action_elements = (
            output.action_velocity.float() - action_target.float()
        ).square()
        if self.args.action_loss_horizon > 0:
            action_elements = action_elements[:, : self.args.action_loss_horizon]
        action_loss = action_elements.mean()
        if self.args.video_loss_weight <= 0:
            return action_loss
        if flow is None:
            raise ValueError("video loss is unavailable for the regression objective")
        video_loss = (
            output.video_velocity.float() - flow.video_target.float()
        ).square().mean()
        return action_loss + self.args.video_loss_weight * video_loss

    def save(self, path: Path, *, step: int, validation_loss: float | None) -> None:
        from fastwam.models.h3wam import h3_lora_state_dict

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "adapter": self.adapter.state_dict(),
                "h3_lora": h3_lora_state_dict(self.h3_model),
                "action_dim": self.adapter.action_dim,
                "state_dim": self.adapter.state_dim,
                "direct_action_conditioning": self.adapter.direct_conditioning,
                "direct_action_residual": self.adapter.direct_action_residual,
                "action_loss_horizon": self.args.action_loss_horizon,
                "objective": self.args.objective,
                "phase_only": self.args.phase_only,
                "phase_sequence": self.args.phase_sequence,
                "phase_length": self.phase_length,
                "random_action_offset": self.args.random_action_offset,
                "fixed_context": self.args.fixed_context,
                "fixed_first_frame": self.args.fixed_first_frame,
                "train_episode": self.args.train_episode,
                "optimizer": self.optimizer.state_dict(),
                "step": step,
                "validation_loss": validation_loss,
                "normalization": self.stats,
            },
            path,
        )


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.validation_every <= 0:
        raise ValueError("steps and validation-every must be positive")
    if args.action_loss_horizon < 0:
        raise ValueError("action-loss-horizon must be non-negative")
    if args.random_action_offset and not args.phase_only:
        raise ValueError("random-action-offset currently requires phase-only")
    if args.phase_sequence and not args.phase_only:
        raise ValueError("phase-sequence requires phase-only")
    if args.phase_sequence and args.random_action_offset:
        raise ValueError("phase-sequence and random-action-offset are mutually exclusive")
    if args.objective == "regression" and args.video_loss_weight > 0:
        raise ValueError("regression objective does not support video loss")
    sys.path.insert(0, str(args.comfy_root.resolve()))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    items = read_manifest(args.manifest.resolve())
    if args.task_groups is not None:
        selected_groups = set(args.task_groups)
        items = [
            item for item in items if int(item["task_group"]) in selected_groups
        ]
        if not items:
            raise ValueError(f"no manifest items matched task groups {sorted(selected_groups)}")
    train_items, validation_items = split_by_episode(items, args.val_episodes_per_task)
    if args.train_episode is not None:
        selected_episodes = set(args.train_episode)
        train_items = [
            item for item in train_items if int(item["episode"]) in selected_episodes
        ]
        if not train_items:
            raise ValueError("train-episode filters removed every training window")
    experiment = CachedExperiment(args, items)
    experiment.refine_contexts()

    selected_validation = validation_items[: args.validation_windows]
    best_validation = float("inf")
    train_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(experiment.device)
    experiment.bridge.train()
    for step in range(1, args.steps + 1):
        item = random.choice(train_items)
        experiment.optimizer.zero_grad(set_to_none=True)
        loss = experiment.loss_for_item(item)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            experiment.trainable_parameters, args.gradient_clip
        )
        experiment.optimizer.step()

        if step == 1 or step % args.validation_every == 0 or step == args.steps:
            experiment.bridge.eval()
            with torch.inference_mode():
                validation_losses = [
                    float(
                        experiment.loss_for_item(
                            validation_item,
                            validation_seed=args.seed + index + 1,
                        ).item()
                    )
                    for index, validation_item in enumerate(selected_validation)
                ]
            experiment.bridge.train()
            validation_loss = sum(validation_losses) / len(validation_losses)
            elapsed = time.perf_counter() - train_started
            record = {
                "step": step,
                "train_loss": float(loss.detach().item()),
                "validation_loss": validation_loss,
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": elapsed,
                "seconds_per_train_step": elapsed / step,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(experiment.device) / 2**30,
                "lora_parameters": 0
                if experiment.lora_report is None
                else experiment.lora_report.parameters,
            }
            print(json.dumps(record), flush=True)
            if args.save_validation_checkpoints:
                experiment.save(
                    args.output.with_name(
                        f"{args.output.stem}_step{step:06d}{args.output.suffix}"
                    ),
                    step=step,
                    validation_loss=validation_loss,
                )
            if validation_loss < best_validation:
                best_validation = validation_loss
                experiment.save(
                    args.output.with_name(args.output.stem + "_best" + args.output.suffix),
                    step=step,
                    validation_loss=validation_loss,
                )

    experiment.save(args.output, step=args.steps, validation_loss=best_validation)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "train_windows": len(train_items),
                "validation_windows": len(validation_items),
                "best_validation_loss": best_validation,
                "total_seconds": time.perf_counter() - train_started,
            }
        )
    )


if __name__ == "__main__":
    main()
