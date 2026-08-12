#!/usr/bin/env python3
"""Fine-tune H3 attention LoRA through the successful feature-action policy."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument("--lora-last-blocks", type=int, default=10)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--action-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--validation-windows", type=int, default=16)
    parser.add_argument("--val-episodes-per-task", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--freeze-action-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Adapt only H3 LoRA instead of jointly updating the initialized action head.",
    )
    return parser.parse_args()


def load_training_helpers():
    path = Path(__file__).with_name("train_libero_h3_action.py")
    spec = importlib.util.spec_from_file_location("train_libero_h3_action", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.validation_every <= 0 or args.checkpoint_every <= 0:
        raise ValueError("steps and checkpoint intervals must be positive")
    if args.lora_rank <= 0 or args.lora_last_blocks <= 0:
        raise ValueError("LoRA rank and selected block count must be positive")

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(args.comfy_root.resolve()))
    import comfy.model_management as model_management
    import comfy.sd

    from fastwam.models.h3wam import (
        H3BlockFeatureCapture,
        H3FeatureActionTransformer,
        enable_comfy_h3_autograd,
        h3_lora_parameters,
        h3_lora_state_dict,
        inject_h3_attention_lora,
        make_first_frame_payload,
    )

    helpers = load_training_helpers()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_management.in_training = True
    enable_comfy_h3_autograd(checkpoint_blocks=True)
    device = model_management.get_torch_device()
    if device.type != "cuda":
        raise RuntimeError(f"CUDA is required, got {device}")

    items = helpers.read_manifest(args.manifest.resolve())
    train_items, validation_items = helpers.split_by_episode(
        items, args.val_episodes_per_task
    )
    if len(validation_items) > args.validation_windows:
        indices = torch.linspace(
            0, len(validation_items) - 1, args.validation_windows
        ).round().long()
        validation_items = [validation_items[int(index)] for index in indices]

    initial = torch.load(
        args.init_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    if initial.get("policy_type") != "h3_feature_action":
        raise ValueError("init checkpoint is not an H3 feature-action policy")
    if initial.get("objective", "regression") != "regression":
        raise ValueError("the first LoRA experiment requires a regression policy")
    if int(initial.get("num_action_modes", 1)) != 1:
        raise ValueError("the first LoRA experiment requires a single action mode")
    if int(initial.get("action_horizon", 32)) != 1:
        raise ValueError("the first LoRA experiment requires a Horizon-1 checkpoint")

    stats = initial["normalization"]
    action_model = H3FeatureActionTransformer(
        action_dim=int(initial["action_dim"]),
        state_dim=int(initial["state_dim"]),
        h3_feature_dim=tuple(initial["feature_shape"])[-1],
        hidden_dim=int(initial["hidden_dim"]),
        num_layers=int(initial["num_layers"]),
        num_heads=int(initial["num_heads"]),
        ffn_dim=int(initial["ffn_dim"]),
        num_action_modes=1,
    ).to(device)
    action_model.load_state_dict(initial["model"])
    action_model.requires_grad_(not args.freeze_action_head)

    patcher = comfy.sd.load_diffusion_model(str(args.h3_checkpoint.resolve()))
    model_management.load_models_gpu([patcher])
    h3_model = patcher.model.diffusion_model
    h3_model.requires_grad_(False).eval()
    lora_alpha = float(args.lora_rank if args.lora_alpha is None else args.lora_alpha)
    lora_report = inject_h3_attention_lora(
        h3_model,
        rank=args.lora_rank,
        alpha=lora_alpha,
        last_n_blocks=args.lora_last_blocks,
    )
    lora_parameters = h3_lora_parameters(h3_model)

    feature_layers = tuple(int(index) for index in initial["feature_layers"])
    feature_timestep = float(initial.get("feature_timestep", 1000.0))
    context_id = initial.get("feature_context_id")
    if not context_id:
        raise ValueError("init checkpoint must record a fixed feature_context_id")
    conditioning = torch.load(
        args.cache_root / "refined_contexts" / f"{context_id}.pt",
        map_location="cpu",
        weights_only=False,
    )
    context = conditioning["context"].to(device=device, dtype=torch.bfloat16)
    token_tags = conditioning["token_tags"].to(device)
    text_len = int(context.shape[1])
    gripper_weight = float(initial.get("gripper_loss_weight", 1.0))
    dimension_weights = torch.ones(int(initial["action_dim"]), device=device)
    dimension_weights[-1] = gripper_weight

    parameter_groups = [
        {
            "params": lora_parameters,
            "lr": args.lora_learning_rate,
            "weight_decay": args.weight_decay,
        }
    ]
    action_parameters = [
        parameter for parameter in action_model.parameters() if parameter.requires_grad
    ]
    if action_parameters:
        parameter_groups.append(
            {
                "params": action_parameters,
                "lr": args.action_learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    trainable_parameters = lora_parameters + action_parameters

    def load_target(item: dict) -> tuple[dict, torch.Tensor, torch.Tensor]:
        window = torch.load(
            args.cache_root / "windows" / f"{item['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        action = helpers.minmax_normalize(
            window["actions"][0].float(), stats["action_min"], stats["action_max"]
        ).reshape(1, 1, -1)
        state_parts = []
        if bool(initial.get("use_proprio", False)):
            proprio = helpers.minmax_normalize(
                window["state"].float(), stats["state_min"], stats["state_max"]
            )
            state_parts.append(proprio)
        if bool(initial.get("use_previous_action", False)):
            start = int(item["start"])
            if start == 0:
                previous = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            else:
                previous_id = f"ep{int(item['episode']):06d}_s{start - 1:06d}"
                previous_window = torch.load(
                    args.cache_root / "windows" / f"{previous_id}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                previous = previous_window["actions"][0].float()
            state_parts.append(
                helpers.minmax_normalize(
                    previous, stats["action_min"], stats["action_max"]
                )
            )
        if bool(initial.get("include_phase", True)):
            phase = 2.0 * float(item["start"]) / max(int(item["length"]) - 1, 1) - 1.0
            state_parts.append(torch.tensor([min(phase, 1.0)]))
        if state_parts:
            state = torch.cat(state_parts)
        else:
            state = torch.zeros(1)
        return window, action.to(device), state.reshape(1, 1, -1).to(device)

    def extract_features(window: dict, *, differentiable: bool) -> torch.Tensor:
        first_frame = window["first_frame_latents"].to(
            device=device, dtype=torch.bfloat16
        )
        frame_rows = int(
            first_frame.shape[2]
            * (first_frame.shape[3] // 2)
            * (first_frame.shape[4] // 2)
        )
        capture = H3BlockFeatureCapture(
            feature_layers,
            token_start=text_len,
            token_stop=text_len + frame_rows,
            detach=not differentiable,
        )
        payload = make_first_frame_payload(
            first_frame, frame_count=int(window.get("h3_frame_count", 39))
        )
        payload["text_token_tags"] = token_tags
        h3_model(
            [
                torch.zeros(
                    (1, 24, 12, first_frame.shape[-2], first_frame.shape[-1]),
                    device=device,
                    dtype=torch.bfloat16,
                ),
                torch.zeros((1, 32, 2, 1), device=device, dtype=torch.float32),
            ],
            torch.tensor([feature_timestep], device=device),
            context,
            transformer_options=capture.transformer_options(),
            minimax_payload=payload,
        )
        return capture.stacked().unsqueeze(0)

    def item_loss(item: dict, *, differentiable: bool) -> torch.Tensor:
        window, target, state = load_target(item)
        features = extract_features(window, differentiable=differentiable)
        prediction = action_model(
            torch.zeros_like(target),
            state=state,
            h3_features=features,
            video_sigma=torch.zeros(1, device=device),
        )
        per_dimension = (prediction - target).square()
        return (
            per_dimension * dimension_weights.reshape(1, 1, -1)
        ).sum(dim=-1).div(dimension_weights.sum()).mean()

    @torch.inference_mode()
    def validation_loss() -> float:
        action_model.eval()
        losses = [float(item_loss(item, differentiable=False).item()) for item in validation_items]
        action_model.train(not args.freeze_action_head)
        return sum(losses) / len(losses)

    def checkpoint(step: int, value: float, baseline_value: float) -> dict:
        result = {
            key: value for key, value in initial.items()
            if key not in ("model", "optimizer", "h3_lora")
        }
        result.update(
            {
                "model": action_model.state_dict(),
                "h3_lora": h3_lora_state_dict(h3_model),
                "h3_lora_rank": args.lora_rank,
                "h3_lora_alpha": lora_alpha,
                "h3_lora_last_blocks": args.lora_last_blocks,
                "h3_lora_trainable_parameters": lora_report.parameters,
                "h3_lora_joint_action_head": not args.freeze_action_head,
                "h3_lora_init_checkpoint": str(args.init_checkpoint.resolve()),
                "h3_lora_baseline_validation_loss": baseline_value,
                "step": step,
                "validation_loss": value,
            }
        )
        return result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    action_model.eval()
    baseline_validation = validation_loss()
    best_validation = baseline_validation
    torch.save(
        checkpoint(0, baseline_validation, baseline_validation),
        args.output.with_name(args.output.stem + "_step000000" + args.output.suffix),
    )
    print(
        json.dumps(
            {
                "step": 0,
                "validation_loss": baseline_validation,
                "lora_parameters": lora_report.parameters,
                "action_parameters": sum(p.numel() for p in action_parameters),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
        ),
        flush=True,
    )

    started = time.perf_counter()
    action_model.train(not args.freeze_action_head)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = item_loss(random.choice(train_items), differentiable=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, args.gradient_clip
        )
        optimizer.step()
        if step % args.validation_every == 0 or step == args.steps:
            value = validation_loss()
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_loss": float(loss.detach().item()),
                        "validation_loss": value,
                        "gradient_norm": float(gradient_norm),
                        "elapsed_seconds": elapsed,
                        "seconds_per_train_step": elapsed / step,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    }
                ),
                flush=True,
            )
            if value < best_validation:
                best_validation = value
                torch.save(
                    checkpoint(step, value, baseline_validation),
                    args.output.with_name(args.output.stem + "_best" + args.output.suffix),
                )
            action_model.train(not args.freeze_action_head)
        if step % args.checkpoint_every == 0 and step != args.steps:
            torch.save(
                checkpoint(step, value if "value" in locals() else float("nan"), baseline_validation),
                args.output.with_name(
                    args.output.stem + f"_step{step:06d}" + args.output.suffix
                ),
            )

    torch.save(checkpoint(args.steps, best_validation, baseline_validation), args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "train_windows": len(train_items),
                "validation_windows": len(validation_items),
                "baseline_validation_loss": baseline_validation,
                "best_validation_loss": best_validation,
                "total_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
