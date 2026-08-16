#!/usr/bin/env python3
"""One-GPU real D0 mechanical probe for C57; never writes a candidate checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastwam.models.h3wam.c57_lingbot_interfaces import (  # noqa: E402
    LingBotTeacherForcedFeedback,
    forward_teacher_forced_history,
)
from fastwam.models.h3wam.dreamwam_kv_carrier import (  # noqa: E402
    H3DreamWAMKVCarrierPolicy,
)
from fastwam.models.h3wam.lingbot_persistent_kv import (  # noqa: E402
    H3LingBotPersistentKVPolicy,
    LingBotPersistentKVState,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--kv-subdir", default="h3_int8_dreamwam_kv_5x32_dense_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executed-action-steps", type=int, default=8)
    parser.add_argument("--actions-per-observation-frame", type=int, default=4)
    parser.add_argument("--persistent-window-chunks", type=int, default=15)
    parser.add_argument("--optimizer-step", action="store_true")
    return parser.parse_args()


def load_trainer_module():
    path = Path(__file__).with_name("train_h3_int8_dreamwam_kv_carrier.py")
    spec = importlib.util.spec_from_file_location("_c57_parent_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load parent trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_arguments(model_spec: dict, carrier_layers: tuple[int, ...], source_mode: str) -> dict:
    return {
        "enabled": True,
        "carrier_layers": carrier_layers,
        "carrier_source_mode": source_mode,
        "action_dim": int(model_spec["action_dim"]),
        "proprio_dim": int(model_spec["proprio_dim"]),
        "context_dim": int(model_spec["context_dim"]),
        "hidden_dim": int(model_spec["hidden_dim"]),
        "ffn_dim": int(model_spec["ffn_dim"]),
        "num_heads": int(model_spec["num_heads"]),
        "attn_head_dim": int(model_spec["attn_head_dim"]),
        "freq_dim": int(model_spec["freq_dim"]),
        "history_action_steps": 0,
    }


def clone_batch(batch: dict, index: int) -> dict:
    result = {}
    for key, value in batch.items():
        if key == "video_kv_cache":
            result[key] = {
                layer: {name: tensor[index : index + 1] for name, tensor in item.items()}
                for layer, item in value.items()
            }
        elif torch.is_tensor(value):
            result[key] = value[index : index + 1]
        elif key == "sample_ids":
            result[key] = [value[index]]
        else:
            result[key] = value
    return result


def forward(
    model,
    batch: dict,
    noisy: torch.Tensor,
    timestep: torch.Tensor,
    *,
    persistent_state=None,
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(
            noisy,
            timestep,
            text_context=batch["text_context"],
            proprio=batch["proprio"],
            video_kv_cache=batch["video_kv_cache"],
            text_mask=batch["text_mask"],
            persistent_state=persistent_state,
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the real C57 probe requires CUDA")
    if min(args.executed_action_steps, args.actions_per_observation_frame) <= 0:
        raise ValueError("executed/action-per-observation steps must be positive")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    payload = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 2:
        raise ValueError("C57 requires a schema-2 D0 checkpoint")
    contract = payload["contract"]
    if contract.get("candidate") != "D0" or contract.get("carrier_source_mode") != "repeat_layer49":
        raise ValueError("C57 parent must be the fixed D0 repeat-layer49 carrier")
    if int(contract.get("history_action_steps", 0)) != 0:
        raise ValueError("C57 cannot initialize from the legacy history adapter")
    carrier_layers = tuple(int(value) for value in contract["kv_layers"])
    model_spec = contract["model_spec"]
    common = model_arguments(model_spec, carrier_layers, "repeat_layer49")

    trainer = load_trainer_module()
    dataset = trainer.CachedDreamWAMKVDataset(
        args.manifest,
        args.cache_root,
        args.kv_subdir,
        source_manifest=args.source_manifest,
        carrier_layers=carrier_layers,
        capture_token_count=int(contract["kv_tokens"]),
        kv_pool_strategy=str(contract["kv_strategy"]),
        num_heads=int(contract["kv_num_heads"]),
        attn_head_dim=int(contract["kv_attn_head_dim"]),
        action_horizon=int(contract["action_horizon"]),
        limit=2,
    )
    batch = trainer.move_batch(
        trainer.collate_cached_batch([dataset[0], dataset[1]]), device, dtype
    )
    history_batch = clone_batch(batch, 0)
    current_batch = clone_batch(batch, 1)
    horizon = int(current_batch["actions"].shape[1])
    if args.executed_action_steps > horizon:
        raise ValueError("executed-action-steps exceeds parent action horizon")
    generator = torch.Generator(device=device).manual_seed(57001)
    noisy = torch.randn(
        current_batch["actions"].shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    timestep = torch.full((1,), 650.25, device=device, dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats(device)

    parent = H3DreamWAMKVCarrierPolicy(**common).to(device=device, dtype=dtype).eval()
    disabled = H3LingBotPersistentKVPolicy(
        persistent_enabled=False,
        persistent_window_chunks=args.persistent_window_chunks,
        observation_tokens_per_chunk=int(contract["kv_tokens"]),
        action_tokens_per_chunk=args.actions_per_observation_frame,
        **common,
    ).to(device=device, dtype=dtype).eval()
    parent.load_state_dict(payload["model"], strict=True)
    disabled.load_state_dict(payload["model"], strict=True)
    torch.cuda.synchronize(device)
    disabled_started = time.perf_counter()
    with torch.no_grad():
        parent_prediction = trainer.forward_policy(parent, current_batch, noisy, timestep)
        disabled_prediction = trainer.forward_policy(disabled, current_batch, noisy, timestep)
    torch.cuda.synchronize(device)
    disabled_seconds = time.perf_counter() - disabled_started
    disabled_max_abs = float(
        (parent_prediction.float() - disabled_prediction.float()).abs().max().item()
    )
    if disabled_max_abs != 0.0:
        raise AssertionError(f"disabled C57 differs from D0: {disabled_max_abs}")

    persistent = H3LingBotPersistentKVPolicy(
        persistent_enabled=True,
        persistent_window_chunks=args.persistent_window_chunks,
        observation_tokens_per_chunk=int(contract["kv_tokens"]),
        action_tokens_per_chunk=args.actions_per_observation_frame,
        **common,
    ).to(device=device, dtype=dtype).eval()
    persistent.load_state_dict(payload["model"], strict=True)
    feedback = LingBotTeacherForcedFeedback(
        observation_kv=history_batch["video_kv_cache"],
        observed_frame_count=1,
        executed_actions=history_batch["actions"][:, : args.executed_action_steps],
        proprio=history_batch["proprio"],
    )
    torch.cuda.synchronize(device)
    persistent_started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        persistent_prediction, state = forward_teacher_forced_history(
            persistent,
            episode_key="c57-real-probe",
            history=[feedback],
            noisy_actions=noisy,
            timestep=timestep,
            text_context=current_batch["text_context"],
            proprio=current_batch["proprio"],
            current_observation_kv=current_batch["video_kv_cache"],
            text_mask=current_batch["text_mask"],
        )
    torch.cuda.synchronize(device)
    persistent_seconds = time.perf_counter() - persistent_started
    if not torch.isfinite(persistent_prediction).all():
        raise FloatingPointError("persistent C57 prediction is non-finite")

    restored = H3LingBotPersistentKVPolicy(
        persistent_enabled=True,
        persistent_window_chunks=args.persistent_window_chunks,
        observation_tokens_per_chunk=int(contract["kv_tokens"]),
        action_tokens_per_chunk=args.actions_per_observation_frame,
        **common,
    ).to(device=device, dtype=dtype).eval()
    restored.load_state_dict(persistent.state_dict(), strict=True)
    restored_state = LingBotPersistentKVState.from_snapshot(
        state.snapshot(), device=device, dtype=dtype
    )
    with torch.no_grad():
        restored_prediction = forward(
            restored,
            current_batch,
            noisy,
            timestep,
            persistent_state=restored_state,
        )
    restore_max_abs = float(
        (persistent_prediction.float() - restored_prediction.float()).abs().max().item()
    )
    if restore_max_abs != 0.0:
        raise AssertionError(f"C57 model/runtime restore differs: {restore_max_abs}")

    persistent.train()
    persistent.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    gradient_started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=dtype):
        train_prediction, train_state = forward_teacher_forced_history(
            persistent,
            episode_key="c57-gradient-probe",
            history=[feedback],
            noisy_actions=noisy,
            timestep=timestep,
            text_context=current_batch["text_context"],
            proprio=current_batch["proprio"],
            current_observation_kv=current_batch["video_kv_cache"],
            text_mask=current_batch["text_mask"],
        )
    loss = F.mse_loss(train_prediction.float(), current_batch["actions"].float())
    loss.backward()
    torch.cuda.synchronize(device)
    gradient_seconds = time.perf_counter() - gradient_started
    history_k_grad = persistent.action_expert.blocks[0].self_attn.k.weight.grad
    gradient_norm = float(history_k_grad.float().norm().item())
    if not torch.isfinite(history_k_grad).all() or gradient_norm <= 0:
        raise FloatingPointError("executed-action K/V gradient is absent or non-finite")
    parameter_delta = None
    if args.optimizer_step:
        before = persistent.action_expert.blocks[0].self_attn.k.weight.detach().clone()
        optimizer = torch.optim.AdamW(persistent.parameters(), lr=1.0e-3)
        optimizer.step()
        parameter_delta = float(
            (persistent.action_expert.blocks[0].self_attn.k.weight - before)
            .float()
            .abs()
            .max()
            .item()
        )
        if parameter_delta <= 0 or not torch.isfinite(
            torch.tensor(parameter_delta)
        ):
            raise FloatingPointError("one-step optimizer probe did not update the model")

    report = {
        "status": "PASS_C57_REAL_MECHANICAL_PROBE",
        "checkpoint": str(args.checkpoint.resolve()),
        "completed_steps": int(payload["completed_steps"]),
        "sample_ids": batch["sample_ids"],
        "persistent_window_chunks": args.persistent_window_chunks,
        "persistent_token_capacity": persistent.persistent_token_capacity,
        "executed_action_steps": args.executed_action_steps,
        "actions_per_observation_frame": args.actions_per_observation_frame,
        "disabled_d0_max_abs": disabled_max_abs,
        "persistent_shape": list(persistent_prediction.shape),
        "persistent_finite": True,
        "persistent_vs_disabled_max_abs": float(
            (persistent_prediction.float() - disabled_prediction.float()).abs().max().item()
        ),
        "history_k_gradient_norm": gradient_norm,
        "loss": float(loss.item()),
        "restore_max_abs": restore_max_abs,
        "runtime_state": train_state.audit(),
        "optimizer_step": bool(args.optimizer_step),
        "optimizer_parameter_max_abs_delta": parameter_delta,
        "disabled_two_forward_seconds": disabled_seconds,
        "persistent_teacher_forced_forward_seconds": persistent_seconds,
        "persistent_forward_backward_seconds": gradient_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "provisional_5000_step_hours_at_ten_microsteps": (
            5000 * gradient_seconds * 10 / 3600
        ),
        "throughput_warning": "Single-GPU one-history probe; run an 8-GPU ten-step canary before long allocation.",
        "candidate_checkpoint_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
