#!/usr/bin/env python3
"""Real C58-parent mechanical gate for the C62 MiniWorld context route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

from fastwam.models.h3wam.c62_miniworld_context import (
    C62MiniWorldRollingContextPolicy,
    MiniWorldRollingContextState,
    verify_miniworld_execution_source,
)
from fastwam.models.h3wam.fastwam_full_tower import (
    H3FastWAMFullTowerPolicy,
    LAYERWISE_H3_50_TO_ACTION_30,
)


C58_CHECKPOINT_SHA256 = (
    "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
)
EXPECTED_FASTWAM_COMMIT = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
EXPECTED_ACTION_DIT_SHA256 = (
    "1301d9224149de43bb701f620a5d41858ecc63c6b19a573ec32edd45a3bdb0a2"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_c58_parent_payload(payload: dict) -> dict:
    """Fail closed on the fields actually serialized by C58b schema v2."""

    contract = payload.get("contract", {})
    model_spec = contract.get("model_spec", {})
    expected = LAYERWISE_H3_50_TO_ACTION_30
    checks = {
        "schema_version": payload.get("schema_version") == 1,
        "completed_steps": int(payload.get("completed_steps", -1)) == 10_000,
        "candidate": contract.get("candidate") == "C58B_FASTWAM_FULL30_H3_LAYERWISE",
        "classification": contract.get("classification")
        == "action-only-on-frozen-layerwise-h3-kv_backbone_port",
        "fastwam_commit": contract.get("fastwam_commit") == EXPECTED_FASTWAM_COMMIT,
        "action_dit_sha256": contract.get("fastwam_action_dit_sha256")
        == EXPECTED_ACTION_DIT_SHA256,
        "carrier_source_mode": contract.get("carrier_source_mode")
        == "uniform_h3_50_to_action30",
        "kv_layers": tuple(contract.get("kv_layers", ())) == expected,
        "block_mapping": tuple(contract.get("action_block_to_h3_layer", ()))
        == expected,
        "model_spec_layers": tuple(model_spec.get("carrier_layers", ())) == expected,
        "model_spec_depth": model_spec.get("action_layers") == 30,
        "model_spec_mode": model_spec.get("carrier_source_mode")
        == "uniform_h3_50_to_action30",
        "online_frozen_h3": contract.get("h3_execution")
        == "online_frozen_int8_per_rank_v1",
        "no_disk_kv": contract.get("disk_kv_training_input") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"checkpoint is not the fixed C58 layer-wise champion parent: {failed}")
    return checks


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--miniworld-source", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--expected-checkpoint-sha256", default=C58_CHECKPOINT_SHA256)
    result.add_argument("--seed", type=int, default=62017)
    result.add_argument("--history-chunks", type=int, default=3)
    result.add_argument("--max-cache-chunks", type=int, default=3)
    return result


def build_parent(device: torch.device, dtype: torch.dtype) -> H3FastWAMFullTowerPolicy:
    return H3FastWAMFullTowerPolicy(
        enabled=True,
        carrier_layers=LAYERWISE_H3_50_TO_ACTION_30,
        action_dim=7,
        proprio_dim=8,
        context_dim=5120,
        hidden_dim=1024,
        ffn_dim=4096,
        num_heads=56,
        attn_head_dim=128,
        freq_dim=256,
        num_layers=30,
        use_gradient_checkpointing=False,
        action_block_to_h3_layer=LAYERWISE_H3_50_TO_ACTION_30,
    ).to(device=device, dtype=dtype)


def carrier(device: torch.device, seed: int) -> dict[int, dict[str, torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return {
        layer: {
            name: torch.randn(
                1,
                32,
                56,
                128,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            for name in ("k", "v")
        }
        for layer in LAYERWISE_H3_50_TO_ACTION_30
    }


def main() -> None:
    args = parser().parse_args()
    started = time.perf_counter()
    source = args.miniworld_source.resolve()
    sys.path.insert(0, str(source))
    source_report = verify_miniworld_execution_source(source)

    checkpoint = args.checkpoint.resolve()
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != args.expected_checkpoint_sha256:
        raise RuntimeError(f"C58 checkpoint SHA256 mismatch: {actual_sha256}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    parent_identity_checks = validate_c58_parent_payload(payload)

    if not torch.cuda.is_available():
        raise RuntimeError("real C58 C62 probe requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = torch.bfloat16
    parent = build_parent(device, dtype)
    parent_restore = parent.load_state_dict(payload["model"], strict=True)
    if parent_restore.missing_keys or parent_restore.unexpected_keys:
        raise RuntimeError(f"non-strict C58 restore: {parent_restore}")
    del payload
    model = C62MiniWorldRollingContextPolicy(
        parent,
        context_enabled=True,
        max_cache_chunks=args.max_cache_chunks,
    ).to(device=device, dtype=dtype)

    current = carrier(device, args.seed + 100)
    noisy = torch.randn(1, 32, 7, device=device, dtype=dtype)
    timestep = torch.tensor([500.0], device=device)
    text = torch.randn(1, 4, 5120, device=device, dtype=dtype)
    proprio = torch.randn(1, 8, device=device, dtype=dtype)
    text_mask = torch.ones(1, 4, device=device, dtype=torch.bool)
    common = {
        "text_context": text,
        "proprio": proprio,
        "text_mask": text_mask,
    }

    parent.eval()
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        parent_prediction = parent(
            noisy, timestep, video_kv_cache=current, **common
        ).float()
        disabled_prediction = model(
            noisy,
            timestep,
            video_kv_cache=current,
            use_context=False,
            **common,
        ).float()
        empty_state = model.new_context_state("empty")
        empty_context_prediction = model(
            noisy,
            timestep,
            video_kv_cache=current,
            context_state=empty_state,
            actions_before_current=None,
            use_context=True,
            **common,
        ).float()
    disabled_max_abs = float((disabled_prediction - parent_prediction).abs().max())
    empty_context_max_abs = float(
        (empty_context_prediction - parent_prediction).abs().max()
    )
    if disabled_max_abs != 0.0 or empty_context_max_abs != 0.0:
        raise RuntimeError(
            "C62 default/empty-context parent parity failed: "
            f"{disabled_max_abs}/{empty_context_max_abs}"
        )

    state = model.new_context_state("real-c58-mechanical")
    for index in range(args.history_chunks):
        model.commit_real_observation(
            state,
            observation_kv=carrier(device, args.seed + index),
            actions_before_observation=(
                None
                if index == 0
                else torch.randn(1, 8, 7, device=device, dtype=dtype)
            ),
        )
    actions_before_current = torch.randn(1, 8, 7, device=device, dtype=dtype)
    rolling = model._rolling_carrier(state, current, actions_before_current)
    expected_tokens = (len(state.entries) + 1) * 32
    if any(item["k"].shape[1] != expected_tokens for item in rolling.values()):
        raise RuntimeError("C62 rolling K/V token shape mismatch")
    h3_source_inputs_require_grad = any(
        tensor.requires_grad
        for item in current.values()
        for tensor in item.values()
    ) or any(
        tensor.requires_grad
        for entry in state.entries
        for item in entry.observation_kv.values()
        for tensor in item.values()
    )
    bridge_outputs_require_grad = all(
        tensor.requires_grad
        for item in rolling.values()
        for tensor in item.values()
    )
    if h3_source_inputs_require_grad or not bridge_outputs_require_grad:
        raise RuntimeError(
            "C62 gradient boundary failed: frozen H3 sources must be detached "
            "and trainable bridge outputs must retain gradients"
        )

    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=dtype):
        prediction = model(
            noisy,
            timestep,
            video_kv_cache=current,
            context_state=state,
            actions_before_current=actions_before_current,
            use_context=True,
            **common,
        )
        action_loss = prediction.float().square().mean()
    action_loss.backward()
    parent_gradients = [
        float(block.self_attn.o.weight.grad.float().norm())
        for block in model.parent.action_expert.blocks
    ]
    refiner_gradients = [
        float(refiner[-1].weight.grad.float().norm())
        for refiner in model.modulator.layer_refiners.values()
    ]
    shared_gradient = float(model.modulator.shared_modulation.weight.grad.float().norm())
    finite_positive = lambda values: all(math.isfinite(value) and value > 0 for value in values)
    if not finite_positive(parent_gradients):
        raise RuntimeError("action loss did not reach all 30 C58 action blocks")
    if not finite_positive(refiner_gradients) or not math.isfinite(shared_gradient) or shared_gradient <= 0:
        raise RuntimeError("action loss did not reach the MiniWorld action modulation route")

    # Restore bridge parameters and episode runtime independently.  The C58
    # parent itself was already strictly restored from the fixed 12.18 GB
    # champion checkpoint above; no new candidate checkpoint is written here.
    bridge_state = {
        key: value.detach().cpu().clone()
        for key, value in model.modulator.state_dict().items()
    }
    runtime_snapshot = state.snapshot()
    model.modulator.load_state_dict(bridge_state, strict=True)
    restored_runtime = MiniWorldRollingContextState.from_snapshot(
        runtime_snapshot, device=device, dtype=dtype
    )
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        before_restore = model(
            noisy,
            timestep,
            video_kv_cache=current,
            context_state=state,
            actions_before_current=actions_before_current,
            use_context=True,
            **common,
        ).float()
        after_restore = model(
            noisy,
            timestep,
            video_kv_cache=current,
            context_state=restored_runtime,
            actions_before_current=actions_before_current,
            use_context=True,
            **common,
        ).float()
    restore_max_abs = float((before_restore - after_restore).abs().max())
    history_delta = float((before_restore - parent_prediction).abs().max())
    if restore_max_abs != 0.0 or history_delta <= 0.0:
        raise RuntimeError(
            f"C62 restore/history gate failed: {restore_max_abs}/{history_delta}"
        )

    report = {
        "event": "h3_c62_miniworld_rolling_context_real_c58_mechanical_gate",
        "status": "PASS_MECHANICAL_GATE",
        "effect_status": "NOT_EVIDENCE_READY",
        "permission": "NO_GO_LONG",
        "classification": "intentional_composition_world_context_into_action_policy",
        "source": source_report,
        "c58_checkpoint": str(checkpoint),
        "c58_checkpoint_sha256": actual_sha256,
        "c58_completed_steps": 10_000,
        "parent_strict_restore": True,
        "parent_identity_checks": parent_identity_checks,
        "default_off_parent_max_abs": disabled_max_abs,
        "empty_context_parent_max_abs": empty_context_max_abs,
        "history_changes_action_max_abs": history_delta,
        "rolling_state": state.audit(),
        "rolling_tokens_per_layer": expected_tokens,
        "action_loss": float(action_loss.detach()),
        "parent_block_gradient_norms": parent_gradients,
        "bridge_shared_gradient_norm": shared_gradient,
        "bridge_layer_refiner_gradient_norms": refiner_gradients,
        "runtime_and_bridge_restore_max_abs": restore_max_abs,
        "h3_source_inputs_require_grad": h3_source_inputs_require_grad,
        "bridge_outputs_require_grad": bridge_outputs_require_grad,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "elapsed_seconds": time.perf_counter() - started,
        "boundary": (
            "MiniWorld remains a video world model; C58 remains the action policy. "
            "This gate authorizes only causal-data/optimizer canaries, not long training."
        ),
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
