#!/usr/bin/env python3
"""No-optimizer mechanical gate for the C64 MiniWorld framewise bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from safetensors import safe_open
import torch

from fastwam.models.h3wam.c62_miniworld_context import (
    MiniWorldRollingContextState,
    verify_miniworld_execution_source,
)
from fastwam.models.h3wam.c64_miniworld_framewise_context import (
    C64MiniWorldFramewiseContextPolicy,
)
from fastwam.models.h3wam.fastwam_full_tower import (
    H3FastWAMFullTowerPolicy,
    LAYERWISE_H3_50_TO_ACTION_30,
)
from fastwam.models.h3wam.int8_online import _official_layout_functions


C58_CHECKPOINT_SHA256 = (
    "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
)
H3_CHECKPOINT_SHA256 = (
    "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
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


def validate_c58_parent_payload(payload: dict) -> dict[str, bool]:
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
        raise ValueError(f"checkpoint is not the fixed C58 champion: {failed}")
    return checks


def build_parent(device: torch.device) -> H3FastWAMFullTowerPolicy:
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
    ).to(device=device, dtype=torch.bfloat16)


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


def aligned_chunk_max_abs(
    left: torch.Tensor, right: torch.Tensor, chunk_order: tuple[int, ...]
) -> float:
    """Compare left chunks to a permuted right tensor after undoing its order."""

    chunks = right.chunk(len(chunk_order), dim=1)
    inverse = [chunk_order.index(index) for index in range(len(chunk_order))]
    restored = torch.cat([chunks[index] for index in inverse], dim=1)
    return float((left.float() - restored.float()).abs().max())


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--h3-checkpoint", type=Path, required=True)
    result.add_argument("--miniworld-source", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--seed", type=int, default=64017)
    return result


def main() -> None:
    args = parser().parse_args()
    started = time.perf_counter()
    source = args.miniworld_source.resolve()
    sys.path.insert(0, str(source))
    source_report = verify_miniworld_execution_source(source)

    # Execute H3's released layout builder.  A real v7 dual-view first frame is
    # 14x28 latent cells and produces 98 condition rows after the 2x2 patch. The
    # fixed real-window probe has 14 text rows, so its condition anchor is t=14.
    layout = _official_layout_functions()
    packed = layout.build_packed_sequence(
        text_token_tags=torch.zeros(14, dtype=torch.long),
        num_latent_frames=12,
        latent_height=14,
        latent_width=28,
        num_audio_latents=32,
        patch_size=(1, 2, 2),
        audio_channels=2,
        video_tag=0,
        audio_tag=2,
        keyframe_anchors=("first",),
    )
    position_ids, _, video_indices, _, _, condition_rows, _ = packed
    condition_indices = video_indices[:condition_rows].long()
    condition_positions = position_ids[condition_indices]
    unique_condition_times = torch.unique(condition_positions[:, 0]).tolist()
    layout_checks = {
        "condition_rows_98": int(condition_rows) == 98,
        "position_width_3": tuple(condition_positions.shape) == (98, 3),
        "single_condition_time": unique_condition_times == [14.0],
    }
    if not all(layout_checks.values()):
        raise RuntimeError(
            f"released H3 layout no longer matches the C58 condition contract: {layout_checks}"
        )

    h3_checkpoint = args.h3_checkpoint.resolve()
    h3_sha256 = sha256_file(h3_checkpoint)
    if h3_sha256 != H3_CHECKPOINT_SHA256:
        raise RuntimeError(f"H3 checkpoint SHA256 mismatch: {h3_sha256}")
    with safe_open(h3_checkpoint, framework="pt", device="cpu") as handle:
        inv_freq = handle.get_tensor("rope.inv_freq").float().clone()
    if tuple(inv_freq.shape) != (16,) or not torch.isfinite(inv_freq).all():
        raise RuntimeError("H3 rope.inv_freq is not finite shape [16]")

    checkpoint = args.checkpoint.resolve()
    c58_sha256 = sha256_file(checkpoint)
    if c58_sha256 != C58_CHECKPOINT_SHA256:
        raise RuntimeError(f"C58 checkpoint SHA256 mismatch: {c58_sha256}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    parent_identity_checks = validate_c58_parent_payload(payload)

    if not torch.cuda.is_available():
        raise RuntimeError("the real C58 C64 probe requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    parent = build_parent(device)
    restored = parent.load_state_dict(payload["model"], strict=True)
    if restored.missing_keys or restored.unexpected_keys:
        raise RuntimeError(f"non-strict C58 restore: {restored}")
    del payload
    parent.requires_grad_(False)
    parent.eval()
    model = C64MiniWorldFramewiseContextPolicy(
        parent,
        temporal_inv_freq=inv_freq,
        context_enabled=True,
        max_cache_frames=3,
    ).to(device=device, dtype=torch.bfloat16)

    current = carrier(device, args.seed + 100)
    noisy = torch.randn(1, 32, 7, device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([500.0], device=device)
    text = torch.randn(1, 4, 5120, device=device, dtype=torch.bfloat16)
    proprio = torch.randn(1, 8, device=device, dtype=torch.bfloat16)
    text_mask = torch.ones(1, 4, device=device, dtype=torch.bool)
    common = {"text_context": text, "proprio": proprio, "text_mask": text_mask}

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        parent_prediction = parent(
            noisy, timestep, video_kv_cache=current, **common
        ).float()
        disabled_prediction = model(
            noisy, timestep, video_kv_cache=current, use_context=False, **common
        ).float()
        empty_prediction = model(
            noisy,
            timestep,
            video_kv_cache=current,
            context_state=model.new_context_state("empty"),
            four_actions_before_current=None,
            use_context=True,
            **common,
        ).float()
    disabled_max_abs = float((disabled_prediction - parent_prediction).abs().max())
    empty_max_abs = float((empty_prediction - parent_prediction).abs().max())
    if disabled_max_abs != 0.0 or empty_max_abs != 0.0:
        raise RuntimeError(f"C64 parent parity failed: {disabled_max_abs}/{empty_max_abs}")

    frames = [carrier(device, args.seed + index) for index in range(3)]
    actions = [
        None,
        torch.randn(1, 4, 7, device=device, dtype=torch.bfloat16),
        torch.randn(1, 4, 7, device=device, dtype=torch.bfloat16),
    ]
    state = model.new_context_state("ordered")
    for frame, action in zip(frames, actions):
        model.commit_real_observation(
            state,
            observation_kv=frame,
            four_actions_before_observation=action,
        )
    current_actions = torch.randn(1, 4, 7, device=device, dtype=torch.bfloat16)
    rolling = model._rolling_carrier(state, current, current_actions)
    if any(item["k"].shape[1] != 128 for item in rolling.values()):
        raise RuntimeError("C64 rolling cache is not four frames x 32 tokens")

    # Stage A: 4-action/frame alignment with no temporal reindex.  Swapping the
    # two post-sink frame pairs only permutes K/V chunks, so aligned tensors are
    # exactly equal.  Stage B changes only temporal reindex: the same raw frame
    # receives a different temporal phase after the swap.
    swapped = model.new_context_state("swapped")
    for frame, action in (
        (frames[0], actions[0]),
        (frames[2], actions[2]),
        (frames[1], actions[1]),
    ):
        model.commit_real_observation(
            swapped,
            observation_kv=frame,
            four_actions_before_observation=action,
        )
    swapped_rolling = model._rolling_carrier(swapped, current, current_actions)
    first_layer = LAYERWISE_H3_50_TO_ACTION_30[0]
    order = (0, 2, 1, 3)
    reindexed_aligned_delta = aligned_chunk_max_abs(
        rolling[first_layer]["k"], swapped_rolling[first_layer]["k"], order
    )
    reindexed_v_aligned_delta = aligned_chunk_max_abs(
        rolling[first_layer]["v"], swapped_rolling[first_layer]["v"], order
    )

    # Construct the Stage-A no-reindex tensors from the same initialized
    # modulator.  This is a mechanical ablation only, not a trainable candidate.
    def unindexed(layer: int, candidate_state: MiniWorldRollingContextState):
        items = [
            model.modulator(
                layer,
                entry.observation_kv[layer],
                entry.actions_before_observation,
            )
            for entry in candidate_state.entries
        ]
        items.append(model.modulator(layer, current[layer], current_actions))
        return {
            name: torch.cat([item[name] for item in items], dim=1)
            for name in ("k", "v")
        }

    unindexed_ordered = unindexed(first_layer, state)
    unindexed_swapped = unindexed(first_layer, swapped)
    unindexed_k_aligned_delta = aligned_chunk_max_abs(
        unindexed_ordered["k"], unindexed_swapped["k"], order
    )
    unindexed_v_aligned_delta = aligned_chunk_max_abs(
        unindexed_ordered["v"], unindexed_swapped["v"], order
    )
    if (
        unindexed_k_aligned_delta != 0.0
        or unindexed_v_aligned_delta != 0.0
        or reindexed_aligned_delta <= 0.0
        or reindexed_v_aligned_delta != 0.0
    ):
        raise RuntimeError(
            "C64 temporal-order falsification failed: "
            f"unindexed={unindexed_k_aligned_delta}/{unindexed_v_aligned_delta}, "
            f"reindexed={reindexed_aligned_delta}/{reindexed_v_aligned_delta}"
        )

    # Freeze C58 and prove that the unchanged action objective reaches the new
    # bridge in all 30 action/world layer pairs.  No optimizer step is taken.
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(
            noisy,
            timestep,
            video_kv_cache=current,
            context_state=state,
            four_actions_before_current=current_actions,
            use_context=True,
            **common,
        )
        action_loss = prediction.float().square().mean()
    action_loss.backward()
    shared_gradient = float(model.modulator.shared_modulation.weight.grad.float().norm())
    refiner_gradients = [
        float(refiner[-1].weight.grad.float().norm())
        for refiner in model.modulator.layer_refiners.values()
    ]
    parent_has_gradient = any(
        parameter.grad is not None for parameter in model.parent.parameters()
    )
    finite_positive = lambda values: all(
        math.isfinite(value) and value > 0 for value in values
    )
    if (
        parent_has_gradient
        or not math.isfinite(shared_gradient)
        or shared_gradient <= 0
        or not finite_positive(refiner_gradients)
    ):
        raise RuntimeError("C64 frozen-parent/bridge gradient boundary failed")

    # Sink+FIFO stores raw H3 tensors and rebuilds contiguous phases, rather
    # than accumulating repeated destructive rotations.
    fifo = model.new_context_state("fifo")
    for index in range(5):
        model.commit_real_observation(
            fifo,
            observation_kv=carrier(device, args.seed + 200 + index),
            four_actions_before_observation=(
                None
                if index == 0
                else torch.full(
                    (1, 4, 7),
                    float(index),
                    device=device,
                    dtype=torch.bfloat16,
                )
            ),
        )
    fifo_ids = fifo.audit()["update_ids"]
    if fifo_ids != [0, 3, 4]:
        raise RuntimeError(f"C64 sink/FIFO identity failed: {fifo_ids}")

    bridge_state = {
        key: value.detach().cpu().clone()
        for key, value in model.modulator.state_dict().items()
    }
    runtime_snapshot = state.snapshot()
    model.modulator.load_state_dict(bridge_state, strict=True)
    restored_runtime = MiniWorldRollingContextState.from_snapshot(
        runtime_snapshot, device=device, dtype=torch.bfloat16
    )
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        before_restore = model(
            noisy,
            timestep,
            video_kv_cache=current,
            context_state=state,
            four_actions_before_current=current_actions,
            use_context=True,
            **common,
        ).float()
        after_restore = model(
            noisy,
            timestep,
            video_kv_cache=current,
            context_state=restored_runtime,
            four_actions_before_current=current_actions,
            use_context=True,
            **common,
        ).float()
    restore_max_abs = float((before_restore - after_restore).abs().max())
    if restore_max_abs != 0.0:
        raise RuntimeError(f"C64 bridge/runtime restore failed: {restore_max_abs}")

    report = {
        "event": "h3_c64_miniworld_framewise_context_mechanical_gate",
        "status": "PASS_MECHANICAL_GATE",
        "effect_status": "NOT_EVIDENCE_READY",
        "permission": "NO_GO_OPTIMIZER",
        "classification": "source_faithful_temporal_contract_composition",
        "source": source_report,
        "c58_checkpoint": str(checkpoint),
        "c58_checkpoint_sha256": c58_sha256,
        "c58_parent_identity_checks": parent_identity_checks,
        "c58_parent_strict_restore": True,
        "h3_checkpoint": str(h3_checkpoint),
        "h3_checkpoint_sha256": h3_sha256,
        "h3_rope_inv_freq": inv_freq.tolist(),
        "h3_layout": {
            "checks": layout_checks,
            "latent_height": 14,
            "latent_width": 28,
            "condition_rows": int(condition_rows),
            "unique_condition_temporal_positions": unique_condition_times,
            "first_three_condition_positions": condition_positions[:3].tolist(),
        },
        "default_off_parent_max_abs": disabled_max_abs,
        "empty_context_parent_max_abs": empty_max_abs,
        "frame_alignment": {
            "actions_per_observation": 4,
            "temporal_mean": False,
            "learned_null_for_true_sink": True,
            "rolling_tokens_per_layer": 128,
        },
        "staged_single_variable_falsification": {
            "c64a_unindexed_aligned_k_max_abs": unindexed_k_aligned_delta,
            "c64a_unindexed_aligned_v_max_abs": unindexed_v_aligned_delta,
            "c64b_reindexed_aligned_k_max_abs": reindexed_aligned_delta,
            "c64b_reindexed_aligned_v_max_abs": reindexed_v_aligned_delta,
            "interpretation": (
                "C64A framewise K/V remains only a chunk permutation after pair-order swap; "
                "adding only exact H3 temporal RoPE reindex in C64B makes the same raw keys "
                "position-sensitive while values remain unchanged."
            ),
        },
        "action_loss": float(action_loss.detach()),
        "frozen_c58_parent_has_gradient": parent_has_gradient,
        "bridge_shared_gradient_norm": shared_gradient,
        "bridge_layer_refiner_gradient_norms": refiner_gradients,
        "bridge_refiners_with_positive_gradient": sum(
            value > 0 for value in refiner_gradients
        ),
        "sink_fifo_update_ids": fifo_ids,
        "runtime_and_bridge_restore_max_abs": restore_max_abs,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "elapsed_seconds": time.perf_counter() - started,
        "boundary": (
            "This gate validates source identity, shapes, temporal semantics, frozen-parent "
            "gradient routing and restore only. It performs zero optimizer steps and does not "
            "authorize training, fusion, or a LIBERO effect claim."
        ),
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
