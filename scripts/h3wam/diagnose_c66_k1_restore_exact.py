#!/usr/bin/env python3
"""Read-only C66-k1 state-restore determinism diagnostic.

No optimizer is constructed.  The script replays the official heldout ordering,
checks snapshot tensors/metadata exactly, and separates serialization error from
repeat-forward numerical nondeterminism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from safetensors import safe_open
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_c66_k1_bounded_mechanism_canary as K1  # noqa: E402


C66 = K1.C66
FORMAT = "h3wam-c66-k1-restore-diagnostic-v2"
CHECKPOINT_SHA256 = "861e95d891ca9128c2cb3bcc514243104fe70fb05c01fc9c0076d384a9201eeb"
REPORT_SHA256 = "70975e1b9de6612f6bdb65ff8d0bbeb9fdff3530b82e6b22cc4a7c781aba908a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--dense-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=66017)
    return parser.parse_args()


def predict_from_state(policy, sequence, noisy, timesteps, state):
    current = sequence["current"]
    return policy(
        noisy,
        timesteps,
        text_context=current["text_context"],
        proprio=current["proprio"],
        video_kv_cache=sequence["current_kv"],
        text_mask=current["text_mask"],
        persistent_state=state,
    )


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def snapshot_exact(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, float]:
    scalar_keys = {
        "schema_version", "layers", "token_capacity", "episode_key",
        "frame_st_id", "action_st_id", "next_update_id",
    }
    if any(left[key] != right[key] for key in scalar_keys):
        return False, float("inf")
    if set(left["entries"]) != set(right["entries"]):
        return False, float("inf")
    maximum = 0.0
    exact = True
    metadata = {
        "kind", "update_id", "frame_start", "frame_count", "action_start",
        "action_count", "predicted",
    }
    for layer in left["entries"]:
        rows_left = left["entries"][layer]
        rows_right = right["entries"][layer]
        if len(rows_left) != len(rows_right):
            return False, float("inf")
        for first, second in zip(rows_left, rows_right, strict=True):
            if any(first[key] != second[key] for key in metadata):
                return False, float("inf")
            for name in ("key", "value"):
                a, b = first[name], second[name]
                exact = exact and torch.equal(a, b)
                maximum = max(maximum, max_abs(a, b))
    return exact, maximum


def main() -> None:
    args = parse_args()
    rank, world_size, device = C66.C58.distributed_setup()
    if world_size != 8:
        raise ValueError("C66-k1 restore diagnostic requires exactly eight ranks")
    started = time.perf_counter()
    if args.seed != 66017:
        raise ValueError("C66-k1 restore diagnostic seed is frozen at 66017")
    identities = None
    if rank == 0:
        identities = {
            "checkpoint": sha256_file(args.checkpoint.resolve()),
            "report": sha256_file(args.report.resolve()),
            "plan": sha256_file(args.plan.resolve()),
            "heldout": sha256_file(args.heldout_manifest.resolve()),
        }
        if identities["checkpoint"] != CHECKPOINT_SHA256 or identities["report"] != REPORT_SHA256:
            raise ValueError("C66-k1 formal artifact identity mismatch")
    shared = [identities]
    dist.broadcast_object_list(shared, src=0)
    identities = shared[0]

    heldout = C66.SequenceDataset(
        args.heldout_manifest,
        args.dense_manifest,
        args.source_manifest,
        args.cache_root,
        args.h3_checkpoint,
    )
    if len(heldout) != 64:
        raise ValueError("C66-k1 restore diagnostic requires heldout64")
    dtype = torch.bfloat16
    policy = C66.build_policy(device, dtype)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    restored_model = policy.load_state_dict(payload["model"], strict=True)
    if restored_model.missing_keys or restored_model.unexpected_keys:
        raise RuntimeError("C66-k1 checkpoint strict restore failed")
    del payload
    policy.eval()
    provider = C66.C58OnlineFrozenH3Provider(
        args.h3_checkpoint, layers=C66.LAYERWISE_H3_50_TO_ACTION_30
    ).to(device).eval()
    provider.requires_grad_(False)
    with safe_open(args.h3_checkpoint, framework="pt", device="cpu") as handle:
        inv_freq = handle.get_tensor("rope.inv_freq").float().to(device)
    flow = C66.C58.PARENT.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    local = []
    with torch.no_grad():
        for index in range(rank, len(heldout), world_size):
            sequence = K1.materialize_sequence(
                provider, heldout[index], inv_freq, device, dtype
            )
            noisy, _, timesteps = C66.C58.PARENT.PARENT.deterministic_flow_batch(
                sequence["current"]["actions"], flow,
                seed=args.seed + 1_000_000 + index,
            )
            with torch.autocast("cuda", dtype=dtype):
                clean, state = K1.predict_context(
                    policy, sequence, noisy, timesteps, shuffle_actions=False
                )
                # Preserve official ordering before its restore comparison.
                K1.predict_context(
                    policy, sequence, noisy, timesteps, shuffle_actions=True
                )
                C66.predict_off(policy, sequence, noisy, timesteps)
            before = state.snapshot()
            restored_state = C66.LingBotPersistentKVState.from_snapshot(
                before, device=device, dtype=dtype
            )
            after = restored_state.snapshot()
            # This is the formal trainer's actual comparison: clean was
            # produced inside CUDA BF16 autocast, while restore was recomputed
            # after leaving that scope.
            official_restored_outside = predict_from_state(
                policy, sequence, noisy, timesteps, restored_state
            )
            same_state_outside = predict_from_state(
                policy, sequence, noisy, timesteps, state
            )
            restored_outside_repeat = predict_from_state(
                policy, sequence, noisy, timesteps, restored_state
            )
            same_state_outside_repeat = predict_from_state(
                policy, sequence, noisy, timesteps, state
            )
            with torch.autocast("cuda", dtype=dtype):
                same_state_inside = predict_from_state(
                    policy, sequence, noisy, timesteps, state
                )
                restored_inside = predict_from_state(
                    policy, sequence, noisy, timesteps, restored_state
                )
                same_state_inside_repeat = predict_from_state(
                    policy, sequence, noisy, timesteps, state
                )
            exact, snapshot_max = snapshot_exact(before, after)
            local.append(
                {
                    "index": index,
                    "id": sequence["id"],
                    "snapshot_exact": exact,
                    "snapshot_max_abs": snapshot_max,
                    "formal_clean_inside_vs_restored_outside_max_abs": max_abs(
                        clean, official_restored_outside
                    ),
                    "same_state_outside_vs_restored_outside_max_abs": max_abs(
                        same_state_outside, restored_outside_repeat
                    ),
                    "same_state_inside_vs_restored_inside_max_abs": max_abs(
                        same_state_inside, restored_inside
                    ),
                    "clean_inside_vs_same_state_inside_max_abs": max_abs(
                        clean, same_state_inside
                    ),
                    "same_state_outside_repeat_max_abs": max_abs(
                        same_state_outside, same_state_outside_repeat
                    ),
                    "same_state_inside_repeat_max_abs": max_abs(
                        same_state_inside, same_state_inside_repeat
                    ),
                    "restored_outside_repeat_max_abs": max_abs(
                        official_restored_outside, restored_outside_repeat
                    ),
                    "coordinates": {
                        "frame_st_id": state.frame_st_id,
                        "action_st_id": state.action_st_id,
                        "next_update_id": state.next_update_id,
                    },
                }
            )
    gathered: list[list[dict[str, Any]] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    rows = sorted(
        [row for group in gathered if group for row in group],
        key=lambda row: row["index"],
    )
    if rank == 0:
        metric_names = [
            "snapshot_max_abs",
            "formal_clean_inside_vs_restored_outside_max_abs",
            "same_state_outside_vs_restored_outside_max_abs",
            "same_state_inside_vs_restored_inside_max_abs",
            "clean_inside_vs_same_state_inside_max_abs",
            "same_state_outside_repeat_max_abs",
            "same_state_inside_repeat_max_abs",
            "restored_outside_repeat_max_abs",
        ]
        maxima = {name: max(row[name] for row in rows) for name in metric_names}
        serialization_exact = all(row["snapshot_exact"] for row in rows)
        coordinates_exact = all(
            row["coordinates"] == {
                "frame_st_id": 15, "action_st_id": 56, "next_update_id": 14
            }
            for row in rows
        )
        autocast_scope_mismatch = (
            maxima["formal_clean_inside_vs_restored_outside_max_abs"] > 0
            and maxima["same_state_outside_vs_restored_outside_max_abs"] == 0
            and maxima["same_state_inside_vs_restored_inside_max_abs"] == 0
            and maxima["clean_inside_vs_same_state_inside_max_abs"] == 0
        )
        classification = (
            "EVALUATION_AUTOCAST_SCOPE_MISMATCH_NOT_SERIALIZATION_OR_K1_PREFIX"
            if serialization_exact and coordinates_exact and autocast_scope_mismatch
            else "UNRESOLVED_RESTORE_OR_PREFIX_DEFECT"
        )
        result = {
            "format": FORMAT,
            "status": "PASS_READ_ONLY_DIAGNOSTIC",
            "permission": "DIAGNOSTIC_ONLY_NO_RETRAIN_NO_THRESHOLD_CHANGE_NO_ROLLOUT",
            "classification": classification,
            "optimizer_steps": 0,
            "training_checkpoints_written": 0,
            "heldout_samples": len(rows),
            "serialization_exact_all_samples": serialization_exact,
            "absolute_coordinates_exact_all_samples": coordinates_exact,
            "maxima": maxima,
            "identities": identities,
            "rows": rows,
            "elapsed_seconds": time.perf_counter() - started,
            "boundary": (
                "This diagnostic does not alter the formal C66-k1 FAIL result, "
                "its preregistered exact gate, or any training/rollout permission."
            ),
        }
        atomic_json(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
