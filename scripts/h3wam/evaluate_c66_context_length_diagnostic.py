#!/usr/bin/env python3
"""Diagnose whether C66 harm is structural, optimization-driven, or context-length driven.

This is an analysis-only paired evaluation.  It compares the promoted C58 parent
and the failed C66 s100 checkpoint on the same frozen heldout rows and noise.  No
result from this script authorizes training, rollout, or model promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

# Reuse the exact frozen dataset, online-H3 materialization, policy constructor,
# and deterministic flow contract from the preregistered C66 canary.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_c66_lingbot_c58_persistent_canary as C66  # noqa: E402

from fastwam.models.h3wam.lingbot_persistent_kv import (  # noqa: E402
    LingBotPersistentKVState,
)


FORMAT = "h3wam-c66-context-length-diagnostic-v1"
PARENT_SHA256 = C66.C58_SHA256
H3_SHA256 = C66.H3_SHA256
WINDOWS = (1, 3, 7)


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
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--c66-checkpoint", type=Path, required=True)
    parser.add_argument("--c66-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=66017)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def truncated_state(
    state: LingBotPersistentKVState, history_chunks: int, device: torch.device
) -> LingBotPersistentKVState:
    """Keep the newest committed chunks while preserving absolute coordinates."""

    if history_chunks not in WINDOWS:
        raise ValueError(f"unsupported history window: {history_chunks}")
    snapshot = state.snapshot()
    # Each committed chunk contributes one observation and one action update.
    first_update = max(0, int(snapshot["next_update_id"]) - 2 * history_chunks)
    entries = snapshot["entries"]
    for layer, rows in entries.items():
        entries[layer] = [
            row for row in rows if int(row["update_id"]) >= first_update
        ]
        if not entries[layer]:
            raise RuntimeError("context truncation removed every persistent token")
    restored = LingBotPersistentKVState.from_snapshot(
        snapshot, device=device, dtype=torch.bfloat16
    )
    if restored.frame_st_id != state.frame_st_id or restored.action_st_id != state.action_st_id:
        raise RuntimeError("context truncation changed absolute rollout coordinates")
    return restored


@torch.no_grad()
def committed_state(policy, sequence) -> LingBotPersistentKVState:
    state = policy.new_persistent_state(
        f"{sequence['suite']}:{sequence['episode']}"
    )
    current = sequence["current"]
    for feedback in sequence["history"]:
        policy.commit_executed_feedback(
            state,
            observation_kv=feedback["observation_kv"],
            observed_frame_count=feedback["observed_frame_count"],
            executed_actions=feedback["executed_actions"],
            text_context=current["text_context"],
            proprio=feedback["proprio"],
            text_mask=current["text_mask"],
        )
    if state.next_update_id != 14:
        raise RuntimeError("C66 diagnostic did not commit all seven history chunks")
    return state


@torch.no_grad()
def predict(policy, sequence, noisy, timesteps, state=None):
    current = sequence["current"]
    if state is None:
        return C66.predict_off(policy, sequence, noisy, timesteps)
    return policy(
        noisy,
        timesteps,
        text_context=current["text_context"],
        proprio=current["proprio"],
        video_kv_cache=sequence["current_kv"],
        text_mask=current["text_mask"],
        persistent_state=state,
    )


def evaluate_checkpoint(
    *,
    name: str,
    checkpoint: Path,
    policy,
    provider,
    heldout,
    inv_freq,
    flow,
    rank: int,
    world_size: int,
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = policy.load_state_dict(payload["model"], strict=True)
    if restored.missing_keys or restored.unexpected_keys:
        raise RuntimeError(f"{name} strict restore failed")
    del payload
    policy.eval()
    local: list[dict[str, Any]] = []
    for index in range(rank, len(heldout), world_size):
        sequence = C66.materialize_sequence(
            provider, heldout[index], inv_freq, device, torch.bfloat16
        )
        noisy, target, timesteps = C66.C58.PARENT.PARENT.deterministic_flow_batch(
            sequence["current"]["actions"], flow, seed=seed + 2_000_000 + index
        )
        valid = (~sequence["current"]["action_is_pad"]).unsqueeze(-1)

        def mse(value: torch.Tensor) -> float:
            value = value.float()
            target_float = target.float()
            return float(
                ((value - target_float).square() * valid).sum()
                / (valid.sum() * value.shape[-1]).clamp_min(1)
            )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            off = predict(policy, sequence, noisy, timesteps)
            full = committed_state(policy, sequence)
            arm_predictions = {
                f"k{window}": predict(
                    policy,
                    sequence,
                    noisy,
                    timesteps,
                    truncated_state(full, window, device),
                )
                for window in WINDOWS
            }
        row = {
            "checkpoint": name,
            "index": index,
            "id": sequence["id"],
            "off_mse": mse(off),
        }
        for arm, value in arm_predictions.items():
            row[f"{arm}_mse"] = mse(value)
            row[f"{arm}_vs_off_prediction_max_abs"] = float(
                (value.float() - off.float()).abs().max()
            )
        local.append(row)
    gathered: list[list[dict[str, Any]] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return [row for rows in gathered if rows for row in rows]


def main() -> None:
    args = parse_args()
    rank, world_size, device = C66.C58.distributed_setup()
    if world_size != 8:
        raise ValueError("C66 context diagnostic requires exactly eight ranks")
    torch.manual_seed(args.seed + rank)
    started = time.perf_counter()
    plan = load_json(args.plan.resolve())
    heldout = C66.SequenceDataset(
        args.heldout_manifest,
        args.dense_manifest,
        args.source_manifest,
        args.cache_root,
        args.h3_checkpoint,
    )
    if (
        len(heldout) != 64
        or plan.get("schema") != C66.PLAN_SCHEMA
        or heldout.manifest_sha256 != plan.get("heldout_manifest_sha256")
    ):
        raise ValueError("C66 diagnostic frozen heldout identity mismatch")

    identities = None
    if rank == 0:
        report = load_json(args.c66_report.resolve())
        identities = {
            "parent": sha256_file(args.parent_checkpoint.resolve()),
            "h3": sha256_file(args.h3_checkpoint.resolve()),
            "c66": sha256_file(args.c66_checkpoint.resolve()),
            "report_c66": report.get("checkpoint_sha256"),
            "report_status": report.get("status"),
            "report_permission": report.get("permission"),
        }
    shared = [identities]
    dist.broadcast_object_list(shared, src=0)
    identities = shared[0]
    if identities != {
        "parent": PARENT_SHA256,
        "h3": H3_SHA256,
        "c66": identities["c66"],
        "report_c66": identities["c66"],
        "report_status": "FAIL_C66_PAIRED_CANARY",
        "report_permission": "NO_GO_C66_LONG_TRAINING",
    }:
        raise ValueError("C66 diagnostic checkpoint/report identity gate failed")

    with safe_open(args.h3_checkpoint.resolve(), framework="pt", device="cpu") as handle:
        inv_freq = handle.get_tensor("rope.inv_freq").float().to(device)
    provider = C66.C58OnlineFrozenH3Provider(
        args.h3_checkpoint, layers=C66.LAYERWISE_H3_50_TO_ACTION_30
    ).to(device).eval()
    provider.requires_grad_(False)
    policy = C66.build_policy(device, torch.bfloat16)
    flow = C66.C58.PARENT.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    rows = []
    for name, checkpoint in (
        ("parent_c58", args.parent_checkpoint.resolve()),
        ("trained_c66_s100", args.c66_checkpoint.resolve()),
    ):
        rows.extend(
            evaluate_checkpoint(
                name=name,
                checkpoint=checkpoint,
                policy=policy,
                provider=provider,
                heldout=heldout,
                inv_freq=inv_freq,
                flow=flow,
                rank=rank,
                world_size=world_size,
                device=device,
                seed=args.seed,
            )
        )

    if rank == 0:
        if len(rows) != 128 or len({(r["checkpoint"], r["id"]) for r in rows}) != 128:
            raise RuntimeError("C66 diagnostic paired coverage is incomplete")
        metrics = ("off_mse", "k1_mse", "k3_mse", "k7_mse")
        by_checkpoint = {}
        for checkpoint in ("parent_c58", "trained_c66_s100"):
            selected = [row for row in rows if row["checkpoint"] == checkpoint]
            means = {
                metric: sum(row[metric] for row in selected) / len(selected)
                for metric in metrics
            }
            by_checkpoint[checkpoint] = {
                "means": means,
                "relative_to_off": {
                    arm: (means[f"{arm}_mse"] - means["off_mse"])
                    / means["off_mse"]
                    for arm in ("k1", "k3", "k7")
                },
                "finite": all(math.isfinite(value) for value in means.values()),
            }
        parent = by_checkpoint["parent_c58"]["means"]
        trained = by_checkpoint["trained_c66_s100"]["means"]
        analysis = {
            "parent_structural_k7_vs_off": (parent["k7_mse"] - parent["off_mse"])
            / parent["off_mse"],
            "training_delta_k7_vs_parent_k7": (trained["k7_mse"] - parent["k7_mse"])
            / parent["k7_mse"],
            "trained_best_context_window": min(
                WINDOWS, key=lambda window: trained[f"k{window}_mse"]
            ),
        }
        result = {
            "format": FORMAT,
            "status": "PASS_C66_CONTEXT_LENGTH_DIAGNOSTIC",
            "permission": "DIAGNOSTIC_ONLY_NO_TRAINING_OR_ROLLOUT_RELEASE",
            "effect_status": "NOT_LIBERO_EVIDENCE",
            "hypothesis": "Paired parent/trained and k1/k3/k7 arms separate structural prefix harm, optimization harm, and excessive-history harm.",
            "world_size": world_size,
            "heldout_samples_per_checkpoint": 64,
            "optimizer_steps": 0,
            "rows": rows,
            "checkpoints": by_checkpoint,
            "analysis": analysis,
            "identities": identities,
            "elapsed_seconds": time.perf_counter() - started,
            "boundary": "This diagnostic selects the next bounded mechanism test only; it cannot promote C66 or authorize long training.",
        }
        atomic_json(args.output.resolve(), result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
