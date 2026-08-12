#!/usr/bin/env python3
"""Audit the LIBERO multi-suite candidate and run its real artifact loader smoke."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--motion-root", type=Path)
    parser.add_argument(
        "--require-complete-motion",
        action="store_true",
        help="Load and validate a motion artifact for every unique window.",
    )
    parser.add_argument(
        "--full-cache-audit",
        action="store_true",
        help="Load and finite-check every H3 window rather than the loader-smoke subset.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def episode_keys(rows: list[dict]) -> set[tuple[str, int]]:
    return {(str(row["suite"]), int(row["episode"])) for row in rows}


def finite(artifact: dict, keys: tuple[str, ...]) -> bool:
    return all(torch.isfinite(artifact[key]).all().item() for key in keys)


def normalize(value: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    scale = (high.float() - low.float()).clamp_min(1.0e-6)
    return ((value.float() - low.float()) / scale * 2.0 - 1.0).clamp(-1.0, 1.0)


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual.float(), expected.float(), rtol=1.0e-5, atol=1.0e-6):
        delta = float((actual.float() - expected.float()).abs().max())
        raise ValueError(f"train-only statistic mismatch for {name}: max_abs_delta={delta}")


def main() -> None:
    args = parse_args()
    candidate = args.candidate.resolve()
    cache = args.cache_root.resolve()
    motion = args.motion_root.resolve() if args.motion_root else None
    all_rows = read_jsonl(candidate / "manifest_all.jsonl")
    unique_train = read_jsonl(candidate / "manifest_train_unique.jsonl")
    candidate_report = json.loads(
        (candidate / "training_candidate.json").read_text(encoding="utf-8")
    )
    sampling_manifest = str(candidate_report["sampling"]["manifest"])
    sampled_train = read_jsonl(candidate / sampling_manifest)
    val_rows = read_jsonl(candidate / "manifest_val.jsonl")
    report = candidate_report
    contexts = json.loads((candidate / "task_contexts.json").read_text(encoding="utf-8"))

    all_ids = {str(row["id"]) for row in all_rows}
    train_ids = {str(row["id"]) for row in unique_train}
    val_ids = {str(row["id"]) for row in val_rows}
    if len(all_ids) != len(all_rows) or train_ids & val_ids or train_ids | val_ids != all_ids:
        raise ValueError("unique train/validation manifests do not partition manifest_all")
    if episode_keys(unique_train) & episode_keys(val_rows):
        raise ValueError("episode leakage between train and validation")
    if any(str(row["id"]) not in train_ids for row in sampled_train):
        raise ValueError("sampling manifest contains a non-training window")

    policy = str(report["sampling"]["policy"])
    target = report["sampling"].get("target_task")
    expected_repeats = report["sampling"].get("target_total_repeats")
    observed_repeats = Counter(str(row["id"]) for row in sampled_train)
    for row in unique_train:
        expected = (
            int(expected_repeats)
            if policy == "target_task_oversampling" and str(row["task"]) == target
            else 1
        )
        if observed_repeats[str(row["id"])] != expected:
            raise ValueError(f"unexpected sampling multiplicity for {row['id']}")

    window_files = {path.stem for path in (cache / "windows").glob("*.pt")}
    context_files = {path.stem for path in (cache / "contexts").glob("*.pt")}
    if all_ids - window_files or set(contexts) - context_files:
        raise FileNotFoundError("H3 cache is incomplete for this candidate")

    # Recompute the owner-declared train-only normalization contract.  This is
    # intentionally based on unique source windows; sampling replicas affect
    # frequency, not the support of the action/state normalization.
    smoke_rows: list[dict] = []
    seen_suites: set[str] = set()
    for row in unique_train:
        suite = str(row["suite"])
        if suite not in seen_suites:
            smoke_rows.append(row)
            seen_suites.add(suite)
    smoke_ids = {str(row["id"]) for row in smoke_rows}
    actions: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    train_artifacts: dict[str, dict] = {}
    for row in unique_train:
        artifact = torch.load(
            cache / "windows" / f"{row['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        actions.append(artifact["actions"][:32].float())
        states.append(artifact["state"].float())
        if str(row["id"]) in smoke_ids:
            train_artifacts[str(row["id"])] = artifact
    action = torch.cat(actions)
    state = torch.stack(states)
    stats = torch.load(cache / "stats.pt", map_location="cpu", weights_only=False)
    if int(stats["num_windows"]) != len(unique_train):
        raise ValueError("stats window count is not the unique training population")
    expected_stats = {
        "action_min": action.amin(dim=0),
        "action_max": action.amax(dim=0),
        "action_mean": action.mean(dim=0),
        "action_std": action.std(dim=0).clamp_min(1.0e-6),
        "state_min": state.amin(dim=0),
        "state_max": state.amax(dim=0),
        "state_mean": state.mean(dim=0),
        "state_std": state.std(dim=0).clamp_min(1.0e-6),
    }
    for name, expected in expected_stats.items():
        assert_close(name, stats[name], expected)

    # A real loader smoke spans every suite, routes the cached task context,
    # applies final normalization, and stacks the exact tensors consumed by
    # the FSDP trainer.
    smoke_shapes = []
    for row in smoke_rows:
        artifact = train_artifacts[str(row["id"])]
        context = torch.load(
            cache / "contexts" / f"{row['context_id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if context.get("text_only") is not True or torch.any(context["token_tags"] != 1):
            raise ValueError(f"context {row['context_id']} is not text-only")
        norm_actions = normalize(artifact["actions"][:32], stats["action_min"], stats["action_max"])
        norm_state = normalize(artifact["state"], stats["state_min"], stats["state_max"])
        if not finite(artifact, ("video_latents", "first_frame_latents", "actions", "state")):
            raise ValueError(f"non-finite H3 artifact: {row['id']}")
        if not torch.isfinite(norm_actions).all() or not torch.isfinite(norm_state).all():
            raise ValueError(f"non-finite normalized sample: {row['id']}")
        smoke_shapes.append(
            {
                "suite": row["suite"],
                "video": list(artifact["video_latents"].shape),
                "first_frame": list(artifact["first_frame_latents"].shape),
                "actions": list(norm_actions.shape),
                "state": list(norm_state.shape),
                "context": list(context["context"].shape),
            }
        )

    if args.full_cache_audit:
        for row in all_rows:
            identifier = str(row["id"])
            artifact = train_artifacts.get(identifier)
            if artifact is None:
                artifact = torch.load(
                    cache / "windows" / f"{identifier}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            if not finite(artifact, ("video_latents", "first_frame_latents", "actions", "state")):
                raise ValueError(f"non-finite H3 artifact: {identifier}")

    motion_checked = 0
    if motion is not None:
        motion_files = {path.stem for path in motion.glob("*.pt")}
        missing_motion = all_ids - motion_files
        if args.require_complete_motion and missing_motion:
            raise FileNotFoundError(f"missing motion artifacts: {len(missing_motion)}")
        rows_to_check = all_rows if args.require_complete_motion else smoke_rows
        for row in rows_to_check:
            path = motion / f"{row['id']}.pt"
            if not path.is_file():
                continue
            flow = torch.load(path, map_location="cpu", weights_only=False)["flow_latents"]
            rgb = train_artifacts.get(str(row["id"]))
            if rgb is None:
                rgb = torch.load(
                    cache / "windows" / f"{row['id']}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            if flow.shape != rgb["video_latents"].shape or not torch.isfinite(flow).all():
                raise ValueError(f"invalid motion artifact: {row['id']}")
            motion_checked += 1

    result = {
        "event": "multisuite_training_candidate_audit_complete",
        "source_demo_episodes": report["source"]["source_demo_episodes"],
        "tasks": report["source"]["tasks"],
        "unique_windows": len(all_rows),
        "train_unique_windows": len(unique_train),
        "sampling_policy": policy,
        "train_sampled_windows": len(sampled_train),
        "validation_windows": len(val_rows),
        "episode_overlap": 0,
        "stats_windows": int(stats["num_windows"]),
        "loader_smoke": smoke_shapes,
        "full_cache_audit": args.full_cache_audit,
        "motion_checked": motion_checked,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
