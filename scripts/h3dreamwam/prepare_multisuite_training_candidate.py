#!/usr/bin/env python3
"""Create an auditable, weighted multi-suite LIBERO training candidate.

The source manifests remain episode-disjoint and unique.  Sampling replicas are
written only to the weighted training manifest, so they cannot be mistaken for
additional demonstrations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-task",
        default="open the top drawer and put the bowl inside",
    )
    parser.add_argument(
        "--target-total-repeats",
        type=int,
        default=1,
        help=(
            "Total occurrences of each target row. The safe default 1 means "
            "uniform multi-task sampling with no task-specific oversampling."
        ),
    )
    parser.add_argument("--sampling-salt", default="h3dreamwam-multisuite-v3-2026-08-08")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_digest(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def episode_keys(rows: list[dict]) -> set[tuple[str, int]]:
    return {(str(row["suite"]), int(row["episode"])) for row in rows}


def main() -> None:
    args = parse_args()
    if args.target_total_repeats < 1:
        raise ValueError("--target-total-repeats must be at least one")

    base = args.base_candidate.resolve()
    cache = args.cache_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    all_rows = read_jsonl(base / "manifest_all.jsonl")
    train_rows = read_jsonl(base / "manifest_train.jsonl")
    val_rows = read_jsonl(base / "manifest_val.jsonl")
    base_report = json.loads((base / "candidate_report.json").read_text(encoding="utf-8"))
    contexts = json.loads((base / "task_contexts.json").read_text(encoding="utf-8"))

    all_ids = [str(row["id"]) for row in all_rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("base manifest contains duplicate window ids")
    if episode_keys(train_rows) & episode_keys(val_rows):
        raise ValueError("train/validation episode leakage detected")
    if set(all_ids) != {str(row["id"]) for row in train_rows + val_rows}:
        raise ValueError("train and validation manifests do not partition manifest_all")
    if any(str(row["context_id"]) not in contexts for row in all_rows):
        raise ValueError("manifest references a missing task context")

    window_dir = cache / "windows"
    context_dir = cache / "contexts"
    missing_windows = [identifier for identifier in all_ids if not (window_dir / f"{identifier}.pt").is_file()]
    missing_contexts = [identifier for identifier in contexts if not (context_dir / f"{identifier}.pt").is_file()]
    if missing_windows or missing_contexts:
        raise FileNotFoundError(
            f"incomplete H3 cache: missing_windows={len(missing_windows)}, "
            f"missing_contexts={len(missing_contexts)}"
        )

    uniform_sampling = args.target_total_repeats == 1
    target_rows = [row for row in train_rows if str(row["task"]) == args.target_task]
    if not uniform_sampling and not target_rows:
        available = sorted({str(row["task"]) for row in train_rows})
        raise ValueError(f"target task not found: {args.target_task!r}; available={available}")

    weighted_entries: list[tuple[str, dict]] = []
    for row in train_rows:
        repeats = args.target_total_repeats if str(row["task"]) == args.target_task else 1
        for replica in range(repeats):
            weighted_entries.append(
                (stable_digest(args.sampling_salt, row["id"], replica), row)
            )
    weighted_rows = [row for _, row in sorted(weighted_entries, key=lambda item: item[0])]

    write_jsonl(output / "manifest_all.jsonl", all_rows)
    write_jsonl(output / "manifest_train_unique.jsonl", train_rows)
    sampling_manifest = (
        "manifest_train_uniform.jsonl"
        if uniform_sampling
        else "manifest_train_weighted.jsonl"
    )
    write_jsonl(output / sampling_manifest, weighted_rows)
    write_jsonl(output / "manifest_train_sampling.jsonl", weighted_rows)
    # Conventional filename always resolves to the final sampling population.
    write_jsonl(output / "manifest_train.jsonl", weighted_rows)
    write_jsonl(output / "manifest_val.jsonl", val_rows)
    shutil.copyfile(base / "task_contexts.json", output / "task_contexts.json")

    unique_train_episodes = len(episode_keys(train_rows))
    val_episodes = len(episode_keys(val_rows))
    weighted_target_windows = sum(str(row["task"]) == args.target_task for row in weighted_rows)
    suite_episodes = Counter((str(row["suite"]), int(row["episode"])) for row in all_rows)
    suite_episode_counts = Counter(suite for suite, _ in suite_episodes)
    suite_window_counts = Counter(str(row["suite"]) for row in all_rows)
    report = {
        "schema_version": 1,
        "status": "candidate_not_frozen",
        "source": {
            "base_candidate": str(base),
            "h3_cache": str(cache),
            "source_demo_episodes": int(base_report["episodes"]),
            "unique_windows": len(all_rows),
            "tasks": len(contexts),
            "suites": {
                suite: {
                    "episodes": suite_episode_counts[suite],
                    "windows": suite_window_counts[suite],
                }
                for suite in sorted(suite_episode_counts)
            },
        },
        "split": {
            "unit": "episode",
            "salt": base_report["split_salt"],
            "train_episodes": unique_train_episodes,
            "validation_episodes": val_episodes,
            "train_unique_windows": len(train_rows),
            "validation_windows": len(val_rows),
            "episode_overlap": 0,
        },
        "sampling": {
            "policy": (
                "uniform_unique_windows"
                if uniform_sampling
                else "target_task_oversampling"
            ),
            "manifest": sampling_manifest,
            "salt": args.sampling_salt,
            "sampled_train_windows": len(weighted_rows),
            "all_task_weights": 1.0 if uniform_sampling else None,
            "target_task": None if uniform_sampling else args.target_task,
            "target_unique_windows": None if uniform_sampling else len(target_rows),
            "target_total_repeats": None if uniform_sampling else args.target_total_repeats,
            "target_weighted_windows": None if uniform_sampling else weighted_target_windows,
            "target_fraction": None if uniform_sampling else weighted_target_windows / len(weighted_rows),
            "note": (
                "Every unique training window occurs exactly once; no task-specific weighting."
                if uniform_sampling
                else "Repeated rows are sampling weights, not additional demonstrations."
            ),
        },
        "contract": {
            "action_horizon": int(base_report["action_horizon"]),
            "windows_per_episode": int(base_report["windows_per_episode"]),
            "window_sampling": str(
                base_report.get("window_sampling", "evenly_spaced")
            ),
            "normalization": "owner min/max, derived from manifest_train_unique.jsonl only",
            "validation_in_normalization": False,
            "cached_video_representation": "MiniMax-H3 VAE latents",
            "conditioning": "task text context plus first-frame latent and robot state",
            "motion_target": "RAFT color-wheel video encoded by MiniMax-H3 VAE",
        },
        "audit": {
            "all_window_cache_files_present": True,
            "all_context_cache_files_present": True,
            "all_window_ids_unique": True,
            "train_validation_episode_disjoint": True,
        },
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (output / "training_candidate.json").write_text(payload, encoding="utf-8")
    (output / "candidate_report.json").write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
