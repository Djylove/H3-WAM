#!/usr/bin/env python3
"""Merge task-local H3 feature caches and teacher actions into one contract."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--source",
        nargs=3,
        action="append",
        metavar=("CACHE_ROOT", "MANIFEST", "TEACHER_TARGETS"),
        required=True,
    )
    parser.add_argument("--feature-subdir", default="h3_official_features_fixedctx")
    return parser.parse_args()


def normalize(value: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return 2.0 * (value.float() - low.float()) / (high.float() - low.float()).clamp_min(1e-6) - 1.0


def denormalize(value: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return (value.float() + 1.0) * 0.5 * (high.float() - low.float()) + low.float()


def atomic_save(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    windows_out = output_root / "windows"
    features_out = output_root / args.feature_subdir
    windows_out.mkdir(parents=True, exist_ok=True)
    features_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    physical_teacher: dict[str, torch.Tensor] = {}
    actions, states = [], []
    seen_ids: set[str] = set()
    task_groups: dict[str, int] = {}
    teacher_sources = []

    for cache_text, manifest_text, teacher_text in args.source:
        cache_root = Path(cache_text).resolve()
        manifest = Path(manifest_text).resolve()
        teacher_path = Path(teacher_text).resolve()
        source_rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
        if teacher.get("format") != "h3-feature-action-teacher-targets-v1":
            raise ValueError(f"unsupported teacher format: {teacher_path}")
        teacher_stats = teacher.get("normalization")
        if teacher_stats is None:
            raise ValueError(f"teacher normalization is missing: {teacher_path}")
        teacher_sources.append(str(teacher_path))
        for row in source_rows:
            item_id = str(row["id"])
            task = str(row["task"])
            if item_id in seen_ids:
                raise ValueError(f"duplicate item id across sources: {item_id}")
            seen_ids.add(item_id)
            if task not in task_groups:
                task_groups[task] = len(task_groups)
            merged = dict(row)
            merged["task_group"] = task_groups[task]
            rows.append(merged)
            source_window = cache_root / "windows" / f"{item_id}.pt"
            source_feature = cache_root / args.feature_subdir / f"{item_id}.pt"
            if not source_window.is_file() or not source_feature.is_file():
                raise FileNotFoundError(f"missing source artifact for {item_id}")
            for source, target in (
                (source_window, windows_out / source_window.name),
                (source_feature, features_out / source_feature.name),
            ):
                if target.exists() or target.is_symlink():
                    if target.resolve() != source:
                        raise ValueError(f"conflicting merged artifact: {target}")
                else:
                    target.symlink_to(source)
            window = torch.load(source_window, map_location="cpu", weights_only=False)
            actions.append(window["actions"].float())
            states.append(window["state"].float())
            normalized_teacher = teacher["targets"].get(item_id)
            if normalized_teacher is None:
                raise ValueError(f"teacher target missing for {item_id}")
            physical_teacher[item_id] = denormalize(
                normalized_teacher,
                teacher_stats["action_min"],
                teacher_stats["action_max"],
            )

    action = torch.cat(actions)
    state = torch.stack(states)
    stats = {
        "num_windows": len(rows),
        "action_min": action.amin(0),
        "action_max": action.amax(0),
        "action_mean": action.mean(0),
        "action_std": action.std(0).clamp_min(1e-6),
        "state_min": state.amin(0),
        "state_max": state.amax(0),
        "state_mean": state.mean(0),
        "state_std": state.std(0).clamp_min(1e-6),
    }
    normalized_teacher = {
        item_id: normalize(value, stats["action_min"], stats["action_max"]).to(torch.float16)
        for item_id, value in physical_teacher.items()
    }
    task_counts = Counter(str(row["task"]) for row in rows)
    for row in rows:
        row["sample_weight"] = 1.0 / task_counts[str(row["task"])]
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_save(stats, output_root / "stats.pt")
    atomic_save(
        {
            "format": "h3-feature-action-teacher-targets-v1",
            "action_horizon": next(iter(normalized_teacher.values())).shape[0],
            "normalization": stats,
            "training_tasks": sorted(task_groups),
            "teacher_sources": teacher_sources,
            "targets": normalized_teacher,
        },
        output_root / "teacher_targets.pt",
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "tasks": task_groups,
                "windows": len(rows),
                "teacher_targets": len(normalized_teacher),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
