#!/usr/bin/env python3
"""Batch-select qnexo episodes and convert them into LeRobot v3 task folders."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_EPISODE_FILES = (
    "metadata.json",
    "action.parquet",
    "action.base.parquet",
    "observation.state.parquet",
    "observation.base_state.parquet",
    "observation.images.camera_top.parquet",
)

TASK_TEXT = (
    "Alternate taking and putting with one hand. "
    "(Testing tasks for the new paradigm of data acquisition)"
)
SUBTASK_TEXT = "Pick the fruit from the table and place it inside the basket."
ATOMIC_SKILL = "Pick"

PICK_LEMON_RIGHT = "Pick the lemon from the table and place it inside the basket on the right."
PICK_TOMATO_LEFT = "Pick the tomato from the table and place it inside the basket on the left."
PICK_POTATO_LEFT = "Pick the potato from the table and place it inside the basket on the left."
PICK_BANANA_RIGHT = "Pick the banana from the table and place it inside the basket on the right."


def combine_expand_tasks(first: str, second: str) -> str:
    return f"{first} And {second[:1].lower()}{second[1:]}"


EXPAND_TASKS_BY_TASK_ID = {
    "1582": combine_expand_tasks(PICK_LEMON_RIGHT, PICK_TOMATO_LEFT),
    "1583": combine_expand_tasks(PICK_LEMON_RIGHT, PICK_POTATO_LEFT),
    "1584": combine_expand_tasks(PICK_BANANA_RIGHT, PICK_POTATO_LEFT),
    "1585": combine_expand_tasks(PICK_BANANA_RIGHT, PICK_TOMATO_LEFT),
    "1586": combine_expand_tasks(PICK_BANANA_RIGHT, PICK_LEMON_RIGHT),
    "1587": combine_expand_tasks(PICK_POTATO_LEFT, PICK_TOMATO_LEFT),
    "1588": combine_expand_tasks(PICK_LEMON_RIGHT, PICK_POTATO_LEFT),
    "1589": combine_expand_tasks(PICK_LEMON_RIGHT, PICK_TOMATO_LEFT),
    "1590": combine_expand_tasks(PICK_BANANA_RIGHT, PICK_POTATO_LEFT),
    "1591": combine_expand_tasks(PICK_BANANA_RIGHT, PICK_TOMATO_LEFT),
}
SKIPPED_TASK_ORDINALS = {
    "1582": {7},
}
TASK_1582_SPECIAL_EXPAND_TASKS = {
    6: PICK_TOMATO_LEFT,
    8: PICK_LEMON_RIGHT,
}


def metadata_task_text(metadata: dict[str, Any]) -> str:
    notes = str(metadata.get("notes") or "").strip()
    if notes:
        return notes
    task_id = metadata.get("task_id", "")
    if task_id != "":
        return f"Task {task_id}"
    return TASK_TEXT


def task_text_for_episode(episode_dir: Path, args: argparse.Namespace) -> str:
    if args.task_text:
        return args.task_text
    if args.task_text_mode == "source":
        return metadata_task_text(read_json(episode_dir / "metadata.json"))
    return TASK_TEXT


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def missing_required_files(path: Path) -> list[str]:
    return [name for name in REQUIRED_EPISODE_FILES if not (path / name).exists()]


def is_episode_dir(path: Path, require_done: bool) -> bool:
    if path.name.startswith("._"):
        return False
    if not path.is_dir() or missing_required_files(path):
        return False
    if require_done and not (path / "DONE").exists():
        return False
    try:
        read_json(path / "metadata.json")
    except (OSError, json.JSONDecodeError):
        return False
    return True


def episode_index(path: Path) -> int:
    try:
        return int(path.name.rsplit("_", 1)[-1])
    except ValueError:
        return sys.maxsize


def episode_sort_key(root: Path, path: Path) -> tuple[str, int, str]:
    try:
        parent = str(path.parent.relative_to(root))
    except ValueError:
        parent = str(path.parent)
    return parent, episode_index(path), path.name


def discover_episode_dirs(root: Path, require_done: bool) -> list[Path]:
    if is_episode_dir(root, require_done):
        return [root]

    episode_dirs = [
        path
        for path in root.rglob("episode_*")
        if is_episode_dir(path, require_done)
    ]
    return sorted(episode_dirs, key=lambda path: episode_sort_key(root, path))


def select_episodes(
    episode_dirs: list[Path],
    root: Path,
    limit: int,
    start: int,
    selection: str,
    seed: int,
) -> list[Path]:
    if start < 0:
        raise ValueError("--start must be >= 0")
    if limit <= 0:
        raise ValueError("--limit must be > 0")

    candidates = episode_dirs[start:]
    if selection == "first":
        return candidates[:limit]

    rng = random.Random(seed)
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    return sorted(shuffled[:limit], key=lambda path: episode_sort_key(root, path))


def filter_episodes_by_task_id(
    episode_dirs: list[Path],
    exclude_task_ids: set[str],
) -> list[Path]:
    if not exclude_task_ids:
        return episode_dirs
    return [
        episode_dir
        for episode_dir in episode_dirs
        if task_id_from_episode(episode_dir) not in exclude_task_ids
    ]


def task_ordinals_by_episode(episode_dirs: list[Path]) -> dict[Path, int]:
    counts: dict[str, int] = defaultdict(int)
    ordinals: dict[Path, int] = {}
    for episode_dir in episode_dirs:
        task_id = task_id_from_episode(episode_dir)
        counts[task_id] += 1
        ordinals[episode_dir] = counts[task_id]
    return ordinals


def filter_skipped_task_ordinals(
    episode_dirs: list[Path],
    ordinals: dict[Path, int],
) -> tuple[list[Path], list[Path]]:
    kept: list[Path] = []
    skipped: list[Path] = []
    for episode_dir in episode_dirs:
        task_id = task_id_from_episode(episode_dir)
        ordinal = ordinals[episode_dir]
        if ordinal in SKIPPED_TASK_ORDINALS.get(task_id, set()):
            skipped.append(episode_dir)
        else:
            kept.append(episode_dir)
    return kept, skipped


def link_selected_episodes(staging_dir: Path, episode_dirs: list[Path]) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)
    for new_index, episode_dir in enumerate(episode_dirs):
        link_path = staging_dir / f"episode_{new_index:09d}"
        link_path.symlink_to(episode_dir, target_is_directory=True)


def expand_task_for_episode(
    episode_dir: Path,
    ordinals: dict[Path, int],
    mode: str,
    task_text: str,
    expand_task_text: str | None = None,
) -> str:
    if expand_task_text is not None:
        return expand_task_text
    if mode == "empty":
        return ""
    if mode == "task":
        return task_text

    task_id = task_id_from_episode(episode_dir)
    ordinal = ordinals.get(episode_dir, 0)
    if task_id == "1582" and ordinal in TASK_1582_SPECIAL_EXPAND_TASKS:
        return TASK_1582_SPECIAL_EXPAND_TASKS[ordinal]
    return EXPAND_TASKS_BY_TASK_ID.get(task_id, task_text)


def task_id_from_episode(episode_dir: Path) -> str:
    metadata = read_json(episode_dir / "metadata.json")
    return str(metadata.get("task_id", "0"))


def task_folder_name(task_id: str, width: int) -> str:
    try:
        return f"task_{int(task_id):0{width}d}"
    except ValueError:
        return f"task_{task_id}"


def group_by_task_id(episode_dirs: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for episode_dir in episode_dirs:
        groups[task_id_from_episode(episode_dir)].append(episode_dir)
    return dict(groups)


def group_selected_episodes(args: argparse.Namespace, selected: list[Path]) -> dict[str, list[Path]]:
    if args.group_mode == "single":
        return {str(args.single_task_id): selected}
    return group_by_task_id(selected)


def build_manifest_episodes(
    selected: list[Path],
    task_output_by_id: dict[str, Path],
    ordinals: dict[Path, int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    episodes = []
    output_counts: dict[str, int] = defaultdict(int)
    for selection_index, episode_dir in enumerate(selected):
        metadata = read_json(episode_dir / "metadata.json")
        source_task_id = str(metadata.get("task_id", "0"))
        output_task_id = str(args.single_task_id) if args.group_mode == "single" else source_task_id
        task_text = task_text_for_episode(episode_dir, args)
        task_episode_index = output_counts[output_task_id]
        output_counts[output_task_id] += 1
        episodes.append(
            {
                "selection_index": selection_index,
                "task_episode_index": task_episode_index,
                "source_task_ordinal": ordinals.get(episode_dir),
                "task_output": str(task_output_by_id[output_task_id]),
                "source_path": str(episode_dir),
                "source_session_id": metadata.get("session_id", episode_dir.parent.name),
                "source_episode_index": metadata.get("episode_index", episode_index(episode_dir)),
                "source_task_id": source_task_id,
                "output_task_id": output_task_id,
                "task_text": task_text,
                "notes": metadata.get("notes", ""),
                "expand_task": expand_task_for_episode(
                    episode_dir,
                    ordinals,
                    args.expand_task_mode,
                    task_text,
                    args.expand_task_text,
                ),
            }
        )
    return episodes


def build_manifest(
    source_root: Path,
    selected: list[Path],
    args: argparse.Namespace,
    task_output_by_id: dict[str, Path],
    ordinals: dict[Path, int],
    skipped: list[Path],
) -> dict[str, Any]:
    task_counts = {
        task_id: len(group)
        for task_id, group in sorted(group_selected_episodes(args, selected).items())
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "output_root": str(args.output.expanduser().resolve()),
        "limit": args.limit,
        "start": args.start,
        "selection": args.selection,
        "seed": args.seed if args.selection == "random" else None,
        "exclude_task_ids": sorted(args.exclude_task_id),
        "group_mode": args.group_mode,
        "single_task_id": args.single_task_id if args.group_mode == "single" else None,
        "skipped_task_ordinals": {
            task_id: sorted(skipped_ordinals)
            for task_id, skipped_ordinals in sorted(SKIPPED_TASK_ORDINALS.items())
        }
        if not args.disable_task_ordinal_skips
        else {},
        "skipped_episodes": [
            {
                "source_path": str(episode_dir),
                "task_id": task_id_from_episode(episode_dir),
                "source_task_ordinal": ordinals.get(episode_dir),
            }
            for episode_dir in skipped
        ],
        "task_text_mode": args.task_text_mode,
        "task_text_override": args.task_text,
        "subtask_text": SUBTASK_TEXT,
        "atomic_skill": ATOMIC_SKILL,
        "expand_task_mode": args.expand_task_mode,
        "expand_task_text_override": args.expand_task_text,
        "episodes_per_batch": args.episodes_per_batch,
        "require_done": not args.allow_unfinished,
        "selected_count": len(selected),
        "task_counts": task_counts,
        "task_outputs": {
            task_id: str(output_path)
            for task_id, output_path in sorted(task_output_by_id.items())
        },
        "episodes": build_manifest_episodes(
            selected,
            task_output_by_id,
            ordinals,
            args,
        ),
    }


def write_manifest(output_root: Path, manifest_name: str, manifest: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / manifest_name
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select episodes from a qnexo root/session and convert them into "
            "task_00xxxx LeRobot v3 folders via convert_episode_to_task_format.py."
        )
    )
    parser.add_argument(
        "--input",
        default=Path("qnexo"),
        type=Path,
        help="qnexo root, session folder, or a single episode_* folder",
    )
    parser.add_argument(
        "--output",
        default=Path("data/anygrasp_v2"),
        type=Path,
        help="Output root that will contain task_00xxxx folders",
    )
    parser.add_argument("--limit", default=100, type=int, help="Number of episodes to convert")
    parser.add_argument("--start", default=0, type=int, help="Skip this many sorted episodes first")
    parser.add_argument(
        "--selection",
        choices=["first", "random"],
        default="first",
        help="Episode selection strategy",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed for --selection random")
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help="Convert fewer than --limit episodes if the input does not contain enough",
    )
    parser.add_argument(
        "--allow-unfinished",
        action="store_true",
        help="Include episodes without a DONE marker",
    )
    parser.add_argument(
        "--exclude-task-id",
        action="append",
        default=[],
        help="Exclude source metadata task_id before selecting episodes. Can be repeated.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for converter")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--batch-name", default="batch_000000")
    parser.add_argument(
        "--episodes-per-batch",
        default=0,
        type=int,
        help=(
            "Split each output task into numbered batch folders with this many "
            "episodes each. 0 keeps one batch per task."
        ),
    )
    parser.add_argument("--task-width", default=6, type=int)
    parser.add_argument(
        "--group-mode",
        choices=["task_id", "single"],
        default="task_id",
        help="Group outputs by source task_id or put all selected episodes in one preview task",
    )
    parser.add_argument(
        "--single-task-id",
        default="0",
        help="Output task_id used when --group-mode single",
    )
    parser.add_argument("--video-codec", choices=["av1", "h264"], default="av1")
    parser.add_argument(
        "--task-text-mode",
        choices=["anygrasp", "source"],
        default="anygrasp",
        help="Use the built-in AnyGrasp task text or source metadata notes/Task id",
    )
    parser.add_argument(
        "--task-text",
        default=None,
        help="Override task text for all generated task folders",
    )
    parser.add_argument(
        "--expand-task-mode",
        choices=["anygrasp", "task", "empty"],
        default="anygrasp",
        help=(
            "Use task-specific AnyGrasp expand tasks, use the task text as placeholder, "
            "or leave expand_task empty"
        ),
    )
    parser.add_argument(
        "--expand-task-text",
        default=None,
        help="Override expand_task for every selected episode. Takes precedence over mode.",
    )
    parser.add_argument(
        "--disable-task-ordinal-skips",
        action="store_true",
        help="Disable built-in per-task ordinal skip rules such as task 1582 ordinal 7",
    )
    parser.add_argument(
        "--ffmpeg-exe",
        default=None,
        type=Path,
        help="Explicit ffmpeg executable passed to the converter/imageio",
    )
    parser.add_argument("--base-state-mode", choices=["zero", "source"], default="zero")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--exact-stats",
        action="store_true",
        help="Compute full image/depth stats instead of fast placeholder stats",
    )
    parser.add_argument(
        "--no-depth-index",
        action="store_true",
        help="Pass through to converter for older v3-compatible output",
    )
    parser.add_argument(
        "--manifest-name",
        default="selected_episodes.json",
        help="Selection manifest written under the output root",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output folder")
    parser.add_argument("--dry-run", action="store_true", help="Print selected episodes only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    source_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    converter = Path(__file__).with_name("convert_episode_to_task_format.py").resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Input path does not exist: {source_root}")
    if not converter.exists():
        raise FileNotFoundError(f"Converter not found: {converter}")

    episode_dirs = discover_episode_dirs(source_root, require_done=not args.allow_unfinished)
    if not episode_dirs:
        raise FileNotFoundError(f"No usable episode_* directories found under {source_root}")
    discovered_count = len(episode_dirs)
    exclude_task_ids = {str(task_id) for task_id in args.exclude_task_id}
    episode_dirs = filter_episodes_by_task_id(episode_dirs, exclude_task_ids)
    if not episode_dirs:
        raise FileNotFoundError(
            f"No usable episode_* directories remain after excluding task_id(s): "
            f"{', '.join(sorted(exclude_task_ids))}"
        )
    after_exclusions_count = len(episode_dirs)
    ordinals = task_ordinals_by_episode(episode_dirs)
    if args.disable_task_ordinal_skips:
        skipped = []
    else:
        episode_dirs, skipped = filter_skipped_task_ordinals(episode_dirs, ordinals)
    if not episode_dirs:
        raise FileNotFoundError("No usable episode_* directories remain after task ordinal skips")

    selected = select_episodes(
        episode_dirs=episode_dirs,
        root=source_root,
        limit=args.limit,
        start=args.start,
        selection=args.selection,
        seed=args.seed,
    )
    if len(selected) < args.limit and not args.allow_fewer:
        raise ValueError(
            f"Only found {len(selected)} eligible episode(s), but --limit is {args.limit}. "
            "Use --allow-fewer to convert what is available."
        )

    print(
        f"Discovered {discovered_count} eligible episode(s); "
        f"{after_exclusions_count} remain after task_id exclusions; "
        f"{len(episode_dirs)} remain after task ordinal skips; "
        f"selected {len(selected)} for conversion."
    )
    for episode_dir in skipped:
        print(
            f"  skipped: task_id={task_id_from_episode(episode_dir)} "
            f"ordinal={ordinals[episode_dir]} path={episode_dir}"
        )
    for output_index, episode_dir in enumerate(selected[:10]):
        print(f"  {output_index:03d}: {episode_dir}")
    if len(selected) > 10:
        print(f"  ... {len(selected) - 10} more")

    groups = group_selected_episodes(args, selected)
    task_output_by_id = {
        task_id: output_root / task_folder_name(task_id, args.task_width)
        for task_id in sorted(groups)
    }
    manifest = build_manifest(source_root, selected, args, task_output_by_id, ordinals, skipped)
    if args.dry_run:
        print("Task folders:")
        for task_id, group in sorted(groups.items()):
            print(f"  {task_output_by_id[task_id]}: {len(group)} episode(s)")
            if args.episodes_per_batch > 0:
                batch_count = (len(group) + args.episodes_per_batch - 1) // args.episodes_per_batch
                for batch_index in range(batch_count):
                    start = batch_index * args.episodes_per_batch
                    end = min(start + args.episodes_per_batch, len(group))
                    print(f"    batch_{batch_index:06d}: {end - start} episode(s)")
        print("Dry run only; no files were written.")
        return 0

    if output_root.exists() and not args.overwrite:
        existing_outputs = [path for path in task_output_by_id.values() if path.exists()]
        existing = "\n  ".join(str(path) for path in existing_outputs or [output_root])
        raise FileExistsError(
            "Task output path(s) already exist. Use --overwrite to replace them:\n"
            f"  {existing}"
        )
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="qnexo_selected_episodes_") as tmp:
        tmp_root = Path(tmp)
        for task_id, group in sorted(groups.items()):
            task_output = task_output_by_id[task_id]
            group_task_text = task_text_for_episode(group[0], args)
            staging_dir = tmp_root / task_folder_name(task_id, args.task_width)
            link_selected_episodes(staging_dir, group)
            expand_task_file = tmp_root / f"{task_folder_name(task_id, args.task_width)}_expand.json"
            write_json(
                expand_task_file,
                {
                    "expand_tasks": [
                        expand_task_for_episode(
                            episode,
                            ordinals,
                            args.expand_task_mode,
                            task_text_for_episode(episode, args),
                            args.expand_task_text,
                        )
                        for episode in group
                    ]
                },
            )

            cmd = [
                args.python,
                str(converter),
                "--input",
                str(staging_dir),
                "--output",
                str(task_output),
                "--batch-name",
                args.batch_name,
                "--episodes-per-batch",
                str(args.episodes_per_batch),
                "--fps",
                str(args.fps),
                "--video-codec",
                args.video_codec,
                "--base-state-mode",
                args.base_state_mode,
                "--task-text",
                group_task_text,
                "--subtask-text",
                SUBTASK_TEXT,
                "--atomic-skill",
                ATOMIC_SKILL,
                "--expand-task-file",
                str(expand_task_file),
                "--overwrite",
            ]
            if args.group_mode == "single":
                cmd.extend(["--task-id-override", str(args.single_task_id)])
            if args.ffmpeg_exe is not None:
                cmd.extend(["--ffmpeg-exe", str(args.ffmpeg_exe.expanduser().resolve())])
            if args.no_video:
                cmd.append("--no-video")
            if not args.exact_stats:
                cmd.append("--fast-stats")
            if args.no_depth_index:
                cmd.append("--no-depth-index")

            print(
                f"Running converter for {task_output.name}: "
                f"{len(group)} episode(s)"
            )
            print("  " + " ".join(cmd))
            subprocess.run(cmd, check=True)

    write_manifest(output_root, args.manifest_name, manifest)
    print(f"Done. Wrote LeRobot v3 task folders under {output_root}")
    print(f"Selection manifest: {output_root / args.manifest_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
