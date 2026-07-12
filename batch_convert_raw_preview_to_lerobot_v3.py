#!/usr/bin/env python3
"""Convert newly collected qnexo episodes to LeRobot v3 preview folders.

This wrapper is for the first pass where videos need to be inspected before
writing final expand_task annotations. It groups by source task_id by default,
but deliberately avoids the AnyGrasp task_id-to-expand_task mapping in
batch_convert_session_to_task_folders.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from batch_convert_session_to_task_folders import main as batch_main


DEFAULT_EXPAND_TASK = (
    "Pick the tomato from the table and place it inside the basket on the left."
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert qnexo raw episodes to LeRobot v3. expand_task is filled "
            "with the configured fixed text by default."
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
        default=Path("data/raw_preview_v3"),
        type=Path,
        help="Output root that will contain task_00xxxx folders",
    )
    parser.add_argument("--limit", default=500, type=int, help="Number of episodes to convert")
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
        dest="allow_unfinished",
        action="store_true",
        default=True,
        help="Include episodes without a DONE marker. This is the preview default.",
    )
    parser.add_argument(
        "--require-done",
        dest="allow_unfinished",
        action="store_false",
        help="Only include episodes with a DONE marker",
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
        help="Number of episodes per batch folder for each task_id. 0 keeps one batch per task_id.",
    )
    parser.add_argument("--task-width", default=6, type=int)
    parser.add_argument(
        "--group-mode",
        choices=["task_id", "single"],
        default="task_id",
        help="Group by source task_id, or put all selected episodes into one preview task",
    )
    parser.add_argument(
        "--expand-task-mode",
        choices=["empty", "task"],
        default="empty",
        help=(
            "Fallback when --expand-task-text is not set: empty leaves expand_task "
            "blank; task uses the task text as a placeholder"
        ),
    )
    parser.add_argument(
        "--expand-task-text",
        default=DEFAULT_EXPAND_TASK,
        help="Fixed expand_task written to every selected episode",
    )
    parser.add_argument(
        "--single-task-id",
        default="0",
        help="Output task_id used when --group-mode single",
    )
    parser.add_argument("--video-codec", choices=["av1", "h264"], default="av1")
    parser.add_argument(
        "--ffmpeg-exe",
        default=None,
        type=Path,
        help="Explicit ffmpeg executable passed to the converter/imageio",
    )
    parser.add_argument(
        "--task-text",
        default=None,
        help="Override task text for all generated task folders",
    )
    parser.add_argument(
        "--task-text-mode",
        choices=["source", "anygrasp"],
        default="source",
        help="source uses metadata notes/Task id. anygrasp uses the old generic task text.",
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


def build_batch_args(args: argparse.Namespace) -> list[str]:
    batch_args = [
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--limit",
        str(args.limit),
        "--start",
        str(args.start),
        "--selection",
        args.selection,
        "--seed",
        str(args.seed),
        "--python",
        args.python,
        "--fps",
        str(args.fps),
        "--batch-name",
        args.batch_name,
        "--episodes-per-batch",
        str(args.episodes_per_batch),
        "--task-width",
        str(args.task_width),
        "--group-mode",
        args.group_mode,
        "--single-task-id",
        str(args.single_task_id),
        "--video-codec",
        args.video_codec,
        "--task-text-mode",
        args.task_text_mode,
        "--expand-task-mode",
        args.expand_task_mode,
        "--base-state-mode",
        args.base_state_mode,
        "--manifest-name",
        args.manifest_name,
        "--disable-task-ordinal-skips",
    ]

    for task_id in args.exclude_task_id:
        batch_args.extend(["--exclude-task-id", str(task_id)])
    if args.allow_fewer:
        batch_args.append("--allow-fewer")
    if args.allow_unfinished:
        batch_args.append("--allow-unfinished")
    if args.ffmpeg_exe is not None:
        batch_args.extend(["--ffmpeg-exe", str(args.ffmpeg_exe)])
    if args.task_text is not None:
        batch_args.extend(["--task-text", args.task_text])
    if args.expand_task_text is not None:
        batch_args.extend(["--expand-task-text", args.expand_task_text])
    if args.no_video:
        batch_args.append("--no-video")
    if args.exact_stats:
        batch_args.append("--exact-stats")
    if args.no_depth_index:
        batch_args.append("--no-depth-index")
    if args.overwrite:
        batch_args.append("--overwrite")
    if args.dry_run:
        batch_args.append("--dry-run")

    return batch_args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return batch_main(build_batch_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
