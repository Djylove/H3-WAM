#!/usr/bin/env python3
"""Select high-value teacher/H3 disagreement states for corrective training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--motion-quantile", type=float, default=0.75)
    parser.add_argument("--neighbor-radius", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.motion_quantile < 1.0:
        raise ValueError("motion-quantile must be between zero and one")
    if args.neighbor_radius < 0:
        raise ValueError("neighbor-radius cannot be negative")
    with np.load(args.input.resolve()) as archive:
        source = {key: archive[key] for key in archive.files}
    for key in ("policy_actions", "teacher_actions", "step"):
        if key not in source:
            raise ValueError(f"input is missing {key}")
    h3 = source["policy_actions"][:, 0]
    teacher = source["teacher_actions"][:, 0]
    if h3.shape[0] != teacher.shape[0] or h3.shape[-1] != 7:
        raise ValueError("policy/teacher first actions must both be [N, 7]")
    motion_l2 = np.linalg.norm(h3[:, :6] - teacher[:, :6], axis=1)
    gripper_mismatch = np.sign(h3[:, 6]) != np.sign(teacher[:, 6])
    threshold = float(np.quantile(motion_l2, args.motion_quantile))
    core = np.flatnonzero((motion_l2 >= threshold) | gripper_mismatch)
    selected_set: set[int] = set()
    for index in core.tolist():
        selected_set.update(
            range(
                max(0, index - args.neighbor_radius),
                min(len(motion_l2), index + args.neighbor_radius + 1),
            )
        )
    selected_indices = np.asarray(sorted(selected_set), dtype=np.int64)
    selected = {key: value[selected_indices] for key, value in source.items()}
    selected["corrective_index"] = selected_indices
    selected["corrective_motion_l2"] = motion_l2[selected_indices].astype(np.float32)
    selected["corrective_gripper_mismatch"] = gripper_mismatch[selected_indices]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **selected)
    temporary.replace(args.output)

    contiguous_runs = []
    if selected_indices.size:
        start = previous = int(selected_indices[0])
        for value in selected_indices[1:].tolist():
            if value != previous + 1:
                contiguous_runs.append([start, previous])
                start = value
            previous = value
        contiguous_runs.append([start, previous])
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "total_states": int(len(motion_l2)),
        "selected_states": int(len(selected_indices)),
        "motion_quantile": args.motion_quantile,
        "motion_threshold": threshold,
        "motion_l2_mean": float(motion_l2.mean()),
        "selected_motion_l2_mean": float(motion_l2[selected_indices].mean()),
        "gripper_mismatch_states": int(gripper_mismatch.sum()),
        "selected_gripper_mismatch_states": int(
            gripper_mismatch[selected_indices].sum()
        ),
        "neighbor_radius": args.neighbor_radius,
        "selected_runs": contiguous_runs,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
