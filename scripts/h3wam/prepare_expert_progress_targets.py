#!/usr/bin/env python3
"""Attach FACT-inspired future time-to-go targets to expert WAM windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


FACT_COMMIT = "618a6c16868699b6d4138941de6a863589ac00dd"
FACT_TRANSFORM_SHA256 = "ed76964b005420e752d15d140156962d6c18abd40e58f9140313857d5ebd7110"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--future-offset", type=int, default=32)
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-val", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def atomic_write(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def convert(path: Path, *, expected_split: str, future_offset: int) -> tuple[str, dict]:
    output = []
    episodes = set()
    suites = Counter()
    contexts = set()
    clipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        if source["split"] != expected_split:
            raise ValueError(f"unexpected split in {path}: {source['split']}")
        start, length = int(source["start"]), int(source["length"])
        if length <= 1 or not 0 <= start < length:
            raise ValueError(f"invalid window geometry: {source['id']}")
        future_state_index = min(start + future_offset, length - 1)
        clipped += int(future_state_index == length - 1 and start + future_offset >= length)
        value_raw = float(length - future_state_index - 1) / float(length - 1)
        record = dict(source)
        record.update(
            {
                "future_state_index": future_state_index,
                "progress_future_offset": future_offset,
                "value_raw": value_raw,
                "value_normalized_fact_range_0_2": value_raw - 1.0,
                "label_status": "expert_success_window",
            }
        )
        output.append(json.dumps(record, sort_keys=True) + "\n")
        episodes.add((source["suite"], source["dataset_root"], int(source["episode"])))
        suites[source["suite"]] += 1
        contexts.add(source["context_id"])
    value = "".join(output)
    return value, {
        "windows": len(output), "episodes": len(episodes), "episode_ids": episodes,
        "contexts": len(contexts), "by_suite": dict(sorted(suites.items())),
        "future_index_clipped_windows": clipped,
        "sha256": hashlib.sha256(value.encode()).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    if args.future_offset <= 0:
        raise ValueError("future offset must be positive")
    train_rows, train = convert(args.train_manifest, expected_split="train", future_offset=args.future_offset)
    val_rows, val = convert(args.val_manifest, expected_split="validation", future_offset=args.future_offset)
    overlap = train.pop("episode_ids") & val.pop("episode_ids")
    if overlap:
        raise ValueError(f"train/validation episode overlap: {len(overlap)}")
    report = {
        "format": "h3wam-expert-progress-targets-v1",
        "source": {
            "project": "FACT", "revision": FACT_COMMIT,
            "wa_transforms_lerobot_sha256": FACT_TRANSFORM_SHA256,
            "official_difference": "local future offset is H3 action horizon 32 rather than FACT NUM_FRAMES 48",
        },
        "future_offset": args.future_offset,
        "target_contract": "future=min(start+offset,length-1); raw=(length-future-1)/(length-1)",
        "train": train, "validation": val, "episode_overlap": 0,
        "gates": {
            "progress_value_diagnostic": "GO_CANARY",
            "action_best_of_n": "NO_GO_UNTIL_ACTION_CONDITIONED_HELD_OUT_RANKING",
        },
    }
    atomic_write(args.output_train.resolve(), train_rows)
    atomic_write(args.output_val.resolve(), val_rows)
    atomic_write(args.output_report.resolve(), json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
