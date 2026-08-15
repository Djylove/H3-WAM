#!/usr/bin/env python3
"""Summarize closed-loop LIBERO failures using recorded object-joint evidence.

The audit deliberately separates simulator success from diagnostic proxies.
Object displacement can identify target contact or an obvious distractor, but
it is never counted as task completion.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


RESULT_PATTERN = re.compile(
    r"(?P<prefix>.+)_(?P<suite>goal|object|spatial|10)_task(?P<task>\d+)_"
    r"trial(?P<trial>\d+)_replan(?P<replan>\d+)/results\.json$"
)
CONTACT_DELTA = 0.08

GOAL_TARGETS = {
    0: ("wooden_cabinet_1_middle_level",),
    1: ("akita_black_bowl_1_joint0",),
    2: ("wine_bottle_1_joint0",),
    3: ("wooden_cabinet_1_top_level", "akita_black_bowl_1_joint0"),
    4: ("akita_black_bowl_1_joint0",),
    5: ("plate_1_joint0",),
    6: ("cream_cheese_1_joint0",),
    7: ("flat_stove_1_button",),
    8: ("akita_black_bowl_1_joint0",),
    9: ("wine_bottle_1_joint0",),
}

LIBERO10_TARGETS = {
    0: ("alphabet_soup_1_joint0", "tomato_sauce_1_joint0"),
    1: ("cream_cheese_1_joint0", "butter_1_joint0"),
    2: ("flat_stove_1_button", "moka_pot_1_joint0"),
    3: ("akita_black_bowl_1_joint0", "wooden_cabinet_1_bottom_level"),
    4: ("white_yellow_mug_1_joint0", "white_mug_1_joint0"),
    5: ("book_1_joint0",),
    6: ("white_mug_1_joint0", "chocolate_pudding_1_joint0"),
    7: ("alphabet_soup_1_joint0", "cream_cheese_1_joint0"),
    8: ("moka_pot_1_joint0", "moka_pot_2_joint0"),
    9: ("white_yellow_mug_1_joint0", "microwave_1_joint"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="d0_h32_s14000")
    parser.add_argument("--replan", type=int, default=8)
    parser.add_argument("--trials", type=int, nargs="+")
    return parser.parse_args()


def object_target(task_language: str) -> str:
    match = re.match(r"pick up the (.+?) and place it in the basket$", task_language)
    if match is None:
        raise ValueError(f"unexpected LIBERO-Object language: {task_language!r}")
    target = match.group(1).replace(" box", "").replace(" ", "_")
    return f"{target}_1_joint0"


def target_keys(suite: str, task_id: int, language: str) -> tuple[str, ...]:
    if suite == "object":
        return (object_target(language),)
    if suite == "spatial":
        # The official LIBERO-Spatial BDDL files bind the relationally selected
        # bowl to akita_black_bowl_1 and the distractor to bowl_2.
        return ("akita_black_bowl_1_joint0",)
    if suite == "goal":
        return GOAL_TARGETS[task_id]
    return LIBERO10_TARGETS[task_id]


def classify(record: dict) -> str:
    if record["success"]:
        return "SUCCESS"
    deltas = record["max_object_joint_delta"]
    targets = record["target_joint_keys"]
    target_deltas = [float(deltas.get(key, 0.0)) for key in targets]
    target_max = max(target_deltas, default=0.0)
    ignored = set(targets) | {
        key for key in deltas if "basket" in key or "plate" in key or "ramekin" in key
    }
    distractor_max = max(
        (float(value) for key, value in deltas.items() if key not in ignored),
        default=0.0,
    )
    if distractor_max >= CONTACT_DELTA and distractor_max > max(
        CONTACT_DELTA, target_max * 1.25
    ):
        return "FAIL_WRONG_OBJECT_EVIDENCE"
    reached = sum(value >= CONTACT_DELTA for value in target_deltas)
    if reached == len(targets) and targets:
        return "FAIL_TARGETS_MOVED_COMPLETION"
    if reached:
        return "FAIL_PARTIAL_REQUIRED_PROGRESS"
    return "FAIL_NO_REQUIRED_CONTACT"


def main() -> None:
    args = parse_args()
    root = args.result_root.resolve()
    trial_filter = None if args.trials is None else set(args.trials)
    records = []
    for path in sorted(root.glob("*/results.json")):
        match = RESULT_PATTERN.fullmatch(f"{path.parent.name}/results.json")
        if match is None:
            continue
        fields = match.groupdict()
        if fields["prefix"] != args.prefix or int(fields["replan"]) != args.replan:
            continue
        trial = int(fields["trial"])
        if trial_filter is not None and trial not in trial_filter:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        task_payload = payload["tasks"][0]
        episode = task_payload["episodes"][0]
        record = {
            "suite": fields["suite"],
            "task_id": int(fields["task"]),
            "trial": trial,
            "task": task_payload["task"],
            "success": bool(episode["success"]),
            "steps": int(episode["steps"]),
            "max_object_joint_delta": episode["max_object_joint_delta"],
        }
        record["target_joint_keys"] = list(
            target_keys(record["suite"], record["task_id"], record["task"])
        )
        record["classification"] = classify(record)
        records.append(record)
    if not records:
        raise ValueError("no matching completed rollout results")

    aggregate: dict[str, dict] = {}
    by_suite: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_suite[record["suite"]].append(record)
    for suite, items in sorted(by_suite.items()):
        classifications = Counter(item["classification"] for item in items)
        aggregate[suite] = {
            "episodes": len(items),
            "successes": sum(item["success"] for item in items),
            "success_rate": sum(item["success"] for item in items) / len(items),
            "classifications": dict(sorted(classifications.items())),
        }
    result = {
        "format": "h3wam-libero-failure-audit-v1",
        "result_root": str(root),
        "prefix": args.prefix,
        "replan": args.replan,
        "trials": sorted({record["trial"] for record in records}),
        "episodes": len(records),
        "simulator_successes": sum(record["success"] for record in records),
        "diagnostic_contact_delta": CONTACT_DELTA,
        "warning": (
            "Joint displacement classifications are diagnostic proxies only; "
            "only the simulator predicate is success."
        ),
        "aggregate": aggregate,
        "records": records,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite failure audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("episodes", "simulator_successes", "aggregate")}, indent=2))


if __name__ == "__main__":
    main()
