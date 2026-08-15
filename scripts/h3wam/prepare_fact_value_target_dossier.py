#!/usr/bin/env python3
"""Freeze the value-only targets that incumbent rollouts can actually support.

Failed episodes are right-censored because these rollouts do not record FACT's
``failure_active_from_frame``.  They are counted but never assigned a made-up
penalty target.  Successful causal transitions use FACT's exact future-state
time-to-go convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np


RESULT_PATTERN = re.compile(
    r"(?P<prefix>.+)_(?P<suite>goal|object|spatial|10)_task(?P<task>\d+)_"
    r"trial(?P<trial>\d+)_replan(?P<replan>\d+)/results\.json$"
)
FACT_COMMIT = "618a6c16868699b6d4138941de6a863589ac00dd"
FACT_TRANSFORM_SHA256 = "ed76964b005420e752d15d140156962d6c18abd40e58f9140313857d5ebd7110"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--prefix", default="d0_h32_s14000")
    parser.add_argument("--replan", type=int, default=8)
    parser.add_argument("--validation-trial", type=int, default=3)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    records: list[dict] = []
    censored = []
    episode_counts = Counter()
    state_counts = Counter()
    for result_path in sorted(args.result_root.resolve().glob("*/results.json")):
        match = RESULT_PATTERN.fullmatch(f"{result_path.parent.name}/results.json")
        if match is None:
            continue
        fields = match.groupdict()
        if fields["prefix"] != args.prefix or int(fields["replan"]) != args.replan:
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        task = result["tasks"][0]
        episode = task["episodes"][0]
        trial = int(fields["trial"])
        split = "val" if trial == args.validation_trial else "train"
        trajectory_path = Path(episode["trajectory"]).resolve()
        with np.load(trajectory_path, allow_pickle=False) as trajectory:
            replans = int(trajectory["step"].shape[0])
            if trajectory["policy_actions"].shape != (replans, 32, 7):
                raise ValueError(f"bad policy action shape in {trajectory_path}")
        key = (split, fields["suite"], bool(episode["success"]))
        episode_counts[key] += 1
        state_counts[key] += replans
        if not episode["success"]:
            censored.append(
                {"suite": fields["suite"], "task_id": int(fields["task"]),
                 "trial": trial, "split": split, "replans": replans,
                 "reason": "failure_onset_unknown"}
            )
            continue
        if replans < 2:
            continue
        for state_index in range(replans - 1):
            future_state_index = state_index + 1
            value_raw = float(replans - future_state_index - 1) / float(replans - 1)
            records.append(
                {
                    "id": f"{fields['suite']}_task{fields['task']}_trial{trial}_state{state_index}",
                    "suite": fields["suite"], "task_id": int(fields["task"]),
                    "trial": trial, "split": split, "language": str(task["task"]),
                    "result": str(result_path.resolve()), "trajectory": str(trajectory_path),
                    "state_index": state_index, "future_state_index": future_state_index,
                    "action_source": "policy_actions[state_index]",
                    "outcome_source": "observation[future_state_index]",
                    "value_raw": value_raw,
                    "value_normalized_fact_range_0_2": value_raw - 1.0,
                    "label_status": "uncensored_success_transition",
                }
            )
    if not records:
        raise ValueError("no successful causal transitions")
    rows = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    train = [record for record in records if record["split"] == "train"]
    val = [record for record in records if record["split"] == "val"]
    report = {
        "format": "h3wam-fact-value-target-dossier-v1",
        "source": {
            "project": "FACT", "revision": FACT_COMMIT,
            "wa_transforms_lerobot_sha256": FACT_TRANSFORM_SHA256,
            "target_contract": "future_state_index=i+1; raw=(N-future_state_index-1)/(N-1)",
            "normalization_contract": "FACT default value min=0 max=2; normalized=raw-1 for successes",
        },
        "manifest_sha256": sha256_bytes(rows.encode()),
        "transitions": len(records), "train_transitions": len(train), "val_transitions": len(val),
        "successful_train_episodes": len({(r["suite"], r["task_id"], r["trial"]) for r in train}),
        "successful_val_episodes": len({(r["suite"], r["task_id"], r["trial"]) for r in val}),
        "validation_suites": dict(sorted(Counter(r["suite"] for r in val).items())),
        "censored_failure_episodes": len(censored),
        "episode_counts": {"|".join(map(str, key)): value for key, value in sorted(episode_counts.items())},
        "state_counts": {"|".join(map(str, key)): value for key, value in sorted(state_counts.items())},
        "gates": {
            "progress_value_diagnostic": "GO_CANARY",
            "failed_episode_penalty_training": "NO_GO_FAILURE_ONSET_UNKNOWN",
            "best_of_n_action_ranking": "NO_GO_NO_COUNTERFACTUAL_OUTCOMES",
        },
        "boundary": (
            "This manifest can test whether H3 features encode monotonic task progress on held-out trials. "
            "It cannot calibrate alternative-action ranking or FACT failure penalties."
        ),
    }
    atomic_write(args.output_manifest.resolve(), rows)
    atomic_write(args.output_report.resolve(), json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
