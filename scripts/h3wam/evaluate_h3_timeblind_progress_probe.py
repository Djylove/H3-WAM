#!/usr/bin/env python3
"""Fit the C18 H3 progress probe with the absolute-time shortcut removed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from fastwam.models.h3wam import (
    PROGRESS_FEATURE_CONTRACT,
    TIMEBLIND_PROGRESS_DESIGN_CONTRACT,
    TIMEBLIND_PROGRESS_PROBE_FORMAT,
)
from scripts.h3wam.evaluate_h3_progress_probe import (
    load,
    metrics,
    ridge_fit_predict,
    ridge_predict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--val", type=Path, nargs="+", required=True)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-probe", type=Path, required=True)
    return parser.parse_args()


def design(data: dict, contexts: list[str], include_h3: bool) -> torch.Tensor:
    context = torch.zeros((len(data["ids"]), len(contexts)), dtype=torch.double)
    index = {value: row for row, value in enumerate(contexts)}
    for row, value in enumerate(data["context_id"]):
        if value in index:
            context[row, index[value]] = 1.0
    return torch.cat((context, data["features"]), dim=1) if include_h3 else context


def main() -> None:
    args = parse_args()
    train, val = load(args.train), load(args.val)
    contexts = sorted(set(train["context_id"]))
    baseline = ridge_predict(
        design(train, contexts, False), train["target"],
        design(val, contexts, False), args.ridge,
    )
    h3, state = ridge_fit_predict(
        design(train, contexts, True), train["target"],
        design(val, contexts, True), args.ridge,
    )
    baseline_metrics = metrics(val["target"], baseline, val["suite"])
    h3_metrics = metrics(val["target"], h3, val["suite"])
    ratio = h3_metrics["all"]["mae"] / baseline_metrics["all"]["mae"]
    suites = sorted(set(val["suite"]))
    guard = all(
        h3_metrics[suite]["mae"] <= 1.05 * baseline_metrics[suite]["mae"]
        for suite in suites
    )
    passed = ratio <= 0.95 and guard
    report = {
        "format": "h3wam-frozen-h3-timeblind-progress-probe-v1",
        "experiment_class": "controlled_ablation",
        "parent": "C17 task+absolute-step+H3 ridge",
        "only_variable": "remove absolute_step/400 from baseline and H3 probe",
        "train_samples": len(train["ids"]),
        "validation_samples": len(val["ids"]),
        "baseline_task_only": baseline_metrics,
        "h3_plus_task_timeblind": h3_metrics,
        "mae_ratio_h3_over_baseline": ratio,
        "promotion": "overall MAE ratio <=0.95 and no suite regresses >5%",
        "status": (
            "PASS_TIMEBLIND_PROGRESS_FEATURE_GATE"
            if passed else "FAIL_TIMEBLIND_PROGRESS_FEATURE_GATE"
        ),
        "boundary": "Frozen-feature diagnostic only; no action reranking authorization.",
    }
    output = args.output.resolve()
    probe = args.save_probe.resolve()
    if output.exists() or probe.exists():
        raise FileExistsError("refusing to overwrite C18 output")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    payload = {
        "format": TIMEBLIND_PROGRESS_PROBE_FORMAT,
        "ridge": args.ridge,
        "contexts": contexts,
        "design_contract": TIMEBLIND_PROGRESS_DESIGN_CONTRACT,
        "feature_contract": PROGRESS_FEATURE_CONTRACT,
        "mean": state["mean"], "std": state["std"], "weights": state["weights"],
        "train_selection_sha256": train["selection_sha256"],
        "validation_selection_sha256": val["selection_sha256"],
        "validation_status": report["status"],
    }
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe_temporary = probe.with_name(f".{probe.name}.{os.getpid()}.partial")
    torch.save(payload, probe_temporary)
    os.replace(probe_temporary, probe)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
