#!/usr/bin/env python3
"""Fit fixed ridge probes and gate frozen-H3 task-progress information."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--val", type=Path, nargs="+", required=True)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(paths: list[Path]) -> dict:
    shards = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    hashes = {shard["selection_sha256"] for shard in shards}
    if len(hashes) != 1:
        raise ValueError("shards do not share one deterministic selection")
    ids = sum((shard["ids"] for shard in shards), [])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate samples across shards")
    return {
        "ids": ids,
        "suite": sum((shard["suite"] for shard in shards), []),
        "context_id": sum((shard["context_id"] for shard in shards), []),
        "start": torch.cat([shard["start"] for shard in shards]).double(),
        "target": torch.cat([shard["target"] for shard in shards]).double(),
        "features": torch.cat([shard["features"] for shard in shards]).double(),
        "selection_sha256": next(iter(hashes)),
    }


def ridge_predict(train_x, train_y, val_x, ridge):
    mean, std = train_x.mean(0), train_x.std(0, unbiased=False).clamp_min(1e-6)
    train_x = (train_x - mean) / std; val_x = (val_x - mean) / std
    train_x = torch.cat((torch.ones((len(train_x), 1), dtype=train_x.dtype), train_x), dim=1)
    val_x = torch.cat((torch.ones((len(val_x), 1), dtype=val_x.dtype), val_x), dim=1)
    penalty = torch.eye(train_x.shape[1], dtype=train_x.dtype) * ridge; penalty[0, 0] = 0
    weights = torch.linalg.solve(train_x.T @ train_x + penalty, train_x.T @ train_y)
    return (val_x @ weights).clamp(0, 1)


def metrics(target, prediction, suites):
    def one(mask):
        error = prediction[mask] - target[mask]
        denominator = ((target[mask] - target[mask].mean()) ** 2).sum().clamp_min(1e-12)
        return {"samples": int(mask.sum()), "mae": float(error.abs().mean()),
                "rmse": float((error.square().mean()).sqrt()),
                "r2": float(1 - error.square().sum() / denominator)}
    report = {"all": one(torch.ones(len(target), dtype=torch.bool))}
    for suite in sorted(set(suites)):
        report[suite] = one(torch.tensor([value == suite for value in suites]))
    return report


def design(data, contexts, include_h3):
    context = torch.zeros((len(data["ids"]), len(contexts)), dtype=torch.double)
    index = {value: i for i, value in enumerate(contexts)}
    for row, value in enumerate(data["context_id"]):
        if value in index:
            context[row, index[value]] = 1
    columns = [context, (data["start"] / 400.0).unsqueeze(1)]
    if include_h3:
        columns.append(data["features"])
    return torch.cat(columns, dim=1)


def main() -> None:
    args = parse_args()
    train, val = load(args.train), load(args.val)
    contexts = sorted(set(train["context_id"]))
    baseline = ridge_predict(design(train, contexts, False), train["target"], design(val, contexts, False), args.ridge)
    h3 = ridge_predict(design(train, contexts, True), train["target"], design(val, contexts, True), args.ridge)
    baseline_metrics, h3_metrics = metrics(val["target"], baseline, val["suite"]), metrics(val["target"], h3, val["suite"])
    ratio = h3_metrics["all"]["mae"] / baseline_metrics["all"]["mae"]
    per_suite_guard = all(h3_metrics[s]["mae"] <= 1.05 * baseline_metrics[s]["mae"] for s in sorted(set(val["suite"])))
    report = {
        "format": "h3wam-frozen-h3-progress-probe-v1", "ridge": args.ridge,
        "train_samples": len(train["ids"]), "validation_samples": len(val["ids"]),
        "train_selection_sha256": train["selection_sha256"], "validation_selection_sha256": val["selection_sha256"],
        "baseline_task_plus_absolute_step": baseline_metrics,
        "h3_plus_task_plus_absolute_step": h3_metrics,
        "mae_ratio_h3_over_baseline": ratio,
        "promotion": "overall MAE ratio <=0.95 and no suite regresses >5%",
        "status": "PASS_PROGRESS_FEATURE_GATE" if ratio <= 0.95 and per_suite_guard else "FAIL_PROGRESS_FEATURE_GATE",
        "boundary": "Frozen-feature diagnostic only; no action conditioning or best-of-N authorization.",
    }
    output = args.output.resolve()
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n"); os.replace(temporary, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
