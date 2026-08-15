#!/usr/bin/env python3
"""Correct the C38 restore-mode audit without changing any effect gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from fastwam.models.h3wam.fact_lite_consequence import (
    TemporalFutureH3ConsequenceModel,
)


EXPECTED_SEEDS = {161803, 271828, 8675309, 20260815}
EFFECT_GATES = (
    "all_metrics_finite",
    "conditioned_beats_shuffled_train",
    "within_state_shuffle_hurts_conditioned",
    "conditioned_beats_paired_null",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_checkpoint(path: Path, seed: int, device: torch.device) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["model_variant"] != "temporal":
        raise ValueError(f"C38 checkpoint is not temporal: {path}")
    state_dict = payload["models"]["conditioned"]
    first = TemporalFutureH3ConsequenceModel(**payload["model_kwargs"]).to(device)
    second = TemporalFutureH3ConsequenceModel(**payload["model_kwargs"]).to(device)
    first.load_state_dict(state_dict, strict=True)
    second.load_state_dict(state_dict, strict=True)
    first.eval()
    second.eval()
    restored_weights_exact = all(
        torch.equal(first.state_dict()[key].cpu(), value)
        and torch.equal(first.state_dict()[key], second.state_dict()[key])
        for key, value in state_dict.items()
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    kwargs = payload["model_kwargs"]
    state = torch.randn(7, kwargs["state_dim"], generator=generator).to(device)
    current = torch.randn(7, kwargs["target_dim"], generator=generator).to(device)
    actions = torch.randn(
        7, kwargs["action_horizon"], kwargs["action_dim"], generator=generator
    ).to(device)
    with torch.inference_mode():
        expected = first.forward_projected(state, current, actions)
        restored = second.forward_projected(state, current, actions)
    max_abs = float((expected - restored).abs().max())
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "restored_weights_exact": restored_weights_exact,
        "eval_to_eval_max_abs": max_abs,
        "passed": restored_weights_exact and max_abs == 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    device = torch.device(args.device)
    rows = []
    for path in sorted(args.root.glob("*/report.json")):
        report = json.loads(path.read_text())
        seed = int(report["optimization"]["seed"])
        gates = report["mechanism"]["gate"]
        if not all(gates.get(name) is True for name in EFFECT_GATES):
            raise ValueError(f"C38 effect gate failed independently of restore: {path}")
        if gates.get("fresh_restore_exact") is not False:
            raise ValueError(f"C38 report is not the known restore-only failure: {path}")
        checkpoint = Path(report["checkpoint"])
        restore = audit_checkpoint(checkpoint, seed, device)
        rows.append({
            "seed": seed,
            "report": str(path.resolve()),
            "report_sha256": sha256_file(path),
            "old_restore_max_abs": report["mechanism"]["fresh_restore_max_abs"],
            "minimum_effect_gates_unchanged": True,
            "paired_null_gain": report["mechanism"]["conditioned_gain_over_paired_null"],
            "shuffle_degradation": report["mechanism"]["conditioned_within_state_shuffle_degradation"],
            "shuffled_train_gain": report["mechanism"]["conditioned_gain_over_shuffled_train"],
            "restore": restore,
        })
    if {row["seed"] for row in rows} != EXPECTED_SEEDS or len(rows) != 4:
        raise ValueError("restore audit requires all four preregistered C38 seeds")
    passed = all(row["restore"]["passed"] for row in rows)
    result = {
        "experiment_id": "h3_c38_restore_mode_corrected_audit_v1",
        "status": "PASS_C38_CORRECTED_MECHANICAL_AUDIT" if passed else "FAIL_C38_CORRECTED_MECHANICAL_AUDIT",
        "permission": "GO_FRESH_RANKING_VALIDATION" if passed else "NO_GO_FRESH_RANKING_VALIDATION",
        "correction": "Original model was eval while restored model remained train; both are eval here. No threshold, effect metric, checkpoint or weight changes.",
        "claim_boundary": "Mechanical correction plus consumed selection metrics only; C33 remains untouched final ranking validation.",
        "runs": sorted(rows, key=lambda row: row["seed"]),
        "minimum_paired_null_gain": min(row["paired_null_gain"] for row in rows),
        "minimum_shuffle_degradation": min(row["shuffle_degradation"] for row in rows),
        "minimum_shuffled_train_gain": min(row["shuffled_train_gain"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"status": result["status"], "permission": result["permission"]}))


if __name__ == "__main__":
    main()
