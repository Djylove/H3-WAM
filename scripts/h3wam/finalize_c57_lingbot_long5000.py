#!/usr/bin/env python3
"""Fail-closed final gate for C57's pre-registered step-5000 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch


FINAL_STEP = 5_000
EXPECTED_PLAN_SHA256 = "7d69a2aded4753985ac31c44f25ba0e88fab1fa47906621390df0fc5de07f73a"
EXPECTED_SELECTED_SAMPLES = 80


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"empty or invalid JSONL: {path}")
    return rows


def require_plan(plan: dict[str, Any], plan_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    if plan.get("schema") != "c57_heldout_eval_plan_v1":
        raise ValueError("C57 final plan schema mismatch")
    if plan.get("candidate") != "C57" or plan.get("control") != "D0_frozen_source_checkpoint":
        raise ValueError("C57 final plan candidate/control mismatch")
    if int(plan.get("promotion_checkpoint", -1)) != FINAL_STEP:
        raise ValueError("only the pre-registered step5000 may be promoted")
    if list(plan.get("checkpoint_milestones", ())) != list(range(200, FINAL_STEP + 1, 200)):
        raise ValueError("C57 checkpoint schedule changed")
    if sha256_file(plan_path) != EXPECTED_PLAN_SHA256:
        raise ValueError("C57 frozen plan identity changed")
    selected = Path(plan["selected_manifest"]).resolve()
    if not selected.is_file() or sha256_file(selected) != plan.get("selected_manifest_sha256"):
        raise ValueError("C57 selected heldout manifest identity changed")
    rows = load_jsonl(selected)
    if len(rows) != EXPECTED_SELECTED_SAMPLES or int(plan.get("samples", -1)) != len(rows):
        raise ValueError("C57 final heldout must contain exactly 80 samples")
    suites = Counter(str(row["suite"]) for row in rows)
    if set(suites.values()) != {20} or len(suites) != 4:
        raise ValueError(f"C57 heldout is not balanced 20-per-suite: {dict(suites)}")
    pairs = [(str(row["current_id"]), int(row["eval_flow_seed"])) for row in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError("C57 heldout sample/seed pairs are not unique")
    return selected, rows


def require_checkpoint(checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    required = {"schema_version", "completed_steps", "model", "optimizer", "lr_scheduler", "contract"}
    if set(payload) != required:
        raise ValueError("C57 checkpoint top-level schema mismatch")
    contract = payload.get("contract", {})
    expected = {
        "candidate": "C57",
        "classification": "action-only-on-frozen-h3-kv",
        "method": "lingbot_persistent_observation_action_kv",
        "sequence_schema": "c57_lingbot_replan8_v1",
        "replan": 8,
        "observe_every": 4,
    }
    mismatch = {key: (contract.get(key), value) for key, value in expected.items() if contract.get(key) != value}
    if payload.get("schema_version") != 1 or int(payload.get("completed_steps", -1)) != FINAL_STEP or mismatch:
        raise ValueError(f"C57 checkpoint identity mismatch: {mismatch}")
    if not isinstance(payload.get("model"), dict) or not payload["model"]:
        raise ValueError("C57 checkpoint model state is empty")
    return payload


def require_train_report(report: dict[str, Any]) -> None:
    if report.get("event") != "c57_lingbot_persistent_kv_training":
        raise ValueError("C57 train report event mismatch")
    if report.get("status") != "PASS" or report.get("gate") != "PASS":
        raise ValueError("C57 train report did not pass mechanical training")
    if int(report.get("completed_steps", -1)) != FINAL_STEP or int(report.get("steps", -1)) != FINAL_STEP:
        raise ValueError("C57 train report did not complete step5000")
    history = report.get("history")
    if not isinstance(history, list) or len(history) != FINAL_STEP:
        raise ValueError("C57 train history is incomplete")
    if [row.get("step") for row in history] != list(range(1, FINAL_STEP + 1)):
        raise ValueError("C57 train history step sequence is incomplete")
    for row in history:
        values = [row.get("loss"), row.get("gradient_norm"), row.get("head_update_max_abs")]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError(f"nonfinite C57 training record at step {row.get('step')}")
        if row["gradient_norm"] <= 0 or row["head_update_max_abs"] <= 0:
            raise ValueError(f"dead C57 update at step {row.get('step')}")


def require_heldout(
    report: dict[str, Any], checkpoint: Path, plan_path: Path, selected_rows: list[dict[str, Any]]
) -> None:
    if report.get("schema") != "c57_paired_heldout_eval_v1":
        raise ValueError("C57 heldout report schema mismatch")
    if int(report.get("checkpoint_step", -1)) != FINAL_STEP:
        raise ValueError("C57 final heldout did not evaluate step5000")
    if Path(report.get("checkpoint", "")).resolve() != checkpoint:
        raise ValueError("C57 heldout checkpoint identity mismatch")
    if Path(report.get("plan", "")).resolve() != plan_path:
        raise ValueError("C57 heldout plan path mismatch")
    if report.get("plan_sha256") != EXPECTED_PLAN_SHA256:
        raise ValueError("C57 heldout plan hash mismatch")
    if report.get("strict_restore") is not True:
        raise ValueError("C57 final heldout lacks an explicit strict restore")
    details = report.get("strict_restore_details", {})
    if details.get("c57_policy_load_state_dict") != "strict=True" or details.get("all_heldout_forwards_completed") is not True:
        raise ValueError("C57 final strict restore/forward evidence is incomplete")
    samples = report.get("samples")
    if not isinstance(samples, list) or int(report.get("sample_count", -1)) != len(selected_rows) or len(samples) != len(selected_rows):
        raise ValueError("C57 final heldout sample count mismatch")
    expected_pairs = [(str(row["current_id"]), int(row["eval_flow_seed"])) for row in selected_rows]
    actual_pairs = [(str(row["current_id"]), int(row["flow_seed"])) for row in samples]
    if actual_pairs != expected_pairs:
        raise ValueError("C57 final heldout changed sample order or frozen RNG")
    for row in samples:
        if not all(math.isfinite(float(row[key])) for key in ("c57_loss", "d0_loss", "c57_minus_d0")):
            raise ValueError("C57 final heldout contains nonfinite metrics")


def finalize(checkpoint: Path, train_path: Path, heldout_path: Path, plan_path: Path) -> dict[str, Any]:
    for path in (checkpoint, train_path, heldout_path, plan_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    plan = load_json(plan_path)
    selected_path, selected_rows = require_plan(plan, plan_path)
    payload = require_checkpoint(checkpoint)
    train = load_json(train_path)
    require_train_report(train)
    heldout = load_json(heldout_path)
    require_heldout(heldout, checkpoint, plan_path, selected_rows)
    contract = payload["contract"]
    if int(contract.get("world_size", -1)) != int(train.get("world_size", -2)):
        raise ValueError("C57 checkpoint/train world size mismatch")
    passed = heldout.get("gate") == "GO_CLOSED_LOOP_CANARY"
    return {
        "format": "h3wam-c57-lingbot-long5000-final-v1",
        "status": "PASS_C57_FINAL_OFFLINE_GATE" if passed else "C57_FINAL_OFFLINE_NO_GO",
        "permission": "GO_FRESH_LIBERO_CANARY" if passed else "NO_GO",
        "effect_status": "NOT_EVIDENCE_READY",
        "completed_steps": FINAL_STEP,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_report": str(train_path),
        "train_report_sha256": sha256_file(train_path),
        "heldout_report": str(heldout_path),
        "heldout_report_sha256": sha256_file(heldout_path),
        "heldout_plan": str(plan_path),
        "heldout_plan_sha256": EXPECTED_PLAN_SHA256,
        "selected_manifest": str(selected_path),
        "selected_manifest_sha256": plan["selected_manifest_sha256"],
        "strict_restore": True,
        "heldout_metrics": {
            "c57_mean_loss": heldout["c57_mean_loss"],
            "d0_mean_loss": heldout["d0_mean_loss"],
            "relative_improvement": heldout["relative_improvement"],
            "sample_win_fraction": heldout["sample_win_fraction"],
        },
        "claim_boundary": (
            "The offline gate authorizes only the pre-registered fresh paired LIBERO canary. "
            "It does not establish closed-loop improvement or benchmark generalization."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--heldout-report", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize(
        args.checkpoint.resolve(), args.train_report.resolve(),
        args.heldout_report.resolve(), args.plan.resolve(),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
