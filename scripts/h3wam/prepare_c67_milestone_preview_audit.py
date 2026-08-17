#!/usr/bin/env python3
"""Audit one completed C67 milestone before the full 20k trajectory ends."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


def load_sibling(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FINAL = load_sibling("_c67_preview_finalizer", "finalize_c67_c60_budget_ablation_20k.py")
FORMAT = "h3wam-c67-milestone-preview-audit-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare(root: Path, milestone: int) -> dict[str, Any]:
    root = root.resolve()
    if milestone not in FINAL.MILESTONES:
        raise ValueError("C67 milestone must be s1000..s20000")
    audit, contract = FINAL.validate_milestone(root, milestone, None)
    checkpoint = Path(audit["checkpoint"]).resolve()
    return {
        "format": FORMAT,
        "status": "PASS_C67_MILESTONE_PREVIEW_AUDIT",
        "permission": "PREVIEW_EVALUATION_ONLY",
        "effect_status": "NOT_EVIDENCE_READY",
        "milestone": milestone,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "milestone_audit": audit,
        "claim_boundary": (
            "This authorizes fixed offline evaluation of one completed milestone only. "
            "It cannot stop or alter the preregistered 20k trajectory, select a "
            "checkpoint, authorize rollout, or establish an effect."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--milestone", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.train_root, args.milestone)
    atomic_json(args.output.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "permission": report["permission"],
        "milestone": report["milestone"],
        "checkpoint_sha256": report["checkpoint_sha256"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
