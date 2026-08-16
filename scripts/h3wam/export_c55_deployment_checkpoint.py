#!/usr/bin/env python3
"""Export a C55 carrier into an evaluator-compatible, non-resumable D0 file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


PARENT_SHA256 = "36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def carrier_state(c55: dict, parent_keys: set[str]) -> dict[str, torch.Tensor]:
    state = c55.get("model")
    if not isinstance(state, dict):
        raise ValueError("C55 checkpoint model state is missing")
    arm = c55.get("contract", {}).get("arm")
    if arm == "action_only":
        result = state
    elif arm == "joint_aux":
        result = {
            key.removeprefix("carrier."): value
            for key, value in state.items()
            if key.startswith("carrier.")
        }
    else:
        raise ValueError(f"unsupported C55 arm: {arm!r}")
    if set(result) != parent_keys:
        raise ValueError(
            "C55 exported carrier key mismatch: "
            f"missing={sorted(parent_keys - set(result))[:5]}, "
            f"extra={sorted(set(result) - parent_keys)[:5]}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c55-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    report_path = args.report.resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite C55 deployment export")
    parent_path = args.parent_checkpoint.resolve()
    c55_path = args.c55_checkpoint.resolve()
    if sha256_file(parent_path) != PARENT_SHA256:
        raise ValueError("C55 deployment parent identity mismatch")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    c55 = torch.load(c55_path, map_location="cpu", weights_only=False)
    if c55.get("schema_version") != 1:
        raise ValueError("C55 deployment source schema mismatch")
    if c55.get("contract", {}).get("parent_sha256") != PARENT_SHA256:
        raise ValueError("C55 deployment source parent mismatch")
    exported = dict(parent)
    exported["model"] = carrier_state(c55, set(parent["model"]))
    exported["completed_steps"] = int(parent["completed_steps"]) + int(
        c55["completed_steps"]
    )
    exported["probe_prediction"] = c55["probe_prediction"].clone()
    exported["data_state"] = dict(parent["data_state"])
    # Existing evaluator and rollout code accept the exact D0 schema/contract.
    # The training loader explicitly rejects this resume mode, preventing stale
    # parent optimizer slots from being used with the updated carrier weights.
    exported["data_state"]["resume_mode"] = "deployment_only_c55_v1"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    torch.save(exported, temporary)
    os.replace(temporary, output)
    result = {
        "format": "h3wam-c55-deployment-export-v1",
        "arm": c55["contract"]["arm"],
        "source_checkpoint": str(c55_path),
        "source_checkpoint_sha256": sha256_file(c55_path),
        "parent_checkpoint_sha256": PARENT_SHA256,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "completed_steps": exported["completed_steps"],
        "model_keys": len(exported["model"]),
        "resume_permission": "NO_GO_DEPLOYMENT_ONLY",
        "evaluation_permission": "GO_MECHANICAL_EVALUATOR_RESTORE",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_name(
        f".{report_path.name}.{os.getpid()}.partial"
    )
    temporary_report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_report, report_path)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
