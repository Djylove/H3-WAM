#!/usr/bin/env python3
"""Strictly aggregate the four-suite C58b trial-33 fresh canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(root: Path, gate: Path) -> dict[str, Any]:
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    if gate_payload.get("permission") != "GO_FRESH_LIBERO":
        raise ValueError("balanced80 gate did not authorize fresh LIBERO")
    rows = []
    sources = {}
    for suite in SUITES:
        path = root / suite / "results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("policy") != "h3_fastwam_online_int8"
            or payload.get("suite") != suite
            or payload.get("task_ids") != list(range(10))
            or payload.get("trial_indices") != [33]
            or payload.get("replan_steps") != 8
            or payload.get("action_horizon") != 32
            or payload.get("model_evaluations") != 10
            or payload.get("use_action_ensembler") is not False
        ):
            raise ValueError(f"C58b fresh rollout contract mismatch: {suite}")
        if Path(payload.get("checkpoint", "")).resolve() != Path(
            gate_payload["checkpoint"]
        ).resolve():
            raise ValueError(f"C58b rollout checkpoint mismatch: {suite}")
        episodes = [
            episode
            for task in payload.get("tasks", [])
            for episode in task.get("episodes", [])
        ]
        if len(episodes) != 10 or any(episode.get("trial") != 33 for episode in episodes):
            raise ValueError(f"C58b fresh suite is incomplete: {suite}")
        successes = sum(bool(episode.get("success")) for episode in episodes)
        rows.append({
            "suite": suite,
            "episodes": 10,
            "successes": successes,
            "success_rate": successes / 10,
        })
        sources[suite] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    total_successes = sum(row["successes"] for row in rows)
    return {
        "format": "h3wam-c58b-online-fresh-libero-trial33-v1",
        "status": "COMPLETE",
        "effect_status": "CLOSED_LOOP_EVIDENCE",
        "candidate": "C58B_FASTWAM_FULL30_H3_LAYERWISE",
        "checkpoint": gate_payload["checkpoint"],
        "checkpoint_sha256": gate_payload["checkpoint_sha256"],
        "balanced80_gate": str(gate.resolve()),
        "balanced80_gate_sha256": sha256_file(gate),
        "protocol": gate_payload["closed_loop_protocol"],
        "episodes": 40,
        "successes": total_successes,
        "success_rate": total_successes / 40,
        "suites": rows,
        "sources": sources,
        "claim_boundary": (
            "Fresh trial-33 single-seed canary across all 40 LIBERO tasks; a full "
            "multi-trial benchmark or paired superiority claim remains unproven."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.root.resolve(), args.gate.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
