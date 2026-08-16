#!/usr/bin/env python3
"""Audit mechanics of the C60 expanded canary without exposing its outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from aggregate_c58b_expanded_paired_eval import (
    episode_map, initial_state_digest, validate_result_contract,
)
from prepare_c56b_fact_expanded_eval import C60_SHA256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path) -> dict:
    root = root.resolve()
    prepared_path = root / "PREPARED.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if (
        prepared.get("format") != "h3wam-c56b-fact-expanded-prepared-v1"
        or prepared.get("candidate_checkpoint_sha256") != C60_SHA256
        or prepared.get("jobs") != 640
    ):
        raise ValueError("C60 preparation mismatch")
    result_path = root / "mechanical-canary/results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = Path(prepared["candidate_checkpoint"]).resolve()
    validate_result_contract(
        payload, policy="h3_fact_online_int8", checkpoint=checkpoint,
        suite="libero_spatial", tasks=[0], trials=[34], save_trajectories=True,
    )
    episodes = episode_map(payload)
    if set(episodes) != {(0, 34)}:
        raise ValueError("canary episode identity mismatch")
    trajectory = Path(episodes[(0, 34)].get("trajectory", "")).resolve()
    initial_digest = initial_state_digest(trajectory)
    policy_log = root / "mechanical-canary/policy_server.log"
    text = policy_log.read_text(encoding="utf-8")
    if '"stage": "ready"' not in text or '"policy": "h3_fact_online_int8"' not in text:
        raise ValueError("real policy startup/restore did not reach ready")
    return {
        "format": "h3wam-c56b-fact-expanded-mechanical-canary-v1",
        "status": "PASS_REAL_POLICY_STARTUP_RESTORE_AND_ONE_EPISODE",
        "permission": "GO_8GPU_640_FRESH_PROCESSES_NO_INTERMEDIATE_STOP",
        "checkpoint_sha256": C60_SHA256,
        "prepared_sha256": sha256_file(prepared_path),
        "result_sha256": sha256_file(result_path),
        "trajectory_sha256": sha256_file(trajectory),
        "initial_state_sha256": initial_digest,
        "policy_log_sha256": sha256_file(policy_log),
        "success_redacted": True,
        "claim_boundary": "Mechanical infrastructure canary only; outcome was not used.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
