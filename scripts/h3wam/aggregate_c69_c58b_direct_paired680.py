#!/usr/bin/env python3
"""Finalize the direct C69 versus C58b paired LIBERO confirmation rerun."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("_direct_pair_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load aggregation base: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-script", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    base = load_base(args.base_script.resolve())
    root = args.root.resolve()
    auth_path, manifest_path = root / "AUTHORIZATION.json", root / "jobs.jsonl"
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    jobs = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    if (
        auth.get("format") != "h3wam-c69-c58b-direct-paired680-authorization-v1"
        or auth.get("status") != "AUTHORIZED_DIRECT_PAIRED_RECHECK"
        or auth.get("jobs") != 1360 or auth.get("pairs") != 680
        or len(jobs) != 1360 or sha256(manifest_path) != auth.get("manifest_sha256")
        or any(not (root / f"SHARD_{index:02d}_COMPLETE.json").is_file() for index in range(5))
    ):
        raise ValueError("direct-pair authorization, manifest or shard completion mismatch")

    expected = {(arm, suite, task, trial) for arm in ("c69_action_only", "c58b_fastwam") for suite in base.SUITES for task in range(10) for trial in range(33, 50)}
    actual = {(row["arm"], row["suite"], row["tasks"][0], row["trials"][0]) for row in jobs}
    if actual != expected:
        raise ValueError("direct-pair job grid mismatch")
    c69_gate = Path(auth["inner_gates"]["c69"]["path"]).resolve()
    c69_gate_sha = auth["inner_gates"]["c69"]["sha256"]

    def inspect(job: dict) -> dict:
        result = Path(job["output"]) / "results.json"
        payload = json.loads(result.read_text(encoding="utf-8"))
        policy = "h3_fact_online_int8" if job["arm"] == "c69_action_only" else "h3_fastwam_online_int8"
        base.validate_result_contract(payload, policy=policy, checkpoint=Path(job["checkpoint"]), suite=job["suite"], tasks=job["tasks"], trials=job["trials"], save_trajectories=True)
        if job["arm"] == "c69_action_only" and (
            Path(payload.get("c67_c69_attribution_authorization", "")).resolve() != c69_gate
            or payload.get("c67_c69_attribution_authorization_sha256") != c69_gate_sha
        ):
            raise ValueError(f"C69 inner authorization mismatch: {result}")
        key = (job["tasks"][0], job["trials"][0])
        episode = base.episode_map(payload)[key]
        trajectory = Path(episode["trajectory"]).resolve()
        return {
            "pair_id": job["pair_id"], "arm": job["arm"], "suite": job["suite"],
            "task": key[0], "trial": key[1], "success": bool(episode["success"]),
            "initial_object_joints": episode["initial_object_joints"],
            "initial_state_sha256": base.initial_state_digest(trajectory),
            "result": str(result), "result_sha256": sha256(result),
            "trajectory": str(trajectory), "trajectory_sha256": sha256(trajectory),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(inspect, jobs))
    mapped = {(row["arm"], row["suite"], row["task"], row["trial"]): row for row in rows}
    pairs, evidence = [], []
    for trial in range(33, 50):
        for suite in base.SUITES:
            for task in range(10):
                c69 = mapped[("c69_action_only", suite, task, trial)]
                c58 = mapped[("c58b_fastwam", suite, task, trial)]
                if c69["pair_id"] != c58["pair_id"] or c69["initial_state_sha256"] != c58["initial_state_sha256"] or not base.same_object_joints(c69["initial_object_joints"], c58["initial_object_joints"]):
                    raise ValueError(f"paired initial-state mismatch: {suite}/{task}/{trial}")
                pairs.append({"trial": trial, "suite": suite, "task": task, "candidate": c69["success"], "control": c58["success"]})
                evidence.append({
                    "trial": trial, "suite": suite, "task": task,
                    "c69_result": c69["result"], "c69_result_sha256": c69["result_sha256"],
                    "c69_trajectory": c69["trajectory"], "c69_trajectory_sha256": c69["trajectory_sha256"],
                    "c58b_result": c58["result"], "c58b_result_sha256": c58["result_sha256"],
                    "c58b_trajectory": c58["trajectory"], "c58b_trajectory_sha256": c58["trajectory_sha256"],
                    "initial_state_sha256": c69["initial_state_sha256"],
                })
    overall = base.paired_summary(pairs)
    per_suite = {suite: base.paired_summary([row for row in pairs if row["suite"] == suite]) for suite in base.SUITES}
    gates = {
        "overall_delta_at_least_3pp": overall["success_rate_delta"] >= 0.03,
        "net_wins_at_least_20": overall["candidate_wins"] - overall["control_wins"] >= 20,
        "one_sided_p_at_most_0_05": overall["one_sided_p_candidate_better"] <= 0.05,
        "suite_floor_at_least_minus_3pp": all(row["success_rate_delta"] >= -0.03 for row in per_suite.values()),
    }
    evidence_path = root / "PAIR_EVIDENCE_DIRECT.jsonl"
    temporary = evidence_path.with_name(f".{evidence_path.name}.{os.getpid()}.partial")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in evidence), encoding="utf-8")
    os.replace(temporary, evidence_path)
    report = {
        "format": "h3wam-c69-c58b-direct-paired680-result-v1",
        "status": "PASS_DIRECT_PAIRED_680_EVIDENCE",
        "decision": "PROMOTE_C69" if all(gates.values()) else "KEEP_C58B_PENDING_ANALYSIS",
        "candidate": "C69_ACTION_ONLY_S20000", "control": "C58B_FASTWAM_S10000",
        "overall": overall, "per_suite": per_suite, "directional_gates": gates,
        "authorization": str(auth_path), "authorization_sha256": sha256(auth_path),
        "pair_evidence": str(evidence_path), "pair_evidence_sha256": sha256(evidence_path),
        "claim_boundary": auth["claim_boundary"],
    }
    atomic_json(root / "RESULTS_DIRECT.json", report)
    atomic_json(root / "COMPLETED.json", {"format": "h3wam-c69-c58b-direct-paired680-complete-v1", "status": "COMPLETE", "jobs": 1360, "pairs": 680, "results": str(root / "RESULTS_DIRECT.json"), "results_sha256": sha256(root / "RESULTS_DIRECT.json")})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
