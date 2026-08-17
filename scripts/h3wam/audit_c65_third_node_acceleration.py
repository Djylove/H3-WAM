#!/usr/bin/env python3
"""Outcome-blind audit of adding a live third worker node to C65.

The audit reads only manifest/source bytes and filesystem metadata.  It never
opens a branch results.json, trajectory, log, or outcome field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


FORMAT = "h3wam-c65-third-node-live-acceleration-audit-v1"
PREPARED_SHA256 = "a883db2662acbb8a2bb31fa9ebbd7ff344ab01d1af5626d5d02def07a0e1158a"
JOBS_SHA256 = "c9a13ede1ea111450ff4bd4f893fd729fc55190f92ce45e76e0240a9001b52cf"
LEGACY_LAUNCHER_SHA256 = "8d36e2803e662e8af9a7abe37a99ba9229e17f891cf7a95ec21e0ebdee854970"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_jobs(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_legacy_launcher(source: str) -> dict[str, Any]:
    skip = 'if [[ -s "${out}/results.json" ]] && compgen -G "${out}/*trajectory.npz"'
    refuse = '[[ ! -e "${out}" ]] || {'
    create = 'mkdir -p "${out}"'
    positions = {name: source.find(token) for name, token in (
        ("complete_skip", skip), ("partial_refuse", refuse), ("create", create)
    )}
    exact_legacy_sequence = (
        all(value >= 0 for value in positions.values())
        and positions["complete_skip"] < positions["partial_refuse"] < positions["create"]
    )
    claim_tokens = (
        "flock ", "renameat2", "RENAME_NOREPLACE", "/claims/", ".claim", "claim_root"
    )
    present = [token for token in claim_tokens if token in source]
    return {
        "complete_skip_partial_refuse_then_mkdir": exact_legacy_sequence,
        "shared_claim_tokens_present": present,
        "all_workers_honor_atomic_claim": False if exact_legacy_sequence and not present else None,
        "check_then_mkdir_toctou": exact_legacy_sequence,
        "positions": positions,
    }


def metadata_inventory(root: Path, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    by_ordinal = {int(row["ordinal"]): row for row in jobs}
    created: set[int] = set()
    complete: set[int] = set()
    unknown_dirs: list[str] = []
    for path in (root / "runs").iterdir():
        if not path.is_dir():
            continue
        match = re.match(r"^(\d+)_g\d+_c\d+_", path.name)
        if not match:
            unknown_dirs.append(path.name)
            continue
        ordinal = int(match.group(1))
        if ordinal not in by_ordinal:
            unknown_dirs.append(path.name)
            continue
        created.add(ordinal)
        result = path / "results.json"
        # This reproduces the legacy launcher's metadata-only completion test.
        if result.is_file() and result.stat().st_size > 0 and any(path.glob("*trajectory.npz")):
            complete.add(ordinal)
    def by_suite(values: set[int]) -> dict[str, int]:
        return dict(sorted(Counter(str(by_ordinal[value]["suite"]) for value in values).items()))
    return {
        "total_jobs": len(jobs),
        "created_output_dirs": len(created),
        "complete_outputs": len(complete),
        "inflight_output_dirs": len(created - complete),
        "uncreated_jobs": len(jobs) - len(created),
        "created_by_suite": by_suite(created),
        "complete_by_suite": by_suite(complete),
        "unknown_run_directories": sorted(unknown_dirs),
        "outcome_files_opened": 0,
        "trajectory_files_opened": 0,
        "logs_opened": 0,
    }


def decide(
    launcher: dict[str, Any], inventory: dict[str, Any], *, active_legacy_launchers: int
) -> dict[str, Any]:
    unsafe_live_mix = (
        active_legacy_launchers > 0
        and launcher["complete_skip_partial_refuse_then_mkdir"]
        and launcher["all_workers_honor_atomic_claim"] is False
        and inventory["uncreated_jobs"] > 0
    )
    return {
        "status": (
            "NO_GO_C65_THIRD_NODE_LIVE_ACCELERATION_LEGACY_UNCLAIMED_QUEUE"
            if unsafe_live_mix else "REQUIRES_SEPARATE_REVIEW"
        ),
        "permission": "DO_NOT_LAUNCH_NODE_30137" if unsafe_live_mix else "NOT_GRANTED",
        "proof": {
            "direct_main_root": (
                "Third worker creates a partial output directory; when a legacy worker reaches "
                "that ordinal it sees incomplete+existing and returns failure, terminating that worker."
            ),
            "staging_publish_before_legacy_check": "Safe only if publication is already complete.",
            "staging_publish_between_legacy_check_and_mkdir": (
                "Unsafe TOCTOU: legacy already observed absence and mkdir -p does not establish ownership."
            ),
            "staging_publish_after_legacy_mkdir": (
                "No-replace publication must decline and gives no acceleration; replacement would corrupt an active output."
            ),
            "third_only_lock": (
                "Insufficient because the two active legacy launchers never acquire or inspect it."
            ),
            "tail_timing": (
                "Not a correctness proof: branch duration is outcome/state dependent and the eight legacy workers advance asynchronously."
            ),
        },
        "safe_transition_requires": [
            "Every active worker must acquire the same atomic per-ordinal claim before testing/creating output.",
            "Each job must write to owner-specific staging and publish complete output with atomic no-replace.",
            "Migration of this live root requires coordinated quiescence/restart of legacy workers and updated marker accounting.",
        ],
        "scope": (
            "This blocks only live acceleration of the frozen C65 root. The idle node remains safe for a disjoint experiment/root."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    prepared = root / "PREPARED.json"
    jobs_path = root / "jobs.jsonl"
    launcher_path = args.legacy_launcher.resolve()
    identities = {
        "prepared_sha256": sha256_file(prepared),
        "jobs_sha256": sha256_file(jobs_path),
        "legacy_launcher_sha256": sha256_file(launcher_path),
    }
    expected = {
        "prepared_sha256": PREPARED_SHA256,
        "jobs_sha256": JOBS_SHA256,
        "legacy_launcher_sha256": LEGACY_LAUNCHER_SHA256,
    }
    if identities != expected:
        raise ValueError(f"C65 identity mismatch: {identities}")
    jobs = load_jobs(jobs_path)
    if len(jobs) != 3072 or [int(row["ordinal"]) for row in jobs] != list(range(3072)):
        raise ValueError("C65 job inventory is not exact and contiguous")
    launcher = audit_legacy_launcher(launcher_path.read_text())
    if not launcher["complete_skip_partial_refuse_then_mkdir"]:
        raise ValueError("legacy launcher control flow changed; this audit is not applicable")
    inventory = metadata_inventory(root, jobs)
    decision = decide(
        launcher, inventory, active_legacy_launchers=args.active_legacy_launchers
    )
    return {
        "format": FORMAT,
        "effect_status": "NOT_EVALUATED",
        "read_only": True,
        "new_node": {
            "ssh": "ssh -p 30137 dev@117.50.181.177",
            "gpus": "8x NVIDIA A800-SXM4-80GB",
            "shared_mnt": True,
            "launches_started": 0,
        },
        "identities": identities,
        "active_legacy_launchers": args.active_legacy_launchers,
        "legacy_launcher_contract": launcher,
        "inventory_snapshot": inventory,
        "markers": {
            "n1_complete": (root / "node-n1-spatial-object.COMPLETED").exists(),
            "n2_complete": (root / "node-n2-goal-10.COMPLETED").exists(),
            "data_gate_exists": (root / "DATA_GATE.json").exists(),
        },
        "decision": decision,
        "claim_boundary": (
            "Filesystem/process metadata only; no result JSON, trajectory, log, success, or score was read."
        ),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--legacy-launcher", type=Path, required=True)
    parser.add_argument("--active-legacy-launchers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing C65 acceleration audit: {output}")
    report = run(args)
    atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
