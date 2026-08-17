#!/usr/bin/env python3
"""Rebind fixed C67 preview metrics to final training-complete evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MILESTONES = tuple(range(1_000, 20_001, 1_000))
PREVIEW_AUDIT_FORMAT = "h3wam-c67-milestone-preview-audit-v1"
REPORT_FORMAT = "h3wam-c67-fact-milestone-balanced80-v1"
COMPLETE_FORMAT = "h3wam-c67-c60-budget-ablation-training-complete-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def seal(
    preview_root: Path,
    train_root: Path,
    training_complete_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    preview_root = preview_root.resolve()
    train_root = train_root.resolve()
    training_complete_path = training_complete_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    complete = load_json(training_complete_path)
    complete_sha = sha256_file(training_complete_path)
    audits = complete.get("milestone_audits", [])
    by_milestone = {
        int(audit.get("milestone", -1)): audit
        for audit in audits if isinstance(audit, dict)
    }
    if (
        complete.get("format") != COMPLETE_FORMAT
        or complete.get("status") != "PASS_C67_BUDGET_TRAINING_COMPLETE"
        or complete.get("permission") != "READY_FOR_PREREGISTERED_OFFLINE_ONLY"
        or complete.get("effect_status") != "NOT_EVIDENCE_READY"
        or complete.get("completed_steps") != 20_000
        or complete.get("global_batch") != 8
        or complete.get("training_samples") != 160_000
        or set(by_milestone) != set(MILESTONES)
    ):
        raise ValueError("C67 final training-complete gate failed before preview sealing")

    staging = output_root.with_name(f".{output_root.name}.{os.getpid()}.partial")
    if staging.exists():
        raise FileExistsError(staging)
    (staging / "reports").mkdir(parents=True)
    sealed_sha: dict[str, str] = {}
    try:
        for milestone in MILESTONES:
            preview_audit_path = preview_root / f"preview-audit/s{milestone}.json"
            preview_report_path = preview_root / f"reports/s{milestone}.json"
            final_audit_path = train_root / f"milestone-audit/s{milestone}.json"
            preview_audit = load_json(preview_audit_path)
            report = load_json(preview_report_path)
            final_audit = load_json(final_audit_path)
            checkpoint = (train_root / f"checkpoints/c67_online_s{milestone}.pt").resolve()
            checkpoint_sha = sha256_file(checkpoint)
            if (
                preview_audit.get("format") != PREVIEW_AUDIT_FORMAT
                or preview_audit.get("status") != "PASS_C67_MILESTONE_PREVIEW_AUDIT"
                or preview_audit.get("permission") != "PREVIEW_EVALUATION_ONLY"
                or preview_audit.get("effect_status") != "NOT_EVIDENCE_READY"
                or preview_audit.get("milestone") != milestone
                or preview_audit.get("milestone_audit") != by_milestone[milestone]
                or final_audit != by_milestone[milestone]
                or preview_audit.get("checkpoint_sha256") != checkpoint_sha
                or Path(preview_audit.get("checkpoint", "")).resolve() != checkpoint
                or preview_audit.get("training_contract_sha256")
                != complete.get("contract_sha256")
            ):
                raise ValueError(f"C67 preview/final audit mismatch: s{milestone}")
            if (
                report.get("format") != REPORT_FORMAT
                or report.get("permission")
                != "PREVIEW_ONLY_PENDING_TRAINING_COMPLETE_REBIND"
                or report.get("effect_status")
                != "PREVIEW_NOT_EVIDENCE_NOT_FOR_EARLY_STOPPING"
                or report.get("milestone") != milestone
                or report.get("checkpoint_sha256") != checkpoint_sha
                or Path(report.get("checkpoint", "")).resolve() != checkpoint
                or report.get("restore_audit_sha256") != sha256_file(preview_audit_path)
                or Path(report.get("restore_audit", "")).resolve()
                != preview_audit_path.resolve()
                or report.get("training_complete") is not None
                or report.get("training_complete_sha256") is not None
                or report.get("training_contract_sha256")
                != complete.get("contract_sha256")
            ):
                raise ValueError(f"invalid C67 preview report binding: s{milestone}")
            sealed = dict(report)
            sealed.update({
                "permission": "DIAGNOSTIC_ONLY_PENDING_FIXED_AGGREGATION",
                "effect_status": "DIAGNOSTIC_NOT_CHECKPOINT_SELECTION",
                "restore_audit": str(final_audit_path.resolve()),
                "restore_audit_sha256": sha256_file(final_audit_path),
                "training_complete": str(training_complete_path),
                "training_complete_sha256": complete_sha,
                "preview_provenance": {
                    "preview_audit": str(preview_audit_path.resolve()),
                    "preview_audit_sha256": sha256_file(preview_audit_path),
                    "preview_report": str(preview_report_path.resolve()),
                    "preview_report_sha256": sha256_file(preview_report_path),
                    "rebound_without_model_reevaluation": True,
                },
            })
            output = staging / f"reports/s{milestone}.json"
            atomic_json(output, sealed)
            sealed_sha[str(milestone)] = sha256_file(output)
        manifest = {
            "format": "h3wam-c67-sealed-preview-manifest-v1",
            "status": "PASS_C67_PREVIEWS_REBOUND_TO_TRAINING_COMPLETE",
            "permission": "READY_FOR_PREREGISTERED_20_POINT_AGGREGATION_ONLY",
            "effect_status": "NOT_EVIDENCE_READY",
            "training_complete": str(training_complete_path),
            "training_complete_sha256": complete_sha,
            "milestones": list(MILESTONES),
            "reports_sha256": sealed_sha,
            "model_reevaluations_during_seal": 0,
            "claim_boundary": (
                "Cryptographic rebinding only. Effect and rollout permission are "
                "determined solely by the unchanged 20-point aggregator."
            ),
        }
        atomic_json(staging / "SEALED.json", manifest)
        os.replace(staging, output_root)
        return manifest
    except Exception:
        # Keep a failed partial tree for forensic inspection; never publish it as output_root.
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--training-complete", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = seal(
        args.preview_root,
        args.train_root,
        args.training_complete,
        args.output_root,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
