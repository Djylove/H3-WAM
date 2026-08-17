#!/usr/bin/env python3
"""Rebind C70 preview metrics to its final 20k evidence without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MILESTONES = tuple(range(1_000, 20_001, 1_000))
PREVIEW_AUDIT_FORMAT = "h3wam-c70-milestone-preview-audit-v1"
REPORT_FORMAT = "h3wam-c70-sampler-coverage-milestone-balanced80-v1"
COMPLETE_FORMAT = "h3wam-c70-sampler-coverage-training-complete-v1"
PREVIEWS_COMPLETE_FORMAT = "h3wam-c70-sampler-coverage-preview-complete-v1"
SELECTED_IDS_SHA256 = "26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42"


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
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _training_complete(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    complete = load_json(path)
    audits = complete.get("milestone_audits", [])
    by_milestone = {
        int(audit.get("milestone", -1)): audit
        for audit in audits
        if isinstance(audit, dict)
    }
    if (
        complete.get("format") != COMPLETE_FORMAT
        or complete.get("status") != "PASS_C70_SAMPLER_TRAINING_COMPLETE"
        or complete.get("permission") != "READY_FOR_PREREGISTERED_OFFLINE_ONLY"
        or complete.get("effect_status") != "NOT_EVIDENCE_READY"
        or complete.get("completed_steps") != 20_000
        or complete.get("training_samples") != 160_000
        or complete.get("matched_control", {}).get("variant") != "C67_4_2_1_1"
        or not isinstance(complete.get("contract_sha256"), str)
        or len(complete["contract_sha256"]) != 64
        or len(audits) != 20
        or set(by_milestone) != set(MILESTONES)
    ):
        raise ValueError("C70 final training-complete gate failed before preview sealing")
    for milestone, audit in by_milestone.items():
        if (
            audit.get("format") != "h3wam-c70-sampler-milestone-restore-audit-v1"
            or audit.get("status") != "PASS_C70_SAMPLER_MILESTONE_STRICT_RESTORE"
            or audit.get("milestone") != milestone
            or not audit.get("gate")
            or not all(audit["gate"].values())
        ):
            raise ValueError(f"C70 final strict-restore audit failed: s{milestone}")
    return complete, by_milestone


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
    complete, by_milestone = _training_complete(training_complete_path)
    complete_sha = sha256_file(training_complete_path)
    preview_marker_path = preview_root / "PREVIEWS_COMPLETE.json"
    preview_marker = load_json(preview_marker_path)
    if (
        preview_marker.get("format") != PREVIEWS_COMPLETE_FORMAT
        or preview_marker.get("status") != "PASS_C70_ALL_20_PREVIEWS_COMPLETE"
        or preview_marker.get("permission") != "WAIT_FOR_FIXED_C67_VS_C70_AGGREGATION"
        or preview_marker.get("effect_status") != "NOT_EVIDENCE_READY"
        or set(preview_marker.get("reports_sha256", {}))
        != {str(step) for step in MILESTONES}
    ):
        raise ValueError("C70 preview-complete marker failed before sealing")

    staging = output_root.with_name(f".{output_root.name}.{os.getpid()}.partial")
    if staging.exists():
        raise FileExistsError(staging)
    (staging / "reports").mkdir(parents=True)
    (staging / "final-audit").mkdir()
    sealed_sha: dict[str, str] = {}
    audit_sha: dict[str, str] = {}
    for milestone in MILESTONES:
        preview_audit_path = preview_root / f"preview-audit/s{milestone}.json"
        preview_report_path = preview_root / f"reports/s{milestone}.json"
        preview_audit = load_json(preview_audit_path)
        report = load_json(preview_report_path)
        embedded = by_milestone[milestone]
        checkpoint = (train_root / f"checkpoints/c70_sampler_s{milestone}.pt").resolve()
        checkpoint_sha = sha256_file(checkpoint)
        if (
            Path(embedded.get("checkpoint", "")).resolve() != checkpoint
            or preview_audit.get("format") != PREVIEW_AUDIT_FORMAT
            or preview_audit.get("status") != "PASS_C70_MILESTONE_PREVIEW_AUDIT"
            or preview_audit.get("permission") != "PREVIEW_EVALUATION_ONLY"
            or preview_audit.get("effect_status") != "NOT_EVIDENCE_READY"
            or preview_audit.get("milestone") != milestone
            or preview_audit.get("milestone_audit") != embedded
            or preview_audit.get("checkpoint_sha256") != checkpoint_sha
            or Path(preview_audit.get("checkpoint", "")).resolve() != checkpoint
            or preview_audit.get("training_contract_sha256")
            != complete.get("contract_sha256")
        ):
            raise ValueError(f"C70 preview/final audit mismatch: s{milestone}")
        arm = report.get("arm", {})
        gates = report.get("conditioning_gates", {})
        if (
            report.get("format") != REPORT_FORMAT
            or report.get("variant") != "c70"
            or report.get("status")
            not in {"PASS_FIXED_BALANCED80", "FAIL_CONDITIONING_COLLAPSE"}
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
            or arm.get("checkpoint_completed_steps") != milestone
            or arm.get("strict_fresh_restore", {}).get("max_abs") != 0.0
            or arm.get("evaluated_ids_sha256") != SELECTED_IDS_SHA256
            or len(arm.get("per_sample", {})) != 80
            or not gates
            or any(not isinstance(value, bool) for value in gates.values())
            or preview_marker["reports_sha256"].get(str(milestone))
            != sha256_file(preview_report_path)
        ):
            raise ValueError(f"invalid C70 preview report binding: s{milestone}")

        final_audit_path = staging / f"final-audit/s{milestone}.json"
        published_final_audit = output_root / f"final-audit/s{milestone}.json"
        atomic_json(final_audit_path, embedded)
        audit_sha[str(milestone)] = sha256_file(final_audit_path)
        sealed = dict(report)
        sealed.update({
            "permission": "DIAGNOSTIC_ONLY_PENDING_FIXED_CROSS_ARM_AGGREGATION",
            "effect_status": "DIAGNOSTIC_NOT_CHECKPOINT_SELECTION",
            "restore_audit": str(published_final_audit),
            "restore_audit_sha256": audit_sha[str(milestone)],
            "training_complete": str(training_complete_path),
            "training_complete_sha256": complete_sha,
            "preview_provenance": {
                "preview_audit": str(preview_audit_path.resolve()),
                "preview_audit_sha256": sha256_file(preview_audit_path),
                "preview_report": str(preview_report_path.resolve()),
                "preview_report_sha256": sha256_file(preview_report_path),
                "previews_complete": str(preview_marker_path.resolve()),
                "previews_complete_sha256": sha256_file(preview_marker_path),
                "rebound_without_model_reevaluation": True,
            },
        })
        output = staging / f"reports/s{milestone}.json"
        atomic_json(output, sealed)
        sealed_sha[str(milestone)] = sha256_file(output)

    manifest = {
        "format": "h3wam-c70-sealed-preview-manifest-v1",
        "status": "PASS_C70_PREVIEWS_REBOUND_TO_TRAINING_COMPLETE",
        "permission": "READY_FOR_FIXED_C67_VS_C70_S20_AGGREGATION_ONLY",
        "effect_status": "NOT_EVIDENCE_READY",
        "training_complete": str(training_complete_path),
        "training_complete_sha256": complete_sha,
        "previews_complete": str(preview_marker_path.resolve()),
        "previews_complete_sha256": sha256_file(preview_marker_path),
        "milestones": list(MILESTONES),
        "reports_sha256": sealed_sha,
        "final_audits_sha256": audit_sha,
        "model_reevaluations_during_seal": 0,
        "claim_boundary": (
            "Cryptographic rebinding only. No checkpoint is selected and no model "
            "is evaluated. Only the fixed C67-s20000 versus C70-s20000 cross-arm "
            "aggregator may authorize an attribution rollout."
        ),
    }
    atomic_json(staging / "SEALED.json", manifest)
    os.replace(staging, output_root)
    return manifest


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
