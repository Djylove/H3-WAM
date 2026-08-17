#!/usr/bin/env python3
"""Independently reproduce the completed C67 evidence chain without inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MILESTONES = tuple(range(1_000, 20_001, 1_000))


def load_sibling(name: str, filename: str):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FINAL = load_sibling(
    "_c67_final_evidence_training", "finalize_c67_c60_budget_ablation_20k.py"
)
SEAL = load_sibling("_c67_final_evidence_seal", "seal_c67_milestone_previews.py")
AGG = load_sibling(
    "_c67_final_evidence_aggregate", "aggregate_c67_fact_milestone_balanced80.py"
)


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


def _expected_sealed_report(
    raw: dict[str, Any],
    *,
    preview_audit_path: Path,
    preview_report_path: Path,
    final_audit_path: Path,
    training_complete_path: Path,
    training_complete_sha256: str,
) -> dict[str, Any]:
    expected = dict(raw)
    expected.update({
        "permission": "DIAGNOSTIC_ONLY_PENDING_FIXED_AGGREGATION",
        "effect_status": "DIAGNOSTIC_NOT_CHECKPOINT_SELECTION",
        "restore_audit": str(final_audit_path.resolve()),
        "restore_audit_sha256": sha256_file(final_audit_path),
        "training_complete": str(training_complete_path.resolve()),
        "training_complete_sha256": training_complete_sha256,
        "preview_provenance": {
            "preview_audit": str(preview_audit_path.resolve()),
            "preview_audit_sha256": sha256_file(preview_audit_path),
            "preview_report": str(preview_report_path.resolve()),
            "preview_report_sha256": sha256_file(preview_report_path),
            "rebound_without_model_reevaluation": True,
        },
    })
    return expected


def audit(
    train_root: Path,
    preview_root: Path,
    sealed_root: Path,
    c58_ready: Path,
) -> dict[str, Any]:
    train_root = train_root.resolve()
    preview_root = preview_root.resolve()
    sealed_root = sealed_root.resolve()
    c58_ready = c58_ready.resolve()
    training_complete_path = train_root / "TRAINING_COMPLETE.json"
    sealed_manifest_path = sealed_root / "SEALED.json"
    results_path = sealed_root / "RESULTS.json"
    for path in (
        train_root, preview_root, sealed_root, c58_ready,
        training_complete_path, sealed_manifest_path, results_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    # Reuse the frozen training finalizer as a pure validator. It loads checkpoint
    # state on CPU/mmap only; it does not instantiate a model, run inference, or write.
    reproduced_complete = FINAL.finalize(train_root, c58_ready)
    published_complete = load_json(training_complete_path)
    if reproduced_complete != published_complete:
        raise ValueError("C67 independently reproduced TRAINING_COMPLETE mismatch")
    complete_sha = sha256_file(training_complete_path)
    complete_audits = {
        int(row.get("milestone", -1)): row
        for row in published_complete.get("milestone_audits", [])
        if isinstance(row, dict)
    }
    if set(complete_audits) != set(MILESTONES):
        raise ValueError("C67 final audit milestone set mismatch")

    manifest = load_json(sealed_manifest_path)
    if (
        manifest.get("format") != "h3wam-c67-sealed-preview-manifest-v1"
        or manifest.get("status")
        != "PASS_C67_PREVIEWS_REBOUND_TO_TRAINING_COMPLETE"
        or manifest.get("permission")
        != "READY_FOR_PREREGISTERED_20_POINT_AGGREGATION_ONLY"
        or manifest.get("effect_status") != "NOT_EVIDENCE_READY"
        or manifest.get("training_complete") != str(training_complete_path)
        or manifest.get("training_complete_sha256") != complete_sha
        or manifest.get("milestones") != list(MILESTONES)
        or manifest.get("model_reevaluations_during_seal") != 0
    ):
        raise ValueError("C67 sealed manifest contract mismatch")

    identities: dict[str, dict[str, Any]] = {}
    sealed_report_sha: dict[str, str] = {}
    for milestone in MILESTONES:
        checkpoint = train_root / f"checkpoints/c67_online_s{milestone}.pt"
        train_path = train_root / f"reports/train_s{milestone}.json"
        restore_path = train_root / f"restore/restore_s{milestone}.json"
        final_audit_path = train_root / f"milestone-audit/s{milestone}.json"
        preview_audit_path = preview_root / f"preview-audit/s{milestone}.json"
        preview_report_path = preview_root / f"reports/s{milestone}.json"
        sealed_report_path = sealed_root / f"reports/s{milestone}.json"
        for path in (
            checkpoint, train_path, restore_path, final_audit_path,
            preview_audit_path, preview_report_path, sealed_report_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        checkpoint_sha = sha256_file(checkpoint)
        final_audit = load_json(final_audit_path)
        preview_audit = load_json(preview_audit_path)
        raw_preview = load_json(preview_report_path)
        sealed_report = load_json(sealed_report_path)
        if final_audit != complete_audits[milestone]:
            raise ValueError(f"C67 final milestone audit mismatch: s{milestone}")
        if (
            preview_audit.get("format") != SEAL.PREVIEW_AUDIT_FORMAT
            or preview_audit.get("status") != "PASS_C67_MILESTONE_PREVIEW_AUDIT"
            or preview_audit.get("permission") != "PREVIEW_EVALUATION_ONLY"
            or preview_audit.get("effect_status") != "NOT_EVIDENCE_READY"
            or preview_audit.get("milestone") != milestone
            or preview_audit.get("milestone_audit") != final_audit
            or Path(preview_audit.get("checkpoint", "")).resolve()
            != checkpoint.resolve()
            or preview_audit.get("checkpoint_sha256") != checkpoint_sha
            or preview_audit.get("training_contract_sha256")
            != published_complete.get("contract_sha256")
        ):
            raise ValueError(f"C67 preview audit identity mismatch: s{milestone}")
        if (
            raw_preview.get("format") != SEAL.REPORT_FORMAT
            or raw_preview.get("milestone") != milestone
            or raw_preview.get("permission")
            != "PREVIEW_ONLY_PENDING_TRAINING_COMPLETE_REBIND"
            or raw_preview.get("effect_status")
            != "PREVIEW_NOT_EVIDENCE_NOT_FOR_EARLY_STOPPING"
            or raw_preview.get("checkpoint_sha256") != checkpoint_sha
            or raw_preview.get("restore_audit_sha256")
            != sha256_file(preview_audit_path)
            or raw_preview.get("training_complete") is not None
            or raw_preview.get("training_complete_sha256") is not None
        ):
            raise ValueError(f"C67 raw preview identity mismatch: s{milestone}")
        expected_sealed = _expected_sealed_report(
            raw_preview,
            preview_audit_path=preview_audit_path,
            preview_report_path=preview_report_path,
            final_audit_path=final_audit_path,
            training_complete_path=training_complete_path,
            training_complete_sha256=complete_sha,
        )
        if sealed_report != expected_sealed:
            raise ValueError(f"C67 sealed report was not a pure rebind: s{milestone}")
        sealed_sha = sha256_file(sealed_report_path)
        sealed_report_sha[str(milestone)] = sealed_sha
        identities[str(milestone)] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "train_report_sha256": sha256_file(train_path),
            "strict_restore_report_sha256": sha256_file(restore_path),
            "final_milestone_audit_sha256": sha256_file(final_audit_path),
            "preview_audit_sha256": sha256_file(preview_audit_path),
            "preview_report_sha256": sha256_file(preview_report_path),
            "sealed_report_sha256": sealed_sha,
        }
    if manifest.get("reports_sha256") != sealed_report_sha:
        raise ValueError("C67 sealed report SHA manifest mismatch")

    reproduced_results = AGG.aggregate(sealed_root, training_complete_path)
    published_results = load_json(results_path)
    if reproduced_results != published_results:
        raise ValueError("C67 independently reproduced aggregate RESULTS mismatch")
    if (
        published_results.get("format") != AGG.RESULT_FORMAT
        or published_results.get("status") not in {
            "PASS_C67_BUDGET_BALANCED80_GATE", "FAIL_C67_BUDGET_BALANCED80_GATE",
        }
        or published_results.get("permission") not in {
            "GO_C67_PAIRED_680_ROLLOUT", "NO_C67_PAIRED_680_ROLLOUT",
        }
        or published_results.get("milestones") != list(MILESTONES)
        or published_results.get("total_model_sample_evaluations") != 1_600
        or set(published_results.get("report_sha256", {})) != {
            str(step) for step in MILESTONES
        }
    ):
        raise ValueError("C67 final aggregate identity contract mismatch")
    endpoints = published_results.get("endpoint_identity", {})
    for name, milestone in (("matched_control", 10_000), ("treatment", 20_000)):
        endpoint = endpoints.get(name, {})
        expected = identities[str(milestone)]
        if (
            endpoint.get("milestone") != milestone
            or endpoint.get("checkpoint") != expected["checkpoint"]
            or endpoint.get("checkpoint_sha256") != expected["checkpoint_sha256"]
            or endpoint.get("restore_audit_sha256")
            != expected["final_milestone_audit_sha256"]
        ):
            raise ValueError(f"C67 fixed endpoint identity mismatch: {name}")

    return {
        "format": "h3wam-c67-independent-final-evidence-audit-v1",
        "status": "PASS_C67_FINAL_EVIDENCE_INDEPENDENTLY_REPRODUCED",
        "permission": "READ_ONLY_AUDIT_COMPLETE_NO_ROLLOUT_AUTHORIZATION",
        "effect_status": "AUDIT_ONLY_PRESERVES_PUBLISHED_RESULT",
        "training_complete": {
            "path": str(training_complete_path), "sha256": complete_sha,
        },
        "sealed_manifest": {
            "path": str(sealed_manifest_path),
            "sha256": sha256_file(sealed_manifest_path),
        },
        "published_results": {
            "path": str(results_path), "sha256": sha256_file(results_path),
            "status": published_results["status"],
            "permission": published_results["permission"],
        },
        "fixed_endpoints": endpoints,
        "milestone_identities": identities,
        "checks": {
            "all_20_training_segments_reproduced": True,
            "all_20_checkpoint_bytes_rehashed": True,
            "all_20_strict_restore_reports_bound": True,
            "all_20_previews_purely_rebound": True,
            "fixed_s10000_s20000_endpoints_exact": True,
            "published_aggregate_exactly_reproduced": True,
            "model_reevaluations": 0,
            "rollouts_started": 0,
        },
        "claim_boundary": (
            "Read-only independent evidence reproduction only. This audit neither "
            "changes the published C67 result nor grants or starts a rollout."
        ),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--preview-root", type=Path, required=True)
    parser.add_argument("--sealed-root", type=Path, required=True)
    parser.add_argument("--c58-ready", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(
        args.train_root, args.preview_root, args.sealed_root, args.c58_ready
    )
    atomic_json(args.output.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "permission": report["permission"],
        "published_results": report["published_results"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
