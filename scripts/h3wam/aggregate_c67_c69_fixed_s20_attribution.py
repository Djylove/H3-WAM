#!/usr/bin/env python3
"""Gate the fixed C67-s20 versus C69-s20 attribution rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


MILESTONES = tuple(range(1_000, 20_001, 1_000))
SELECTED_IDS_SHA256 = "26b0326d9694825dac3d6e1cccd0b55db03c7d0b78e56a441927e31d1eb99c42"
C67_REPORT_FORMAT = "h3wam-c67-fact-milestone-balanced80-v1"
C69_REPORT_FORMAT = "h3wam-c69-action-only-milestone-balanced80-v1"
C67_COMPLETE_FORMAT = "h3wam-c67-c60-budget-ablation-training-complete-v1"
C69_COMPLETE_FORMAT = "h3wam-c69-matched-action-only-training-complete-v1"
ALLOWED_CONTRACT_DIFFERENCES = frozenset({
    "objective_mode", "loss_weights", "frozen_auxiliary_parameters",
})
AUXILIARY_PREFIXES = (
    "future_state_encoder.", "value_encoder.", "future_representation_encoder.",
    "future_state_decoder.", "value_decoder.", "future_representation_decoder.",
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


def contract_sha256(contract: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _complete(
    arm: str, path: Path
) -> tuple[dict[str, Any], str, dict[int, dict[str, Any]]]:
    complete = load_json(path)
    audits = complete.get("milestone_audits", [])
    by_milestone = {
        int(audit.get("milestone", -1)): audit
        for audit in audits
        if isinstance(audit, dict)
    }
    fixed = (
        {
            "format": C67_COMPLETE_FORMAT,
            "status": "PASS_C67_BUDGET_TRAINING_COMPLETE",
            "permission": "READY_FOR_PREREGISTERED_OFFLINE_ONLY",
            "effect_status": "NOT_EVIDENCE_READY",
            "completed_steps": 20_000,
            "training_samples": 160_000,
            "global_batch": 8,
        }
        if arm == "c67"
        else {
            "format": C69_COMPLETE_FORMAT,
            "status": "PASS_C69_MATCHED_ACTION_ONLY_TRAINING_COMPLETE",
            "permission": "READY_FOR_PREREGISTERED_OFFLINE_ONLY",
            "effect_status": "NOT_EVIDENCE_READY",
            "completed_steps": 20_000,
            "training_samples": 160_000,
            "matched_joint_arm": "C67-s20000",
        }
    )
    mismatch = [key for key, expected in fixed.items() if complete.get(key) != expected]
    if (
        mismatch
        or len(audits) != 20
        or set(by_milestone) != set(MILESTONES)
        or not isinstance(complete.get("contract_sha256"), str)
        or len(complete["contract_sha256"]) != 64
    ):
        raise ValueError(f"{arm.upper()} training-complete gate failed: {mismatch}")
    expected_status = (
        "PASS_C67_BUDGET_MILESTONE_STRICT_RESTORE"
        if arm == "c67" else "PASS_C69_MILESTONE_STRICT_RESTORE"
    )
    for milestone, audit in by_milestone.items():
        if (
            audit.get("milestone") != milestone
            or audit.get("status") != expected_status
            or not audit.get("gate")
            or not all(audit["gate"].values())
        ):
            raise ValueError(f"{arm.upper()} strict audit failed: s{milestone}")
    return complete, sha256_file(path), by_milestone


def _sealed_manifest(
    arm: str, root: Path, training_complete: Path, complete_sha: str
) -> dict[str, Any]:
    path = root / "SEALED.json"
    manifest = load_json(path)
    fixed = (
        {
            "format": "h3wam-c67-sealed-preview-manifest-v1",
            "status": "PASS_C67_PREVIEWS_REBOUND_TO_TRAINING_COMPLETE",
            "permission": "READY_FOR_PREREGISTERED_20_POINT_AGGREGATION_ONLY",
        }
        if arm == "c67"
        else {
            "format": "h3wam-c69-sealed-preview-manifest-v1",
            "status": "PASS_C69_PREVIEWS_REBOUND_TO_TRAINING_COMPLETE",
            "permission": "READY_FOR_FIXED_C67_VS_C69_S20_AGGREGATION_ONLY",
        }
    )
    if (
        any(manifest.get(key) != value for key, value in fixed.items())
        or manifest.get("effect_status") != "NOT_EVIDENCE_READY"
        or Path(manifest.get("training_complete", "")).resolve()
        != training_complete.resolve()
        or manifest.get("training_complete_sha256") != complete_sha
        or manifest.get("milestones") != list(MILESTONES)
        or set(manifest.get("reports_sha256", {}))
        != {str(step) for step in MILESTONES}
        or manifest.get("model_reevaluations_during_seal") != 0
    ):
        raise ValueError(f"{arm.upper()} sealed preview manifest gate failed")
    for milestone in MILESTONES:
        report = root / f"reports/s{milestone}.json"
        if manifest["reports_sha256"][str(milestone)] != sha256_file(report):
            raise ValueError(f"{arm.upper()} sealed report SHA drift: s{milestone}")
    return manifest


def _contracts(
    arm: str,
    train_root: Path,
    complete: dict[str, Any],
) -> dict[str, Any]:
    audits = {
        int(audit.get("milestone", -1)): audit
        for audit in complete.get("milestone_audits", [])
        if isinstance(audit, dict)
    }
    expected = None
    for milestone in MILESTONES:
        report = load_json(train_root / f"reports/train_s{milestone}.json")
        contract = report.get("contract")
        if (
            report.get("status") != "PASS_C56B_ONLINE_TRAINING_INVOCATION"
            or report.get("completed_steps") != milestone
            or Path(report.get("checkpoint", "")).resolve()
            != Path(audits[milestone].get("checkpoint", "")).resolve()
            or not isinstance(contract, dict)
        ):
            raise ValueError(f"{arm.upper()} train report identity failed: s{milestone}")
        if contract_sha256(contract) != complete["contract_sha256"]:
            raise ValueError(f"{arm.upper()} train contract hash drift: s{milestone}")
        if expected is None:
            expected = contract
        elif contract != expected:
            raise ValueError(f"{arm.upper()} train contract drift: s{milestone}")
    assert expected is not None
    return expected


def _contract_attribution(c67: dict[str, Any], c69: dict[str, Any]) -> dict[str, Any]:
    raw_c67_sha = contract_sha256(c67)
    raw_c69_sha = contract_sha256(c69)
    historical_defaults = {
        "objective_mode": "fact_joint",
        "frozen_auxiliary_parameters": [],
    }
    defaulted = sorted(key for key in historical_defaults if key not in c67)
    normalized_c67 = dict(c67)
    for key, value in historical_defaults.items():
        normalized_c67.setdefault(key, value)
    if set(normalized_c67) != set(c69):
        raise ValueError("C67/C69 training contract key sets differ")
    differences = {
        key for key in normalized_c67 if normalized_c67[key] != c69[key]
    }
    if differences != ALLOWED_CONTRACT_DIFFERENCES:
        raise ValueError(
            "C67/C69 contract changed outside the preregistered attribution: "
            f"{sorted(differences)}"
        )
    frozen67 = normalized_c67.get("frozen_auxiliary_parameters")
    frozen69 = c69.get("frozen_auxiliary_parameters")
    if (
        normalized_c67.get("objective_mode") != "fact_joint"
        or c69.get("objective_mode") != "action_only"
        or normalized_c67.get("loss_weights") != [10.0, 1.0, 0.4, 0.4]
        or c69.get("loss_weights") != [10.0, 0.0, 0.0, 0.0]
        or frozen67 != []
        or not isinstance(frozen69, list)
        or not frozen69
        or len(frozen69) != len(set(frozen69))
        or any(not name.startswith(AUXILIARY_PREFIXES) for name in frozen69)
        or any(
            not any(name.startswith(prefix) for name in frozen69)
            for prefix in AUXILIARY_PREFIXES
        )
    ):
        raise ValueError("C67/C69 preregistered objective/freeze contract mismatch")
    return {
        "only_allowed_fields_differ": True,
        "allowed_fields": sorted(ALLOWED_CONTRACT_DIFFERENCES),
        "c67_contract_sha256": raw_c67_sha,
        "c69_contract_sha256": raw_c69_sha,
        "historical_c67_defaults_applied": defaulted,
        "declared_values": {
            "c67": {
                "objective_mode": normalized_c67["objective_mode"],
                "loss_weights": normalized_c67["loss_weights"],
                "frozen_auxiliary_parameters": normalized_c67["frozen_auxiliary_parameters"],
            },
            "c69": {
                "objective_mode": c69["objective_mode"],
                "loss_weights": c69["loss_weights"],
                "frozen_auxiliary_parameters": c69["frozen_auxiliary_parameters"],
            },
        },
    }


def _report(
    arm: str,
    root: Path,
    milestone: int,
    training_complete: Path,
    complete_sha: str,
    contract_sha: str,
    embedded_audit: dict[str, Any],
) -> dict[str, Any]:
    path = root / f"reports/s{milestone}.json"
    report = load_json(path)
    expected_format = C67_REPORT_FORMAT if arm == "c67" else C69_REPORT_FORMAT
    expected_permission = (
        "DIAGNOSTIC_ONLY_PENDING_FIXED_AGGREGATION"
        if arm == "c67"
        else "DIAGNOSTIC_ONLY_PENDING_FIXED_CROSS_ARM_AGGREGATION"
    )
    restore_path = Path(report.get("restore_audit", "")).resolve()
    restore = load_json(restore_path)
    arm_report = report.get("arm", {})
    selection = report.get("data", {}).get("selection", {})
    gates = report.get("conditioning_gates", {})
    expected_status = (
        "PASS_FIXED_BALANCED80"
        if gates and all(gates.values())
        else "FAIL_CONDITIONING_COLLAPSE"
    )
    if (
        report.get("format") != expected_format
        or report.get("variant", "c67") != arm
        or report.get("status") != expected_status
        or report.get("permission") != expected_permission
        or report.get("effect_status") != "DIAGNOSTIC_NOT_CHECKPOINT_SELECTION"
        or report.get("milestone") != milestone
        or Path(report.get("training_complete", "")).resolve()
        != training_complete.resolve()
        or report.get("training_complete_sha256") != complete_sha
        or report.get("training_contract_sha256") != contract_sha
        or report.get("restore_audit_sha256") != sha256_file(restore_path)
        or restore != embedded_audit
        or Path(embedded_audit.get("checkpoint", "")).resolve()
        != Path(report.get("checkpoint", "")).resolve()
        or arm_report.get("checkpoint_completed_steps") != milestone
        or arm_report.get("strict_fresh_restore", {}).get("max_abs") != 0.0
        or arm_report.get("evaluated_ids_sha256") != SELECTED_IDS_SHA256
        or len(arm_report.get("per_sample", {})) != 80
        or selection.get("selected_ids_sha256") != SELECTED_IDS_SHA256
        or selection.get("selected_task_count") != 40
        or any(count != 2 for count in selection.get("task_counts", {}).values())
        or not gates
        or any(not isinstance(value, bool) for value in gates.values())
    ):
        raise ValueError(f"invalid sealed {arm.upper()} report: s{milestone}")
    if arm == "c69" and embedded_audit.get("checkpoint_sha256") != report.get("checkpoint_sha256"):
        raise ValueError(f"C69 checkpoint SHA differs from final audit: s{milestone}")
    if not _safety(report)["all_metrics_finite"]:
        raise ValueError(f"non-finite sealed {arm.upper()} metrics: s{milestone}")
    return report


def _paired(c69: dict[str, Any], c67: dict[str, Any], metric: str) -> dict[str, Any]:
    if set(c69) != set(c67) or len(c69) != 80:
        raise ValueError("C67/C69 fixed s20 sample identities differ or are incomplete")
    c67_wins = c69_wins = ties = 0
    deltas: list[float] = []
    for sample_id in sorted(c69):
        left = float(c69[sample_id][metric])
        right = float(c67[sample_id][metric])
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError(f"non-finite fixed s20 paired error: {sample_id}/{metric}")
        delta = right - left
        deltas.append(delta)
        if abs(delta) <= 1e-12:
            ties += 1
        elif right < left:
            c67_wins += 1
        else:
            c69_wins += 1
    return {
        "c67_fact_joint_wins": c67_wins,
        "c69_action_only_wins": c69_wins,
        "ties": ties,
        "c67_win_rate_all_80": c67_wins / 80,
        "c67_minus_c69_mean": sum(deltas) / 80,
    }


def _safety(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["arm"]["metrics"]
    values = {
        "normalized_action_mse": float(
            metrics["normalized_clip5_model_domain"]["action_mse"]
        ),
        "physical_action_mse": float(
            metrics["denormalized_official_minmax_clamp"]["action_mse"]
        ),
        "prediction_std": float(
            metrics["normalized_clip5_model_domain"]["prediction_std"]
        ),
        "gripper_macro_f1": float(metrics["gripper_sign"]["macro_f1"]),
        "language_mean_abs_delta": float(
            metrics["language_replacement_end_to_end_h3_and_action"]
            ["mean_abs_prediction_delta"]
        ),
        "visual_shuffle_action_mse": float(
            metrics["visual_feature_shuffle_baseline_delta"]["action_mse"]
        ),
    }
    return {
        **values,
        "all_metrics_finite": all(math.isfinite(value) for value in values.values()),
        "conditioning_gates": report["conditioning_gates"],
        "conditioning_safe": all(report["conditioning_gates"].values()),
    }


def aggregate(
    c67_root: Path,
    c67_train_root: Path,
    c67_training_complete: Path,
    c69_root: Path,
    c69_train_root: Path,
    c69_training_complete: Path,
) -> dict[str, Any]:
    c67_root, c69_root = c67_root.resolve(), c69_root.resolve()
    c67_train_root, c69_train_root = c67_train_root.resolve(), c69_train_root.resolve()
    c67_training_complete = c67_training_complete.resolve()
    c69_training_complete = c69_training_complete.resolve()
    c67_complete, c67_complete_sha, c67_audits = _complete(
        "c67", c67_training_complete
    )
    c69_complete, c69_complete_sha, c69_audits = _complete(
        "c69", c69_training_complete
    )
    _sealed_manifest("c67", c67_root, c67_training_complete, c67_complete_sha)
    _sealed_manifest("c69", c69_root, c69_training_complete, c69_complete_sha)
    c67_contract = _contracts("c67", c67_train_root, c67_complete)
    c69_contract = _contracts("c69", c69_train_root, c69_complete)
    contract_attribution = _contract_attribution(c67_contract, c69_contract)

    reports: dict[str, dict[int, dict[str, Any]]] = {"c67": {}, "c69": {}}
    shared_evaluation_identity = None
    all_report_sha256: dict[str, dict[str, str]] = {"c67": {}, "c69": {}}
    for milestone in MILESTONES:
        for arm, root, complete_path, complete_sha, complete, audits in (
            ("c67", c67_root, c67_training_complete, c67_complete_sha, c67_complete, c67_audits),
            ("c69", c69_root, c69_training_complete, c69_complete_sha, c69_complete, c69_audits),
        ):
            report = _report(
                arm, root, milestone, complete_path, complete_sha,
                complete["contract_sha256"], audits[milestone],
            )
            identity = {
                "data": report["data"],
                "execution": report["execution"],
                "evaluated_ids_sha256": report["arm"]["evaluated_ids_sha256"],
                "per_sample_ids": sorted(report["arm"]["per_sample"]),
            }
            if shared_evaluation_identity is None:
                shared_evaluation_identity = identity
            elif identity != shared_evaluation_identity:
                raise ValueError(
                    f"cross-arm data/execution/noise/solver identity drift: {arm}/s{milestone}"
                )
            reports[arm][milestone] = report
            all_report_sha256[arm][str(milestone)] = sha256_file(
                root / f"reports/s{milestone}.json"
            )
    assert shared_evaluation_identity is not None

    c67_s20, c69_s20 = reports["c67"][20_000], reports["c69"][20_000]
    endpoint_checks = {
        "c67_training_complete_endpoint": (
            c67_complete.get("treatment", {}).get("milestone") == 20_000
            and c67_complete.get("treatment", {}).get("training_samples") == 160_000
            and Path(c67_complete.get("treatment", {}).get("checkpoint", "")).resolve()
            == Path(c67_s20["checkpoint"]).resolve()
            and c67_complete.get("treatment", {}).get("checkpoint_sha256")
            == c67_s20["checkpoint_sha256"]
        ),
        "c69_training_complete_endpoint": (
            Path(c69_complete.get("final_checkpoint", "")).resolve()
            == Path(c69_s20["checkpoint"]).resolve()
            and c69_complete.get("final_checkpoint_sha256")
            == c69_s20["checkpoint_sha256"]
        ),
        "c67_endpoint_bytes_match_hash": (
            sha256_file(Path(c67_s20["checkpoint"]).resolve())
            == c67_s20["checkpoint_sha256"]
        ),
        "c69_endpoint_bytes_match_hash": (
            sha256_file(Path(c69_s20["checkpoint"]).resolve())
            == c69_s20["checkpoint_sha256"]
        ),
    }
    if not all(endpoint_checks.values()):
        raise ValueError(f"fixed s20 endpoint identity failed: {endpoint_checks}")

    c67_all_safe = all(
        all(reports["c67"][step]["conditioning_gates"].values())
        for step in MILESTONES
    )
    c69_all_safe = all(
        all(reports["c69"][step]["conditioning_gates"].values())
        for step in MILESTONES
    )
    c67_safety, c69_safety = _safety(c67_s20), _safety(c69_s20)
    evidence_gates = {
        "c67_all_20_training_and_strict_restore_complete": len(c67_audits) == 20,
        "c69_all_20_training_and_strict_restore_complete": len(c69_audits) == 20,
        "c67_all_20_sealed_reports_complete": len(reports["c67"]) == 20,
        "c69_all_20_sealed_reports_complete": len(reports["c69"]) == 20,
        "same_80_ids_data_execution_noise_solver_all_40_reports": True,
        "only_preregistered_objective_loss_frozen_aux_differ": True,
        "fixed_s20_endpoint_hashes_match_training_complete": all(endpoint_checks.values()),
        "c67_all_20_conditioning_safe": c67_all_safe,
        "c69_all_20_conditioning_safe": c69_all_safe,
        "c67_s20_metrics_finite": bool(c67_safety["all_metrics_finite"]),
        "c69_s20_metrics_finite": bool(c69_safety["all_metrics_finite"]),
    }
    safe = all(evidence_gates.values())
    paired = {
        metric: _paired(
            c69_s20["arm"]["per_sample"],
            c67_s20["arm"]["per_sample"],
            metric,
        )
        for metric in ("normalized_action_mse", "physical_action_mse")
    }
    return {
        "format": "h3wam-c67-c69-fixed-s20-attribution-gate-v1",
        "status": (
            "PASS_C67_C69_FIXED_S20_ATTRIBUTION_CHAIN"
            if safe else "FAIL_C67_C69_CONDITIONING_SAFETY"
        ),
        "permission": (
            "GO_C67_VS_C69_FIXED_S20_PAIRED_LIBERO_ATTRIBUTION"
            if safe else "NO_C67_VS_C69_PAIRED_LIBERO_ATTRIBUTION"
        ),
        "effect_status": "OFFLINE_ATTRIBUTION_NOT_WINNER_NOT_CLOSED_LOOP_EVIDENCE",
        "hypothesis": (
            "At identical 20k budgets, any closed-loop C67-versus-C69 difference "
            "is attributable only to the preregistered FACT consequence objective "
            "and its auxiliary-head freeze contract."
        ),
        "fixed_comparison": {
            "c67": "fact_joint_s20000",
            "c69": "action_only_s20000",
            "checkpoint_selection": False,
            "intermediate_milestones_used_for_selection": False,
        },
        "training_complete": {
            "c67": {"path": str(c67_training_complete), "sha256": c67_complete_sha},
            "c69": {"path": str(c69_training_complete), "sha256": c69_complete_sha},
        },
        "fixed_endpoint_identity": {
            "c67": {
                "milestone": 20_000,
                "checkpoint": c67_s20["checkpoint"],
                "checkpoint_sha256": c67_s20["checkpoint_sha256"],
                "restore_audit": c67_s20["restore_audit"],
                "restore_audit_sha256": c67_s20["restore_audit_sha256"],
            },
            "c69": {
                "milestone": 20_000,
                "checkpoint": c69_s20["checkpoint"],
                "checkpoint_sha256": c69_s20["checkpoint_sha256"],
                "restore_audit": c69_s20["restore_audit"],
                "restore_audit_sha256": c69_s20["restore_audit_sha256"],
            },
        },
        "contract_attribution": contract_attribution,
        "evaluation_identity": shared_evaluation_identity,
        "report_sha256": all_report_sha256,
        "evidence_gates": evidence_gates,
        "condition_safety": {"c67_s20000": c67_safety, "c69_s20000": c69_safety},
        "paired_s20000_c69_action_only_vs_c67_fact_joint": paired,
        "claim_boundary": (
            "This offline report never declares a winner and never selects a "
            "checkpoint. A pass authorizes only the separately fixed paired LIBERO "
            "C67-s20000 versus C69-s20000 attribution evaluation. Only that closed-loop "
            "result may support an incremental consequence-objective claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c67-root", type=Path, required=True)
    parser.add_argument("--c67-train-root", type=Path, required=True)
    parser.add_argument("--c67-training-complete", type=Path, required=True)
    parser.add_argument("--c69-root", type=Path, required=True)
    parser.add_argument("--c69-train-root", type=Path, required=True)
    parser.add_argument("--c69-training-complete", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report = aggregate(
        args.c67_root,
        args.c67_train_root,
        args.c67_training_complete,
        args.c69_root,
        args.c69_train_root,
        args.c69_training_complete,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({
        "status": report["status"],
        "permission": report["permission"],
        "fixed_endpoint_identity": report["fixed_endpoint_identity"],
        "evidence_gates": report["evidence_gates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
