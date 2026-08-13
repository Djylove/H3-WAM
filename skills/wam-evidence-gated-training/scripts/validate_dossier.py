#!/usr/bin/env python3
"""Validate evidence gates before allocating GPU time to a WAM experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ALIGNMENT_SECTIONS = (
    "architecture", "data", "action", "objective", "optimization", "evaluation"
)
ALIGNMENT_STATUSES = {
    "EXACT", "EQUIVALENT", "INTENTIONAL_DEVIATION", "MISMATCH", "UNKNOWN"
}
CANARY_GATES = (
    "source_identity", "architecture_contract", "data_contract", "action_contract",
    "objective_contract", "optimization_contract", "evaluation_contract", "gradient_path",
)
LONG_GATES = CANARY_GATES + (
    "smoke_finite", "checkpoint_restore", "parent_baseline",
)
CLAIM_GATES = LONG_GATES + ("mechanism_signal", "closed_loop_canary")
CLASSIFICATIONS = {
    "reproduction", "backbone_port", "controlled_ablation", "novel_composition"
}


def nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate(payload: dict, target: str) -> list[str]:
    errors: list[str] = []
    for key in ("experiment_id", "hypothesis", "single_variable", "parent_baseline"):
        if not nonempty_text(payload.get(key)):
            errors.append(f"missing non-empty {key}")

    if payload.get("classification") not in CLASSIFICATIONS:
        errors.append(f"classification must be one of {sorted(CLASSIFICATIONS)}")

    references = payload.get("references")
    if not isinstance(references, list):
        errors.append("references must be a list")
        references = []
    kinds = {item.get("kind") for item in references if isinstance(item, dict)}
    for kind in ("paper", "upstream_code", "local_implementation"):
        if kind not in kinds:
            errors.append(f"missing reference kind {kind}")
    for index, item in enumerate(references):
        if not isinstance(item, dict):
            errors.append(f"references[{index}] must be an object")
            continue
        for key in ("kind", "title", "locator", "revision"):
            if not nonempty_text(item.get(key)):
                errors.append(f"references[{index}] missing {key}")
    if target == "long" and not any(
        isinstance(item, dict)
        and item.get("kind") == "upstream_code"
        and item.get("official") is True
        and item.get("revision") != "PAPER_ONLY"
        for item in references
    ):
        errors.append("GO_LONG requires official code at a fixed revision")

    alignment = payload.get("alignment")
    if not isinstance(alignment, dict):
        errors.append("alignment must be an object")
        alignment = {}
    high_risk: list[str] = []
    for section in ALIGNMENT_SECTIONS:
        rows = alignment.get(section)
        if not isinstance(rows, list) or not rows:
            errors.append(f"alignment.{section} must be a non-empty list")
            continue
        for index, row in enumerate(rows):
            prefix = f"alignment.{section}[{index}]"
            if not isinstance(row, dict):
                errors.append(f"{prefix} must be an object")
                continue
            if not nonempty_text(row.get("field")):
                errors.append(f"{prefix} missing field")
            status = row.get("status")
            if status not in ALIGNMENT_STATUSES:
                errors.append(f"{prefix} invalid status {status!r}")
            if status in {"MISMATCH", "UNKNOWN"}:
                high_risk.append(f"{section}:{row.get('field', index)}={status}")
            evidence = row.get("evidence")
            if not isinstance(evidence, list) or not any(nonempty_text(x) for x in evidence):
                errors.append(f"{prefix} needs evidence")
            if status in {"EQUIVALENT", "INTENTIONAL_DEVIATION"} and not nonempty_text(row.get("rationale")):
                errors.append(f"{prefix} status {status} needs rationale")

    gates = payload.get("gates")
    if not isinstance(gates, dict):
        errors.append("gates must be an object")
        gates = {}
    required_gates = {
        "canary": CANARY_GATES,
        "long": LONG_GATES,
        "claim": CLAIM_GATES,
    }[target]
    for name in required_gates:
        gate = gates.get(name)
        if not isinstance(gate, dict):
            errors.append(f"missing gate {name}")
            continue
        if gate.get("status") != "PASS":
            errors.append(f"gate {name} is not PASS")
        evidence = gate.get("evidence")
        if not isinstance(evidence, list) or not any(nonempty_text(x) for x in evidence):
            errors.append(f"gate {name} needs evidence")
    if high_risk:
        errors.append("unresolved high-risk alignment: " + ", ".join(high_risk))

    budget = payload.get("budget")
    budget_keys = (
        "global_batch", "optimizer_steps", "training_samples", "unique_windows",
        "effective_epochs", "gpu_count", "estimated_hours",
    )
    if not isinstance(budget, dict):
        errors.append("budget must be an object")
    else:
        for key in budget_keys:
            value = budget.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                errors.append(f"budget.{key} must be finite and > 0")
        if all(isinstance(budget.get(key), (int, float)) for key in (
            "global_batch", "optimizer_steps", "training_samples"
        )):
            expected = float(budget["global_batch"]) * float(budget["optimizer_steps"])
            actual = float(budget["training_samples"])
            if abs(expected - actual) > max(1.0, expected * 1e-6):
                errors.append(
                    f"training_samples={actual:g} but global_batch*optimizer_steps={expected:g}"
                )

    decision = payload.get("decision")
    expected_decision = {
        "canary": "GO_CANARY",
        "long": "GO_LONG",
        "claim": "EVIDENCE_READY",
    }[target]
    if not isinstance(decision, dict):
        errors.append("decision must be an object")
    else:
        if decision.get("status") != expected_decision:
            errors.append(
                f"decision.status must be {expected_decision} for --target {target}"
            )
        if not nonempty_text(decision.get("rationale")):
            errors.append("decision.rationale is required")
        if not nonempty_text(decision.get("launch_command")):
            errors.append("decision.launch_command is required for GO")
    return errors


def self_test() -> None:
    row = {
        "field": "x", "paper": "x", "upstream_code": "x", "local": "x",
        "status": "EXACT", "evidence": ["artifact:1"], "rationale": "exact",
    }
    payload = {
        "experiment_id": "self-test", "classification": "controlled_ablation",
        "hypothesis": "x improves y", "single_variable": "x", "parent_baseline": "p",
        "references": [
            {"kind": "paper", "title": "p", "locator": "u", "revision": "v1"},
            {"kind": "upstream_code", "title": "c", "locator": "u", "revision": "abc", "official": True},
            {"kind": "local_implementation", "title": "l", "locator": "/x", "revision": "def"},
        ],
        "alignment": {name: [dict(row)] for name in ALIGNMENT_SECTIONS},
        "gates": {name: {"status": "PASS", "evidence": ["artifact"]} for name in LONG_GATES},
        "budget": {"global_batch": 8, "optimizer_steps": 10, "training_samples": 80,
                   "unique_windows": 80, "effective_epochs": 1, "gpu_count": 8,
                   "estimated_hours": 1},
        "decision": {"status": "GO_LONG", "rationale": "all pass", "launch_command": "true"},
    }
    errors = validate(payload, "long")
    if errors:
        raise AssertionError(errors)
    payload["gates"]["mechanism_signal"] = {
        "status": "FAIL", "evidence": ["diagnostic target"]
    }
    payload["gates"]["closed_loop_canary"] = {
        "status": "FAIL", "evidence": ["not yet evaluated"]
    }
    if validate(payload, "long"):
        raise AssertionError("effectiveness gates incorrectly blocked diagnostic long run")
    payload["alignment"]["data"][0]["status"] = "UNKNOWN"
    if not validate(payload, "long"):
        raise AssertionError("UNKNOWN alignment was not rejected")
    print("self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dossier", nargs="?", type=Path)
    parser.add_argument(
        "--target", choices=("canary", "long", "claim"), default="canary"
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.dossier is None:
        parser.error("dossier is required unless --self-test is used")
    payload = json.loads(args.dossier.read_text())
    errors = validate(payload, args.target)
    print(json.dumps({"valid": not errors, "target": args.target, "errors": errors}, indent=2, ensure_ascii=False))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
