#!/usr/bin/env python3
"""Validate locked, graded, and selected WAM evolution epoch bundles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any


HEX64 = re.compile(r"^[0-9a-f]{64}$")
CLASSIFICATIONS = {"reproduction", "backbone_port", "controlled_ablation", "novel_composition"}
LOCK_STATUSES = {"LOCKED", "EXECUTING", "EVIDENCE_FROZEN", "GRADED", "SELECTED"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0


def validate_lock(bundle: dict[str, Any]) -> tuple[list[str], set[str]]:
    require(bundle.get("format") == "wam-evolution-epoch-v1", "format mismatch")
    require(isinstance(bundle.get("epoch_id"), str) and bundle["epoch_id"].strip(), "epoch_id is required")
    require(bundle.get("status") in LOCK_STATUSES, "epoch is not locked")
    require(isinstance(bundle.get("hypothesis"), str) and bundle["hypothesis"].strip(), "hypothesis is required")
    require(bundle.get("classification") in CLASSIFICATIONS, "classification mismatch")
    lock = bundle.get("kernel_lock")
    expected_locks = {
        "objective_sha256", "source_tree_sha256", "data_manifest_sha256",
        "action_contract_sha256", "evaluator_sha256", "runtime_sha256",
        "tool_manifest_sha256",
    }
    require(isinstance(lock, dict) and set(lock) == expected_locks, "kernel_lock fields mismatch")
    require(all(isinstance(value, str) and HEX64.fullmatch(value) for value in lock.values()), "kernel lock values must be lowercase SHA256")
    benchmark = bundle.get("benchmark")
    require(isinstance(benchmark, dict), "benchmark is required")
    for field in ("primary_metric", "direction", "score_formula", "automatic_failures", "tie_break"):
        require(benchmark.get(field), f"benchmark.{field} is required")
    require(benchmark.get("direction") in {"higher", "lower"}, "benchmark direction mismatch")
    require(benchmark.get("self_score_ignored") is True, "candidate self-score must be ignored")
    boundaries = bundle.get("boundaries")
    required_boundaries = {"kernel_not_candidate_writable", "execution_source_immutable", "grade_frozen_evidence_only", "mutation_after_grade_only"}
    require(isinstance(boundaries, dict) and required_boundaries.issubset(boundaries), "boundary fields mismatch")
    require(all(boundaries[name] is True for name in required_boundaries), "all integrity boundaries must be true")
    cohort = bundle.get("cohort")
    require(isinstance(cohort, list) and len(cohort) >= 2, "cohort needs at least two candidates")
    ids: list[str] = []
    signatures: list[str] = []
    for index, candidate in enumerate(cohort):
        require(isinstance(candidate, dict), f"cohort[{index}] is not an object")
        candidate_id = candidate.get("candidate_id")
        require(isinstance(candidate_id, str) and candidate_id.strip(), f"cohort[{index}].candidate_id is required")
        ids.append(candidate_id)
        signature = candidate.get("genome_signature")
        require(isinstance(signature, str) and HEX64.fullmatch(signature), f"{candidate_id} genome signature mismatch")
        signatures.append(signature)
        require(isinstance(candidate.get("sole_mutation"), str) and candidate["sole_mutation"].strip(), f"{candidate_id} sole_mutation is required")
        require(isinstance(candidate.get("dossier"), str) and candidate["dossier"].strip(), f"{candidate_id} dossier is required")
        budget = candidate.get("budget")
        require(isinstance(budget, dict), f"{candidate_id} budget is required")
        for field in ("global_batch", "optimizer_steps", "training_samples", "unique_windows", "effective_epochs", "estimated_hours"):
            require(positive_number(budget.get(field)), f"{candidate_id} budget.{field} must be positive")
        require(int(budget["training_samples"]) == int(budget["global_batch"]) * int(budget["optimizer_steps"]), f"{candidate_id} training_samples arithmetic mismatch")
    require(len(ids) == len(set(ids)), "candidate ids must be unique")
    require(len(signatures) == len(set(signatures)), "duplicate genome signatures are forbidden")
    id_set = set(ids)
    for candidate in cohort:
        parent = candidate.get("parent_candidate_id")
        require(parent is None or parent in id_set, f"{candidate['candidate_id']} parent is outside cohort")
        if parent is None:
            require(candidate["sole_mutation"] == "none_control", f"root {candidate['candidate_id']} must be an unchanged control")
        else:
            require(parent != candidate["candidate_id"], f"{candidate['candidate_id']} cannot parent itself")
    return ids, id_set


def validate_grade(bundle: dict[str, Any], ids: list[str], id_set: set[str]) -> None:
    require(bundle.get("status") in {"EVIDENCE_FROZEN", "GRADED", "SELECTED"}, "grade target requires frozen evidence")
    evidence = bundle.get("evidence")
    require(isinstance(evidence, list) and len(evidence) == len(ids), "evidence must contain one row per candidate")
    seen = set()
    for row in evidence:
        require(isinstance(row, dict) and row.get("candidate_id") in id_set, "evidence candidate mismatch")
        seen.add(row["candidate_id"])
        require(row.get("execution_stopped") is True and row.get("frozen_read_only") is True, "evidence was not frozen after execution")
        require(isinstance(row.get("artifact_manifest_sha256"), str) and HEX64.fullmatch(row["artifact_manifest_sha256"]), "artifact manifest SHA mismatch")
        require(row.get("self_score_ignored") is True, "self score was not ignored")
        if bundle.get("status") in {"GRADED", "SELECTED"}:
            require(row.get("verified_outcome") in {"success", "failed", "infra"}, "verified outcome mismatch")
            require(isinstance(row.get("metric"), (int, float)) and math.isfinite(float(row["metric"])), "graded metric must be finite")
    require(seen == id_set, "evidence candidate coverage mismatch")


def validate_select(bundle: dict[str, Any], id_set: set[str]) -> None:
    require(bundle.get("status") == "SELECTED", "select target requires SELECTED status")
    selection = bundle.get("selection")
    require(isinstance(selection, list) and len(selection) == len(id_set), "selection must cover the cohort")
    seen = set()
    for row in selection:
        require(isinstance(row, dict) and row.get("candidate_id") in id_set, "selection candidate mismatch")
        seen.add(row["candidate_id"])
        require(row.get("disposition") in {"elite", "middle", "culled"}, "selection disposition mismatch")
        require(isinstance(row.get("reason"), str) and row["reason"].strip(), "selection reason is required")
        if row["disposition"] == "middle":
            require(isinstance(row.get("next_single_mutation"), str) and row["next_single_mutation"].strip(), "middle candidate needs one next mutation")
        if row["disposition"] == "culled":
            require(isinstance(row.get("memory_record"), str) and row["memory_record"].strip(), "culled candidate needs retained failure memory")
    require(seen == id_set, "selection candidate coverage mismatch")
    require(any(row["disposition"] == "elite" for row in selection), "selection needs a verified elite; otherwise stop the lineage")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--target", choices=("lock", "grade", "select"), required=True)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    require(isinstance(bundle, dict), "bundle must be a JSON object")
    ids, id_set = validate_lock(bundle)
    if args.target in {"grade", "select"}:
        validate_grade(bundle, ids, id_set)
    if args.target == "select":
        validate_select(bundle, id_set)
    print(json.dumps({"status": "PASS", "target": args.target, "epoch_id": bundle["epoch_id"], "candidates": len(ids)}, sort_keys=True))


if __name__ == "__main__":
    main()
