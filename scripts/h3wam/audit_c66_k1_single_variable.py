#!/usr/bin/env python3
"""Fail-closed source/data/evidence audit for the C66-k1 canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PINNED_SOURCE_HASHES = {
    "scripts/h3wam/train_c66_k1_bounded_mechanism_canary.py": "b9ca8cf03e978009468d594781982df8426d3d51e25afd56c63fe0c92f94e530",
    "scripts/h3wam/train_c66_lingbot_c58_persistent_canary.py": "e7a24522a47a660025c664b78f19bd35565b8e24cb9b5338155784e3b1632638",
    "scripts/h3wam/evaluate_c66_context_length_diagnostic.py": "40c33106c9aab088464d1fb3bb6d5ec74d035e8e66b6c51f416de9ab949e75a5",
    "scripts/h3wam/freeze_c66_lingbot_c58_canary.py": "0f857fdb805a1e5d99b2f846fe80a4db57359dd69a40b392396b74284d894f87",
    "src/fastwam/models/h3wam/c66_lingbot_fastwam_persistent.py": "bd61718de773aef7789ebeb2d4ac733a5e0691bad7eceb2e9256b937cd0ff7b4",
}
PLAN_SHA256 = "f01371397b8e31b111a487ea98ac02b8762e16b768d75dc3db1798951e52b70a"
TRAIN_SHA256 = "d81b6909ca64cc2c179ef23e88aafd4ff44cefd3769ab1629f65af66cbaf53e7"
HELDOUT_SHA256 = "dab4705726cad062f7b2ea0aaf1c03b69fd69e3fe25eed655936a2b752d0b02e"
PARENT_SHA256 = "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
H3_SHA256 = "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
C66_REPORT_SHA256 = "55abab8d6f4e71d52941f84eea7725a4f83615a7a56c890446ba50439fc88c34"
C66_CHECKPOINT_SHA256 = "9fffce1f3844bfbe25642364cdc7946d3f8d4f74a950416bd2acc6ac0830fd46"
DIAGNOSTIC_SHA256 = "50a726dd6bc69fa185c9c9bf17cac9ed138d9d8ef6a229b886d44af76c241237"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"{label} identity mismatch: {actual} != {expected}")
    return {"path": str(path.resolve()), "sha256": actual}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--c66-root", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    source = {
        relative: require_hash(project / relative, expected, f"source {relative}")
        for relative, expected in PINNED_SOURCE_HASHES.items()
    }
    plan_path = args.plan_root / "PLAN.json"
    train_path = args.plan_root / "manifest_train800.jsonl"
    heldout_path = args.plan_root / "manifest_heldout64.jsonl"
    identities = {
        "plan": require_hash(plan_path, PLAN_SHA256, "frozen plan"),
        "train800": require_hash(train_path, TRAIN_SHA256, "frozen train800"),
        "heldout64": require_hash(heldout_path, HELDOUT_SHA256, "frozen heldout64"),
        "parent_c58": require_hash(args.parent_checkpoint, PARENT_SHA256, "C58 parent"),
        "h3_int8": require_hash(args.h3_checkpoint, H3_SHA256, "INT8 H3"),
        "c66_report": require_hash(args.c66_root / "report.json", C66_REPORT_SHA256, "C66 report"),
        "c66_checkpoint": require_hash(args.c66_root / "c66_s00100.pt", C66_CHECKPOINT_SHA256, "C66 checkpoint"),
        "context_diagnostic": require_hash(args.diagnostic, DIAGNOSTIC_SHA256, "context diagnostic"),
    }
    plan = json.loads(plan_path.read_text())
    report = json.loads((args.c66_root / "report.json").read_text())
    diagnostic = json.loads(args.diagnostic.read_text())
    exact_contract = {
        "plan_schema": plan.get("schema") == "h3wam-c66-lingbot-c58-canary-plan-v1",
        "seed_66017": plan.get("seed") == 66017,
        "world_size_8": plan.get("budget", {}).get("world_size") == 8,
        "steps_100": plan.get("budget", {}).get("steps") == 100,
        "training_samples_800": plan.get("budget", {}).get("training_samples") == 800,
        "heldout_samples_64": plan.get("heldout_rows") == 64,
        "episode_disjoint": plan.get("episode_intersection") == 0,
        "source_history_7_15_56": (
            plan.get("history_chunks"),
            plan.get("history_observation_frames"),
            plan.get("history_executed_actions"),
        ) == (7, 15, 56),
        "c66_failed_no_go": (
            report.get("status"), report.get("permission")
        ) == ("FAIL_C66_PAIRED_CANARY", "NO_GO_C66_LONG_TRAINING"),
        "diagnostic_only": diagnostic.get("permission") == "DIAGNOSTIC_ONLY_NO_TRAINING_OR_ROLLOUT_RELEASE",
        "diagnostic_selects_k1": diagnostic.get("analysis", {}).get("trained_best_context_window") == 1,
    }
    if not all(exact_contract.values()):
        failed = [name for name, passed in exact_contract.items() if not passed]
        raise SystemExit(f"C66-k1 source/evidence contract failed: {failed}")

    result = {
        "format": "h3wam-c66-k1-single-variable-audit-v1",
        "status": "PASS_C66_K1_SOURCE_DATA_GATE",
        "permission": "GO_BOUNDED_S100_MECHANISM_CANARY_ONLY",
        "source": source,
        "identities": identities,
        "exact_contract": exact_contract,
        "single_variable_matrix": {
            "parent": {"c66_full": "C58 fresh", "c66_k1": "C58 fresh", "changed": False},
            "train_heldout": {"c66_full": "800/64", "c66_k1": "800/64", "changed": False},
            "seed": {"c66_full": 66017, "c66_k1": 66017, "changed": False},
            "optimizer": {"c66_full": "AdamW lr1e-5 beta0.9/0.95 wd0.01", "c66_k1": "AdamW lr1e-5 beta0.9/0.95 wd0.01", "changed": False},
            "steps_world": {"c66_full": "100/8", "c66_k1": "100/8", "changed": False},
            "model_h3": {"c66_full": "same C66 model/frozen INT8 H3", "c66_k1": "same C66 model/frozen INT8 H3", "changed": False},
            "committed_history_chunks": {"c66_full": 7, "c66_k1": 1, "changed": True},
        },
        "boundary": "This gate authorizes only one fresh-parent s100 k1 mechanism canary; never long training or rollout.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
