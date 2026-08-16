#!/usr/bin/env python3
"""Strictly finalize C58b online-H3 s10000 after an independent restore."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


FASTWAM_COMMIT = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
H3_SHA256 = "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
D0_SHA256 = "36c5615746fcd57f834db4cdbedd7a124174fca634786e1353871ded6b6e6de3"
LAYERS = (0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 20, 22, 24,
          25, 27, 29, 30, 32, 34, 35, 37, 39, 41, 42, 44, 46, 47, 49)
CHECKPOINT_KEYS = {
    "schema_version", "completed_steps", "model", "optimizer", "lr_scheduler",
    "contract", "probe_prediction", "probe_sample_ids", "rng_states", "data_state",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def json_canonical(value: Any) -> Any:
    """Match torch checkpoint containers to their lossless JSON report form."""

    if isinstance(value, dict):
        return {str(key): json_canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_canonical(item) for item in value]
    return value


def require_train_report(report: dict[str, Any], checkpoint: Path) -> None:
    if report.get("event") != "h3_c58b_online_frozen_h3_full30_train":
        raise ValueError("s10000 report event mismatch")
    if report.get("completed_steps") != 10_000 or report.get("world_size") != 8:
        raise ValueError("s10000 report milestone/topology mismatch")
    if Path(report.get("saved_checkpoint", "")).resolve() != checkpoint:
        raise ValueError("s10000 report checkpoint identity mismatch")
    history = report.get("history")
    if not isinstance(history, list) or len(history) != 1_000:
        raise ValueError("s10000 stage must contain exactly 1000 training steps")
    if [row.get("step") for row in history] != list(range(9_001, 10_001)):
        raise ValueError("s10000 history is not the exact final stage")
    for row in history:
        gradients = row.get("block_gradient_norms")
        if (
            not isinstance(gradients, list)
            or len(gradients) != 30
            or not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in gradients)
        ):
            raise ValueError(f"invalid 30-layer gradient record at step {row.get('step')}")


def require_restore_report(report: dict[str, Any], checkpoint: Path) -> None:
    if report.get("event") != "h3_c58b_online_frozen_h3_full30_train":
        raise ValueError("restore report event mismatch")
    if report.get("completed_steps") != 10_000 or report.get("world_size") != 8:
        raise ValueError("restore milestone/topology mismatch")
    if Path(report.get("loaded_checkpoint", "")).resolve() != checkpoint:
        raise ValueError("restore loaded checkpoint identity mismatch")
    if report.get("restore_probe_max_abs") != 0.0:
        raise ValueError("s10000 restore is not bit-exact")
    if report.get("training_samples") != 0 or report.get("history") != []:
        raise ValueError("restore-check-only unexpectedly trained")


def require_contract(contract: dict[str, Any]) -> None:
    required = {
        "candidate": "C58B_FASTWAM_FULL30_H3_LAYERWISE",
        "fastwam_commit": FASTWAM_COMMIT,
        "d0_parent_sha256": D0_SHA256,
        "h3_checkpoint_sha256": H3_SHA256,
        "h3_execution": "online_frozen_int8_per_rank_v1",
        "disk_kv_training_input": False,
        "kv_subdir": None,
        "action_horizon": 32,
        "action_shift": 5.0,
    }
    mismatches = {
        key: {"actual": contract.get(key), "expected": value}
        for key, value in required.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"C58b online contract mismatch: {mismatches}")
    spec = contract.get("model_spec", {})
    if spec.get("action_layers") != 30 or tuple(spec.get("carrier_layers", ())) != LAYERS:
        raise ValueError("C58b model depth/carrier identity mismatch")
    if tuple(contract.get("action_block_to_h3_layer", ())) != LAYERS:
        raise ValueError("C58b 30-layer mapping mismatch")


def finalize(root: Path) -> dict[str, Any]:
    checkpoint = (root / "checkpoints/c58b_online_s10000.pt").resolve()
    train_path = (root / "reports/train_s10000.json").resolve()
    restore_path = (root / "reports/restore_s10000.json").resolve()
    for path in (checkpoint, train_path, restore_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    train = load_json(train_path)
    restore = load_json(restore_path)
    require_train_report(train, checkpoint)
    require_restore_report(restore, checkpoint)
    if train.get("contract") != restore.get("contract"):
        raise ValueError("train and restore contracts differ")
    require_contract(train["contract"])

    # mmap validates the serialized checkpoint identity without duplicating a
    # 12 GB state in host RAM. SHA256 below still authenticates every byte.
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    if set(checkpoint_payload) != CHECKPOINT_KEYS:
        raise ValueError("checkpoint top-level schema mismatch")
    if checkpoint_payload.get("completed_steps") != 10_000:
        raise ValueError("checkpoint milestone mismatch")
    if json_canonical(checkpoint_payload.get("contract")) != train["contract"]:
        raise ValueError("checkpoint/report contract mismatch")
    probe_ids = checkpoint_payload.get("probe_sample_ids")
    if not isinstance(probe_ids, list) or len(probe_ids) != 1:
        raise ValueError("checkpoint must freeze exactly one restore probe")
    if not isinstance(checkpoint_payload.get("model"), dict) or not checkpoint_payload["model"]:
        raise ValueError("checkpoint model state is empty")

    checkpoint_bytes = checkpoint.stat().st_size
    if checkpoint_bytes < 10 * 1024**3:
        raise ValueError("checkpoint is unexpectedly small")
    hashes = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_report_sha256": sha256_file(train_path),
        "restore_report_sha256": sha256_file(restore_path),
    }
    return {
        "format": "h3wam-c58b-online-long10000-ready-v1",
        "status": "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE",
        "permission": "READY_FOR_CHILD_BRANCH_AND_LIBERO_EVAL",
        "completed_steps": 10_000,
        "world_size": 8,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint_bytes,
        **hashes,
        "probe_sample_ids": probe_ids,
        "restore_probe_max_abs": 0.0,
        "all_final_stage_30_layer_gradients_positive": True,
        "contract_identity": {
            "fastwam_commit": FASTWAM_COMMIT,
            "h3_checkpoint_sha256": H3_SHA256,
            "d0_parent_sha256": D0_SHA256,
            "h3_execution": "online_frozen_int8_per_rank_v1",
            "action_block_to_h3_layer": list(LAYERS),
        },
        "claim_boundary": (
            "Proves completed online training and exact restore only; LIBERO "
            "closed-loop effectiveness remains unproven."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize(args.root.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
