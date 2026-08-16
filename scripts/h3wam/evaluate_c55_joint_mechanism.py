#!/usr/bin/env python3
"""Evaluate C55 consequence heads on fixed episode-disjoint validation rows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_trainer():
    path = Path(__file__).with_name("train_c55_fact_joint_action.py")
    spec = importlib.util.spec_from_file_location("_c55_mechanism_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = load_trainer()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_balanced_indices(rows: list[dict], per_outcome: int) -> list[int]:
    result = []
    for success in (True, False):
        candidates = [
            index for index, row in enumerate(rows) if bool(row["success"]) is success
        ]
        candidates.sort(
            key=lambda index: hashlib.sha256(
                f"c55-mechanism-v1:{rows[index]['sample_id']}".encode()
            ).digest()
        )
        if len(candidates) < per_outcome:
            raise ValueError(f"not enough validation rows for success={success}")
        result.extend(candidates[:per_outcome])
    result.sort(
        key=lambda index: hashlib.sha256(
            f"c55-mechanism-order-v1:{rows[index]['sample_id']}".encode()
        ).digest()
    )
    return result


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("empty C55 mechanism metric")
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--rollout-dataset", type=Path, required=True)
    parser.add_argument("--rollout-projected-features", type=Path, required=True)
    parser.add_argument("--rollout-kv-root", type=Path, required=True)
    parser.add_argument("--demo-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-outcome", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.per_outcome <= 0:
        raise ValueError("per-outcome must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite C55 mechanism report: {output}")
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C55 mechanism evaluation requires CUDA")

    dataset = TRAIN.C55RolloutDataset(
        args.rollout_dataset,
        args.rollout_projected_features,
        args.rollout_kv_root,
        args.demo_cache_root,
        split="validation",
    )
    indices = fixed_balanced_indices(dataset.rows, args.per_outcome)
    mapping = {index: indices[(position + 1) % len(indices)] for position, index in enumerate(indices)}
    if any(index == replacement for index, replacement in mapping.items()):
        raise RuntimeError("C55 shuffled action control contains a self-map")

    checkpoint_path = args.checkpoint.resolve()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = saved.get("contract", {})
    if (
        saved.get("schema_version") != TRAIN.CHECKPOINT_SCHEMA
        or contract.get("format") != TRAIN.FORMAT
        or contract.get("arm") != "joint_aux"
        or contract.get("future_h3_target_norm_sha256")
        != dataset.future_h3_target_norm_sha256
    ):
        raise ValueError("C55 joint mechanism checkpoint contract mismatch")
    spec = TRAIN.D0.ModelSpec(
        carrier_layers=TRAIN.DEFAULT_H3_CARRIER_LAYERS,
        carrier_source_mode=TRAIN.REPEAT_LAYER49_CARRIER_SOURCE,
    )
    carrier = TRAIN.D0.build_model(spec, device=device, dtype=torch.bfloat16)
    model = TRAIN.H3FactJointAuxPolicy(
        carrier, hidden_dim=1024, future_h3_dim=256, future_state_dim=8
    ).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()

    metrics = {
        "future_h3_clean": [],
        "future_h3_shuffled_action": [],
        "future_state_clean": [],
        "future_state_shuffled_action": [],
        "value_clean_success_only": [],
        "value_shuffled_action_success_only": [],
    }
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in indices:
            item = dataset[index]
            batch = TRAIN.move_targets(TRAIN.collate_rollout(item), device)
            replacement_row = dataset.rows[mapping[index]]
            replacement_actions = TRAIN.environment_actions_to_dataset(
                replacement_row["executed_actions"].float()
            )
            replacement_actions = TRAIN.PARENT.normalize_minmax(
                replacement_actions, dataset.action_min, dataset.action_max
            ).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            timestep = torch.zeros(1, device=device, dtype=torch.float32)
            clean = model.forward_joint(
                batch["actions"], timestep,
                clean_executed_actions=batch["actions"],
                text_context=batch["text_context"], proprio=batch["proprio"],
                video_kv_cache=batch["video_kv_cache"], text_mask=batch["text_mask"],
            )
            shuffled = model.forward_joint(
                batch["actions"], timestep,
                clean_executed_actions=replacement_actions,
                text_context=batch["text_context"], proprio=batch["proprio"],
                video_kv_cache=batch["video_kv_cache"], text_mask=batch["text_mask"],
            )
            metrics["future_h3_clean"].append(
                torch.nn.functional.mse_loss(clean["future_h3"].float(), batch["future_h3"]).item()
            )
            metrics["future_h3_shuffled_action"].append(
                torch.nn.functional.mse_loss(shuffled["future_h3"].float(), batch["future_h3"]).item()
            )
            metrics["future_state_clean"].append(
                torch.nn.functional.mse_loss(clean["future_state"].float(), batch["future_state"]).item()
            )
            metrics["future_state_shuffled_action"].append(
                torch.nn.functional.mse_loss(shuffled["future_state"].float(), batch["future_state"]).item()
            )
            if item["success"]:
                metrics["value_clean_success_only"].append(
                    torch.nn.functional.mse_loss(clean["value"].float(), batch["value"]).item()
                )
                metrics["value_shuffled_action_success_only"].append(
                    torch.nn.functional.mse_loss(shuffled["value"].float(), batch["value"]).item()
                )

    reduced = {key: mean(values) for key, values in metrics.items()}
    reduced["future_h3_shuffle_degradation"] = (
        reduced["future_h3_shuffled_action"] - reduced["future_h3_clean"]
    )
    reduced["future_state_shuffle_degradation"] = (
        reduced["future_state_shuffled_action"] - reduced["future_state_clean"]
    )
    reduced["value_shuffle_degradation_success_only"] = (
        reduced["value_shuffled_action_success_only"]
        - reduced["value_clean_success_only"]
    )
    selection_sha = hashlib.sha256(
        "\n".join(str(dataset.rows[index]["sample_id"]) for index in indices).encode()
    ).hexdigest()
    report = {
        "format": "h3wam-c55-joint-mechanism-eval-v1",
        "status": "MECHANISM_EVALUATED_NOT_CLOSED_LOOP",
        "effect_status": "NOT_EVIDENCE_READY",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "completed_steps": int(saved["completed_steps"]),
        "validation_rows": len(indices),
        "success_rows": args.per_outcome,
        "failure_rows": args.per_outcome,
        "selection_sha256": selection_sha,
        "shuffle_contract": "sha256-fixed-selection circular-right-shift-1; self_map=0",
        "future_h3_target_norm_sha256": dataset.future_h3_target_norm_sha256,
        "metrics": reduced,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
