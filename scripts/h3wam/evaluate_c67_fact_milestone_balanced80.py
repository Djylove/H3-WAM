#!/usr/bin/env python3
"""Evaluate one frozen C67 milestone on the immutable balanced-80 split."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_sibling(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PAIRED = load_sibling("_c67_offline_paired", "evaluate_c56b_fact_online_paired.py")
ONLINE = PAIRED.ONLINE
TRAIN = PAIRED.TRAIN
PROTOCOL = PAIRED.PROTOCOL
LAYERS = tuple(PAIRED.LAYERWISE_H3_50_TO_ACTION_30)
FORMAT = "h3wam-c67-fact-milestone-balanced80-v1"
TRAINING_COMPLETE_FORMAT = "h3wam-c67-c60-budget-ablation-training-complete-v1"
RESTORE_FORMAT = "h3wam-c67-budget-milestone-restore-audit-v1"
SELECTED_IDS_SHA256 = PAIRED.SELECTED_IDS_SHA256
MILESTONES = tuple(range(1_000, 20_001, 1_000))
C72_MILESTONES = tuple(range(1_000, 30_001, 1_000)) + (30_195,)

C58_SHA256 = "2e6294712f7944037c3982ae7e6b8b87adbdaab190e1972ff4a3d592cc99e541"
C60_DATASET_SHA256 = "1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4"
C60_OBSERVATIONS_SHA256 = "b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55"
FIXED_SHARED_DATA_SHA256 = {
    "demo_manifest_sha256": "b0d611c21059fa7da6fb08162b03efadd59aff68354bb101be41d3ae20d98eb1",
    "source_manifest_sha256": "cab8876f067114dce41d16ca52cb0bafddf17da33c92d0adde5f11d7ac9555b9",
    "demo_stats_sha256": "6f7e9f4a2232a798e4e30ad26f5748e71aeeda7fa54cb6ea2d0a3ec7d290e814",
    "c48_dataset_sha256": "d416d86c09ba334fae449a131510b84fa1d111e665a77eabfb248f1c79a5bc61",
    "c48_observations_sha256": "399d93f31a8f26297145942387a233b9667049efc60ac1f46514a3f7ce77a638",
    "c59_completed_sha256": "4e67bb95b69ada2a854d3b2bf4ba434c6b3072c2bba11a91df2c30c6de5eeb99",
    "c59_sample_labels_sha256": "f2be6801cac2f1c5b680b30c5e089f47e2bf428f179ee13c1ae283e2d47a9d53",
}


@dataclass(frozen=True)
class Config:
    checkpoint: Path
    milestone: int
    h3_checkpoint: Path
    source_manifest: Path
    train_manifest: Path
    val_manifest: Path
    cache_root: Path
    output: Path
    variant: str = "c67"
    restore_audit: Path | None = None
    training_complete: Path | None = None
    preview_audit: Path | None = None
    device: str = "cuda:0"
    seed: int = 42
    inference_steps: int = 10


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


def require_c67_contract(contract: dict[str, Any], *, variant: str = "c67") -> None:
    if variant not in {"c67", "c69", "c70", "c72"}:
        raise ValueError(f"unsupported milestone variant: {variant}")
    rank_categories = (
        list(TRAIN.rank_schedule_contract(TRAIN.C70_RANK_SCHEDULE)["rank_categories"])
        if variant == "c70" else list(TRAIN.RANK_CATEGORIES)
    )
    fixed = {
        "format": TRAIN.FORMAT,
        "classification": "FACT_full_backbone_port_online_frozen_int8_h3",
        "rank_categories": rank_categories,
        "loss_weights": (
            [10.0, 1.0, 0.4, 0.4]
            if variant in {"c67", "c70"}
            else [10.0, 0.0, 0.0, 0.0]
        ),
        "target_norm_sha256": "95df1f65eba1b1c3bfb9cebea90983ca54dffa69f60e6135354eb67e8551d000",
        "h3_sha256": TRAIN.EXPECTED_H3_SHA256,
        "d0_sha256": TRAIN.EXPECTED_D0_SHA256,
        "c58_parent_sha256": C58_SHA256,
        "causal_failure_dataset_sha256": C60_DATASET_SHA256,
        "causal_failure_observations_sha256": C60_OBSERVATIONS_SHA256,
        "base_lr": 2e-5,
        "action_lr": 2e-4,
        "warmup_steps": 500,
        "scheduler_horizon": 30_195 if variant == "c72" else 20_000,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "seed": 20260816,
        "gradient_checkpointing": True,
        "action_horizon": 32,
        "action_shift": 5.0,
        "h3_carrier_layers": list(LAYERS),
        "h3_execution": "online_frozen_int8_per_rank_v1",
        "no_kv_cache": True,
        **FIXED_SHARED_DATA_SHA256,
    }
    if variant == "c70":
        fixed["rank_schedule"] = TRAIN.rank_schedule_contract(
            TRAIN.C70_RANK_SCHEDULE
        )["rank_schedule"]
    mismatch = {
        key: {"actual": contract.get(key), "expected": expected}
        for key, expected in fixed.items()
        if contract.get(key) != expected
    }
    if mismatch:
        raise ValueError(f"C67 fixed training contract mismatch: {mismatch}")
    if contract.get("initialization") != {
        "initialization_contract": "strict_online_c58b_parent_v1",
        "c58_completed_steps": 10_000,
    }:
        raise ValueError("C67 fixed C58 initialization mismatch")
    expected_objective = (
        "fact_joint" if variant in {"c67", "c70"} else "action_only"
    )
    if contract.get("objective_mode", "fact_joint") != expected_objective:
        raise ValueError(f"{variant.upper()} objective-mode contract mismatch")
    frozen = contract.get("frozen_auxiliary_parameters")
    if variant in {"c67", "c70"} and frozen not in (None, []):
        raise ValueError(f"{variant.upper()} unexpectedly froze auxiliary parameters")
    if variant in {"c69", "c72"}:
        prefixes = (
            "future_state_encoder.", "value_encoder.",
            "future_representation_encoder.", "future_state_decoder.",
            "value_decoder.", "future_representation_decoder.",
        )
        if (
            not isinstance(frozen, list) or not frozen
            or any(not name.startswith(prefixes) for name in frozen)
            or any(not any(name.startswith(prefix) for name in frozen) for prefix in prefixes)
        ):
            raise ValueError(f"{variant.upper()} auxiliary freeze contract mismatch")


def load_checkpoint(config: Config) -> tuple[dict[str, Any], str, dict[str, Any]]:
    allowed_milestones = C72_MILESTONES if config.variant == "c72" else MILESTONES
    if config.milestone not in allowed_milestones:
        raise ValueError("milestone is outside the fixed variant trajectory")
    checkpoint = config.checkpoint.resolve()
    checkpoint_sha = sha256_file(checkpoint)
    final_mode = config.training_complete is not None
    preview_mode = config.preview_audit is not None
    if final_mode == preview_mode:
        raise ValueError("select exactly one C67 final or preview binding")
    if config.variant == "c69" and final_mode:
        raise ValueError("C69 final rebinding requires the fixed cross-arm aggregator")
    if config.variant == "c70" and final_mode:
        raise ValueError("C70 final rebinding requires the fixed sampler aggregator")
    if config.variant == "c72" and final_mode:
        raise ValueError("C72 final rebinding requires its fixed budget aggregator")
    if final_mode:
        if config.restore_audit is None:
            raise ValueError("final C67 evaluation requires --restore-audit")
        restore_path = config.restore_audit.resolve()
        complete_path = config.training_complete.resolve()
        complete_sha = sha256_file(complete_path)
        complete = load_json(complete_path)
        audits = complete.get("milestone_audits", [])
        by_milestone = {
            int(audit.get("milestone", -1)): audit
            for audit in audits if isinstance(audit, dict)
        }
        if (
            complete.get("format") != TRAINING_COMPLETE_FORMAT
            or complete.get("status") != "PASS_C67_BUDGET_TRAINING_COMPLETE"
            or complete.get("permission") != "READY_FOR_PREREGISTERED_OFFLINE_ONLY"
            or complete.get("effect_status") != "NOT_EVIDENCE_READY"
            or complete.get("completed_steps") != 20_000
            or complete.get("global_batch") != 8
            or complete.get("training_samples") != 160_000
            or set(by_milestone) != set(MILESTONES)
        ):
            raise ValueError("C67 training-complete gate failed")
        restore = load_json(restore_path)
        embedded = by_milestone[config.milestone]
        if restore != embedded:
            raise ValueError("C67 restore audit differs from training-complete evidence")
        binding = {
            "mode": "final",
            "restore_audit": str(restore_path),
            "restore_audit_sha256": sha256_file(restore_path),
            "training_complete": str(complete_path),
            "training_complete_sha256": complete_sha,
            "training_contract_sha256": complete.get("contract_sha256"),
        }
    else:
        preview_path = config.preview_audit.resolve()
        preview = load_json(preview_path)
        restore = preview.get("milestone_audit", {})
        expected_preview = {
            "c67": (
                "h3wam-c67-milestone-preview-audit-v1",
                "PASS_C67_MILESTONE_PREVIEW_AUDIT",
            ),
            "c69": (
                "h3wam-c69-milestone-preview-audit-v1",
                "PASS_C69_MILESTONE_PREVIEW_AUDIT",
            ),
            "c70": (
                "h3wam-c70-milestone-preview-audit-v1",
                "PASS_C70_MILESTONE_PREVIEW_AUDIT",
            ),
            "c72": (
                "h3wam-c72-milestone-preview-audit-v1",
                "PASS_C72_MILESTONE_PREVIEW_AUDIT",
            ),
        }[config.variant]
        if (
            preview.get("format") != expected_preview[0]
            or preview.get("status") != expected_preview[1]
            or preview.get("permission") != "PREVIEW_EVALUATION_ONLY"
            or preview.get("effect_status") != "NOT_EVIDENCE_READY"
            or preview.get("milestone") != config.milestone
            or preview.get("checkpoint_sha256") != checkpoint_sha
            or not isinstance(restore, dict)
        ):
            raise ValueError("C67 preview audit binding failed")
        restore_path = preview_path
        binding = {
            "mode": "preview",
            "restore_audit": str(preview_path),
            "restore_audit_sha256": sha256_file(preview_path),
            "training_complete": None,
            "training_complete_sha256": None,
            "training_contract_sha256": preview.get("training_contract_sha256"),
        }
    restore_identity_ok = (
        (
            restore.get("format") == RESTORE_FORMAT
            and restore.get("status") == "PASS_C67_BUDGET_MILESTONE_STRICT_RESTORE"
            and restore.get("effect_status") == "NOT_EVIDENCE_READY"
        )
        if config.variant == "c67"
        else (
            restore.get("status") == (
                "PASS_C69_MILESTONE_STRICT_RESTORE"
                if config.variant == "c69"
                else "PASS_C72_MILESTONE_STRICT_RESTORE"
            )
            if config.variant in {"c69", "c72"}
            else (
                restore.get("format") == "h3wam-c70-sampler-milestone-restore-audit-v1"
                and restore.get("status") == "PASS_C70_SAMPLER_MILESTONE_STRICT_RESTORE"
                and restore.get("effect_status") == "NOT_EVIDENCE_READY"
            )
        )
    )
    if (
        not restore_identity_ok
        or restore.get("milestone") != config.milestone
        or Path(restore.get("checkpoint", "")).resolve() != checkpoint
        or (
            config.variant in {"c67", "c70"}
            and restore.get("checkpoint_size_bytes") != checkpoint.stat().st_size
        )
        or (
            config.variant in {"c67", "c70"}
            and restore.get("restore_max_abs") != 0.0
        )
        or not restore.get("gate")
        or not all(restore["gate"].values())
    ):
        raise ValueError("C67 milestone strict-restore gate failed")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    required = {
        "schema_version", "completed_steps", "model", "optimizer",
        "lr_scheduler", "contract", "probe_step", "probe_predictions",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != 1
        or payload.get("completed_steps") != config.milestone
        or payload.get("probe_step") != config.milestone
        or not isinstance(payload.get("probe_predictions"), list)
        or len(payload["probe_predictions"]) != 8
    ):
        raise ValueError("C67 milestone checkpoint schema/step mismatch")
    contract = payload.get("contract", {})
    require_c67_contract(contract, variant=config.variant)
    if contract_sha256(contract) != binding["training_contract_sha256"]:
        raise ValueError("C67 checkpoint contract differs from its evaluation binding")
    if final_mode:
        endpoint = "matched_control" if config.milestone == 10_000 else (
            "treatment" if config.milestone == 20_000 else None
        )
        if endpoint is not None:
            endpoint_evidence = complete.get(endpoint, {})
            if (
                endpoint_evidence.get("milestone") != config.milestone
                or endpoint_evidence.get("training_samples") != config.milestone * 8
                or Path(endpoint_evidence.get("checkpoint", "")).resolve() != checkpoint
                or endpoint_evidence.get("checkpoint_sha256") != checkpoint_sha
            ):
                raise ValueError(
                    f"C67 {endpoint} checkpoint differs from training-complete evidence"
                )
    return payload, checkpoint_sha, binding


def run(config: Config) -> dict[str, Any]:
    if config.variant not in {"c67", "c69", "c70", "c72"}:
        raise ValueError("milestone variant must be c67, c69, c70 or c72")
    if config.seed != 42 or config.inference_steps != 10:
        raise ValueError("C67 balanced80 seed/solver is fixed")
    payload, checkpoint_sha, binding = load_checkpoint(config)
    source_rows = PROTOCOL.read_jsonl(config.source_manifest.resolve())
    train_rows = PROTOCOL.read_jsonl(config.train_manifest.resolve())
    val_rows = PROTOCOL.read_jsonl(config.val_manifest.resolve())
    split = PROTOCOL.validate_episode_disjoint_manifests(source_rows, train_rows, val_rows)
    selected, selection = PROTOCOL.select_validation_rows(val_rows, samples_per_task=2)
    if (
        len(selected) != 80
        or selection["selected_ids_sha256"] != SELECTED_IDS_SHA256
        or selection["selected_task_count"] != 40
        or any(count != 2 for count in selection["task_counts"].values())
    ):
        raise ValueError("C67 balanced80 selection drifted")
    contract = payload["contract"]
    actual_data = {
        "source_manifest_sha256": sha256_file(config.source_manifest.resolve()),
        "demo_manifest_sha256": sha256_file(config.train_manifest.resolve()),
        "demo_stats_sha256": sha256_file(config.cache_root.resolve() / "stats.pt"),
    }
    if any(contract.get(key) != value for key, value in actual_data.items()):
        raise ValueError("heldout data differs from C67 training contract")
    visual_mapping, visual_contract = PROTOCOL.build_visual_feature_shuffle(selected)
    dataset = ONLINE.OnlineC58bValidationDataset(
        selected, cache_root=config.cache_root.resolve(), action_horizon=32,
        visual_mapping=visual_mapping,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=ONLINE.collate_online,
    )
    device, dtype = PROTOCOL._resolve_device_dtype(config.device)
    if device.type != "cuda":
        raise ValueError("online INT8 H3 C67 evaluation requires CUDA")
    if sha256_file(config.h3_checkpoint.resolve()) != TRAIN.EXPECTED_H3_SHA256:
        raise ValueError("C67 evaluator H3 checkpoint identity mismatch")
    provider = ONLINE.C58OnlineFrozenH3Provider(
        config.h3_checkpoint.resolve(), layers=LAYERS
    ).to(device=device).eval()
    scheduler = PROTOCOL.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    started = time.perf_counter()
    arm = PAIRED.evaluate_arm(
        name=f"{config.variant}_s{config.milestone}", payload=payload, loader=loader,
        provider=provider, dataset=dataset, scheduler=scheduler,
        device=device, dtype=dtype, seed=config.seed,
    )
    gates = PAIRED._conditioning_gate(arm)
    gc.collect()
    torch.cuda.empty_cache()
    report = {
        "format": {
            "c67": FORMAT,
            "c69": "h3wam-c69-action-only-milestone-balanced80-v1",
            "c70": "h3wam-c70-sampler-coverage-milestone-balanced80-v1",
            "c72": "h3wam-c72-action-only-milestone-balanced80-v1",
        }[config.variant],
        "status": "PASS_FIXED_BALANCED80" if all(gates.values()) else "FAIL_CONDITIONING_COLLAPSE",
        "permission": (
            "DIAGNOSTIC_ONLY_PENDING_FIXED_AGGREGATION"
            if binding["mode"] == "final"
            else "PREVIEW_ONLY_PENDING_TRAINING_COMPLETE_REBIND"
        ),
        "effect_status": (
            "DIAGNOSTIC_NOT_CHECKPOINT_SELECTION"
            if binding["mode"] == "final"
            else "PREVIEW_NOT_EVIDENCE_NOT_FOR_EARLY_STOPPING"
        ),
        "milestone": config.milestone,
        "variant": config.variant,
        "checkpoint": str(config.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "restore_audit": binding["restore_audit"],
        "restore_audit_sha256": binding["restore_audit_sha256"],
        "training_complete": binding["training_complete"],
        "training_complete_sha256": binding["training_complete_sha256"],
        "training_contract_sha256": contract_sha256(contract),
        "data": {
            **actual_data,
            "validation_manifest_sha256": sha256_file(config.val_manifest.resolve()),
            "selection": selection,
            "split_audit": split,
            "visual_shuffle": visual_contract,
        },
        "execution": {
            "h3": "online_frozen_int8",
            "h3_checkpoint_sha256": TRAIN.EXPECTED_H3_SHA256,
            "disk_kv_read": False,
            "disk_kv_write": False,
            "disk_feature_read": False,
            "carrier_layers": list(LAYERS),
            "same_selected_samples_noise_solver_normalization": True,
            "seed": 42,
            "inference_steps": 10,
            "shift": 5.0,
        },
        "arm": arm,
        "conditioning_gates": gates,
        "timing": {"elapsed_seconds": time.perf_counter() - started},
        "claim_boundary": (
            f"One fixed offline {config.variant.upper()} milestone only. This report cannot select a "
            "checkpoint, alter the fixed trajectory, or authorize LIBERO without "
            "training-complete rebinding and the preregistered aggregate."
        ),
    }
    output = config.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return report


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--restore-audit", type=Path)
    binding = parser.add_mutually_exclusive_group(required=True)
    binding.add_argument("--training-complete", type=Path)
    binding.add_argument("--preview-audit", type=Path)
    parser.add_argument("--milestone", type=int, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("c67", "c69", "c70", "c72"), default="c67"
    )
    parser.add_argument("--device", default="cuda:0")
    values = parser.parse_args()
    return Config(**vars(values))


def main() -> None:
    report = run(parse_args())
    metrics = report["arm"]["metrics"]
    print(json.dumps({
        "status": report["status"],
        "milestone": report["milestone"],
        "normalized": metrics["normalized_clip5_model_domain"],
        "physical": metrics["denormalized_official_minmax_clamp"],
        "gripper": metrics["gripper_sign"],
        "conditioning_gates": report["conditioning_gates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
