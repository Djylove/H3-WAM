#!/usr/bin/env python3
"""Paired held-out C57-versus-D0 flow evaluation with frozen per-sample RNG."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_trainer():
    path = Path(__file__).with_name("train_c57_lingbot_persistent_kv.py")
    spec = importlib.util.spec_from_file_location("_c57_eval_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load C57 trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAINER = load_trainer()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-source-manifest", type=Path, required=True)
    parser.add_argument("--kv-subdir", default="h3_int8_dreamwam_kv_5x32_dense_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    selected = Path(plan["selected_manifest"])
    heldout_source = Path(plan["heldout_source_manifest"])
    if sha256(selected) != plan["selected_manifest_sha256"]:
        raise ValueError("heldout selected manifest changed after plan freeze")
    if sha256(heldout_source) != plan["heldout_source_sha256"]:
        raise ValueError("heldout source manifest changed after plan freeze")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("contract", {}).get("candidate") != "C57":
        raise ValueError("paired evaluator requires C57 checkpoint")
    step = int(checkpoint["completed_steps"])
    if step not in plan["checkpoint_milestones"]:
        raise ValueError(f"checkpoint step {step} is outside the frozen queue")
    dataset_args = SimpleNamespace(
        sequence_manifest=selected,
        source_manifest=heldout_source,
        cache_source_manifest=args.cache_source_manifest,
        cache_root=args.cache_root,
        kv_subdir=args.kv_subdir,
        limit=0,
        hidden_dim=1024,
        ffn_dim=4096,
        num_heads=56,
        attn_head_dim=128,
        freq_dim=256,
    )
    dataset = TRAINER.C57SequenceDataset(dataset_args)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    c57 = TRAINER.C57TrainingModel(dataset_args).to(device=device, dtype=dtype)
    c57.policy.load_state_dict(checkpoint["model"], strict=True)
    c57.eval()
    d0_path = Path(checkpoint["contract"]["source_checkpoint"])
    d0_payload = torch.load(d0_path, map_location="cpu", weights_only=False)
    d0_contract = d0_payload["contract"]
    spec_dict = d0_contract["model_spec"]
    d0_spec = TRAINER.PARENT.ModelSpec(
        action_dim=int(spec_dict["action_dim"]),
        proprio_dim=int(spec_dict["proprio_dim"]),
        context_dim=int(spec_dict["context_dim"]),
        hidden_dim=int(spec_dict["hidden_dim"]),
        ffn_dim=int(spec_dict["ffn_dim"]),
        num_heads=int(spec_dict["num_heads"]),
        attn_head_dim=int(spec_dict["attn_head_dim"]),
        freq_dim=int(spec_dict["freq_dim"]),
        carrier_layers=tuple(spec_dict["carrier_layers"]),
        carrier_source_mode=str(spec_dict["carrier_source_mode"]),
        history_action_steps=int(spec_dict.get("history_action_steps", 0)),
    )
    d0 = TRAINER.PARENT.build_model(d0_spec, device=device, dtype=dtype)
    d0.load_state_dict(d0_payload["model"], strict=True)
    d0.eval()
    scheduler = TRAINER.PARENT.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    records = []
    for index, row in enumerate(dataset.rows):
        batch = TRAINER.move_batch(
            TRAINER.collate_one([dataset[index]]), device, dtype
        )
        seed = int(row["eval_flow_seed"])
        noisy, target, timesteps = TRAINER.PARENT.PARENT.deterministic_flow_batch(
            batch["actions"], scheduler, seed=seed
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            c57_prediction = c57(batch, noisy, timesteps)
            d0_prediction = TRAINER.PARENT.forward_policy(
                d0, batch, noisy, timesteps
            )
            c57_loss = TRAINER.PARENT.flow_matching_loss(
                c57_prediction, target, timesteps, scheduler,
                is_pad_mask=batch["action_is_pad"],
            )
            d0_loss = TRAINER.PARENT.flow_matching_loss(
                d0_prediction, target, timesteps, scheduler,
                is_pad_mask=batch["action_is_pad"],
            )
        values = (float(c57_loss), float(d0_loss))
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError(f"nonfinite heldout loss for {row['current_id']}")
        records.append(
            {
                "current_id": str(row["current_id"]),
                "suite": str(row["suite"]),
                "episode": str(row["episode"]),
                "flow_seed": seed,
                "history_chunks": len(row["history"]),
                "c57_loss": values[0],
                "d0_loss": values[1],
                "c57_minus_d0": values[0] - values[1],
            }
        )
    c57_mean = sum(row["c57_loss"] for row in records) / len(records)
    d0_mean = sum(row["d0_loss"] for row in records) / len(records)
    win_fraction = sum(row["c57_loss"] < row["d0_loss"] for row in records) / len(records)
    relative = (d0_mean - c57_mean) / max(abs(d0_mean), 1e-12)
    gate = plan["offline_gate"]
    passed = (
        relative >= gate["paired_mean_loss_relative_improvement_min"]
        and win_fraction >= gate["paired_sample_win_fraction_min"]
        and step == int(plan["promotion_checkpoint"])
    )
    report = {
        "schema": "c57_paired_heldout_eval_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "plan": str(args.plan.resolve()),
        "plan_sha256": sha256(args.plan),
        "sample_count": len(records),
        "c57_mean_loss": c57_mean,
        "d0_mean_loss": d0_mean,
        "relative_improvement": relative,
        "sample_win_fraction": win_fraction,
        "gate": "GO_CLOSED_LOOP_CANARY" if passed else "NO_GO",
        "note": "offline loss can authorize only a real traced LIBERO canary",
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "samples"}))


if __name__ == "__main__":
    main()
