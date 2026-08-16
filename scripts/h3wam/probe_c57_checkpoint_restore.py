#!/usr/bin/env python3
"""Strict C57 checkpoint schema/state/prediction restore gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_trainer():
    path = Path(__file__).with_name("train_c57_lingbot_persistent_kv.py")
    spec = importlib.util.spec_from_file_location("_c57_restore_trainer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


T = load_trainer()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sequence-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cache-source-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--kv-subdir", required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.limit = 1
    args.num_heads = 56
    args.attn_head_dim = 128
    args.hidden_dim = 1024
    args.ffn_dim = 4096
    args.freq_dim = 256
    device, dtype = torch.device("cuda:0"), torch.bfloat16
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != T.CHECKPOINT_SCHEMA:
        raise ValueError("C57 checkpoint schema mismatch")
    if int(payload.get("completed_steps", -1)) <= 0:
        raise ValueError("C57 checkpoint has no completed optimizer steps")
    if not payload.get("optimizer", {}).get("state"):
        raise ValueError("C57 checkpoint optimizer state is empty")
    dataset = T.C57SequenceDataset(args)
    batch = T.move_batch(T.collate_one([dataset[0]]), device, dtype)
    flow = T.PARENT.FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    noisy, _, timesteps = T.PARENT.PARENT.deterministic_flow_batch(
        batch["actions"], flow, seed=57000010
    )
    predictions = []
    for _ in range(2):
        model = T.C57TrainingModel(args).to(device=device, dtype=dtype).eval()
        missing, unexpected = model.policy.load_state_dict(payload["model"], strict=True)
        if missing or unexpected:
            raise ValueError(f"strict restore mismatch: {missing}, {unexpected}")
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            prediction = model(batch, noisy, timesteps).float().cpu()
        predictions.append(prediction)
        del model
        torch.cuda.empty_cache()
    max_abs = float((predictions[0] - predictions[1]).abs().max())
    if max_abs != 0.0 or not torch.isfinite(predictions[0]).all():
        raise RuntimeError(f"C57 prediction restore mismatch: {max_abs}")
    report = {
        "status": "PASS",
        "completed_steps": int(payload["completed_steps"]),
        "restore_max_abs": max_abs,
        "optimizer_state_entries": len(payload["optimizer"]["state"]),
        "probe_sample_id": batch["sample_ids"][0],
        "prediction_std": float(predictions[0].std()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": T.PARENT.sha256_file(args.checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
