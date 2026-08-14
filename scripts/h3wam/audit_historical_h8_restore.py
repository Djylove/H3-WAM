#!/usr/bin/env python3
"""Strictly restore a historical H8 regression head and audit one feature sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fastwam.models.h3wam import H3FeatureActionTransformer
from fastwam.models.h3wam.deployment import libero_environment_actions, minmax_normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature", type=Path, required=True)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--reference-feature", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_model(checkpoint: dict) -> H3FeatureActionTransformer:
    model = H3FeatureActionTransformer(
        action_dim=int(checkpoint["action_dim"]),
        state_dim=int(checkpoint["state_dim"]),
        h3_feature_dim=int(checkpoint["feature_shape"][-1]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        num_heads=int(checkpoint["num_heads"]),
        ffn_dim=int(checkpoint["ffn_dim"]),
        num_action_modes=int(checkpoint.get("num_action_modes", 1)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def policy_state(checkpoint: dict, window: dict, start: int) -> torch.Tensor:
    horizon = int(checkpoint["action_horizon"])
    parts = []
    if checkpoint.get("use_proprio", False):
        stats = checkpoint["normalization"]
        proprio = minmax_normalize(
            window["state"].float(), stats["state_min"], stats["state_max"]
        )
        parts.append(proprio.reshape(1, 8).expand(horizon, -1).clone())
    if checkpoint.get("use_previous_action", False):
        raise ValueError("this audit requires a checkpoint without previous-action input")
    if checkpoint.get("include_phase", True):
        phase_length = int(checkpoint["phase_length"])
        steps = torch.arange(horizon, dtype=torch.float32).add(start).clamp_max(
            phase_length - 1
        )
        parts.append((2.0 * steps / max(phase_length - 1, 1) - 1.0).unsqueeze(-1))
    state = torch.cat(parts, dim=-1) if parts else torch.zeros(horizon, 1)
    expected = (horizon, int(checkpoint["state_dim"]))
    if tuple(state.shape) != expected:
        raise ValueError(f"state shape mismatch: expected {expected}, got {tuple(state.shape)}")
    return state.unsqueeze(0)


def tensor_metrics(left: torch.Tensor, right: torch.Tensor) -> dict:
    a = left.float().reshape(-1)
    b = right.float().reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0)
    return {
        "mse": float((a - b).square().mean()),
        "max_abs": float((a - b).abs().max()),
        "relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-12)),
        "cosine": float(cosine),
    }


def layerwise_feature_metrics(left: torch.Tensor, right: torch.Tensor) -> list[dict]:
    if left.ndim != 4 or right.shape != left.shape:
        raise ValueError("layerwise feature comparison requires equal [B,L,S,D] tensors")
    return [
        {"layer_position": index, **tensor_metrics(left[:, index], right[:, index])}
        for index in range(left.shape[1])
    ]


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_item = torch.load(args.feature, map_location="cpu", weights_only=False)
    window = torch.load(args.window, map_location="cpu", weights_only=False)
    features = feature_item["features"].float().unsqueeze(0)
    expected_shape = tuple(int(value) for value in checkpoint["feature_shape"])
    actual_shape = tuple(int(value) for value in features.shape[1:])
    if actual_shape != expected_shape:
        raise ValueError(f"feature shape mismatch: expected {expected_shape}, got {actual_shape}")
    expected_layers = tuple(int(value) for value in checkpoint["feature_layers"])
    actual_layers = tuple(int(value) for value in feature_item["layers"])
    if actual_layers != expected_layers:
        raise ValueError(f"feature layers mismatch: expected {expected_layers}, got {actual_layers}")

    model_a = build_model(checkpoint)
    model_b = build_model(
        torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    )
    horizon = int(checkpoint["action_horizon"])
    state = policy_state(checkpoint, window, int(feature_item["start"]))
    noisy_actions = torch.zeros(1, horizon, int(checkpoint["action_dim"]))
    sigma = torch.zeros(1)
    with torch.inference_mode():
        output = model_a(
            noisy_actions, state=state, h3_features=features, video_sigma=sigma
        )
        restored_output = model_b(
            noisy_actions, state=state, h3_features=features, video_sigma=sigma
        )
        zero_output = model_a(
            noisy_actions,
            state=state,
            h3_features=torch.zeros_like(features),
            video_sigma=sigma,
        )
    if not isinstance(output, torch.Tensor):
        raise TypeError("historical regression head unexpectedly returned mixture output")
    if not torch.isfinite(output).all() or not torch.isfinite(zero_output).all():
        raise FloatingPointError("non-finite historical action output")

    stats = checkpoint["normalization"]
    environment = libero_environment_actions(
        output[0], stats["action_min"], stats["action_max"], binarize_gripper=True
    )
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "feature": str(args.feature.resolve()),
        "window": str(args.window.resolve()),
        "strict_restore": {
            "status": "PASS",
            "exact_output_equal": bool(torch.equal(output, restored_output)),
            "max_abs_output_difference": float((output - restored_output).abs().max()),
        },
        "checkpoint_contract": {
            "policy_type": checkpoint.get("policy_type"),
            "objective": checkpoint.get("objective", "regression"),
            "feature_shape": list(expected_shape),
            "feature_layers": list(expected_layers),
            "feature_timestep": checkpoint.get("feature_timestep"),
            "feature_context_id": checkpoint.get("feature_context_id"),
            "action_horizon": horizon,
            "state_dim": int(checkpoint["state_dim"]),
        },
        "observed_feature_contract": {
            key: feature_item.get(key)
            for key in (
                "layers",
                "timestep",
                "condition_video_timestep",
                "context_id",
                "context_width",
                "context_mode",
                "action_horizon",
                "capture_token_count",
                "capture_token_strategy",
                "capture_compatibility",
                "backbone",
                "quantization",
                "checkpoint",
            )
        },
        "feature_stats": {
            "mean": float(features.mean()),
            "std": float(features.std()),
            "absmax": float(features.abs().max()),
        },
        "normalized_action": output[0].tolist(),
        "environment_action": environment.tolist(),
        "zero_feature_action": zero_output[0].tolist(),
        "feature_vs_zero_action": tensor_metrics(output, zero_output),
    }
    if args.reference_feature is not None:
        reference_item = torch.load(
            args.reference_feature, map_location="cpu", weights_only=False
        )
        reference = reference_item["features"].float().unsqueeze(0)
        if tuple(reference.shape[1:]) != expected_shape:
            raise ValueError("reference feature shape differs from checkpoint contract")
        with torch.inference_mode():
            reference_output = model_a(
                noisy_actions, state=state, h3_features=reference, video_sigma=sigma
            )
        report["reference_feature"] = str(args.reference_feature.resolve())
        report["feature_vs_reference"] = tensor_metrics(features, reference)
        report["feature_vs_reference_by_layer"] = layerwise_feature_metrics(
            features, reference
        )
        report["action_vs_reference"] = tensor_metrics(output, reference_output)
        report["reference_vs_zero_action"] = tensor_metrics(
            reference_output, zero_output
        )
        report["reference_normalized_action"] = reference_output[0].tolist()
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
