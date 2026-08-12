#!/usr/bin/env python3
"""Cache video-LoRA H3 features for FastWAM-relabeled corrective states."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--feature-subdir", required=True)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--h3-lora-checkpoint",
        type=Path,
        help="Optional H3 LoRA checkpoint. Omit for base-H3 feature caches.",
    )
    parser.add_argument("--video-vae", type=Path, required=True)
    parser.add_argument("--context-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-id", type=int, default=-1)
    parser.add_argument(
        "--item-prefix",
        default="corrective",
        help="Unique cache item prefix when combining multiple recovery roll-ins.",
    )
    parser.add_argument("--phase-length", type=int, default=198)
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=1,
        help="Number of leading FastWAM teacher actions to cache (1-32).",
    )
    parser.add_argument("--sample-weight", type=float, default=20.0)
    parser.add_argument("--layers", type=int, nargs="+", default=[9, 19, 29, 39, 49])
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase_length <= 0 or args.sample_weight <= 0:
        raise ValueError("phase-length and sample-weight must be positive")
    if not 1 <= args.action_horizon <= 32:
        raise ValueError("action-horizon must be between 1 and 32")
    sys.path.insert(0, str(args.comfy_root.resolve()))
    import comfy.model_management as model_management
    import comfy.sd
    import comfy.utils

    from fastwam.models.h3wam import (
        H3BlockFeatureCapture,
        inject_h3_attention_lora,
        libero_observation_state,
        load_h3_lora_state_dict,
        make_first_frame_payload,
        preprocess_libero_cameras,
    )

    with np.load(args.input.resolve()) as archive:
        source = {key: archive[key] for key in archive.files}
    required = {
        "step",
        "agentview_image",
        "wristview_image",
        "eef_pos",
        "eef_quat",
        "gripper_qpos",
        "teacher_actions",
    }
    missing = sorted(required - set(source))
    if missing:
        raise ValueError(f"corrective input is missing fields: {missing}")
    count = int(source["step"].shape[0])
    if any(int(source[key].shape[0]) != count for key in required):
        raise ValueError("corrective fields have inconsistent first dimensions")

    cache_root = args.cache_root.resolve()
    conditioning = torch.load(
        cache_root / "refined_contexts" / f"{args.context_id}.pt",
        map_location="cpu",
        weights_only=False,
    )
    patcher = comfy.sd.load_diffusion_model(str(args.h3_checkpoint.resolve()))
    model_management.load_models_gpu([patcher])
    model = patcher.model.diffusion_model
    resolved_lora = None
    if args.h3_lora_checkpoint is not None:
        resolved_lora = args.h3_lora_checkpoint.resolve()
        lora = torch.load(resolved_lora, map_location="cpu", weights_only=False)
        h3_lora = lora.get("h3_lora")
        if not h3_lora:
            raise ValueError("H3 LoRA checkpoint does not contain h3_lora")
        rank = int(lora["h3_lora_rank"])
        include_mlp = bool(lora.get("h3_lora_include_mlp", False))
        if include_mlp:
            model_management.in_training = True
        inject_h3_attention_lora(
            model,
            rank=rank,
            alpha=float(lora.get("h3_lora_alpha", rank)),
            last_n_blocks=int(lora["h3_lora_last_blocks"]),
            include_mlp=include_mlp,
        )
        load_h3_lora_state_dict(model, h3_lora)
    model.eval()
    vae_state = comfy.utils.load_torch_file(str(args.video_vae.resolve()))
    video_vae = comfy.sd.VAE(sd=vae_state)
    del vae_state

    device = model_management.get_torch_device()
    context = conditioning["context"].to(device=device, dtype=torch.bfloat16)
    token_tags = conditioning["token_tags"].to(device)
    layers = tuple(sorted(set(args.layers)))
    if not layers or layers[0] < 0 or layers[-1] >= len(model.blocks):
        raise ValueError(f"invalid feature layers {layers}")
    feature_root = cache_root / args.feature_subdir
    window_root = cache_root / "windows"
    feature_root.mkdir(parents=True, exist_ok=True)
    window_root.mkdir(parents=True, exist_ok=True)

    manifest_items = []
    started = time.perf_counter()
    for position in range(count):
        step = int(source["step"][position])
        item_id = f"{args.item_prefix}_ep{args.episode_id}_s{step:06d}"
        feature_path = feature_root / f"{item_id}.pt"
        window_path = window_root / f"{item_id}.pt"
        item = {
            "id": item_id,
            "episode": args.episode_id,
            "start": step,
            "length": args.phase_length,
            "task": args.task,
            "task_group": 0,
            "sample_weight": args.sample_weight,
            "corrective": True,
        }
        manifest_items.append(item)
        if feature_path.exists() and window_path.exists() and not args.overwrite:
            continue

        pixels = preprocess_libero_cameras(
            source["agentview_image"][position],
            source["wristview_image"][position],
        )
        with torch.inference_mode():
            first_frame = video_vae.encode(pixels).to(
                device=device, dtype=torch.bfloat16
            )
        model_management.load_models_gpu([patcher])
        text_len = int(context.shape[1])
        frame_rows = int(
            first_frame.shape[2]
            * (first_frame.shape[3] // 2)
            * (first_frame.shape[4] // 2)
        )
        capture = H3BlockFeatureCapture(
            layers, token_start=text_len, token_stop=text_len + frame_rows
        )
        payload = make_first_frame_payload(first_frame, frame_count=39)
        payload["text_token_tags"] = token_tags
        with torch.inference_mode():
            model(
                [
                    torch.zeros(
                        (1, 24, 12, first_frame.shape[-2], first_frame.shape[-1]),
                        device=device,
                        dtype=torch.bfloat16,
                    ),
                    torch.zeros((1, 32, 2, 1), device=device, dtype=torch.float32),
                ],
                torch.tensor([1000.0], device=device),
                context,
                transformer_options=capture.transformer_options(),
                minimax_payload=payload,
            )
        features = capture.stacked().to(device="cpu", dtype=torch.bfloat16)

        available_horizon = int(source["teacher_actions"].shape[1])
        if available_horizon < args.action_horizon:
            raise ValueError(
                f"teacher horizon {available_horizon} is shorter than requested "
                f"action horizon {args.action_horizon}"
            )
        teacher_action = torch.from_numpy(
            source["teacher_actions"][
                position, : args.action_horizon
            ].astype(np.float32, copy=True)
        ).reshape(args.action_horizon, 7)
        # FastWAM evaluator returns LIBERO gripper {-1=open,+1=close}; cached
        # training actions use {1=open,0=close}.
        teacher_action[:, -1] = (1.0 - teacher_action[:, -1]) * 0.5
        observation = {
            "eef_pos": source["eef_pos"][position],
            "eef_quat": source["eef_quat"][position],
            "gripper_qpos": source["gripper_qpos"][position],
        }
        window = {
            "actions": teacher_action,
            "state": libero_observation_state(observation),
            "task": args.task,
            "task_index": 0,
            "episode": args.episode_id,
            "start": step,
            "corrective": True,
            "teacher": "official_fastwam",
        }
        feature = {
            "features": features,
            "layers": layers,
            "token_start": text_len,
            "token_stop": text_len + frame_rows,
            "episode": args.episode_id,
            "start": step,
            "context_id": args.context_id,
            "timestep": 1000.0,
            "action_horizon": args.action_horizon,
            "h3_lora_checkpoint": (
                str(resolved_lora) if resolved_lora is not None else None
            ),
            "corrective": True,
        }
        for path, artifact in ((window_path, window), (feature_path, feature)):
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(artifact, temporary)
            temporary.replace(path)
        completed = position + 1
        if completed % args.progress_every == 0 or completed == count:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": count,
                        "seconds_per_state": round(elapsed / completed, 3),
                        "peak_allocated_gib": round(
                            torch.cuda.max_memory_allocated(device) / 2**30, 3
                        ),
                    }
                ),
                flush=True,
            )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        "".join(json.dumps(item) + "\n" for item in manifest_items),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"manifest": str(args.manifest.resolve()), "states": count},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
