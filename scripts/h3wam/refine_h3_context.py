#!/usr/bin/env python3
"""Project a cached raw MiniMax context into the H3 diffusion width."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("context", type=Path)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--h3-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy_root.resolve()))
    import comfy.model_management as model_management
    import comfy.sd

    source = torch.load(args.context, map_location="cpu", weights_only=False)
    patcher = comfy.sd.load_diffusion_model(str(args.h3_checkpoint.resolve()))
    model_management.load_models_gpu([patcher])
    model = patcher.model.diffusion_model
    device = model_management.get_torch_device()
    with torch.inference_mode():
        context = model.preprocess_text_embeds(
            source["context"].to(device=device, dtype=torch.bfloat16)
        )
    artifact = {"context": context.cpu(), "token_tags": source["token_tags"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "context_shape": list(context.shape),
                "token_tags_shape": list(source["token_tags"].shape),
            }
        )
    )


if __name__ == "__main__":
    main()
