#!/usr/bin/env python3
"""Exercise the official H3 forward/backward through the intended FSDP stack."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3TransformerBlock,
)
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    patchify_video_latents,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    generator = torch.Generator(device=device).manual_seed(1000 + rank)

    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=4,
        attention_head_dim=16,
        hidden_size=64,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=128,
        in_channels=4,
        audio_in_channels=8,
        patch_size=(1, 2, 2),
        text_dim=32,
        freq_dim=16,
        time_embed_hidden_dim=64,
        time_embed_dim=32,
        rope_freq_dim=2,
    )
    for name, module in model.named_children():
        if name not in {
            "proj_in",
            "audio_proj_in",
            "time_proj",
            "time_embedder",
            "rope",
            "proj_out",
            "audio_proj_out",
        }:
            module.to(torch.bfloat16)
    model.requires_grad_(False)
    model.transformer_blocks[-1].requires_grad_(True)
    model.enable_gradient_checkpointing()
    replicated_modules = [
        child
        for name, child in model.named_children()
        if name != "transformer_blocks"
    ]
    for module in replicated_modules:
        module.to(device)
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({MiniMaxH3TransformerBlock}),
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
        ignored_modules=replicated_modules,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)

    clean = torch.randn((1, 4, 2, 4, 4), generator=generator, device=device)
    first = clean[:, :, :1]
    noise = torch.randn(clean.shape, generator=generator, device=device)
    timestep = 0.2
    noisy = timestep * clean + (1.0 - timestep) * noise
    condition_noise = torch.randn(first.shape, generator=generator, device=device)
    condition = 0.999 * first + 0.001 * condition_noise
    target = patchify_video_latents(clean - noise, (1, 2, 2))[None]
    video_rows = torch.cat(
        [
            patchify_video_latents(condition, (1, 2, 2)),
            patchify_video_latents(noisy, (1, 2, 2)),
        ]
    )[None]
    context = torch.randn((1, 5, 32), generator=generator, device=device)
    text_tags = torch.ones(5, dtype=torch.long)
    layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_tags, 2, 4, 4, 2, (1, 2, 2), 2, 2, 0, ("first",)
    )
    position_ids, token_tags, video_indices, audio_indices, text_indices, ncv, nca = layout
    unique_t, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices, audio_indices, ncv, nca, text_indices.numel(), timestep, 0.0, 0.999, 1.0
    )
    audio = torch.randn((1, 4, 8), generator=generator, device=device)

    output = model(
        hidden_states=video_rows,
        audio_hidden_states=audio,
        encoder_hidden_states=context,
        timestep=unique_t.to(device),
        timestep_indices=timestep_indices.to(device),
        token_tags=token_tags.to(device),
        position_ids=position_ids.to(device),
        video_indices=video_indices.to(device),
        audio_indices=audio_indices.to(device),
        text_indices=text_indices.to(device),
    )
    loss = F.mse_loss(output.sample[:, ncv:].float(), target.float())
    loss.backward()
    grad_norm = model.clip_grad_norm_(1.0)
    optimizer.step()
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(grad_norm)):
        raise FloatingPointError(f"non-finite result: loss={loss}, grad_norm={grad_norm}")
    values = torch.tensor([loss.item(), grad_norm.item()], device=device)
    dist.all_reduce(values)
    values /= dist.get_world_size()
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "world_size": dist.get_world_size(),
                    "loss": float(values[0]),
                    "gradient_norm": float(values[1]),
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
