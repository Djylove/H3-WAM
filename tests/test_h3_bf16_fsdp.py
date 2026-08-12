from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "h3wam" / "train_h3_bf16_fsdp.py"
SPEC = importlib.util.spec_from_file_location("train_h3_bf16_fsdp", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_audio_latent_count_matches_h3_rate() -> None:
    assert MODULE.audio_latent_count(24) == 40
    assert MODULE.audio_latent_count(22) == 37


def test_validate_cached_sample_accepts_raw_context() -> None:
    window = {
        "video_latents": torch.zeros(1, 24, 7, 14, 28),
        "first_frame_latents": torch.zeros(1, 24, 1, 14, 28),
    }
    conditioning = {
        "context": torch.zeros(1, 115, 5120),
        "token_tags": torch.cat(
            [torch.zeros(20, dtype=torch.long), torch.ones(95, dtype=torch.long)]
        ),
    }
    MODULE.validate_cached_sample(window, conditioning)


def test_validate_cached_sample_rejects_comfy_refined_context() -> None:
    window = {
        "video_latents": torch.zeros(1, 24, 7, 14, 28),
        "first_frame_latents": torch.zeros(1, 24, 1, 14, 28),
    }
    conditioning = {
        "context": torch.zeros(1, 115, 5376),
        "token_tags": torch.ones(115, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="5120"):
        MODULE.validate_cached_sample(window, conditioning)


def test_set_trainable_tail_only_selects_final_blocks() -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj_in = torch.nn.Linear(2, 2)
            self.transformer_blocks = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2) for _ in range(4)]
            )
            self.proj_out = torch.nn.Linear(2, 2)

    model = Tiny()
    names = MODULE.set_trainable_tail(model, 2)
    assert names
    assert all(
        name.startswith(("transformer_blocks.2.", "transformer_blocks.3."))
        for name in names
    )
    assert not model.proj_in.weight.requires_grad
    assert not model.transformer_blocks[1].weight.requires_grad
    assert model.transformer_blocks[2].weight.requires_grad
    assert not model.proj_out.weight.requires_grad


def test_replicated_modules_exclude_large_block_stack() -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj_in = torch.nn.Linear(2, 2)
            self.transformer_blocks = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2) for _ in range(4)]
            )
            self.proj_out = torch.nn.Linear(2, 2)

    model = Tiny()
    modules = MODULE.replicated_non_block_modules(model)
    assert modules == [model.proj_in, model.proj_out]
    assert model.transformer_blocks not in modules
