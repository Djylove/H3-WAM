"""StarWAM ActionDiT adapter for frozen MiniMax-H3 observation features.

The ActionDiT implementation itself is imported from the pinned StarWAM
submodule.  This file only adapts H3's 5376-wide cached observation tokens and
5120-wide text/proprio context to StarWAM's feature-conditioned call contract.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional

import torch
from torch import nn


STARWAM_ACTION_DIT_SHA256 = "b6cd067cac448d8f4dba20f3778bae9bef622f58bdd854b1dfccc190d9dcf8b1"
STARWAM_WAN_BLOCK_SHA256 = "303344329ba63692616494e40dd3b2288945d329d905e38c1bdcc26af5467524"


def _verified_source(path: Path, expected_sha256: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"pinned StarWAM source hash mismatch for {path}: {actual}"
        )
    return path


def _load_pinned_starwam_action_dit():
    """Load only StarWAM's two ActionDiT source files.

    StarWAM's package ``__init__`` eagerly imports its dataset stack, including
    PyArrow.  Cached H3 training needs none of that.  Loading the pinned action
    modules directly keeps the upstream implementation byte-identical and the
    standalone runtime small.
    """

    cached = sys.modules.get("starwam.modules.action_dit")
    if cached is not None:
        return cached.ActionDiT
    repo_root = Path(__file__).resolve().parents[4]
    source_root = repo_root / "third_party/StarWAM/starwam"
    wan_block_path = _verified_source(
        source_root / "modules/wan_block.py", STARWAM_WAN_BLOCK_SHA256
    )
    action_dit_path = _verified_source(
        source_root / "modules/action_dit.py", STARWAM_ACTION_DIT_SHA256
    )
    starwam_package = sys.modules.setdefault("starwam", types.ModuleType("starwam"))
    starwam_package.__path__ = [str(source_root)]
    modules_package = sys.modules.setdefault(
        "starwam.modules", types.ModuleType("starwam.modules")
    )
    modules_package.__path__ = [str(source_root / "modules")]
    setattr(starwam_package, "modules", modules_package)

    for module_name, source_path in (
        ("starwam.modules.wan_block", wan_block_path),
        ("starwam.modules.action_dit", action_dit_path),
    ):
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load pinned StarWAM module {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        setattr(modules_package, module_name.rsplit(".", 1)[-1], module)
    return sys.modules["starwam.modules.action_dit"].ActionDiT


class H3StarWAMFeatureActionPolicy(nn.Module):
    """Frozen-feature policy that keeps StarWAM's action expert unchanged."""

    def __init__(
        self,
        *,
        action_dim: int = 7,
        proprio_dim: int = 8,
        h3_feature_dim: int = 5376,
        context_dim: int = 5120,
        hidden_dim: int = 1024,
        ffn_dim: int = 4096,
        num_heads: int = 40,
        attn_head_dim: int = 128,
        num_layers: int = 30,
        freq_dim: int = 256,
        eps: float = 1e-6,
        max_seq_len: int = 64,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        ActionDiT = _load_pinned_starwam_action_dit()
        if min(action_dim, proprio_dim, h3_feature_dim, context_dim) <= 0:
            raise ValueError("action, proprio, feature and context dimensions must be positive")
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.h3_feature_dim = int(h3_feature_dim)
        self.context_dim = int(context_dim)
        self.feature_projector = nn.Linear(h3_feature_dim, context_dim)
        self.proprio_encoder = nn.Linear(proprio_dim, context_dim)
        self.action_expert = ActionDiT(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            ffn_dim=ffn_dim,
            text_dim=context_dim,
            freq_dim=freq_dim,
            eps=eps,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    def compose_context(
        self,
        text_context: torch.Tensor,
        h3_features: torch.Tensor,
        proprio: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose ``[text, proprio, pooled H3 features]`` like StarWAM."""

        if h3_features.ndim == 4:
            if h3_features.shape[1] != 1:
                raise ValueError("StarWAM last32 expects exactly one captured H3 layer")
            h3_features = h3_features[:, 0]
        if h3_features.ndim != 3 or h3_features.shape[-1] != self.h3_feature_dim:
            raise ValueError("H3 features must be [B,T,h3_feature_dim]")
        if text_context.ndim != 3 or text_context.shape[-1] != self.context_dim:
            raise ValueError("text context must be [B,L,context_dim]")
        if proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError("proprio must be [B,proprio_dim]")
        batch = text_context.shape[0]
        if h3_features.shape[0] != batch or proprio.shape[0] != batch:
            raise ValueError("text, H3 feature and proprio batches must match")
        if text_mask is None:
            text_mask = torch.ones(
                text_context.shape[:2], dtype=torch.bool, device=text_context.device
            )
        elif tuple(text_mask.shape) != tuple(text_context.shape[:2]):
            raise ValueError("text mask must match text token dimensions")

        feature_param = self.feature_projector.weight
        features = self.feature_projector(
            h3_features.to(device=feature_param.device, dtype=feature_param.dtype)
        )
        proprio_param = self.proprio_encoder.weight
        proprio_token = self.proprio_encoder(
            proprio.to(device=proprio_param.device, dtype=proprio_param.dtype)
        ).unsqueeze(1)
        context = torch.cat(
            (
                text_context.to(device=features.device, dtype=features.dtype),
                proprio_token,
                features,
            ),
            dim=1,
        )
        mask = torch.cat(
            (
                text_mask.to(device=context.device, dtype=torch.bool),
                torch.ones((batch, 1 + features.shape[1]), dtype=torch.bool, device=context.device),
            ),
            dim=1,
        )
        return context, mask

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        text_context: torch.Tensor,
        h3_features: torch.Tensor,
        proprio: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context, context_mask = self.compose_context(
            text_context, h3_features, proprio, text_mask
        )
        return self.action_expert(noisy_actions, timestep, context, context_mask)
