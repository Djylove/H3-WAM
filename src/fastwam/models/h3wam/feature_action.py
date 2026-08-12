"""Independent action expert conditioned on frozen MiniMax H3 video tokens."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class H3MixtureActionOutput:
    actions: torch.Tensor
    mode_logits: torch.Tensor


class H3BlockFeatureCapture:
    """Capture selected packed-token rows after chosen H3 transformer blocks.

    Cached-feature generation and inference should keep ``detach=True`` (the
    default).  End-to-end feature-policy adaptation can set it to ``False`` so
    an action loss reaches LoRA branches inside H3.
    """

    def __init__(
        self,
        layer_indices: Iterable[int],
        token_start: int,
        token_stop: int,
        *,
        detach: bool = True,
    ):
        self.layer_indices = tuple(sorted({int(index) for index in layer_indices}))
        self.token_start = int(token_start)
        self.token_stop = int(token_stop)
        self.detach = bool(detach)
        if not self.layer_indices:
            raise ValueError("at least one H3 layer must be selected")
        if self.token_start < 0 or self.token_stop <= self.token_start:
            raise ValueError("invalid H3 token slice")
        self.features: dict[int, torch.Tensor] = {}

    def clear(self) -> None:
        self.features.clear()

    def _replacement(self, layer_index: int):
        def replace(args: dict, extra: dict) -> dict:
            result = extra["original_block"](args)
            hidden = result["img"]
            if hidden.ndim != 2 or hidden.shape[0] < self.token_stop:
                raise ValueError(
                    f"H3 hidden tokens cannot satisfy slice "
                    f"[{self.token_start}:{self.token_stop}], got {tuple(hidden.shape)}"
                )
            feature = hidden[self.token_start : self.token_stop]
            self.features[layer_index] = feature.detach() if self.detach else feature
            return result

        return replace

    def transformer_options(self) -> dict:
        self.clear()
        replacements = {
            ("double_block", index): self._replacement(index)
            for index in self.layer_indices
        }
        return {"patches_replace": {"dit": replacements}}

    def stacked(self) -> torch.Tensor:
        missing = [index for index in self.layer_indices if index not in self.features]
        if missing:
            raise RuntimeError(f"H3 feature capture missed layers {missing}")
        return torch.stack([self.features[index] for index in self.layer_indices], dim=0)


class H3FeatureActionTransformer(nn.Module):
    """Small action diffusion/regression expert with layerwise H3 cross-attention."""

    def __init__(
        self,
        *,
        action_dim: int,
        state_dim: int,
        h3_feature_dim: int = 5376,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        max_horizon: int = 64,
        time_dim: int = 128,
        num_action_modes: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.h3_feature_dim = int(h3_feature_dim)
        self.time_dim = int(time_dim)
        self.num_action_modes = int(num_action_modes)
        if self.num_action_modes <= 0:
            raise ValueError("num_action_modes must be positive")
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(h3_feature_dim),
            nn.Linear(h3_feature_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position = nn.Parameter(torch.randn(1, max_horizon, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ffn_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim * self.num_action_modes)
        self.mode_head = (
            nn.Linear(hidden_dim, self.num_action_modes)
            if self.num_action_modes > 1
            else None
        )

    def _time_embedding(self, sigma: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=sigma.device, dtype=torch.float32)
            / half
        )
        angles = sigma.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        *,
        state: torch.Tensor,
        h3_features: torch.Tensor,
        video_sigma: torch.Tensor,
    ) -> torch.Tensor | H3MixtureActionOutput:
        batch, horizon, action_dim = noisy_actions.shape
        if action_dim != self.action_dim:
            raise ValueError(f"expected action_dim {self.action_dim}, got {action_dim}")
        if horizon > self.position.shape[1]:
            raise ValueError("action horizon exceeds learned positional capacity")
        if state.ndim == 2:
            state = state.unsqueeze(1).expand(-1, horizon, -1)
        expected_state = (batch, horizon, self.state_dim)
        if tuple(state.shape) != expected_state:
            raise ValueError(
                f"state must have shape {expected_state}, got {tuple(state.shape)}"
            )
        if h3_features.ndim == 4:
            # [B, selected_layers, tokens, H3_dim]. Each action layer receives
            # the nearest available H3 block; the final memory is never pooled.
            if h3_features.shape[0] != batch:
                raise ValueError("H3 feature batch does not match action batch")
            feature_layers = h3_features
        elif h3_features.ndim == 3:
            if h3_features.shape[0] != batch:
                raise ValueError("H3 feature batch does not match action batch")
            feature_layers = h3_features.unsqueeze(1)
        else:
            raise ValueError("h3_features must be [B,S,D] or [B,L,S,D]")
        if feature_layers.shape[-1] != self.h3_feature_dim:
            raise ValueError(
                f"expected H3 feature dim {self.h3_feature_dim}, "
                f"got {feature_layers.shape[-1]}"
            )

        hidden = (
            self.action_projection(noisy_actions.float())
            + self.state_projection(state.float())
            + self.position[:, :horizon]
            + self.time_projection(self._time_embedding(video_sigma)).unsqueeze(1)
        )
        for layer_index, layer in enumerate(self.layers):
            memory_index = round(
                layer_index
                * (feature_layers.shape[1] - 1)
                / max(len(self.layers) - 1, 1)
            )
            memory = self.feature_projection(
                feature_layers[:, memory_index].float()
            )
            hidden = layer(hidden, memory)
        hidden = self.output_norm(hidden)
        output = self.output(hidden)
        if self.num_action_modes == 1:
            return output
        actions = output.reshape(
            batch, horizon, self.num_action_modes, self.action_dim
        ).permute(0, 2, 1, 3)
        assert self.mode_head is not None
        mode_logits = self.mode_head(hidden).mean(dim=1)
        return H3MixtureActionOutput(actions=actions, mode_logits=mode_logits)


class H3MultiLayerActionTransformer(nn.Module):
    """Flow-matching action head with learned fusion over H3 backbone depths.

    Unlike :class:`H3FeatureActionTransformer`, which assigns one cached H3
    layer to each decoder layer, this head learns a soft distribution over all
    captured H3 layers.  A shared feature projection keeps the docking module
    small while still allowing every action layer to choose a different mix of
    low-, mid- and high-level world-model representations.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        state_dim: int,
        num_h3_layers: int,
        h3_feature_dim: int = 5376,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        max_horizon: int = 64,
        time_dim: int = 128,
        language_feature_dim: int | None = None,
        layer_mix_initialization: str = "spaced",
        history_conditioning: bool = False,
        history_adapter_rank: int = 0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_h3_layers <= 0 or num_layers <= 0:
            raise ValueError("H3 and action layer counts must be positive")
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.num_h3_layers = int(num_h3_layers)
        self.h3_feature_dim = int(h3_feature_dim)
        self.time_dim = int(time_dim)
        self.language_feature_dim = (
            None if language_feature_dim is None else int(language_feature_dim)
        )
        self.history_conditioning = bool(history_conditioning)
        self.history_adapter_rank = int(history_adapter_rank)
        if self.history_adapter_rank < 0:
            raise ValueError("history adapter rank cannot be negative")
        if self.history_adapter_rank and not self.history_conditioning:
            raise ValueError("history adapter requires history conditioning")
        if self.language_feature_dim is not None and self.language_feature_dim <= 0:
            raise ValueError("language feature dimension must be positive")
        if layer_mix_initialization not in {"spaced", "uniform"}:
            raise ValueError(
                "layer_mix_initialization must be 'spaced' or 'uniform'"
            )
        self.layer_mix_initialization = layer_mix_initialization
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(h3_feature_dim),
            nn.Linear(h3_feature_dim, hidden_dim),
        )
        self.language_projection = (
            None
            if self.language_feature_dim is None
            else nn.Sequential(
                nn.LayerNorm(self.language_feature_dim),
                nn.Linear(self.language_feature_dim, hidden_dim),
            )
        )
        self.time_projection = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position = nn.Parameter(torch.randn(1, max_horizon, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ffn_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        initial_logits = torch.zeros((num_layers, num_h3_layers))
        if layer_mix_initialization == "spaced":
            # Historical initialization retained for checkpoint compatibility.
            # It is intentionally opt-in for new runs: a four-logit gap gives
            # the selected layer about 86% mass with ten captured layers and
            # was empirically too strong for the small routing gradients.
            initial_logits.fill_(-2.0)
            for action_layer in range(num_layers):
                h3_layer = round(
                    action_layer * (num_h3_layers - 1) / max(num_layers - 1, 1)
                )
                initial_logits[action_layer, h3_layer] = 2.0
        self.layer_mix_logits = nn.Parameter(initial_logits)
        self.history_gate = (
            nn.Parameter(torch.zeros((num_layers, hidden_dim)))
            if self.history_conditioning
            else None
        )
        self.history_down = (
            nn.ModuleList(
                [
                    nn.Linear(hidden_dim, self.history_adapter_rank, bias=False)
                    for _ in range(num_layers)
                ]
            )
            if self.history_adapter_rank
            else None
        )
        self.history_up = (
            nn.ModuleList(
                [
                    nn.Linear(self.history_adapter_rank, hidden_dim, bias=False)
                    for _ in range(num_layers)
                ]
            )
            if self.history_adapter_rank
            else None
        )
        if self.history_up is not None:
            for projection in self.history_up:
                nn.init.zeros_(projection.weight)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def _time_embedding(self, sigma: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=sigma.device, dtype=torch.float32)
            / half
        )
        angles = sigma.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        *,
        state: torch.Tensor,
        h3_features: torch.Tensor,
        action_timestep: torch.Tensor,
        language_features: torch.Tensor | None = None,
        history_h3_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, horizon, action_dim = noisy_actions.shape
        if action_dim != self.action_dim:
            raise ValueError(f"expected action_dim {self.action_dim}, got {action_dim}")
        if horizon > self.position.shape[1]:
            raise ValueError("action horizon exceeds learned positional capacity")
        if state.ndim == 2:
            state = state.unsqueeze(1).expand(-1, horizon, -1)
        expected_state = (batch, horizon, self.state_dim)
        if tuple(state.shape) != expected_state:
            raise ValueError(
                f"state must have shape {expected_state}, got {tuple(state.shape)}"
            )
        expected_prefix = (batch, self.num_h3_layers)
        if h3_features.ndim != 4 or tuple(h3_features.shape[:2]) != expected_prefix:
            raise ValueError(
                "h3_features must be [B,num_h3_layers,tokens,H3_dim], got "
                f"{tuple(h3_features.shape)}"
            )
        if h3_features.shape[-1] != self.h3_feature_dim:
            raise ValueError(
                f"expected H3 feature dim {self.h3_feature_dim}, "
                f"got {h3_features.shape[-1]}"
            )
        if self.history_conditioning:
            if history_h3_features is None or tuple(history_h3_features.shape) != tuple(
                h3_features.shape
            ):
                raise ValueError(
                    "history_h3_features must match h3_features when history "
                    "conditioning is enabled"
                )
        elif history_h3_features is not None:
            raise ValueError(
                "history_h3_features were provided to a head without history "
                "conditioning"
            )
        projected_language = None
        if self.language_projection is not None:
            if (
                language_features is None
                or language_features.ndim != 3
                or language_features.shape[0] != batch
                or language_features.shape[-1] != self.language_feature_dim
            ):
                raise ValueError(
                    "language_features must be "
                    f"[B,T,{self.language_feature_dim}] when explicit language "
                    "conditioning is enabled"
                )
            projected_language = self.language_projection(language_features.float())
        elif language_features is not None:
            raise ValueError(
                "language_features were provided to a head without language projection"
            )

        hidden = (
            self.action_projection(noisy_actions.float())
            + self.state_projection(state.float())
            + self.position[:, :horizon]
            + self.time_projection(self._time_embedding(action_timestep)).unsqueeze(1)
        )
        projected_features = self.feature_projection(h3_features.float())
        projected_history_features = (
            None
            if history_h3_features is None
            else self.feature_projection(history_h3_features.float())
        )
        layer_weights = self.layer_mix_logits.softmax(dim=-1)
        for layer_index, layer in enumerate(self.layers):
            memory = torch.einsum(
                "l,blsd->bsd", layer_weights[layer_index], projected_features
            )
            if projected_history_features is not None:
                history_memory = torch.einsum(
                    "l,blsd->bsd",
                    layer_weights[layer_index],
                    projected_history_features,
                )
                assert self.history_gate is not None
                gate = self.history_gate[layer_index].tanh().reshape(1, 1, -1)
                history_delta = history_memory - memory
                memory = memory + gate * history_delta
                if self.history_down is not None and self.history_up is not None:
                    memory = memory + self.history_up[layer_index](
                        F.gelu(self.history_down[layer_index](history_delta))
                    )
            if projected_language is not None:
                memory = torch.cat((projected_language, memory), dim=1)
            hidden = layer(hidden, memory)
        return self.output(self.output_norm(hidden))


class H3FeatureSwitchGate(nn.Module):
    """State gate over pooled H3 tokens and normalized proprioception.

    The gate deliberately receives no timestep or phase input.  It must infer
    whether recovery is appropriate from the current visual/proprioceptive
    state, and deployment can latch the recovery branch after the first hit.
    """

    def __init__(
        self,
        *,
        h3_feature_dim: int = 5376,
        state_dim: int = 8,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.h3_feature_dim = int(h3_feature_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        if self.h3_feature_dim <= 0 or self.state_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("gate dimensions must be positive")
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(self.h3_feature_dim),
            nn.Linear(self.h3_feature_dim, self.hidden_dim),
        )
        self.state_projection = nn.Linear(self.state_dim, self.hidden_dim)
        self.output = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self, h3_features: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        if h3_features.ndim == 4:
            pooled = h3_features.float().mean(dim=(1, 2))
        elif h3_features.ndim == 3:
            pooled = h3_features.float().mean(dim=1)
        elif h3_features.ndim == 2:
            pooled = h3_features.float()
        else:
            raise ValueError(
                "h3_features must be [B,D], [B,S,D] or [B,L,S,D]"
            )
        if pooled.shape[-1] != self.h3_feature_dim:
            raise ValueError(
                f"expected H3 feature dim {self.h3_feature_dim}, "
                f"got {pooled.shape[-1]}"
            )
        if state.ndim != 2 or state.shape != (pooled.shape[0], self.state_dim):
            raise ValueError(
                f"state must have shape {(pooled.shape[0], self.state_dim)}, "
                f"got {tuple(state.shape)}"
            )
        hidden = self.feature_projection(pooled) + self.state_projection(state.float())
        return self.output(hidden).squeeze(-1)
