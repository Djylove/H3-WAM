"""Robot-action adapters for MiniMax H3's 32-channel audio latent slot."""

from __future__ import annotations

import torch
from torch import nn


class H3ActionAdapter(nn.Module):
    """Map action chunks to and from H3's stereo audio-latent layout.

    H3 represents audio as ``[batch, 32, 2, time]``.  The adapter maps each
    robot action to both H3 streams jointly instead of assuming a particular
    embodiment-specific split such as left arm / right arm.  This keeps the
    first experiment compatible with both 7-D LIBERO and 14-D RoboTwin
    actions.  A structured arm split can be tested later as an ablation.
    """

    def __init__(
        self,
        action_dim: int,
        *,
        state_dim: int = 0,
        latent_dim: int = 32,
        num_streams: int = 2,
        hidden_dim: int = 128,
        context_dim: int = 5376,
        direct_conditioning: bool = False,
        direct_action_residual: bool = False,
    ) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if state_dim < 0:
            raise ValueError(f"state_dim must be non-negative, got {state_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if num_streams <= 0:
            raise ValueError(f"num_streams must be positive, got {num_streams}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.num_streams = int(num_streams)
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.direct_conditioning = bool(direct_conditioning)
        self.direct_action_residual = bool(direct_action_residual)
        flattened_latent_dim = self.latent_dim * self.num_streams

        encoder_input_dim = self.action_dim + self.state_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(encoder_input_dim),
            nn.Linear(encoder_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, flattened_latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(flattened_latent_dim),
            nn.Linear(flattened_latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        if self.direct_conditioning:
            if self.state_dim <= 0:
                raise ValueError("direct conditioning requires a positive state_dim")
            if self.context_dim <= 0:
                raise ValueError("direct conditioning requires a positive context_dim")
            self.decoder_state_projection = nn.Linear(
                self.state_dim, flattened_latent_dim, bias=False
            )
            self.decoder_context_projection = nn.Sequential(
                nn.LayerNorm(self.context_dim),
                nn.Linear(self.context_dim, flattened_latent_dim, bias=False),
            )
            nn.init.zeros_(self.decoder_state_projection.weight)
            nn.init.zeros_(self.decoder_context_projection[-1].weight)
        if self.direct_action_residual:
            if self.state_dim <= 0:
                raise ValueError("direct action residual requires a positive state_dim")
            self.decoder_action_residual = nn.Sequential(
                nn.Linear(self.state_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.action_dim),
            )
            nn.init.zeros_(self.decoder_action_residual[-1].weight)
            nn.init.zeros_(self.decoder_action_residual[-1].bias)

    def encode_actions(
        self,
        actions: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode ``[B, T, action_dim]`` to H3 ``[B, 32, 2, T]``."""

        self._validate_actions(actions)
        batch, horizon, _ = actions.shape
        encoder_input = actions
        if self.state_dim:
            if state is None:
                raise ValueError(f"state with dimension {self.state_dim} is required")
            if state.ndim == 2:
                state = state.unsqueeze(1).expand(-1, horizon, -1)
            expected = (batch, horizon, self.state_dim)
            if tuple(state.shape) != expected:
                raise ValueError(f"state must have shape {expected}, got {tuple(state.shape)}")
            encoder_input = torch.cat(
                (actions, state.to(device=actions.device, dtype=actions.dtype)),
                dim=-1,
            )
        elif state is not None:
            raise ValueError("this adapter was created without state conditioning")
        latents = self.encoder(encoder_input)
        latents = latents.reshape(batch, horizon, self.num_streams, self.latent_dim)
        return latents.permute(0, 3, 2, 1).contiguous()

    def decode_velocity(
        self,
        latent_velocity: torch.Tensor,
        state: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode H3 latent velocity to ``[B, T, action_dim]``."""

        self._validate_latents(latent_velocity)
        batch, _, _, horizon = latent_velocity.shape
        flattened = latent_velocity.permute(0, 3, 2, 1).reshape(batch, horizon, -1)
        if self.direct_conditioning:
            if state is None:
                raise ValueError("direct decoder state is required")
            if state.ndim == 2:
                if tuple(state.shape) != (batch, self.state_dim):
                    raise ValueError(
                        f"direct decoder state must have shape {(batch, self.state_dim)}, "
                        f"got {tuple(state.shape)}"
                    )
                direct = self.decoder_state_projection(state.float()).unsqueeze(1)
            elif state.ndim == 3:
                expected = (batch, horizon, self.state_dim)
                if tuple(state.shape) != expected:
                    raise ValueError(
                        f"direct decoder state must have shape {expected}, "
                        f"got {tuple(state.shape)}"
                    )
                direct = self.decoder_state_projection(state.float())
            else:
                raise ValueError(
                    "direct decoder state must have shape [B, D] or [B, T, D], "
                    f"got {tuple(state.shape)}"
                )
            if context is None or context.ndim != 3 or context.shape[0] != batch:
                raise ValueError(
                    "direct decoder context must have shape [batch, tokens, context_dim]"
                )
            if context.shape[-1] != self.context_dim:
                raise ValueError(
                    f"expected context_dim={self.context_dim}, got {context.shape[-1]}"
                )
            direct = direct + self.decoder_context_projection(
                context.float().mean(dim=1)
            ).unsqueeze(1)
            flattened = flattened + direct.to(flattened.dtype)
        decoded = self.decoder(flattened)
        if self.direct_action_residual:
            if state is None:
                raise ValueError("direct action residual state is required")
            if state.ndim == 2:
                residual_state = state.unsqueeze(1).expand(-1, horizon, -1)
            elif state.ndim == 3:
                residual_state = state
            else:
                raise ValueError("direct action residual state must be [B, D] or [B, T, D]")
            expected = (batch, horizon, self.state_dim)
            if tuple(residual_state.shape) != expected:
                raise ValueError(
                    f"direct action residual state must have shape {expected}, "
                    f"got {tuple(residual_state.shape)}"
                )
            decoded = decoded + self.decoder_action_residual(residual_state.float())
        return decoded

    def forward(
        self,
        actions: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoencode actions; useful for adapter-only smoke tests."""

        if self.direct_conditioning:
            raise ValueError("forward() needs context when direct conditioning is enabled")
        return self.decode_velocity(self.encode_actions(actions, state))

    def _validate_actions(self, actions: torch.Tensor) -> None:
        if actions.ndim != 3:
            raise ValueError(
                "actions must have shape [batch, horizon, action_dim], "
                f"got {tuple(actions.shape)}"
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected action_dim={self.action_dim}, got {actions.shape[-1]}"
            )

    def _validate_latents(self, latents: torch.Tensor) -> None:
        if latents.ndim != 4:
            raise ValueError(
                "latent velocity must have shape [batch, latent_dim, streams, horizon], "
                f"got {tuple(latents.shape)}"
            )
        expected = (self.latent_dim, self.num_streams)
        actual = (latents.shape[1], latents.shape[2])
        if actual != expected:
            raise ValueError(f"expected latent channel/stream shape {expected}, got {actual}")
