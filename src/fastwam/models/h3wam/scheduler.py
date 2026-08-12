"""Flow-matching schedule for robot actions carried by H3's audio slot."""

from __future__ import annotations

import torch


class H3ActionFlowScheduler:
    """Match MiniMax H3's coupled video/audio sigma schedules.

    ComfyUI drives H3 with the video sigma (shift 12 by default), while H3's
    audio tokens use the same base time with shift 3.  Since actions replace
    audio latents in H3-WAM, action noising and velocity targets must follow
    the audio schedule.  Returned velocities are expressed with respect to
    the video sigma, hence the derivative factor in :meth:`training_target`.
    """

    def __init__(
        self,
        *,
        video_shift: float = 12.0,
        action_shift: float = 3.0,
        timestep_scale: float = 1000.0,
    ) -> None:
        if video_shift <= 0:
            raise ValueError(f"video_shift must be positive, got {video_shift}")
        if action_shift <= 0:
            raise ValueError(f"action_shift must be positive, got {action_shift}")
        if timestep_scale <= 0:
            raise ValueError(f"timestep_scale must be positive, got {timestep_scale}")
        self.video_shift = float(video_shift)
        self.action_shift = float(action_shift)
        self.timestep_scale = float(timestep_scale)

    @staticmethod
    def shift(base_sigma: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * base_sigma / (1.0 + (shift - 1.0) * base_sigma)

    @staticmethod
    def unshift(sigma: torch.Tensor, shift: float) -> torch.Tensor:
        return sigma / (shift + sigma * (1.0 - shift))

    def sample_training_sigmas(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return coupled ``(video_sigma, action_sigma)`` for a batch."""

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        base = torch.rand(batch_size, device=device, dtype=torch.float32)
        video = self.shift(base, self.video_shift)
        action = self.shift(base, self.action_shift)
        return video.to(dtype=dtype), action.to(dtype=dtype)

    def action_sigma(self, video_sigma: torch.Tensor) -> torch.Tensor:
        base = self.unshift(video_sigma, self.video_shift)
        return self.shift(base, self.action_shift)

    def action_slope(self, video_sigma: torch.Tensor) -> torch.Tensor:
        """Return ``d(action_sigma) / d(video_sigma)`` analytically."""

        base = self.unshift(video_sigma, self.video_shift)
        numerator = self.action_shift * (
            1.0 + (self.video_shift - 1.0) * base
        ).square()
        denominator = self.video_shift * (
            1.0 + (self.action_shift - 1.0) * base
        ).square()
        return numerator / denominator

    def timestep(self, video_sigma: torch.Tensor) -> torch.Tensor:
        return video_sigma * self.timestep_scale

    def inference_schedule(
        self,
        model_evaluations: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return video sigmas and Euler deltas ending at clean sigma zero."""

        if model_evaluations <= 0:
            raise ValueError(f"model_evaluations must be positive, got {model_evaluations}")
        base = torch.linspace(
            1.0,
            0.0,
            model_evaluations + 1,
            device=device,
            dtype=torch.float32,
        )
        sigmas = self.shift(base, self.video_shift)
        return sigmas[:-1].to(dtype=dtype), (sigmas[1:] - sigmas[:-1]).to(dtype=dtype)

    def action_inference_delta(
        self,
        video_sigma: torch.Tensor,
        video_delta: torch.Tensor,
    ) -> torch.Tensor:
        """Exact action-sigma step coupled to one video-sigma step."""

        return self.action_sigma(video_sigma + video_delta) - self.action_sigma(
            video_sigma
        )

    def add_action_noise(
        self,
        actions: torch.Tensor,
        noise: torch.Tensor,
        video_sigma: torch.Tensor,
    ) -> torch.Tensor:
        if actions.shape != noise.shape:
            raise ValueError(
                f"actions and noise must have the same shape, got {actions.shape} and {noise.shape}"
            )
        sigma = self._broadcast(self.action_sigma(video_sigma), actions)
        return (1.0 - sigma) * actions + sigma * noise

    def add_video_noise(
        self,
        video_latents: torch.Tensor,
        noise: torch.Tensor,
        video_sigma: torch.Tensor,
    ) -> torch.Tensor:
        if video_latents.shape != noise.shape:
            raise ValueError(
                "video latents and noise must have the same shape, "
                f"got {video_latents.shape} and {noise.shape}"
            )
        sigma = self._broadcast(video_sigma, video_latents)
        return (1.0 - sigma) * video_latents + sigma * noise

    def training_target(
        self,
        actions: torch.Tensor,
        noise: torch.Tensor,
        video_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity target in H3's externally visible video-sigma domain."""

        if actions.shape != noise.shape:
            raise ValueError(
                f"actions and noise must have the same shape, got {actions.shape} and {noise.shape}"
            )
        slope = self._broadcast(self.action_slope(video_sigma), actions)
        return slope * (noise - actions)

    @staticmethod
    def video_training_target(video_latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if video_latents.shape != noise.shape:
            raise ValueError(
                "video latents and noise must have the same shape, "
                f"got {video_latents.shape} and {noise.shape}"
            )
        return noise - video_latents

    @staticmethod
    def _broadcast(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if values.ndim == 0:
            return values.to(device=reference.device, dtype=reference.dtype)
        if values.ndim != 1 or values.shape[0] != reference.shape[0]:
            raise ValueError(
                "sigma must be scalar or [batch], "
                f"got {tuple(values.shape)} for batch {reference.shape[0]}"
            )
        shape = (values.shape[0],) + (1,) * (reference.ndim - 1)
        return values.to(device=reference.device, dtype=reference.dtype).view(shape)
