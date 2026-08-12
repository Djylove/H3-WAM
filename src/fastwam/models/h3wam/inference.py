"""Few-step joint video/action sampling for H3-WAM deployment tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .bridge import H3ActionBridge
from .scheduler import H3ActionFlowScheduler


@dataclass
class H3WAMSample:
    actions: torch.Tensor
    video_latents: torch.Tensor


@torch.inference_mode()
def sample_h3wam_actions(
    bridge: H3ActionBridge,
    *,
    context: torch.Tensor,
    state: torch.Tensor | None,
    scheduler: H3ActionFlowScheduler,
    action_shape: tuple[int, int, int],
    video_shape: tuple[int, int, int, int, int],
    model_evaluations: int,
    minimax_payload: dict[str, Any] | None = None,
    generator: torch.Generator | None = None,
    initial_action_noise: torch.Tensor | None = None,
    initial_video_noise: torch.Tensor | None = None,
) -> H3WAMSample:
    """Euler-integrate H3's coupled ODE using 2–4 model evaluations."""

    device = context.device
    if initial_action_noise is None:
        actions = torch.randn(
            action_shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
    else:
        if tuple(initial_action_noise.shape) != action_shape:
            raise ValueError("initial action noise shape does not match action_shape")
        actions = initial_action_noise.to(device=device, dtype=torch.float32).clone()
    if initial_video_noise is None:
        video = torch.randn(
            video_shape,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
    else:
        if tuple(initial_video_noise.shape) != video_shape:
            raise ValueError("initial video noise shape does not match video_shape")
        video = initial_video_noise.to(device=device, dtype=torch.bfloat16).clone()

    sigmas, deltas = scheduler.inference_schedule(
        model_evaluations,
        device=device,
        dtype=torch.float32,
    )
    was_training = bridge.training
    bridge.eval()
    try:
        for sigma, delta in zip(sigmas, deltas):
            timestep = scheduler.timestep(sigma).reshape(1)
            output = bridge(
                video_latents=video,
                noisy_actions=actions,
                timestep=timestep,
                context=context,
                state=state,
                minimax_payload=minimax_payload,
            )
            # The model is trained to emit d(action)/d(video_sigma), while the
            # action corruption path is linear in action_sigma.  Reparameterize
            # the vector field and integrate on its natural action-sigma grid;
            # coarse Euler steps on the shifted video grid over-shoot by up to
            # video_shift/action_shift (4x for H3) near pure noise.
            action_delta = scheduler.action_inference_delta(sigma, delta)
            action_slope = scheduler.action_slope(sigma)
            actions = actions + (output.action_velocity / action_slope) * action_delta.to(
                actions.dtype
            )
            video = video + output.video_velocity * delta.to(video.dtype)
    finally:
        bridge.train(was_training)
    return H3WAMSample(actions=actions, video_latents=video)
