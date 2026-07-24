import torch

from experiments.anygrasp.deploy_policy import FastWAMAnyGraspPolicy
from fastwam.models.wan22.fastwam_hierarchical import (
    rtc_guided_velocity,
    rtc_prefix_weights,
)


def test_rtc_prefix_weights_match_official_schedule():
    weights = rtc_prefix_weights(
        2,
        6,
        10,
        "linear",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        weights,
        torch.tensor([1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0]),
    )


def test_rtc_guidance_moves_next_scheduler_step_toward_prefix():
    latents = torch.zeros((1, 4, 2), dtype=torch.float32)
    prefix = torch.ones((1, 2, 2), dtype=torch.float32)
    guided = rtc_guided_velocity(
        action_latents=latents,
        action_timestep=torch.tensor([500.0]),
        denoiser=lambda x: x * 0.0,
        previous_action_chunk=prefix,
        inference_delay=0,
        prefix_horizon=2,
        prefix_attention_schedule="ones",
        max_guidance_weight=5.0,
        num_train_timesteps=1000,
    )

    next_latents = latents + guided * -0.1
    assert bool((next_latents[:, :2] > 0).all())
    torch.testing.assert_close(next_latents[:, 2:], torch.zeros_like(next_latents[:, 2:]))


def test_selected_action_prefix_uses_dataset_normalizer():
    class Normalizer:
        def forward(self, value):
            return value * 2.0

    class Processor:
        normalizer = type("Container", (), {"normalizers": {"action": {"default": Normalizer()}}})()

    policy = FastWAMAnyGraspPolicy.__new__(FastWAMAnyGraspPolicy)
    policy.processor = Processor()
    policy.action_key = "default"
    policy.action_dim = 31
    policy.device = "cpu"
    policy.model_dtype = torch.float32

    prefix = policy._normalize_selected_action(torch.ones((16, 31)))
    assert prefix.shape == (1, 16, 31)
    torch.testing.assert_close(prefix, torch.full_like(prefix, 2.0))


def test_frame_history_records_once_per_action_index():
    policy = FastWAMAnyGraspPolicy.__new__(FastWAMAnyGraspPolicy)
    policy._frame_history = []
    policy._last_frame_action_index = None
    policy._max_keyframe_history = 3
    frame = torch.zeros((3, 2, 2))

    policy._get_padded_frame_history(frame, action_index=0)
    policy._get_padded_frame_history(frame, action_index=0)
    padded = policy._get_padded_frame_history(frame + 1, action_index=16)

    assert len(policy._frame_history) == 2
    assert len(padded) == 3
    torch.testing.assert_close(padded[-1], frame + 1)
