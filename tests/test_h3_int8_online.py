import unittest
from types import SimpleNamespace

import torch
from torch import nn

from fastwam.models.h3wam import (
    H3Int8LayoutFunctions,
    H3Int8OnlineFeatureContract,
    H3Int8OnlineFeatureProvider,
    H3Int8OnlineKVContract,
    H3Int8OnlineKVProvider,
    encode_h3_vae_condition_standalone,
    pool_online_kv_tokens,
)


def _fake_build_packed_sequence(**kwargs):
    text_tags = kwargs["text_token_tags"]
    text_count = int(text_tags.numel())
    condition_count = (kwargs["latent_height"] // 2) * (kwargs["latent_width"] // 2)
    audio_count = kwargs["num_audio_latents"] * kwargs["audio_channels"]
    video_count = kwargs["num_latent_frames"] * condition_count
    condition_start = text_count
    audio_start = condition_start + condition_count
    video_start = audio_start + audio_count
    sequence = video_start + video_count
    text_indices = torch.arange(text_count)
    condition_indices = torch.arange(condition_start, audio_start)
    target_video_indices = torch.arange(video_start, sequence)
    video_indices = torch.cat((condition_indices, target_video_indices))
    audio_indices = torch.arange(audio_start, video_start)
    tags = torch.empty(sequence, dtype=torch.long)
    tags[text_indices] = text_tags
    tags[video_indices] = 0
    tags[audio_indices] = 2
    return (
        torch.zeros(sequence, 3, dtype=torch.float64), tags, video_indices,
        audio_indices, text_indices, condition_count, 0,
    )


def _fake_build_row_timesteps(**kwargs):
    sequence = int(
        kwargs["video_indices"].numel()
        + kwargs["audio_indices"].numel()
        + kwargs["num_text_tokens"]
    )
    inverse = torch.zeros(sequence, dtype=torch.long)
    inverse[kwargs["video_indices"][: kwargs["num_condition_video_rows"]]] = 1
    return torch.tensor(
        [kwargs["video_timestep"], kwargs["condition_video_timestep"]]
    ), inverse


def _fake_patchify(latents, _patch_size):
    batch, channels, frames, height, width = latents.shape
    return torch.zeros(batch * frames * (height // 2) * (width // 2), channels * 4)


FAKE_LAYOUT = H3Int8LayoutFunctions(
    _fake_build_packed_sequence, _fake_build_row_timesteps, _fake_patchify
)


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_call = None

    def forward(self, **kwargs):
        self.last_call = kwargs
        count = int(kwargs["capture_indices"].numel())
        captures = {
            layer: torch.full((1, count, 4), float(layer))
            for layer in kwargs["capture_layers"]
        }
        return SimpleNamespace(captured_features=captures)


class _FakeKVBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_call = None

    def forward(self, **kwargs):
        self.last_call = kwargs
        count = int(kwargs["kv_capture_indices"].numel())
        captures = {
            layer: {
                "k": torch.arange(count * 2 * 3, dtype=torch.float32).reshape(
                    1, count, 2, 3
                ),
                "v": torch.full((1, count, 2, 3), float(layer)),
            }
            for layer in kwargs["capture_kv_layers"]
        }
        return SimpleNamespace(captured_kv=captures)


class _FakePosterior:
    def __init__(self, mean):
        self.mean = mean

    def sample(self, generator):
        return self.mean + torch.randn(
            self.mean.shape, generator=generator, dtype=self.mean.dtype
        )


class _FakeVAE(nn.Module):
    config = SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0])

    def encode(self, pixels, return_dict=False):
        assert return_dict is False
        pooled = pixels.mean(dim=1, keepdim=True)
        return (_FakePosterior(pooled.expand(-1, 2, -1, -1, -1)),)


class H3Int8OnlineFeatureProviderTest(unittest.TestCase):
    def test_historical_contract_uses_t0_cond0999_and_repeats_final_layer(self):
        backbone = _FakeBackbone()
        provider = H3Int8OnlineFeatureProvider(
            backbone,
            H3Int8OnlineFeatureContract(action_horizon=8),
            layout_functions=FAKE_LAYOUT,
        )
        output = provider(
            torch.zeros(1, 24, 1, 14, 28),
            torch.zeros(1, 3, 5376),
            torch.tensor([1, 0, 1]),
        )

        self.assertEqual(tuple(output.shape), (1, 5, 98, 4))
        torch.testing.assert_close(output, torch.full_like(output, 49.0))
        call = backbone.last_call
        assert call is not None
        torch.testing.assert_close(
            call["timestep"], torch.tensor([0.0, 0.999]), atol=0, rtol=0
        )
        self.assertEqual(tuple(call["audio_hidden_states"].shape), (1, 16, 32))
        self.assertEqual(int(call["capture_indices"].numel()), 98)
        torch.testing.assert_close(call["token_tags"][:3], torch.tensor([1, 0, 1]))

    def test_accepts_raw_or_refined_and_rejects_misaligned_context(self):
        provider = H3Int8OnlineFeatureProvider(
            _FakeBackbone(), H3Int8OnlineFeatureContract(), layout_functions=FAKE_LAYOUT
        )
        first = torch.zeros(1, 24, 1, 14, 28)
        raw = provider(
            first, torch.zeros(1, 3, 5120), torch.ones(3, dtype=torch.long)
        )
        refined = provider(
            first, torch.zeros(1, 3, 5376), torch.ones(3, dtype=torch.long)
        )
        self.assertEqual(raw.shape, refined.shape)
        with self.assertRaisesRegex(ValueError, "raw 5120 or refined 5376"):
            provider(first, torch.zeros(1, 3, 5119), torch.ones(3, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "cover every encoder context row"):
            provider(first, torch.zeros(1, 3, 5376), torch.ones(2, dtype=torch.long))

    def test_standalone_vae_recipe_is_seeded_and_cpu_float32(self):
        pixels = torch.full((1, 3, 1, 2, 2), 128, dtype=torch.uint8)
        first = encode_h3_vae_condition_standalone(
            _FakeVAE(), pixels, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
        )
        second = encode_h3_vae_condition_standalone(
            _FakeVAE(), pixels, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
        )
        self.assertEqual(first.device.type, "cpu")
        self.assertEqual(first.dtype, torch.float32)
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_live_kv_contract_matches_candidate_d_timestep_and_pooling(self):
        backbone = _FakeKVBackbone()
        provider = H3Int8OnlineKVProvider(
            backbone,
            H3Int8OnlineKVContract(
                layers=(49,), action_horizon=32, capture_token_count=32
            ),
            layout_functions=FAKE_LAYOUT,
        )
        output = provider(
            torch.zeros(1, 24, 1, 14, 28),
            torch.zeros(1, 3, 5120),
            torch.ones(3, dtype=torch.long),
        )

        self.assertEqual(set(output), {49})
        self.assertEqual(set(output[49]), {"k", "v"})
        self.assertEqual(tuple(output[49]["k"].shape), (1, 32, 2, 3))
        self.assertEqual(output[49]["k"].dtype, torch.bfloat16)
        self.assertNotEqual(
            output[49]["k"].untyped_storage().data_ptr(),
            output[49]["v"].untyped_storage().data_ptr(),
        )
        call = backbone.last_call
        assert call is not None
        self.assertEqual(call["capture_layers"], ())
        self.assertEqual(call["capture_kv_layers"], (49,))
        torch.testing.assert_close(
            call["timestep"], torch.tensor([1.0, 1.0]), atol=0, rtol=0
        )
        self.assertEqual(tuple(call["audio_hidden_states"].shape), (1, 64, 32))
        self.assertEqual(int(call["kv_capture_indices"].numel()), 98)

    def test_online_kv_pooling_rejects_bad_shape_and_preserves_short_sequences(self):
        source = torch.randn(1, 7, 2, 3)
        pooled = pool_online_kv_tokens(source, 32)
        self.assertEqual(tuple(pooled.shape), (7, 2, 3))
        torch.testing.assert_close(pooled, source[0])
        with self.assertRaisesRegex(ValueError, "online K/V tensor"):
            pool_online_kv_tokens(torch.randn(7, 2, 3), 32)


if __name__ == "__main__":
    unittest.main()
