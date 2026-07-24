import inspect
import unittest
from unittest.mock import patch

import torch

from experiments.anygrasp.deploy_policy import FastWAMAnyGraspPolicy
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam_hierarchical import FastWAM_Hierarchical
from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components
from fastwam.models.wan22.wan_video_dit import (
    WanVideoDiT,
    precompute_freqs_cis,
    rope_apply,
    sinusoidal_embedding_1d,
)
from fastwam.runtime import create_fastwam_hierarchical


class TextEncoderOffloadTest(unittest.TestCase):
    def test_hierarchical_model_keeps_offloaded_text_encoder_on_cpu_dtype(self):
        model = FastWAM_Hierarchical.__new__(FastWAM_Hierarchical)
        torch.nn.Module.__init__(model)
        model.mot = torch.nn.Linear(1, 1)
        model.text_encoder = torch.nn.Linear(1, 1)
        model.vae = torch.nn.Linear(1, 1)
        model.offload_text_encoder = True
        model.offload_vae = False

        model.to(dtype=torch.float64)

        self.assertEqual(model.mot.weight.dtype, torch.float64)
        self.assertEqual(model.vae.weight.dtype, torch.float64)
        self.assertEqual(model.text_encoder.weight.dtype, torch.float32)

    def test_text_encoder_offload_is_exposed_through_loader_and_runtime(self):
        self.assertIn(
            "text_encoder_device",
            inspect.signature(load_wan22_ti2v_5b_components).parameters,
        )
        self.assertIn(
            "offload_text_encoder",
            inspect.signature(create_fastwam_hierarchical).parameters,
        )

    def test_same_prompt_reuses_context_and_changed_prompt_reencodes(self):
        self.assertTrue(hasattr(FastWAMAnyGraspPolicy, "_resolve_prompt_inputs"))

        class Model:
            calls = 0

            def encode_prompt(self, prompt):
                self.calls += 1
                return torch.tensor([[self.calls]]), torch.ones((1, 1), dtype=torch.bool)

        policy = FastWAMAnyGraspPolicy.__new__(FastWAMAnyGraspPolicy)
        policy.model = Model()
        policy._cached_prompt = None
        policy._cached_context = None
        policy._cached_context_mask = None
        policy._prompt_cache_hits = 0
        policy._prompt_cache_misses = 0
        signature = {"context": object(), "context_mask": object()}

        first, first_info = policy._resolve_prompt_inputs(prompt="pick", signature=signature)
        second, second_info = policy._resolve_prompt_inputs(prompt="pick", signature=signature)
        third, third_info = policy._resolve_prompt_inputs(prompt="place", signature=signature)

        self.assertIsNone(first["prompt"])
        self.assertIs(first["context"], second["context"])
        self.assertEqual(policy.model.calls, 2)
        self.assertFalse(first_info["hit"])
        self.assertTrue(second_info["hit"])
        self.assertFalse(third_info["hit"])
        self.assertEqual(third_info["hits"], 1)
        self.assertEqual(third_info["misses"], 2)


class HierarchicalCompileTest(unittest.TestCase):
    def _model(self):
        model = FastWAM_Hierarchical.__new__(FastWAM_Hierarchical)
        torch.nn.Module.__init__(model)
        model._hierarchical_compile_cache = {}
        model._hierarchical_compile_disabled = {}
        model._hierarchical_compile_validation = {}
        model.torch_dtype = torch.bfloat16
        return model

    def test_compiled_callable_is_cached_and_outputs_are_cloned(self):
        self.assertTrue(hasattr(FastWAM_Hierarchical, "_call_hierarchical_compiled_or_eager"))
        self.assertIn(
            "compile_mode",
            inspect.signature(FastWAM_Hierarchical._call_hierarchical_compiled_or_eager).parameters,
        )
        model = self._model()
        compile_calls = []

        def fake_compile(fn, **kwargs):
            compile_calls.append(kwargs)
            return fn

        value = torch.tensor([1.0])
        with patch.object(torch, "compile", side_effect=fake_compile):
            first = model._call_hierarchical_compiled_or_eager(
                name="increment",
                fn=lambda x: x + 1,
                kwargs={"x": value},
                compile_hierarchical=True,
                compile_mode="reduce-overhead",
            )
            second = model._call_hierarchical_compiled_or_eager(
                name="increment",
                fn=lambda x: x + 1,
                kwargs={"x": value},
                compile_hierarchical=True,
                compile_mode="reduce-overhead",
            )

        self.assertEqual(len(compile_calls), 1)
        self.assertEqual(compile_calls[0]["dynamic"], False)
        self.assertEqual(compile_calls[0]["mode"], "reduce-overhead")
        self.assertEqual(first.item(), 2.0)
        self.assertEqual(second.item(), 2.0)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_cudagraph_failure_falls_back_to_default_compile(self):
        self.assertTrue(hasattr(FastWAM_Hierarchical, "_call_hierarchical_compiled_or_eager"))
        self.assertIn(
            "compile_mode",
            inspect.signature(FastWAM_Hierarchical._call_hierarchical_compiled_or_eager).parameters,
        )
        model = self._model()
        compile_modes = []

        def fake_compile(fn, **kwargs):
            compile_modes.append(kwargs["mode"])
            if kwargs["mode"] == "reduce-overhead":
                def fail(**_kwargs):
                    raise RuntimeError("CUDA Graph replay failed")

                return fail
            return fn

        with patch.object(torch, "compile", side_effect=fake_compile):
            result = model._call_hierarchical_compiled_or_eager(
                name="increment",
                fn=lambda x: x + 1,
                kwargs={"x": torch.tensor([1.0])},
                compile_hierarchical=True,
                compile_mode="reduce-overhead",
            )

        self.assertEqual(result.item(), 2.0)
        self.assertEqual(compile_modes, ["reduce-overhead", "default"])
        self.assertIn("increment@reduce-overhead", model._hierarchical_compile_disabled)
        self.assertIn("increment@default", model._hierarchical_compile_cache)

    def test_tensorrt_output_is_validated_before_use(self):
        model = self._model()
        cache_key = "tensorrt:identity@default"
        model._hierarchical_compile_cache[cache_key] = lambda x: x + torch.tensor(0.003, dtype=x.dtype)

        for _ in range(5):
            result = model._call_hierarchical_compiled_or_eager(
                name="identity",
                fn=lambda x: x,
                kwargs={"x": torch.ones(2, dtype=torch.bfloat16)},
                compile_hierarchical=True,
                compile_backend="tensorrt",
            )

        self.assertEqual(result.shape, (2,))
        self.assertTrue(model._hierarchical_compile_validation[cache_key]["passed"])
        self.assertNotIn(cache_key, model._hierarchical_compile_disabled)

    def test_tensorrt_validation_failure_disables_engine_and_returns_eager(self):
        model = self._model()
        cache_key = "tensorrt:identity@default"
        model._hierarchical_compile_cache[cache_key] = lambda x: x + 1
        expected = torch.ones(2, dtype=torch.bfloat16)

        result = model._call_hierarchical_compiled_or_eager(
            name="identity",
            fn=lambda x: x,
            kwargs={"x": expected},
            compile_hierarchical=True,
            compile_backend="tensorrt",
        )

        torch.testing.assert_close(result, expected)
        self.assertIn(cache_key, model._hierarchical_compile_disabled)
        self.assertNotIn(cache_key, model._hierarchical_compile_cache)

    def test_compile_option_is_exposed_by_model_and_deploy_wrapper(self):
        self.assertIn("compile_hierarchical", inspect.signature(FastWAM_Hierarchical.infer_action).parameters)
        self.assertIn("compile_cudagraphs", inspect.signature(FastWAM_Hierarchical.infer_action).parameters)
        self.assertIn("optimize_denoise_static", inspect.signature(FastWAM_Hierarchical.infer_action).parameters)
        self.assertIn("inference_backend", inspect.signature(FastWAM_Hierarchical.infer_action).parameters)
        self.assertIn("compile_hierarchical", inspect.signature(FastWAMAnyGraspPolicy.__init__).parameters)
        self.assertIn("compile_cudagraphs", inspect.signature(FastWAMAnyGraspPolicy.__init__).parameters)
        self.assertIn("optimize_denoise_static", inspect.signature(FastWAMAnyGraspPolicy.__init__).parameters)
        self.assertIn("inference_backend", inspect.signature(FastWAMAnyGraspPolicy.__init__).parameters)
        self.assertIn("warmup", inspect.signature(FastWAMAnyGraspPolicy.__init__).parameters)
        self.assertTrue(hasattr(FastWAMAnyGraspPolicy, "warmup"))

    def test_rope_caches_are_nonpersistent_device_buffers(self):
        video = WanVideoDiT(
            hidden_dim=12,
            in_dim=4,
            ffn_dim=24,
            out_dim=4,
            text_dim=8,
            freq_dim=8,
            eps=1e-6,
            patch_size=(1, 1, 1),
            num_heads=2,
            attn_head_dim=6,
            num_layers=0,
            has_image_input=False,
        )
        action = ActionDiT(
            hidden_dim=12,
            action_dim=3,
            ffn_dim=24,
            text_dim=8,
            freq_dim=8,
            eps=1e-6,
            num_heads=2,
            attn_head_dim=6,
            num_layers=0,
        )

        self.assertTrue({"freqs_f", "freqs_h", "freqs_w"} <= set(video._buffers))
        self.assertIn("freqs", action._buffers)
        self.assertIn("time_frequencies", video._buffers)
        self.assertIn("time_frequencies", action._buffers)
        self.assertFalse(any(key.startswith("freqs") for key in video.state_dict()))
        self.assertNotIn("freqs", action.state_dict())
        self.assertNotIn("time_frequencies", video.state_dict())
        self.assertNotIn("time_frequencies", action.state_dict())

        if torch.cuda.is_available():
            video.cuda()
            action.cuda()
            self.assertTrue(all(freqs.is_cuda for freqs in video.freqs))
            self.assertTrue(action.freqs.is_cuda)
            self.assertTrue(video.time_frequencies.is_cuda)
            self.assertTrue(action.time_frequencies.is_cuda)

    def test_cached_time_frequencies_match_regular_embedding(self):
        positions = torch.tensor([0.0, 0.25, 1.0])
        frequencies = torch.pow(
            10000,
            -torch.arange(4, dtype=torch.float64).div(4),
        )
        sinusoid = torch.outer(positions.to(torch.float64), frequencies)
        expected = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(positions.dtype)
        actual = sinusoidal_embedding_1d(8, positions, frequencies)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_real_rope_matches_legacy_complex_rotation(self):
        torch.manual_seed(11)
        x = torch.randn(2, 7, 24)
        freqs = precompute_freqs_cis(6, end=7).view(7, 1, -1)
        actual = rope_apply(x, freqs, num_heads=4)

        x_heads = x.view(2, 7, 4, 6)
        x_complex = torch.view_as_complex(x_heads.to(torch.float64).reshape(2, 7, 4, 3, 2))
        freq_complex = torch.view_as_complex(freqs.reshape(7, 1, 3, 2).contiguous())
        expected = torch.view_as_real(x_complex * freq_complex).flatten(2).to(x.dtype)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_float32_rope_math_preserves_low_precision_outputs(self):
        torch.manual_seed(19)
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(2, 7, 24, dtype=dtype)
            frequencies = precompute_freqs_cis(6, end=7).view(7, 1, -1)
            exact = rope_apply(x, frequencies, num_heads=4)
            fast = rope_apply(x, frequencies.float(), num_heads=4)
            torch.testing.assert_close(fast, exact, rtol=0, atol=0)

    def test_video_precomputed_context_matches_regular_path(self):
        torch.manual_seed(13)
        video = WanVideoDiT(
            hidden_dim=12,
            in_dim=4,
            ffn_dim=24,
            out_dim=4,
            text_dim=8,
            freq_dim=8,
            eps=1e-6,
            patch_size=(1, 1, 1),
            num_heads=2,
            attn_head_dim=6,
            num_layers=0,
            has_image_input=False,
            seperated_timestep=True,
        ).eval()
        x = torch.randn(1, 4, 1, 2, 2)
        timestep = torch.tensor([0.25])
        context = torch.randn(1, 3, 8)
        context_mask = torch.ones(1, 3, dtype=torch.bool)
        expected = video.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
        )
        actual = video.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
            context_embedding=video.text_embedding(context),
        )

        for key in ("tokens", "freqs", "t", "t_mod", "context", "context_mask"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_compiled_attention_mask_does_not_escape_into_python_cache(self):
        model = self._model()
        model._hierarchical_attention_mask_cache = {}
        model.hierarchical_mask_high_predict = False
        model.hierarchical_mask_low_predict = False

        with patch.object(torch.compiler, "is_compiling", return_value=True):
            mask = model._build_hierarchical_training_attention_mask(
                high_frames=2,
                low_frames=1,
                tokens_per_frame=2,
                action_seq_len=3,
                has_proprio_token=True,
                history_high_frames=1,
                low_visible_high_start=0,
                low_visible_high_end=1,
                device=torch.device("cpu"),
            )

        self.assertEqual(mask.shape, (9, 9))
        self.assertEqual(model._hierarchical_attention_mask_cache, {})

    def test_prepared_action_path_matches_existing_action_path(self):
        torch.manual_seed(7)
        action_expert = ActionDiT(
            hidden_dim=12,
            action_dim=3,
            ffn_dim=24,
            text_dim=8,
            freq_dim=8,
            eps=1e-6,
            num_heads=2,
            attn_head_dim=6,
            num_layers=0,
        ).eval()

        class Mot:
            @staticmethod
            def _forward_action_with_video_cache_inner(**kwargs):
                return kwargs["action_tokens"]

        model = self._model()
        model.action_expert = action_expert
        model.mot = Mot()
        model.hierarchical_mask_high_predict = False
        model.hierarchical_mask_low_predict = False
        model._hierarchical_attention_mask_cache = {}

        action_latents = torch.randn(1, 4, 3)
        action_timestep = torch.tensor([0.5])
        context = torch.randn(1, 5, 8)
        context_mask = torch.ones(1, 5, dtype=torch.bool)
        proprio_token = torch.randn(1, 1, 12)

        expected = model._predict_action_with_frozen_video_cache_flat(
            action_latents=action_latents,
            action_timestep=action_timestep,
            context=context,
            context_mask=context_mask,
            video_cache_k=[],
            video_cache_v=[],
            video_seq_len=4,
            tokens_per_frame=2,
            high_frames=1,
            low_frames=1,
            history_high_frames=1,
            visible_high_start=0,
            visible_high_end=0,
            proprio_token=proprio_token,
        )
        static_inputs = model._prepare_action_only_static_inputs(
            action_latents=action_latents,
            context=context,
            context_mask=context_mask,
            action_context_embedding=action_expert.text_embedding(context),
            video_seq_len=4,
            tokens_per_frame=2,
            high_frames=1,
            low_frames=1,
            history_high_frames=1,
            visible_high_start=0,
            visible_high_end=0,
            proprio_token=proprio_token,
        )
        actual = model._predict_action_with_prepared_video_cache_flat(
            action_latents=action_latents,
            action_timestep=action_timestep,
            action_context=static_inputs[0],
            action_context_mask=static_inputs[1],
            action_freqs=static_inputs[2],
            action_attention_mask=static_inputs[3],
            proprio_token=static_inputs[4],
            proprio_t_mod=static_inputs[5],
            action_time_frequencies=static_inputs[6],
            video_cache_k=[],
            video_cache_v=[],
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
