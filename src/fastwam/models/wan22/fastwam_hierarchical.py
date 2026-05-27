from typing import Any, Optional, Sequence, Union
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import sinusoidal_embedding_1d

logger = get_logger(__name__)


class FastWAM_Hierarchical(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        hierarchical_action_horizon: int = 32,
        hierarchical_mask_high_predict: bool = False,
        hierarchical_mask_low_predict: bool = False,
        hierarchical_high_condition_mode: str = "boundary",
        hierarchical_dynamic_skip: bool = False,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            action_encoder = getattr(self.action_expert, "action_encoder", None)
            if action_encoder is None:
                raise ValueError("Action expert has no `action_encoder`; cannot build proprio encoder from action encoder.")
            if int(self.proprio_dim) == int(self.action_expert.action_dim):
                self.proprio_encoder = copy.deepcopy(action_encoder).to(dtype=torch_dtype)
            else:
                self.proprio_encoder = nn.Linear(self.proprio_dim, self.action_expert.hidden_dim).to(dtype=torch_dtype)
                logger.warning(
                    "`proprio_dim` (%d) does not match action_dim (%d); using same-structure linear proprio encoder with random init.",
                    int(self.proprio_dim),
                    int(self.action_expert.action_dim),
                )
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        self.hierarchical_action_horizon = int(hierarchical_action_horizon)
        self.hierarchical_mask_high_predict = bool(hierarchical_mask_high_predict)
        self.hierarchical_mask_low_predict = bool(hierarchical_mask_low_predict)
        self.hierarchical_high_condition_mode = str(hierarchical_high_condition_mode).lower()
        self.hierarchical_dynamic_skip = bool(hierarchical_dynamic_skip)
        if self.hierarchical_action_horizon <= 0:
            raise ValueError("`hierarchical_action_horizon` must be positive.")
        if self.hierarchical_high_condition_mode not in {"history", "boundary", "future"}:
            raise ValueError(
                "`hierarchical_high_condition_mode` must be one of "
                "['history', 'boundary', 'future'], got "
                f"{hierarchical_high_condition_mode!r}."
            )
        self._hierarchical_attention_mask_cache: dict[tuple[Any, ...], torch.Tensor] = {}

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        hierarchical_action_horizon: int = 32,
        hierarchical_mask_high_predict: bool = False,
        hierarchical_mask_low_predict: bool = False,
        hierarchical_high_condition_mode: str = "boundary",
        hierarchical_dynamic_skip: bool = False,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            hierarchical_action_horizon=hierarchical_action_horizon,
            hierarchical_mask_high_predict=hierarchical_mask_high_predict,
            hierarchical_mask_low_predict=hierarchical_mask_low_predict,
            hierarchical_high_condition_mode=hierarchical_high_condition_mode,
            hierarchical_dynamic_skip=hierarchical_dynamic_skip,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def _decode_latents_to_tensor(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)) -> torch.Tensor:
        video_tensor = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return video_tensor.squeeze(0).detach().float().clamp(-1, 1)

    @staticmethod
    def _video_tensor_to_pil(video_tensor: torch.Tensor) -> list[Image.Image]:
        frames = []
        video_uint8 = ((video_tensor.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu()
        for t in range(video_uint8.shape[1]):
            frame = video_uint8[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    @staticmethod
    def _temporal_downsample(video: torch.Tensor, stride: int) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected [B, C, T, H, W], got {tuple(video.shape)}")
        if stride <= 0:
            raise ValueError(f"`stride` must be positive, got {stride}")
        total_t = int(video.shape[2])
        idx = torch.arange(0, total_t, stride, device=video.device)
        if int(idx[-1].item()) != total_t - 1:
            idx = torch.cat([idx, torch.tensor([total_t - 1], device=video.device)])
        return torch.index_select(video, dim=2, index=idx)

    @staticmethod
    def _temporal_downsample_mask(mask: torch.Tensor, stride: int) -> torch.Tensor:
        if mask.ndim != 2:
            raise ValueError(f"Expected [B, T] mask, got {tuple(mask.shape)}")
        if stride <= 0:
            raise ValueError(f"`stride` must be positive, got {stride}")
        total_t = int(mask.shape[1])
        idx = torch.arange(0, total_t, stride, device=mask.device)
        if int(idx[-1].item()) != total_t - 1:
            idx = torch.cat([idx, torch.tensor([total_t - 1], device=mask.device)])
        return torch.index_select(mask, dim=1, index=idx)

    def _predict_video_only_noise(
        self,
        latents_video: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = video_pre["tokens"]
        freqs = video_pre["freqs"]
        t_mod = video_pre["t_mod"]
        ctx = video_pre["context"]
        ctx_mask = video_pre["context_mask"]
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        self_attn_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=x_tokens.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=x_tokens.device,
        )

        for block in self.video_expert.blocks:
            x_tokens = block(
                x_tokens,
                ctx,
                t_mod,
                freqs,
                context_mask=ctx_mask,
                self_attn_mask=self_attn_mask,
            )
        return self.video_expert.post_dit(x_tokens, video_pre)

    def _predict_high_video_noise_masked(
        self,
        *,
        latents_video: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        history_frames: int,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = video_pre["tokens"]
        freqs = video_pre["freqs"]
        t_mod = video_pre["t_mod"]
        ctx = video_pre["context"]
        ctx_mask = video_pre["context_mask"]
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        high_frames = int(latents_video.shape[2])
        self_attn_mask = self._build_high_history_predict_mask(
            high_frames=high_frames,
            history_frames=history_frames,
            tokens_per_frame=tokens_per_frame,
            device=x_tokens.device,
        )

        for block in self.video_expert.blocks:
            x_tokens = block(
                x_tokens,
                ctx,
                t_mod,
                freqs,
                context_mask=ctx_mask,
                self_attn_mask=self_attn_mask,
            )
        return self.video_expert.post_dit(x_tokens, video_pre)

    @staticmethod
    def _build_first_frame_causal_mask(
        *,
        num_frames: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        seq_len = num_frames * tokens_per_frame
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        first_frame_tokens = min(tokens_per_frame, seq_len)
        mask[:first_frame_tokens, first_frame_tokens:] = False
        return mask

    @staticmethod
    def _build_high_history_predict_mask(
        *,
        high_frames: int,
        history_frames: int,
        tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if history_frames <= 0 or history_frames > high_frames:
            raise ValueError(
                f"Invalid high-level history length {history_frames} for high_frames={high_frames}."
            )
        seq_len = high_frames * tokens_per_frame
        history_seq_len = history_frames * tokens_per_frame
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
        mask[:history_seq_len, :history_seq_len] = True
        mask[history_seq_len:, :seq_len] = True
        return mask

    def _build_hierarchical_training_attention_mask(
        self,
        *,
        high_frames: int,
        low_frames: int,
        tokens_per_frame: int,
        action_seq_len: int,
        has_proprio_token: bool,
        history_high_frames: int,
        low_visible_high_start: int,
        low_visible_high_end: int,
        device: torch.device,
    ) -> torch.Tensor:
        high_seq_len = high_frames * tokens_per_frame
        low_seq_len = low_frames * tokens_per_frame
        video_seq_len = high_seq_len + low_seq_len
        total_seq_len = video_seq_len + action_seq_len
        low_base = high_seq_len
        action_base = video_seq_len
        visible_high_start = max(0, min(int(low_visible_high_start), high_frames - 1))
        visible_high_end = max(visible_high_start, min(int(low_visible_high_end), high_frames - 1))
        cache_key = (
            int(high_frames),
            int(low_frames),
            int(tokens_per_frame),
            int(action_seq_len),
            bool(has_proprio_token),
            int(history_high_frames),
            int(visible_high_start),
            int(visible_high_end),
            bool(self.hierarchical_mask_high_predict),
            bool(self.hierarchical_mask_low_predict),
            str(device),
        )
        cached_mask = self._hierarchical_attention_mask_cache.get(cache_key)
        if cached_mask is not None:
            return cached_mask

        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        if self.hierarchical_mask_high_predict:
            high_mask = self._build_high_history_predict_mask(
                high_frames=high_frames,
                history_frames=history_high_frames,
                tokens_per_frame=tokens_per_frame,
                device=device,
            )
        else:
            high_mask = torch.ones((high_seq_len, high_seq_len), dtype=torch.bool, device=device)
        mask[:high_seq_len, :high_seq_len] = high_mask

        if self.hierarchical_mask_low_predict:
            low_mask = self._build_first_frame_causal_mask(
                num_frames=low_frames,
                tokens_per_frame=tokens_per_frame,
                device=device,
            )
        else:
            low_mask = torch.ones((low_seq_len, low_seq_len), dtype=torch.bool, device=device)
        mask[low_base:video_seq_len, low_base:video_seq_len] = low_mask

        col_start = visible_high_start * tokens_per_frame
        col_end = (visible_high_end + 1) * tokens_per_frame
        mask[low_base:video_seq_len, col_start:col_end] = True

        if action_seq_len > 0:
            if has_proprio_token:
                proprio_idx = action_base
                mask[proprio_idx, proprio_idx] = True
                action_token_start = proprio_idx + 1
            else:
                action_token_start = action_base

            if action_token_start < total_seq_len:
                mask[action_token_start:, action_token_start:] = True
                if has_proprio_token:
                    mask[action_token_start:, proprio_idx : proprio_idx + 1] = True

                mask[action_token_start:, col_start:col_end] = True
                if self.hierarchical_mask_low_predict:
                    low_visible_seq_len = min(tokens_per_frame, low_seq_len)
                    mask[action_token_start:, low_base : low_base + low_visible_seq_len] = True
                else:
                    mask[action_token_start:, low_base:video_seq_len] = True
        if len(self._hierarchical_attention_mask_cache) >= 16:
            self._hierarchical_attention_mask_cache.clear()
        self._hierarchical_attention_mask_cache[cache_key] = mask
        return mask

    def _perturb_history_latents_for_training(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        clean_latents: torch.Tensor,
        history_frames: int,
        preserve_initial_frame: bool,
        probability: float = 0.5,
        max_noise_scale: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history_frames <= 0:
            return latents, timesteps
        if clean_latents.shape != latents.shape:
            raise ValueError(
                "History perturbation requires matching latent shapes: "
                f"latents={tuple(latents.shape)}, clean={tuple(clean_latents.shape)}."
            )
        if timesteps.ndim != 2 or timesteps.shape[0] != latents.shape[0] or timesteps.shape[1] != latents.shape[2]:
            raise ValueError(
                "History perturbation requires timestep shape [B,T]: "
                f"timesteps={tuple(timesteps.shape)}, latents={tuple(latents.shape)}."
            )
        history_frames = min(int(history_frames), int(latents.shape[2]))
        history_start = 1 if preserve_initial_frame else 0
        if history_start >= history_frames:
            return latents, timesteps

        batch_size = int(latents.shape[0])
        cond_noise_mask = torch.rand((batch_size,), device=latents.device) < float(probability)
        if not bool(cond_noise_mask.any()):
            latents[:, :, history_start:history_frames] = clean_latents[:, :, history_start:history_frames]
            timesteps[:, history_start:history_frames] = 0
            return latents, timesteps

        sampled_timestep = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=latents.device,
            dtype=latents.dtype,
        ) * float(max_noise_scale)
        history_timestep = torch.where(
            cond_noise_mask,
            sampled_timestep,
            torch.zeros_like(sampled_timestep),
        ) 
        noise_history = torch.randn_like(clean_latents[:, :, history_start:history_frames])
        noised_history = self.train_video_scheduler.add_noise(
            clean_latents[:, :, history_start:history_frames],
            noise_history,
            history_timestep,
        )
        history_selector = cond_noise_mask.view(batch_size, 1, 1, 1, 1)
        latents[:, :, history_start:history_frames] = torch.where(
            history_selector,
            noised_history,
            clean_latents[:, :, history_start:history_frames],
        )
        timesteps[:, history_start:history_frames] = history_timestep.unsqueeze(1).expand(-1, history_frames - history_start)
        return latents, timesteps

    @staticmethod
    def _build_history_predict_timestep(
        *,
        step_timestep: torch.Tensor,
        total_frames: int,
        history_frames: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if history_frames < 0 or history_frames > total_frames:
            raise ValueError(
                f"Invalid history_frames={history_frames} for total_frames={total_frames}."
            )

        step_timestep = step_timestep.to(device=device, dtype=dtype)
        if step_timestep.ndim == 0:
            step_timestep = step_timestep.reshape(1)
        if step_timestep.ndim == 1:
            timestep = step_timestep.unsqueeze(1).expand(-1, total_frames).clone()
        elif step_timestep.ndim == 2:
            if int(step_timestep.shape[1]) == 1:
                timestep = step_timestep.expand(-1, total_frames).clone()
            elif int(step_timestep.shape[1]) == total_frames:
                timestep = step_timestep.clone()
            else:
                raise ValueError(
                    "`step_timestep` second dim must be 1 or `total_frames`, got "
                    f"shape={tuple(step_timestep.shape)}, total_frames={total_frames}."
                )
        else:
            raise ValueError(
                f"`step_timestep` must be scalar/[B]/[B,T], got shape {tuple(step_timestep.shape)}."
            )

        if history_frames > 0:
            timestep[:, :history_frames] = 0
        return timestep

    def _prepare_infer_context(
        self,
        *,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            return self.encode_prompt(prompt)
        if context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must be both provided together.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        return (
            context.to(device=self.device, dtype=self.torch_dtype),
            context_mask.to(device=self.device, dtype=torch.bool),
        )




    def _encode_proprio_action_token(
        self,
        proprio: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.proprio_encoder is None:
            return None
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or int(proprio.shape[1]) != int(self.proprio_dim):
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
        proprio_text = self.proprio_encoder(
            proprio.to(device=self.device, dtype=dtype).unsqueeze(1)
        ).to(dtype=dtype)
        return proprio_text.to(dtype=dtype)

    def _prepend_proprio_to_action_pre(
        self,
        action_pre: dict[str, torch.Tensor],
        proprio_token: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_tokens = action_pre["tokens"]
        action_freqs = action_pre["freqs"]
        action_ctx = action_pre["context"]
        action_ctx_mask = action_pre["context_mask"]
        action_t_mod = action_pre["t_mod"]
        if proprio_token is None:
            return action_tokens, action_freqs, action_ctx, action_ctx_mask, action_t_mod

        if proprio_token.shape[0] != action_tokens.shape[0] or proprio_token.shape[2] != action_tokens.shape[2]:
            raise ValueError(
                "`proprio_token` shape mismatch with action hidden size: "
                f"got {tuple(proprio_token.shape)}, expected [B,1,{action_tokens.shape[2]}]."
            )

        # Proprio is prepended as an extra action query token.
        # Let proprio token share action-query context visibility so it can attend text context.
        proprio_ctx_mask = action_ctx_mask[:, :1].clone()
        # Reuse the first action token RoPE frequency for proprio token to keep action/proprio frequency encoding aligned.
        proprio_freq = action_freqs[:1].clone()
        batch_size = int(action_tokens.shape[0])
        zero_timestep = torch.zeros((batch_size,), device=action_tokens.device, dtype=action_pre["t"].dtype)
        zero_t = self.action_expert.time_embedding(
            sinusoidal_embedding_1d(self.action_expert.freq_dim, zero_timestep)
        )
        zero_t_mod = self.action_expert.time_projection(zero_t).unflatten(1, (6, self.action_expert.hidden_dim))
        if action_t_mod.ndim == 3:
            action_t_mod = action_t_mod.unsqueeze(1).expand(-1, action_tokens.shape[1], -1, -1)
        zero_t_mod = zero_t_mod.unsqueeze(1)
        action_t_mod = torch.cat([zero_t_mod, action_t_mod], dim=1)

        return (
            torch.cat([proprio_token, action_tokens], dim=1),
            torch.cat([proprio_freq, action_freqs], dim=0),
            action_ctx,
            torch.cat([proprio_ctx_mask, action_ctx_mask], dim=1),
            action_t_mod,
        )


    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
        temporal_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if temporal_weights is not None:
            temporal_weights = temporal_weights.to(
                device=video_loss_token.device,
                dtype=video_loss_token.dtype,
            )
            if temporal_weights.shape != video_loss_token.shape:
                raise ValueError(
                    "Temporal weight shape mismatch: "
                    f"weights={tuple(temporal_weights.shape)}, loss={tuple(video_loss_token.shape)}."
                )
        if image_is_pad is None:
            if temporal_weights is None:
                return video_loss_token.mean(dim=1)
            weight_sum = temporal_weights.sum(dim=1).clamp(min=1e-8)
            return (video_loss_token * temporal_weights).sum(dim=1) / weight_sum

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        weight = valid if temporal_weights is None else (valid * temporal_weights)
        weight_sum = weight.sum(dim=1).clamp(min=1e-8)
        return (video_loss_token * weight).sum(dim=1) / weight_sum

    def _predict_low_action_joint_noise(
        self,
        *,
        keyframe_cond_latents: torch.Tensor,
        low_latents: torch.Tensor,
        action_latents: torch.Tensor,
        keyframe_timestep: torch.Tensor,
        low_timestep: torch.Tensor,
        action_timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio_token: Optional[torch.Tensor],
        fuse_vae_embedding_in_latents: bool,
        history_high_frames: Optional[int] = None,
        visible_high_start: Optional[int] = None,
        visible_high_end: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        high_pre = self.video_expert.pre_dit(
            x=keyframe_cond_latents,
            timestep=keyframe_timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        low_pre = self.video_expert.pre_dit(
            x=low_latents,
            timestep=low_timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_latents,
            timestep=action_timestep,
            context=context,
            context_mask=context_mask,
        )

        tokens_per_frame = int(high_pre["meta"]["tokens_per_frame"])
        if int(low_pre["meta"]["tokens_per_frame"]) != tokens_per_frame:
            raise ValueError(
                "Token-per-frame mismatch between keyframe and low video branches: "
                f"keyframe={tokens_per_frame}, low={int(low_pre['meta']['tokens_per_frame'])}."
            )
        action_tokens, action_freqs, action_ctx, action_ctx_mask, action_t_mod = self._prepend_proprio_to_action_pre(
            action_pre,
            proprio_token,
        )
        has_proprio = proprio_token is not None
        action_seq_len = int(action_tokens.shape[1])
        high_frames = int(keyframe_cond_latents.shape[2])
        history_high_frames = high_frames if history_high_frames is None else int(history_high_frames)
        visible_high_start = 0 if visible_high_start is None else int(visible_high_start)
        visible_high_end = high_frames - 1 if visible_high_end is None else int(visible_high_end)
        attention_mask = self._build_hierarchical_training_attention_mask(
            high_frames=high_frames,
            low_frames=int(low_latents.shape[2]),
            tokens_per_frame=tokens_per_frame,
            action_seq_len=action_seq_len,
            has_proprio_token=has_proprio,
            history_high_frames=history_high_frames,
            low_visible_high_start=visible_high_start,
            low_visible_high_end=visible_high_end,
            device=low_latents.device,
        )

        merged_video_tokens = torch.cat([high_pre["tokens"], low_pre["tokens"]], dim=1)
        merged_video_freqs = torch.cat([high_pre["freqs"], low_pre["freqs"]], dim=0)
        merged_video_t_mod = torch.cat([high_pre["t_mod"], low_pre["t_mod"]], dim=1)
        merged_video_context_mask = torch.cat([high_pre["context_mask"], low_pre["context_mask"]], dim=1)

        tokens_out = self.mot(
            embeds_all={
                "video": merged_video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": merged_video_freqs,
                "action": action_freqs,
            },
            context_all={
                "video": {
                    "context": high_pre["context"],
                    "mask": merged_video_context_mask,
                },
                "action": {
                    "context": action_ctx,
                    "mask": action_ctx_mask,
                },
            },
            t_mod_all={
                "video": merged_video_t_mod,
                "action": action_t_mod,
            },
        )

        high_seq_len = int(high_pre["tokens"].shape[1])
        pred_keyframe = self.video_expert.post_dit(tokens_out["video"][:, :high_seq_len], high_pre)
        pred_low = self.video_expert.post_dit(tokens_out["video"][:, high_seq_len:], low_pre)
        pred_action_all = self.action_expert.post_dit(tokens_out["action"], action_pre)
        pred_action = pred_action_all[:, 1:] if has_proprio else pred_action_all
        return pred_keyframe, pred_low, pred_action

    @staticmethod
    def _should_run_diffusion_step(
        *,
        prev_predictions: list[torch.Tensor],
        skip_countdown: int,
        enabled: bool,
    ) -> tuple[bool, int]:
        if not enabled or len(prev_predictions) < 2:
            return True, skip_countdown
        if skip_countdown > 1:
            return False, skip_countdown - 1
        if skip_countdown == 1:
            return True, 0

        v_last = prev_predictions[-1].flatten(1).float()
        v_prev = prev_predictions[-2].flatten(1).float()
        sim = F.cosine_similarity(v_last, v_prev, dim=1).mean()
        for threshold, countdown in ((0.95, 4), (0.93, 2)):
            if bool(sim > threshold):
                return False, countdown
        return True, 0

    def _downstream_high_visible_range(self, history_frames: int) -> tuple[int, int]:
        mode = self.hierarchical_high_condition_mode
        if self.hierarchical_mask_high_predict:
            if mode == "history":
                return 0, history_frames - 1
            return history_frames - 1, history_frames - 1
        if mode == "history":
            return 0, history_frames
        if mode == "boundary":
            return history_frames - 1, history_frames
        return history_frames, history_frames

    def _select_downstream_keyframes(
        self,
        keyframe_latents: torch.Tensor,
        keyframe_timestep: torch.Tensor,
        *,
        history_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
        selected, selected_history_frames = self._downstream_keyframe_indices(history_frames)
        indices = torch.tensor(selected, device=keyframe_latents.device, dtype=torch.long)
        return (
            torch.index_select(keyframe_latents, dim=2, index=indices),
            torch.index_select(keyframe_timestep, dim=1, index=indices),
            selected_history_frames,
            0,
            len(selected) - 1,
        )

    def _downstream_keyframe_indices(self, history_frames: int) -> tuple[list[int], int]:
        mode = self.hierarchical_high_condition_mode
        if self.hierarchical_mask_high_predict:
            if mode == "history":
                return list(range(history_frames)), history_frames
            return [history_frames - 1], 1
        if mode == "history":
            return list(range(history_frames + 1)), history_frames
        if mode == "boundary":
            return [history_frames - 1, history_frames], 1
        return [history_frames], 1

    @staticmethod
    def _concat_video_kv_caches(
        prefix_cache: list[dict[str, torch.Tensor]],
        suffix_cache: list[dict[str, torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        if len(prefix_cache) != len(suffix_cache):
            raise ValueError(
                f"Cache layer mismatch: prefix={len(prefix_cache)}, suffix={len(suffix_cache)}."
            )
        return [
            {
                "k": torch.cat([prefix_layer["k"], suffix_layer["k"]], dim=1),
                "v": torch.cat([prefix_layer["v"], suffix_layer["v"]], dim=1),
            }
            for prefix_layer, suffix_layer in zip(prefix_cache, suffix_cache)
        ]

    @staticmethod
    def _slice_video_kv_cache(
        cache: list[dict[str, torch.Tensor]],
        *,
        frame_indices: list[int],
        tokens_per_frame: int,
    ) -> list[dict[str, torch.Tensor]]:
        token_indices = []
        for frame_idx in frame_indices:
            start = int(frame_idx) * int(tokens_per_frame)
            token_indices.extend(range(start, start + int(tokens_per_frame)))
        if not token_indices:
            raise ValueError("Cannot slice an empty keyframe cache selection.")
        index = None
        sliced: list[dict[str, torch.Tensor]] = []
        for layer in cache:
            if index is None:
                index = torch.tensor(token_indices, device=layer["k"].device, dtype=torch.long)
            sliced.append(
                {
                    "k": torch.index_select(layer["k"], dim=1, index=index),
                    "v": torch.index_select(layer["v"], dim=1, index=index),
                }
            )
        return sliced

    @torch.no_grad()
    def _prefill_keyframe_cache(
        self,
        *,
        keyframe_cond_latents: torch.Tensor,
        keyframe_timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        history_high_frames: int,
        visible_high_start: int,
        visible_high_end: int,
    ) -> tuple[list[dict[str, torch.Tensor]], int, int]:
        key_pre = self.video_expert.pre_dit(
            x=keyframe_cond_latents,
            timestep=keyframe_timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        tokens_per_frame = int(key_pre["meta"]["tokens_per_frame"])
        high_frames = int(keyframe_cond_latents.shape[2])
        attention_mask = self._build_hierarchical_training_attention_mask(
            high_frames=high_frames,
            low_frames=0,
            tokens_per_frame=tokens_per_frame,
            action_seq_len=0,
            has_proprio_token=False,
            history_high_frames=history_high_frames,
            low_visible_high_start=visible_high_start,
            low_visible_high_end=visible_high_end,
            device=key_pre["tokens"].device,
        )
        key_cache = self.mot.prefill_video_cache(
            video_tokens=key_pre["tokens"],
            video_freqs=key_pre["freqs"],
            video_t_mod=key_pre["t_mod"],
            video_context_payload={
                "context": key_pre["context"],
                "mask": key_pre["context_mask"],
            },
            video_attention_mask=attention_mask,
        )
        return key_cache, int(key_pre["tokens"].shape[1]), tokens_per_frame

    @torch.no_grad()
    def _prefill_low_cache_with_keyframe_cache(
        self,
        *,
        keyframe_cache: list[dict[str, torch.Tensor]],
        keyframe_seq_len: int,
        keyframe_frames: int,
        low_latents: torch.Tensor,
        low_timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        tokens_per_frame: int,
        history_high_frames: int,
        visible_high_start: int,
        visible_high_end: int,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor, dict[str, torch.Tensor]]:
        low_pre = self.video_expert.pre_dit(
            x=low_latents,
            timestep=low_timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        if int(low_pre["meta"]["tokens_per_frame"]) != tokens_per_frame:
            raise ValueError(
                "Token-per-frame mismatch between keyframe and low video cache branches: "
                f"keyframe={tokens_per_frame}, low={int(low_pre['meta']['tokens_per_frame'])}."
            )
        low_frames = int(low_latents.shape[2])
        attention_mask = self._build_hierarchical_training_attention_mask(
            high_frames=keyframe_frames,
            low_frames=low_frames,
            tokens_per_frame=tokens_per_frame,
            action_seq_len=0,
            has_proprio_token=False,
            history_high_frames=history_high_frames,
            low_visible_high_start=visible_high_start,
            low_visible_high_end=visible_high_end,
            device=low_pre["tokens"].device,
        )
        low_cache, low_tokens = self.mot.prefill_video_cache_with_prefix(
            video_tokens=low_pre["tokens"],
            video_freqs=low_pre["freqs"],
            video_t_mod=low_pre["t_mod"],
            video_context_payload={
                "context": low_pre["context"],
                "mask": low_pre["context_mask"],
            },
            prefix_kv_cache=keyframe_cache,
            attention_mask=attention_mask,
            prefix_seq_len=keyframe_seq_len,
            return_tokens=True,
        )
        return low_cache, low_tokens, low_pre

    @torch.no_grad()
    def _predict_low_action_with_keyframe_cache(
        self,
        *,
        keyframe_cache: list[dict[str, torch.Tensor]],
        keyframe_seq_len: int,
        keyframe_frames: int,
        low_latents: torch.Tensor,
        action_latents: torch.Tensor,
        low_timestep: torch.Tensor,
        action_timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio_token: Optional[torch.Tensor],
        fuse_vae_embedding_in_latents: bool,
        tokens_per_frame: int,
        history_high_frames: int,
        visible_high_start: int,
        visible_high_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low_pre = self.video_expert.pre_dit(
            x=low_latents,
            timestep=low_timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_latents,
            timestep=action_timestep,
            context=context,
            context_mask=context_mask,
        )
        if int(low_pre["meta"]["tokens_per_frame"]) != tokens_per_frame:
            raise ValueError(
                "Token-per-frame mismatch between keyframe and low video branches: "
                f"keyframe={tokens_per_frame}, low={int(low_pre['meta']['tokens_per_frame'])}."
            )
        action_tokens, action_freqs, action_ctx, action_ctx_mask, action_t_mod = self._prepend_proprio_to_action_pre(
            action_pre,
            proprio_token,
        )
        has_proprio = proprio_token is not None
        low_frames = int(low_latents.shape[2])
        attention_mask = self._build_hierarchical_training_attention_mask(
            high_frames=keyframe_frames,
            low_frames=low_frames,
            tokens_per_frame=tokens_per_frame,
            action_seq_len=int(action_tokens.shape[1]),
            has_proprio_token=has_proprio,
            history_high_frames=history_high_frames,
            low_visible_high_start=visible_high_start,
            low_visible_high_end=visible_high_end,
            device=action_latents.device,
        )
        low_tokens, action_tokens = self.mot.forward_video_action_with_video_cache(
            video_tokens=low_pre["tokens"],
            video_freqs=low_pre["freqs"],
            video_t_mod=low_pre["t_mod"],
            video_context_payload={
                "context": low_pre["context"],
                "mask": low_pre["context_mask"],
            },
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context_payload={
                "context": action_ctx,
                "mask": action_ctx_mask,
            },
            prefix_kv_cache=keyframe_cache,
            attention_mask=attention_mask,
            prefix_seq_len=keyframe_seq_len,
        )
        pred_low = self.video_expert.post_dit(low_tokens, low_pre)
        pred_action_all = self.action_expert.post_dit(action_tokens, action_pre)
        pred_action = pred_action_all[:, 1:] if has_proprio else pred_action_all
        return pred_low, pred_action

    @torch.no_grad()
    def _predict_action_with_frozen_video_cache(
        self,
        *,
        action_latents: torch.Tensor,
        action_timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_cache: list[dict[str, torch.Tensor]],
        video_seq_len: int,
        tokens_per_frame: int,
        high_frames: int,
        low_frames: int,
        history_high_frames: int,
        visible_high_start: int,
        visible_high_end: int,
        proprio_token: Optional[torch.Tensor],
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_latents,
            timestep=action_timestep,
            context=context,
            context_mask=context_mask,
        )
        action_tokens, action_freqs, action_ctx, action_ctx_mask, action_t_mod = self._prepend_proprio_to_action_pre(
            action_pre,
            proprio_token,
        )
        has_proprio = proprio_token is not None
        attention_mask = self._build_hierarchical_training_attention_mask(
            high_frames=high_frames,
            low_frames=low_frames,
            tokens_per_frame=tokens_per_frame,
            action_seq_len=int(action_tokens.shape[1]),
            has_proprio_token=has_proprio,
            history_high_frames=history_high_frames,
            low_visible_high_start=visible_high_start,
            low_visible_high_end=visible_high_end,
            device=action_latents.device,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context_payload={
                "context": action_ctx,
                "mask": action_ctx_mask,
            },
            video_kv_cache=video_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)
        return pred_action[:, 1:] if has_proprio else pred_action

    def _training_loss_hierarchical(self, sample, tiled: bool = False):
        required = {"video", "keyframe", "action", "context", "context_mask"}
        missing = required - set(sample.keys())
        if missing:
            raise ValueError(f"Hierarchical training requires keys {sorted(required)}, missing {sorted(missing)}.")

        video = sample["video"]
        keyframe = sample["keyframe"]
        action = sample["action"]
        proprio = sample.get("proprio", None)
        image_is_pad = sample.get("image_is_pad", None)
        keyframe_is_pad = sample.get("keyframe_is_pad", None)
        action_is_pad = sample.get("action_is_pad", None)
        if video.ndim != 5 or keyframe.ndim != 5 or action.ndim != 3:
            raise ValueError(
                "Hierarchical mode expects video/keyframe/action as "
                f"[B,3,T,H,W]/[B,3,T,H,W]/[B,T,d], got {tuple(video.shape)}, "
                f"{tuple(keyframe.shape)}, {tuple(action.shape)}"
            )
        batch_size = int(video.shape[0])
        action_horizon = int(action.shape[1])
        if action_horizon != int(self.hierarchical_action_horizon):
            raise ValueError(
                "New hierarchical training expects one sample per action horizon. "
                f"Got action_horizon={action_horizon}, configured horizon={self.hierarchical_action_horizon}."
            )
        if video.shape[2] != 9:
            raise ValueError(f"New hierarchical training expects low video with 9 frames, got {video.shape[2]}.")
        if keyframe.shape[2] != 17:
            raise ValueError(f"New hierarchical training expects keyframe with 17 frames, got {keyframe.shape[2]}.")
        if proprio is not None and (proprio.ndim != 3 or int(proprio.shape[1]) != action_horizon):
            raise ValueError(f"Expected proprio shape [B,{action_horizon},d], got {tuple(proprio.shape)}.")

        context = sample["context"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = sample["context_mask"].to(device=self.device, dtype=torch.bool, non_blocking=True)
        video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        keyframe = keyframe.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if proprio is not None:
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if keyframe_is_pad is not None:
            keyframe_is_pad = keyframe_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        keyframe_latents = self._encode_video_latents(keyframe, tiled=tiled)
        low_latents = self._encode_video_latents(video, tiled=tiled)
        if int(keyframe_latents.shape[2]) != 5:
            raise ValueError(f"Expected encoded keyframe to have 5 latent frames, got {keyframe_latents.shape[2]}.")
        if int(low_latents.shape[2]) != 3:
            raise ValueError(f"Expected encoded video to have 3 latent frames, got {low_latents.shape[2]}.")

        history_key_latents = 3
        predict_key_latents = int(keyframe_latents.shape[2]) - history_key_latents
        timestep_key = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=keyframe_latents.dtype,
        )
        noise_key = torch.randn_like(keyframe_latents)
        noisy_key = keyframe_latents.clone()
        noisy_key_predict = self.train_video_scheduler.add_noise(
            keyframe_latents[:, :, history_key_latents:],
            noise_key[:, :, history_key_latents:],
            timestep_key,
        )
        noisy_key[:, :, history_key_latents:] = noisy_key_predict
        target_key = self.train_video_scheduler.training_target(
            keyframe_latents,
            noise_key,
            timestep_key,
        )
        timestep_key_matrix = torch.zeros(
            (batch_size, int(keyframe_latents.shape[2])),
            dtype=keyframe_latents.dtype,
            device=self.device,
        )
        timestep_key_matrix[:, history_key_latents:] = timestep_key.unsqueeze(1).expand(-1, predict_key_latents)
        noisy_key, timestep_key_matrix = self._perturb_history_latents_for_training(
            noisy_key,
            timestep_key_matrix,
            clean_latents=keyframe_latents,
            history_frames=history_key_latents,
            preserve_initial_frame=fuse_flag,
            probability=0.5,
            max_noise_scale=0.5,
        )

        timestep_low = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=low_latents.dtype,
        )
        noise_low = torch.randn_like(low_latents)
        noisy_low = self.train_video_scheduler.add_noise(low_latents, noise_low, timestep_low)
        target_low = self.train_video_scheduler.training_target(low_latents, noise_low, timestep_low)
        low_timestep_matrix = timestep_low.unsqueeze(1).expand(-1, int(low_latents.shape[2])).clone()
        if fuse_flag:
            noisy_low[:, :, 0:1] = low_latents[:, :, 0:1]
            low_timestep_matrix[:, 0] = 0

        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noise_action = torch.randn_like(action)
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        has_proprio = self.proprio_encoder is not None
        proprio_token = None
        if has_proprio:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            proprio_token = self._encode_proprio_action_token(
                proprio=proprio[:, 0, :],
                dtype=context.dtype,
            )

        visible_high_start, visible_high_end = self._downstream_high_visible_range(history_key_latents)
        pred_key, pred_low, pred_action = self._predict_low_action_joint_noise(
            keyframe_cond_latents=noisy_key,
            low_latents=noisy_low,
            action_latents=noisy_action,
            keyframe_timestep=timestep_key_matrix,
            low_timestep=low_timestep_matrix,
            action_timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            proprio_token=proprio_token,
            fuse_vae_embedding_in_latents=fuse_flag,
            history_high_frames=history_key_latents,
            visible_high_start=visible_high_start,
            visible_high_end=visible_high_end,
        )

        key_temporal_weights = torch.zeros(
            (batch_size, int(keyframe_latents.shape[2])),
            dtype=keyframe_latents.dtype,
            device=self.device,
        )
        key_temporal_weights[:, history_key_latents:] = 1.0
        key_loss_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_key,
            target_video=target_key,
            image_is_pad=keyframe_is_pad,
            include_initial_video_step=True,
            temporal_weights=key_temporal_weights,
        )
        low_loss_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_low[:, :, 1:] if fuse_flag else pred_low,
            target_video=target_low[:, :, 1:] if fuse_flag else target_low,
            image_is_pad=image_is_pad,
            include_initial_video_step=not fuse_flag,
        )

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        key_weight = self.train_video_scheduler.training_weight(timestep_key).to(
            device=self.device,
            dtype=key_loss_per_sample.dtype,
        )
        low_weight = self.train_video_scheduler.training_weight(timestep_low).to(
            device=self.device,
            dtype=low_loss_per_sample.dtype,
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            device=self.device,
            dtype=action_loss_per_sample.dtype,
        )

        loss_high = (key_loss_per_sample * key_weight).mean()
        loss_low = (low_loss_per_sample * low_weight).mean()
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_video = loss_high + loss_low
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_video_high": self.loss_lambda_video * float(loss_high.detach().item()),
            "loss_video_low": self.loss_lambda_video * float(loss_low.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    def training_loss(self, sample, tiled: bool = False):
        return self._training_loss_hierarchical(sample, tiled=tiled)








    @staticmethod
    def _truncate_inference_schedule(
        timesteps: torch.Tensor,
        deltas: torch.Tensor,
        keep_steps: Optional[int],
        *,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if keep_steps is None:
            return timesteps, deltas
        keep_steps = int(keep_steps)
        total_steps = int(timesteps.shape[0])
        if keep_steps < 0:
            raise ValueError(f"`{name}` must be non-negative when provided, got {keep_steps}.")
        if keep_steps > total_steps:
            raise ValueError(
                f"`{name}`={keep_steps} exceeds scheduled steps {total_steps}."
            )
        return timesteps[:keep_steps], deltas[:keep_steps]

    @staticmethod
    def _next_schedule_timestep(
        *,
        full_timesteps: torch.Tensor,
        used_steps: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        total_steps = int(full_timesteps.shape[0])
        if used_steps < 0 or used_steps > total_steps:
            raise ValueError(
                f"Invalid used_steps={used_steps} for schedule length {total_steps}."
            )
        if used_steps >= total_steps:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return full_timesteps[used_steps].to(device=device, dtype=dtype)


    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        high_video_inference_steps: Optional[int] = None,
        low_video_inference_steps: Optional[int] = None,
        action_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        return_high_level_video: bool = False,
        test_action_with_infer_action: bool = True,
        gt_video: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        del action, test_action_with_infer_action
        return self.infer_hierarchical(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            gt_video=gt_video,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            high_video_inference_steps=high_video_inference_steps,
            low_video_inference_steps=low_video_inference_steps,
            action_inference_steps=action_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            return_high_level_video=return_high_level_video,
        )




    @torch.no_grad()
    def _build_keyframe_history_video_for_infer(
        self,
        *,
        input_image: torch.Tensor,
        observed_chunk_videos: Optional[list[torch.Tensor]] = None,
        gt_keyframe: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if gt_keyframe is not None:
            keyframe = gt_keyframe
            if keyframe.ndim == 4:
                keyframe = keyframe.unsqueeze(0)
            if keyframe.ndim != 5 or keyframe.shape[0] != 1 or keyframe.shape[1] != 3:
                raise ValueError(f"`gt_keyframe` must be [3,T,H,W] or [1,3,T,H,W], got {tuple(gt_keyframe.shape)}.")
            if int(keyframe.shape[2]) < 9:
                raise ValueError(f"`gt_keyframe` must contain at least 9 history frames, got {keyframe.shape[2]}.")
            return keyframe[:, :, :9].to(device=self.device, dtype=self.torch_dtype)

        frames: list[torch.Tensor] = []
        if observed_chunk_videos is not None:
            for idx, observed in enumerate(observed_chunk_videos):
                if observed is None:
                    continue
                if observed.ndim == 3:
                    if observed.shape[0] != 3:
                        raise ValueError(f"Observed frame at {idx} must be [3,H,W], got {tuple(observed.shape)}.")
                    frames.append(observed.to(device=self.device, dtype=self.torch_dtype))
                elif observed.ndim == 4:
                    if observed.shape[0] == 3:
                        video = observed.unsqueeze(0)
                    elif observed.shape[1] == 3:
                        video = observed.permute(1, 0, 2, 3).unsqueeze(0)
                    else:
                        raise ValueError(f"Observed 4D item at {idx} must be [3,T,H,W] or [T,3,H,W], got {tuple(observed.shape)}.")
                    video = video.to(device=self.device, dtype=self.torch_dtype)
                    for t in range(int(video.shape[2])):
                        frames.append(video[0, :, t])
                elif observed.ndim == 5:
                    if observed.shape[0] != 1 or observed.shape[1] != 3:
                        raise ValueError(f"Observed 5D item at {idx} must be [1,3,T,H,W], got {tuple(observed.shape)}.")
                    video = observed.to(device=self.device, dtype=self.torch_dtype)
                    for t in range(int(video.shape[2])):
                        frames.append(video[0, :, t])
                else:
                    raise ValueError(f"Invalid observed item at {idx}: {tuple(observed.shape)}.")

        current = input_image[0].to(device=self.device, dtype=self.torch_dtype)
        if len(frames) == 0 or not torch.equal(frames[-1], current):
            frames.append(current)
        frames = frames[-9:]
        if len(frames) < 9:
            frames = [frames[0]] * (9 - len(frames)) + frames
        return torch.stack(frames, dim=1).unsqueeze(0)

    @torch.no_grad()
    def _infer_hierarchical_action_horizon(
        self,
        *,
        input_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        proprio: Optional[torch.Tensor],
        observed_chunk_videos: Optional[list[torch.Tensor]],
        gt_keyframe: Optional[torch.Tensor],
        high_video_inference_steps: Optional[int],
        low_video_inference_steps: Optional[int],
        high_denoise_step: Optional[int],
        low_denoise_step: Optional[int],
        action_inference_steps: Optional[int],
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        rand_device: str,
        tiled: bool,
        full_visualization: bool = False,
    ) -> dict[str, Any]:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}.")
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width) or checked_t != num_video_frames:
            raise ValueError(
                f"Invalid infer shape: expected H/W multiples of 16 and T%4==1, got HxW=({height},{width}), T={num_video_frames}."
            )
        if int(num_video_frames) != 9:
            raise ValueError(f"New hierarchical infer expects 9 low video frames, got {num_video_frames}.")
        if int(action_horizon) != int(self.hierarchical_action_horizon):
            raise ValueError(
                f"New hierarchical infer expects action_horizon={self.hierarchical_action_horizon}, got {action_horizon}."
            )

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        key_history_video = self._build_keyframe_history_video_for_infer(
            input_image=input_image,
            observed_chunk_videos=observed_chunk_videos,
            gt_keyframe=gt_keyframe,
        )
        key_history_latents = self._encode_video_latents(key_history_video, tiled=tiled)
        if int(key_history_latents.shape[2]) != 3:
            raise ValueError(f"Expected encoded keyframe history to have 3 latent frames, got {key_history_latents.shape[2]}.")
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        z_dim = self.vae.model.z_dim
        g_video = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        g_action = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)

        key_latents = torch.randn(
            (1, z_dim, 5, latent_h, latent_w),
            generator=g_video,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        key_latents[:, :, :3] = key_history_latents
        fixed_key_history = key_history_latents.clone()

        high_steps = max(1, int(high_video_inference_steps if high_video_inference_steps is not None else num_inference_steps))
        high_timesteps_full, high_deltas_full = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=high_steps,
            device=self.device,
            dtype=key_latents.dtype,
            shift_override=sigma_shift,
        )
        run_high_denoise = full_visualization or not self.hierarchical_mask_high_predict
        effective_high_denoise_step = None if full_visualization else high_denoise_step
        high_timesteps, high_deltas = self._truncate_inference_schedule(
            high_timesteps_full,
            high_deltas_full,
            effective_high_denoise_step,
            name="high_denoise_step",
        )
        if not run_high_denoise:
            high_timesteps = high_timesteps[:0]
            high_deltas = high_deltas[:0]
        high_cache_t = self._next_schedule_timestep(
            full_timesteps=high_timesteps_full,
            used_steps=int(high_timesteps.shape[0]),
            dtype=key_latents.dtype,
            device=self.device,
        )
        high_prev_denoise_predictions: list[torch.Tensor] = []
        high_skip_countdown = 0
        prev_high_pred: Optional[torch.Tensor] = None
        for step_idx_high, (step_t_high, step_d_high) in enumerate(zip(high_timesteps, high_deltas)):
            t_high = self._build_history_predict_timestep(
                step_timestep=step_t_high,
                total_frames=5,
                history_frames=3,
                dtype=key_latents.dtype,
                device=self.device,
            )
            should_run_high, high_skip_countdown = self._should_run_diffusion_step(
                prev_predictions=high_prev_denoise_predictions,
                skip_countdown=high_skip_countdown,
                enabled=self.hierarchical_dynamic_skip,
            )
            if should_run_high:
                if self.hierarchical_mask_high_predict:
                    pred_high = self._predict_high_video_noise_masked(
                        latents_video=key_latents,
                        timestep_video=t_high,
                        context=context,
                        context_mask=context_mask,
                        history_frames=3,
                        fuse_vae_embedding_in_latents=fuse_flag,
                    )
                else:
                    pred_high = self._predict_video_only_noise(
                        latents_video=key_latents,
                        timestep_video=t_high,
                        context=context,
                        context_mask=context_mask,
                        fuse_vae_embedding_in_latents=fuse_flag,
                    )
                prev_high_pred = pred_high
                high_prev_denoise_predictions.append(pred_high[:, :, 3:].detach())
                if len(high_prev_denoise_predictions) > 2:
                    high_prev_denoise_predictions.pop(0)
            else:
                if prev_high_pred is None:
                    raise RuntimeError("Dynamic skip requested before a keyframe prediction is available.")
                pred_high = prev_high_pred
            key_latents = self.infer_video_scheduler.step(pred_high, step_d_high, key_latents)
            key_latents[:, :, :3] = fixed_key_history

        key_timestep = self._build_history_predict_timestep(
            step_timestep=high_cache_t,
            total_frames=5,
            history_frames=3,
            dtype=key_latents.dtype,
            device=self.device,
        )
        (
            downstream_key_latents,
            downstream_key_timestep,
            downstream_history_frames,
            downstream_visible_start,
            downstream_visible_end,
        ) = self._select_downstream_keyframes(
            key_latents,
            key_timestep,
            history_frames=3,
        )
        selected_key_indices, downstream_history_frames = self._downstream_keyframe_indices(3)
        keyframe_cache: list[dict[str, torch.Tensor]]
        keyframe_seq_len: int
        cache_tokens_per_frame: int
        if self.hierarchical_mask_high_predict:
            history_keyframe_cache, _, history_keyframe_tokens_per_frame = self._prefill_keyframe_cache(
                keyframe_cond_latents=key_latents[:, :, :3],
                keyframe_timestep=key_timestep[:, :3],
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                history_high_frames=3,
                visible_high_start=0,
                visible_high_end=2,
            )
            keyframe_cache = self._slice_video_kv_cache(
                history_keyframe_cache,
                frame_indices=selected_key_indices,
                tokens_per_frame=history_keyframe_tokens_per_frame,
            )
            cache_tokens_per_frame = history_keyframe_tokens_per_frame
            keyframe_seq_len = len(selected_key_indices) * cache_tokens_per_frame
        else:
            all_keyframe_cache, _, cache_tokens_per_frame = self._prefill_keyframe_cache(
                keyframe_cond_latents=key_latents,
                keyframe_timestep=key_timestep,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                history_high_frames=3,
                visible_high_start=0,
                visible_high_end=int(key_latents.shape[2]) - 1,
            )
            keyframe_cache = self._slice_video_kv_cache(
                all_keyframe_cache,
                frame_indices=selected_key_indices,
                tokens_per_frame=cache_tokens_per_frame,
            )
            keyframe_seq_len = len(selected_key_indices) * cache_tokens_per_frame
        low_latents = torch.randn(
            (1, z_dim, 3, latent_h, latent_w),
            generator=g_video,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        low_latents[:, :, 0:1] = first_frame_latents.clone()
        action_latents = torch.randn(
            (1, int(action_horizon), self.action_expert.action_dim),
            generator=g_action,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        low_steps = max(1, int(low_video_inference_steps if low_video_inference_steps is not None else num_inference_steps))
        action_steps = max(1, int(action_inference_steps if action_inference_steps is not None else num_inference_steps))
        if full_visualization:
            low_steps = action_steps
        if low_steps < action_steps:
            raise ValueError(
                "`low_video_inference_steps` must be >= `action_inference_steps` for hierarchical infer_action, "
                f"got {low_steps} and {action_steps}."
            )
        effective_low_denoise_step = None if full_visualization else low_denoise_step
        low_actual_steps = action_steps if effective_low_denoise_step is None else int(effective_low_denoise_step)
        if low_actual_steps < 0 or low_actual_steps > action_steps:
            raise ValueError(
                "`low_denoise_step` must be in [0, action_inference_steps] when provided, "
                f"got low_denoise_step={low_actual_steps}, action_inference_steps={action_steps}."
            )
        low_timesteps_full, low_deltas_full = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=low_steps,
            device=self.device,
            dtype=low_latents.dtype,
            shift_override=sigma_shift,
        )
        low_timesteps, low_deltas = self._truncate_inference_schedule(
            low_timesteps_full,
            low_deltas_full,
            low_actual_steps,
            name="low_denoise_step",
        )
        action_timesteps, action_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=action_steps,
            device=self.device,
            dtype=action_latents.dtype,
            shift_override=sigma_shift,
        )

        proprio_token = None
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`proprio` must be provided when `proprio_dim` is enabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and int(proprio.shape[0]) != 1:
                proprio = proprio[:1]
            elif proprio.ndim == 3:
                proprio = proprio[:, 0, :]
            if proprio.ndim != 2 or int(proprio.shape[1]) != int(self.proprio_dim):
                raise ValueError(f"Expected proprio shape [1,{self.proprio_dim}], got {tuple(proprio.shape)}.")
            proprio_token = self._encode_proprio_action_token(
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
                dtype=context.dtype,
            )

        joint_prev_denoise_predictions: list[torch.Tensor] = []
        joint_skip_countdown = 0
        prev_low_pred: Optional[torch.Tensor] = None
        prev_action_pred: Optional[torch.Tensor] = None
        joint_low_steps = 0 if (self.hierarchical_mask_low_predict and not full_visualization) else low_actual_steps
        for step_idx in range(joint_low_steps):
            t_low = low_timesteps[step_idx].reshape(1).to(device=self.device, dtype=low_latents.dtype)
            t_action = action_timesteps[step_idx].reshape(1).to(device=self.device, dtype=action_latents.dtype)
            low_timestep = t_low.unsqueeze(1).expand(-1, 3).clone()
            low_timestep[:, 0] = 0
            should_run_joint, joint_skip_countdown = self._should_run_diffusion_step(
                prev_predictions=joint_prev_denoise_predictions,
                skip_countdown=joint_skip_countdown,
                enabled=self.hierarchical_dynamic_skip,
            )
            if should_run_joint:
                pred_low, pred_action = self._predict_low_action_with_keyframe_cache(
                    keyframe_cache=keyframe_cache,
                    keyframe_seq_len=keyframe_seq_len,
                    keyframe_frames=int(downstream_key_latents.shape[2]),
                    low_latents=low_latents,
                    action_latents=action_latents,
                    low_timestep=low_timestep,
                    action_timestep=t_action,
                    context=context,
                    context_mask=context_mask,
                    proprio_token=proprio_token,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    tokens_per_frame=cache_tokens_per_frame,
                    history_high_frames=downstream_history_frames,
                    visible_high_start=downstream_visible_start,
                    visible_high_end=downstream_visible_end,
                )
                prev_low_pred = pred_low
                joint_prev_denoise_predictions.append(pred_low[:, :, 1:].detach())
                if len(joint_prev_denoise_predictions) > 2:
                    joint_prev_denoise_predictions.pop(0)
            else:
                if prev_low_pred is None or prev_action_pred is None:
                    raise RuntimeError("Dynamic skip requested before a low/action prediction is available.")
                pred_low = prev_low_pred
                pred_action = prev_action_pred
            low_latents = self.infer_video_scheduler.step(pred_low, low_deltas[step_idx], low_latents)
            low_latents[:, :, 0:1] = first_frame_latents.clone()
            action_latents = self.infer_action_scheduler.step(pred_action, action_deltas[step_idx], action_latents)
            prev_action_pred = pred_action

        action_only_start = joint_low_steps
        if action_only_start < action_steps:
            cache_used_steps = 0 if (self.hierarchical_mask_low_predict and not full_visualization) else low_actual_steps
            cache_t_low = self._next_schedule_timestep(
                full_timesteps=low_timesteps_full,
                used_steps=cache_used_steps,
                dtype=low_latents.dtype,
                device=self.device,
            )
            low_cache_timestep = cache_t_low.reshape(1).unsqueeze(1).expand(-1, 3).clone()
            low_cache_timestep[:, 0] = 0
            cache_low_latents = low_latents[:, :, :1] if self.hierarchical_mask_low_predict else low_latents
            cache_low_timestep = low_cache_timestep[:, :1] if self.hierarchical_mask_low_predict else low_cache_timestep
            low_cache, _, _ = self._prefill_low_cache_with_keyframe_cache(
                keyframe_cache=keyframe_cache,
                keyframe_seq_len=keyframe_seq_len,
                keyframe_frames=int(downstream_key_latents.shape[2]),
                low_latents=cache_low_latents,
                low_timestep=cache_low_timestep,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                tokens_per_frame=cache_tokens_per_frame,
                history_high_frames=downstream_history_frames,
                visible_high_start=downstream_visible_start,
                visible_high_end=downstream_visible_end,
            )
            video_cache = self._concat_video_kv_caches(keyframe_cache, low_cache)
            video_seq_len = keyframe_seq_len + int(cache_low_latents.shape[2]) * cache_tokens_per_frame
            for step_idx in range(action_only_start, action_steps):
                t_action = action_timesteps[step_idx].reshape(1).to(device=self.device, dtype=action_latents.dtype)
                pred_action = self._predict_action_with_frozen_video_cache(
                    action_latents=action_latents,
                    action_timestep=t_action,
                    context=context,
                    context_mask=context_mask,
                    video_cache=video_cache,
                    video_seq_len=video_seq_len,
                    tokens_per_frame=cache_tokens_per_frame,
                    high_frames=int(downstream_key_latents.shape[2]),
                    low_frames=int(cache_low_latents.shape[2]),
                    history_high_frames=downstream_history_frames,
                    visible_high_start=downstream_visible_start,
                    visible_high_end=downstream_visible_end,
                    proprio_token=proprio_token,
                )
                action_latents = self.infer_action_scheduler.step(pred_action, action_deltas[step_idx], action_latents)

        return {
            "action": action_latents[0].detach().to(device="cpu", dtype=torch.float32),
            "low_latents": low_latents,
            "keyframe_latents": key_latents,
        }

    @torch.no_grad()
    def _infer_hierarchical_joint_denoise_action_horizon(
        self,
        *,
        input_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        proprio: Optional[torch.Tensor],
        observed_chunk_videos: Optional[list[torch.Tensor]],
        gt_keyframe: Optional[torch.Tensor],
        high_video_inference_steps: Optional[int],
        low_video_inference_steps: Optional[int],
        high_denoise_step: Optional[int],
        low_denoise_step: Optional[int],
        action_inference_steps: Optional[int],
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        rand_device: str,
        tiled: bool,
    ) -> dict[str, Any]:
        if self.hierarchical_mask_high_predict or self.hierarchical_mask_low_predict:
            raise ValueError("Joint denoise infer_action requires both hierarchical masks to be disabled.")
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}.")
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width) or checked_t != num_video_frames:
            raise ValueError(
                f"Invalid infer shape: expected H/W multiples of 16 and T%4==1, got HxW=({height},{width}), T={num_video_frames}."
            )
        if int(num_video_frames) != 9:
            raise ValueError(f"New hierarchical infer expects 9 low video frames, got {num_video_frames}.")
        if int(action_horizon) != int(self.hierarchical_action_horizon):
            raise ValueError(
                f"New hierarchical infer expects action_horizon={self.hierarchical_action_horizon}, got {action_horizon}."
            )

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        key_history_video = self._build_keyframe_history_video_for_infer(
            input_image=input_image,
            observed_chunk_videos=observed_chunk_videos,
            gt_keyframe=gt_keyframe,
        )
        key_history_latents = self._encode_video_latents(key_history_video, tiled=tiled)
        if int(key_history_latents.shape[2]) != 3:
            raise ValueError(f"Expected encoded keyframe history to have 3 latent frames, got {key_history_latents.shape[2]}.")

        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        z_dim = self.vae.model.z_dim
        g_video = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        g_action = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)

        key_latents = torch.randn(
            (1, z_dim, 5, latent_h, latent_w),
            generator=g_video,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        key_latents[:, :, :3] = key_history_latents
        fixed_key_history = key_history_latents.clone()

        low_latents = torch.randn(
            (1, z_dim, 3, latent_h, latent_w),
            generator=g_video,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        low_latents[:, :, 0:1] = first_frame_latents.clone()

        action_latents = torch.randn(
            (1, int(action_horizon), self.action_expert.action_dim),
            generator=g_action,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        high_steps = max(1, int(high_video_inference_steps if high_video_inference_steps is not None else num_inference_steps))
        low_steps = max(1, int(low_video_inference_steps if low_video_inference_steps is not None else num_inference_steps))
        action_steps = max(1, int(action_inference_steps if action_inference_steps is not None else num_inference_steps))
        high_timesteps_full, high_deltas_full = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=high_steps,
            device=self.device,
            dtype=key_latents.dtype,
            shift_override=sigma_shift,
        )
        low_timesteps_full, low_deltas_full = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=low_steps,
            device=self.device,
            dtype=low_latents.dtype,
            shift_override=sigma_shift,
        )
        high_timesteps, high_deltas = self._truncate_inference_schedule(
            high_timesteps_full,
            high_deltas_full,
            high_denoise_step,
            name="high_denoise_step",
        )
        low_timesteps, low_deltas = self._truncate_inference_schedule(
            low_timesteps_full,
            low_deltas_full,
            low_denoise_step,
            name="low_denoise_step",
        )
        action_timesteps, action_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=action_steps,
            device=self.device,
            dtype=action_latents.dtype,
            shift_override=sigma_shift,
        )
        high_actual_steps = int(high_timesteps.shape[0])
        low_actual_steps = int(low_timesteps.shape[0])
        if high_actual_steps > low_actual_steps or low_actual_steps > action_steps:
            raise ValueError(
                "Joint denoise infer_action requires "
                "high_denoise_steps <= low_denoise_steps <= action_inference_steps, "
                f"got {high_actual_steps}, {low_actual_steps}, {action_steps}."
            )

        proprio_token = None
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`proprio` must be provided when `proprio_dim` is enabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and int(proprio.shape[0]) != 1:
                proprio = proprio[:1]
            elif proprio.ndim == 3:
                proprio = proprio[:, 0, :]
            if proprio.ndim != 2 or int(proprio.shape[1]) != int(self.proprio_dim):
                raise ValueError(f"Expected proprio shape [1,{self.proprio_dim}], got {tuple(proprio.shape)}.")
            proprio_token = self._encode_proprio_action_token(
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
                dtype=context.dtype,
            )

        visible_high_start, visible_high_end = self._downstream_high_visible_range(3)
        joint_prev_denoise_predictions: list[torch.Tensor] = []
        joint_skip_countdown = 0
        prev_pred_high: Optional[torch.Tensor] = None
        prev_pred_low: Optional[torch.Tensor] = None
        prev_pred_action: Optional[torch.Tensor] = None

        for step_idx in range(high_actual_steps):
            t_high = self._build_history_predict_timestep(
                step_timestep=high_timesteps[step_idx],
                total_frames=5,
                history_frames=3,
                dtype=key_latents.dtype,
                device=self.device,
            )
            t_low = low_timesteps[step_idx].reshape(1).to(device=self.device, dtype=low_latents.dtype)
            t_action = action_timesteps[step_idx].reshape(1).to(device=self.device, dtype=action_latents.dtype)
            low_timestep = t_low.unsqueeze(1).expand(-1, 3).clone()
            low_timestep[:, 0] = 0
            should_run_joint, joint_skip_countdown = self._should_run_diffusion_step(
                prev_predictions=joint_prev_denoise_predictions,
                skip_countdown=joint_skip_countdown,
                enabled=self.hierarchical_dynamic_skip,
            )
            if should_run_joint:
                pred_high, pred_low, pred_action = self._predict_low_action_joint_noise(
                    keyframe_cond_latents=key_latents,
                    low_latents=low_latents,
                    action_latents=action_latents,
                    keyframe_timestep=t_high,
                    low_timestep=low_timestep,
                    action_timestep=t_action,
                    context=context,
                    context_mask=context_mask,
                    proprio_token=proprio_token,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    history_high_frames=3,
                    visible_high_start=visible_high_start,
                    visible_high_end=visible_high_end,
                )
                prev_pred_high = pred_high
                prev_pred_low = pred_low
                prev_pred_action = pred_action
                joint_prev_denoise_predictions.append(
                    torch.cat([pred_high[:, :, 3:].flatten(1), pred_low[:, :, 1:].flatten(1)], dim=1).detach()
                )
                if len(joint_prev_denoise_predictions) > 2:
                    joint_prev_denoise_predictions.pop(0)
            else:
                if prev_pred_high is None or prev_pred_low is None or prev_pred_action is None:
                    raise RuntimeError("Dynamic skip requested before a joint prediction is available.")
                pred_high = prev_pred_high
                pred_low = prev_pred_low
                pred_action = prev_pred_action

            key_latents = self.infer_video_scheduler.step(pred_high, high_deltas[step_idx], key_latents)
            key_latents[:, :, :3] = fixed_key_history
            low_latents = self.infer_video_scheduler.step(pred_low, low_deltas[step_idx], low_latents)
            low_latents[:, :, 0:1] = first_frame_latents.clone()
            action_latents = self.infer_action_scheduler.step(pred_action, action_deltas[step_idx], action_latents)

        key_cache_t = self._next_schedule_timestep(
            full_timesteps=high_timesteps_full,
            used_steps=high_actual_steps,
            dtype=key_latents.dtype,
            device=self.device,
        )
        key_timestep = self._build_history_predict_timestep(
            step_timestep=key_cache_t,
            total_frames=5,
            history_frames=3,
            dtype=key_latents.dtype,
            device=self.device,
        )
        (
            downstream_key_latents,
            _downstream_key_timestep,
            downstream_history_frames,
            downstream_visible_start,
            downstream_visible_end,
        ) = self._select_downstream_keyframes(
            key_latents,
            key_timestep,
            history_frames=3,
        )
        selected_key_indices, _ = self._downstream_keyframe_indices(3)
        all_keyframe_cache, _, cache_tokens_per_frame = self._prefill_keyframe_cache(
            keyframe_cond_latents=key_latents,
            keyframe_timestep=key_timestep,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
            history_high_frames=3,
            visible_high_start=0,
            visible_high_end=int(key_latents.shape[2]) - 1,
        )
        keyframe_cache = self._slice_video_kv_cache(
            all_keyframe_cache,
            frame_indices=selected_key_indices,
            tokens_per_frame=cache_tokens_per_frame,
        )
        keyframe_seq_len = len(selected_key_indices) * cache_tokens_per_frame

        low_prev_denoise_predictions: list[torch.Tensor] = []
        low_skip_countdown = 0
        prev_low_pred: Optional[torch.Tensor] = None
        prev_action_pred: Optional[torch.Tensor] = None
        for step_idx in range(high_actual_steps, low_actual_steps):
            t_low = low_timesteps[step_idx].reshape(1).to(device=self.device, dtype=low_latents.dtype)
            t_action = action_timesteps[step_idx].reshape(1).to(device=self.device, dtype=action_latents.dtype)
            low_timestep = t_low.unsqueeze(1).expand(-1, 3).clone()
            low_timestep[:, 0] = 0
            should_run_low, low_skip_countdown = self._should_run_diffusion_step(
                prev_predictions=low_prev_denoise_predictions,
                skip_countdown=low_skip_countdown,
                enabled=self.hierarchical_dynamic_skip,
            )
            if should_run_low:
                pred_low, pred_action = self._predict_low_action_with_keyframe_cache(
                    keyframe_cache=keyframe_cache,
                    keyframe_seq_len=keyframe_seq_len,
                    keyframe_frames=int(downstream_key_latents.shape[2]),
                    low_latents=low_latents,
                    action_latents=action_latents,
                    low_timestep=low_timestep,
                    action_timestep=t_action,
                    context=context,
                    context_mask=context_mask,
                    proprio_token=proprio_token,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    tokens_per_frame=cache_tokens_per_frame,
                    history_high_frames=downstream_history_frames,
                    visible_high_start=downstream_visible_start,
                    visible_high_end=downstream_visible_end,
                )
                prev_low_pred = pred_low
                prev_action_pred = pred_action
                low_prev_denoise_predictions.append(pred_low[:, :, 1:].detach())
                if len(low_prev_denoise_predictions) > 2:
                    low_prev_denoise_predictions.pop(0)
            else:
                if prev_low_pred is None or prev_action_pred is None:
                    raise RuntimeError("Dynamic skip requested before a low/action prediction is available.")
                pred_low = prev_low_pred
                pred_action = prev_action_pred

            low_latents = self.infer_video_scheduler.step(pred_low, low_deltas[step_idx], low_latents)
            low_latents[:, :, 0:1] = first_frame_latents.clone()
            action_latents = self.infer_action_scheduler.step(pred_action, action_deltas[step_idx], action_latents)

        if low_actual_steps < action_steps:
            cache_t_low = self._next_schedule_timestep(
                full_timesteps=low_timesteps_full,
                used_steps=low_actual_steps,
                dtype=low_latents.dtype,
                device=self.device,
            )
            low_cache_timestep = cache_t_low.reshape(1).unsqueeze(1).expand(-1, 3).clone()
            low_cache_timestep[:, 0] = 0
            low_cache, _, _ = self._prefill_low_cache_with_keyframe_cache(
                keyframe_cache=keyframe_cache,
                keyframe_seq_len=keyframe_seq_len,
                keyframe_frames=int(downstream_key_latents.shape[2]),
                low_latents=low_latents,
                low_timestep=low_cache_timestep,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                tokens_per_frame=cache_tokens_per_frame,
                history_high_frames=downstream_history_frames,
                visible_high_start=downstream_visible_start,
                visible_high_end=downstream_visible_end,
            )
            video_cache = self._concat_video_kv_caches(keyframe_cache, low_cache)
            video_seq_len = keyframe_seq_len + int(low_latents.shape[2]) * cache_tokens_per_frame
            for step_idx in range(low_actual_steps, action_steps):
                t_action = action_timesteps[step_idx].reshape(1).to(device=self.device, dtype=action_latents.dtype)
                pred_action = self._predict_action_with_frozen_video_cache(
                    action_latents=action_latents,
                    action_timestep=t_action,
                    context=context,
                    context_mask=context_mask,
                    video_cache=video_cache,
                    video_seq_len=video_seq_len,
                    tokens_per_frame=cache_tokens_per_frame,
                    high_frames=int(downstream_key_latents.shape[2]),
                    low_frames=int(low_latents.shape[2]),
                    history_high_frames=downstream_history_frames,
                    visible_high_start=downstream_visible_start,
                    visible_high_end=downstream_visible_end,
                    proprio_token=proprio_token,
                )
                action_latents = self.infer_action_scheduler.step(pred_action, action_deltas[step_idx], action_latents)

        return {
            "action": action_latents[0].detach().to(device="cpu", dtype=torch.float32),
            "low_latents": low_latents,
            "keyframe_latents": key_latents,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        observed_chunk_videos: Optional[list[torch.Tensor]] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        high_video_inference_steps: Optional[int] = None,
        low_video_inference_steps: Optional[int] = None,
        high_denoise_step: Optional[int] = None,
        low_denoise_step: Optional[int] = None,
        action_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        joint_denoise: bool = False,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale
        self.eval()
        context, context_mask = self._prepare_infer_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
        )
        if bool(joint_denoise) and not self.hierarchical_mask_high_predict and not self.hierarchical_mask_low_predict:
            out = self._infer_hierarchical_joint_denoise_action_horizon(
                input_image=input_image,
                context=context,
                context_mask=context_mask,
                action_horizon=int(action_horizon),
                num_video_frames=9 if num_video_frames is None else int(num_video_frames),
                proprio=proprio,
                observed_chunk_videos=observed_chunk_videos,
                gt_keyframe=None,
                high_video_inference_steps=high_video_inference_steps,
                low_video_inference_steps=low_video_inference_steps,
                high_denoise_step=high_denoise_step,
                low_denoise_step=low_denoise_step,
                action_inference_steps=action_inference_steps,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
            )
            return {"action": out["action"]}
        out = self._infer_hierarchical_action_horizon(
            input_image=input_image,
            context=context,
            context_mask=context_mask,
            action_horizon=int(action_horizon),
            num_video_frames=9 if num_video_frames is None else int(num_video_frames),
            proprio=proprio,
            observed_chunk_videos=observed_chunk_videos,
            gt_keyframe=None,
            high_video_inference_steps=high_video_inference_steps,
            low_video_inference_steps=low_video_inference_steps,
            high_denoise_step=high_denoise_step,
            low_denoise_step=low_denoise_step,
            action_inference_steps=action_inference_steps,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            full_visualization=False,
        )
        return {"action": out["action"]}

    @torch.no_grad()
    def infer_hierarchical(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: Optional[int],
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        high_video_inference_steps: Optional[int] = None,
        low_video_inference_steps: Optional[int] = None,
        high_denoise_step: Optional[int] = None,
        low_denoise_step: Optional[int] = None,
        action_inference_steps: Optional[int] = None,
        return_high_level_video: bool = False,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        gt_video: Optional[torch.Tensor] = None,
        gt_keyframe: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale, gt_video
        self.eval()
        context, context_mask = self._prepare_infer_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
        )
        action_horizon_cfg = int(self.hierarchical_action_horizon)
        eval_action_steps = max(1, int(action_inference_steps if action_inference_steps is not None else num_inference_steps))
        out = self._infer_hierarchical_action_horizon(
            input_image=input_image,
            context=context,
            context_mask=context_mask,
            action_horizon=action_horizon_cfg if action_horizon is None else int(action_horizon),
            num_video_frames=int(num_video_frames),
            proprio=proprio,
            observed_chunk_videos=None,
            gt_keyframe=gt_keyframe,
            high_video_inference_steps=high_video_inference_steps,
            low_video_inference_steps=eval_action_steps,
            high_denoise_step=None,
            low_denoise_step=None,
            action_inference_steps=eval_action_steps,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            full_visualization=True,
        )
        low_video = self._decode_latents_to_tensor(out["low_latents"], tiled=tiled)
        result = {
            "video": self._video_tensor_to_pil(low_video),
            "action": out["action"],
        }
        if return_high_level_video:
            key_video = self._decode_latents_to_tensor(out["keyframe_latents"], tiled=tiled)
            result["video_high"] = self._video_tensor_to_pil(key_video)
        return result

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        high_video_inference_steps: Optional[int] = None,
        low_video_inference_steps: Optional[int] = None,
        high_denoise_step: Optional[int] = None,
        low_denoise_step: Optional[int] = None,
        action_inference_steps: Optional[int] = None,
        return_high_level_video: bool = False,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        gt_video: Optional[torch.Tensor] = None,
        gt_keyframe: Optional[torch.Tensor] = None,
    ):
        del action, action_cfg_scale
        return self.infer_hierarchical(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            gt_video=gt_video,
            gt_keyframe=gt_keyframe,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            high_video_inference_steps=high_video_inference_steps,
            low_video_inference_steps=low_video_inference_steps,
            high_denoise_step=high_denoise_step,
            low_denoise_step=low_denoise_step,
            action_inference_steps=action_inference_steps,
            return_high_level_video=return_high_level_video,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
