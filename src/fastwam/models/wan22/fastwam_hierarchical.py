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
        hierarchical_num_chunks: int = 4,
        hierarchical_chunk_action_horizon: int = 32,
        hierarchical_high_stride: int = 8,
        hierarchical_low_stride: int = 4,
        hierarchical_mask_high_predict: bool = False,
        hierarchical_mask_low_predict: bool = False,
        hierarchical_high_select: str = "boundary",
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

        self.hierarchical_num_chunks = int(hierarchical_num_chunks)
        self.hierarchical_chunk_action_horizon = int(hierarchical_chunk_action_horizon)
        self.hierarchical_high_stride = int(hierarchical_high_stride)
        self.hierarchical_low_stride = int(hierarchical_low_stride)
        self.hierarchical_mask_high_predict = bool(hierarchical_mask_high_predict)
        self.hierarchical_mask_low_predict = bool(hierarchical_mask_low_predict)
        self.hierarchical_high_select = self._normalize_high_select(hierarchical_high_select)
        if self.hierarchical_num_chunks <= 0:
            raise ValueError("`hierarchical_num_chunks` must be positive.")
        if self.hierarchical_chunk_action_horizon <= 0:
            raise ValueError("`hierarchical_chunk_action_horizon` must be positive.")
        if self.hierarchical_high_stride <= 0 or self.hierarchical_low_stride <= 0:
            raise ValueError("`hierarchical_high_stride` and `hierarchical_low_stride` must be positive.")

        hidden_dim = int(getattr(self.video_expert, "hidden_dim"))
        self.hierarchical_level_embedding = nn.Embedding(2, hidden_dim).to(dtype=torch_dtype)
        nn.init.normal_(self.hierarchical_level_embedding.weight, mean=0.0, std=hidden_dim ** -0.5)

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
        hierarchical_num_chunks: int = 4,
        hierarchical_chunk_action_horizon: int = 32,
        hierarchical_high_stride: int = 8,
        hierarchical_low_stride: int = 4,
        hierarchical_mask_high_predict: bool = False,
        hierarchical_mask_low_predict: bool = False,
        hierarchical_high_select: str = "boundary",
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
            hierarchical_num_chunks=hierarchical_num_chunks,
            hierarchical_chunk_action_horizon=hierarchical_chunk_action_horizon,
            hierarchical_high_stride=hierarchical_high_stride,
            hierarchical_low_stride=hierarchical_low_stride,
            hierarchical_mask_high_predict=hierarchical_mask_high_predict,
            hierarchical_mask_low_predict=hierarchical_mask_low_predict,
            hierarchical_high_select=hierarchical_high_select,
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

    @staticmethod
    def _global_high_frame_positions(
        *,
        high_frames: int,
        block_chunks: int,
        low_frames_per_chunk: int,
        device: torch.device,
    ) -> torch.Tensor:
        if high_frames <= 0:
            raise ValueError(f"`high_frames` must be positive, got {high_frames}.")
        if high_frames == 1:
            return torch.zeros((1,), device=device, dtype=torch.long)
        low_stride = max(1, int(low_frames_per_chunk) - 1)
        block_end = max(int(high_frames) - 1, max(1, int(block_chunks)) * low_stride)
        return torch.linspace(0, block_end, steps=int(high_frames), device=device).round().long()

    @staticmethod
    def _normalize_high_select(value: str) -> str:
        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {
            "boundary": "boundary",
            "boundaries": "boundary",
            "start_end": "boundary",
            "first_last": "boundary",
            "both": "boundary",
            "end": "end",
            "last": "end",
            "target": "end",
            "tail": "end",
        }
        if normalized not in aliases:
            raise ValueError(
                "`hierarchical_high_select` must be one of "
                "{'boundary', 'end'} (aliases: start_end/first_last/both, last/target), "
                f"got {value!r}."
            )
        return aliases[normalized]

    def _high_cache_boundary_index(self, boundary_idx: int, high_frames: int) -> int:
        if high_frames <= 0:
            raise ValueError(f"`high_frames` must be positive, got {high_frames}.")
        # 根据当前设计：high level帧即为1个首帧 + 每个chunk的末尾帧
        # 故 chunk_idx 的边界直接一一对应 high level 的帧索引
        return max(0, min(int(boundary_idx), int(high_frames) - 1))

    def _select_high_cache_frame_indices(
        self,
        *,
        chunk_idx: int,
        high_frames: int,
        mask_high_predict: Optional[bool] = None,
    ) -> tuple[int, ...]:
        if high_frames <= 0:
            raise ValueError(f"`high_frames` must be positive, got {high_frames}.")
        chunk_idx = max(0, int(chunk_idx))
        start_frame = self._high_cache_boundary_index(chunk_idx, high_frames)
        end_frame = self._high_cache_boundary_index(chunk_idx + 1, high_frames)
        mask_high_predict = self.hierarchical_mask_high_predict if mask_high_predict is None else bool(mask_high_predict)
        if mask_high_predict:
            return (min(chunk_idx, int(high_frames) - 1),)

        if self.hierarchical_high_select == "boundary":
            if start_frame == end_frame:
                return (start_frame,)
            return (start_frame, end_frame)

        if self.hierarchical_high_select == "end":
            return (end_frame,)

        raise ValueError(f"Unsupported hierarchical_high_select={self.hierarchical_high_select!r}.")

    def _slice_high_cache_for_chunk(
        self,
        high_kv_cache: list[dict[str, torch.Tensor]],
        *,
        chunk_idx: int,
        high_frames: int,
        tokens_per_frame: int,
        mask_high_predict: Optional[bool] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], int, tuple[int, ...]]:
        frame_indices = self._select_high_cache_frame_indices(
            chunk_idx=chunk_idx,
            high_frames=high_frames,
            mask_high_predict=mask_high_predict,
        )
        frame_caches = [
            self._slice_video_kv_cache(
                high_kv_cache,
                token_start=int(frame_idx) * tokens_per_frame,
                token_end=(int(frame_idx) + 1) * tokens_per_frame,
            )
            for frame_idx in frame_indices
        ]
        if len(frame_caches) == 1:
            selected_cache = frame_caches[0]
        else:
            selected_cache = frame_caches[0]
            for next_cache in frame_caches[1:]:
                selected_cache = self._concat_video_kv_caches(selected_cache, next_cache)
        return selected_cache, len(frame_indices), frame_indices

    @staticmethod
    def _global_low_frame_positions(
        *,
        chunk_indices: int | torch.Tensor,
        low_frames: int,
        low_frames_per_chunk: int,
        device: torch.device,
    ) -> torch.Tensor:
        if low_frames <= 0:
            raise ValueError(f"`low_frames` must be positive, got {low_frames}.")
        low_stride = max(1, int(low_frames_per_chunk) - 1)
        local = torch.arange(int(low_frames), device=device, dtype=torch.long)
        if isinstance(chunk_indices, torch.Tensor):
            chunk_indices = chunk_indices.to(device=device, dtype=torch.long)
            if chunk_indices.ndim != 1:
                raise ValueError(f"`chunk_indices` must be 1D, got shape {tuple(chunk_indices.shape)}.")
            return chunk_indices.unsqueeze(1) * low_stride + local.unsqueeze(0)
        return int(chunk_indices) * low_stride + local

    def _video_freqs_for_frame_positions(
        self,
        pre_state: dict[str, Any],
        frame_positions: torch.Tensor,
    ) -> torch.Tensor:
        f, h, w = pre_state["meta"]["grid_size"]
        frame_positions = frame_positions.to(dtype=torch.long)
        if frame_positions.ndim == 1:
            if int(frame_positions.shape[0]) != int(f):
                raise ValueError(
                    f"Expected {f} frame positions, got shape {tuple(frame_positions.shape)}."
                )
        elif frame_positions.ndim == 2:
            batch_size = int(pre_state["tokens"].shape[0])
            if int(frame_positions.shape[0]) != batch_size or int(frame_positions.shape[1]) != int(f):
                raise ValueError(
                    "Batch frame-position shape mismatch: "
                    f"positions={tuple(frame_positions.shape)}, expected=({batch_size}, {f})."
                )
        else:
            raise ValueError(f"`frame_positions` must be 1D or 2D, got shape {tuple(frame_positions.shape)}.")

        device = pre_state["tokens"].device
        f_freqs, h_freqs, w_freqs = self.video_expert.freqs
        max_frame_pos = int(frame_positions.max().item())
        if max_frame_pos >= int(f_freqs.shape[0]):
            raise ValueError(
                f"Global video RoPE frame position {max_frame_pos} exceeds cache {int(f_freqs.shape[0])}."
            )
        if int(h) > int(h_freqs.shape[0]) or int(w) > int(w_freqs.shape[0]):
            raise ValueError(
                "Video RoPE spatial size exceeds cache: "
                f"grid=({h},{w}), cache=({int(h_freqs.shape[0])},{int(w_freqs.shape[0])})."
            )

        if frame_positions.ndim == 1:
            f_part = f_freqs.index_select(0, frame_positions.to(device=f_freqs.device))
            freqs = torch.cat([
                f_part.view(f, 1, 1, -1).expand(f, h, w, -1),
                h_freqs[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                w_freqs[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ], dim=-1).reshape(f * h * w, 1, -1)
            return freqs.to(device)

        batch_size = int(frame_positions.shape[0])
        f_part = f_freqs.index_select(0, frame_positions.reshape(-1).to(device=f_freqs.device))
        f_part = f_part.view(batch_size, f, -1)
        freqs = torch.cat([
            f_part.view(batch_size, f, 1, 1, -1).expand(batch_size, f, h, w, -1),
            h_freqs[:h].view(1, 1, h, 1, -1).expand(batch_size, f, h, w, -1),
            w_freqs[:w].view(1, 1, 1, w, -1).expand(batch_size, f, h, w, -1),
        ], dim=-1).reshape(batch_size, f * h * w, 1, -1)
        return freqs.to(device)

    def _add_hierarchical_level_embedding(self, tokens: torch.Tensor, level: str) -> torch.Tensor:
        if level == "high":
            level_idx = 0
        elif level == "low":
            level_idx = 1
        else:
            raise ValueError(f"Unsupported hierarchical level `{level}`.")
        emb = self.hierarchical_level_embedding.weight[level_idx].to(device=tokens.device, dtype=tokens.dtype)
        return tokens + emb.view(1, 1, -1)

    def _predict_video_only_noise(
        self,
        latents_video: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        fuse_vae_embedding_in_latents: bool,
        video_level: Optional[str] = None,
        frame_positions: Optional[torch.Tensor] = None,
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
        if video_level is not None:
            x_tokens = self._add_hierarchical_level_embedding(x_tokens, video_level)
        freqs = (
            self._video_freqs_for_frame_positions(video_pre, frame_positions)
            if frame_positions is not None
            else video_pre["freqs"]
        )
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
        frame_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = self._add_hierarchical_level_embedding(video_pre["tokens"], "high")
        freqs = (
            self._video_freqs_for_frame_positions(video_pre, frame_positions)
            if frame_positions is not None
            else video_pre["freqs"]
        )
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
    def _build_chunk_attention_mask(
        *,
        high_frames: int,
        low_frames: int,
        tokens_per_frame: int,
        action_seq_len: int,
        has_proprio_token: bool,
        device: torch.device,
        bidirectional_video_attention: bool = False,
        visible_high_start: Optional[int] = None,
        visible_high_end: Optional[int] = None,
    ) -> torch.Tensor:
        video_seq_len = (high_frames + low_frames) * tokens_per_frame
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        low_base = high_frames * tokens_per_frame
        low_end = low_base + low_frames * tokens_per_frame
        mask[:low_base, :low_base] = True
        if high_frames > 0:
            if visible_high_start is None or visible_high_end is None:
                high_start = 0
                high_end = high_frames - 1
            else:
                high_start = max(0, min(int(visible_high_start), high_frames - 1))
                high_end = max(high_start, min(int(visible_high_end), high_frames - 1))
            col_start = high_start * tokens_per_frame
            col_end = (high_end + 1) * tokens_per_frame
            mask[low_base:low_end, col_start:col_end] = True
        mask[low_base:low_end, low_base:low_end] = True
        if bidirectional_video_attention and high_frames > 0 and low_frames > 0:
            if visible_high_start is None or visible_high_end is None:
                row_start = 0
                row_end = high_frames - 1
            else:
                row_start = max(0, min(int(visible_high_start), high_frames - 1))
                row_end = max(row_start, min(int(visible_high_end), high_frames - 1))
            row_lo = row_start * tokens_per_frame
            row_hi = (row_end + 1) * tokens_per_frame
            mask[row_lo:row_hi, low_base:low_end] = True

        if action_seq_len > 0:
            if has_proprio_token:
                proprio_idx = video_seq_len
                mask[proprio_idx, proprio_idx] = True
                if action_seq_len > 1:
                    action_begin = proprio_idx + 1
                    if high_frames > 0:
                        mask[action_begin:, col_start:col_end] = True
                    mask[action_begin:, low_base:low_end] = True
                    mask[action_begin:, proprio_idx : proprio_idx + 1] = True
                    mask[action_begin:, action_begin:] = True
            else:
                if high_frames > 0:
                    mask[video_seq_len:, col_start:col_end] = True
                mask[video_seq_len:, low_base:low_end] = True
                mask[video_seq_len:, video_seq_len:] = True
        return mask


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
        low_visible_high_indices: Sequence[int],
        device: torch.device,
    ) -> torch.Tensor:
        high_seq_len = high_frames * tokens_per_frame
        low_seq_len = low_frames * tokens_per_frame
        video_seq_len = high_seq_len + low_seq_len
        total_seq_len = video_seq_len + action_seq_len
        low_base = high_seq_len
        action_base = video_seq_len
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

        visible_high_indices = tuple(
            dict.fromkeys(max(0, min(int(frame_idx), high_frames - 1)) for frame_idx in low_visible_high_indices)
        )
        if len(visible_high_indices) == 0:
            raise ValueError("At least one visible high-level frame is required for hierarchical training.")
        for frame_idx in visible_high_indices:
            col_start = frame_idx * tokens_per_frame
            col_end = (frame_idx + 1) * tokens_per_frame
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

                for frame_idx in visible_high_indices:
                    col_start = frame_idx * tokens_per_frame
                    col_end = (frame_idx + 1) * tokens_per_frame
                    mask[action_token_start:, col_start:col_end] = True
                if self.hierarchical_mask_low_predict:
                    low_visible_seq_len = min(tokens_per_frame, low_seq_len)
                    mask[action_token_start:, low_base : low_base + low_visible_seq_len] = True
                else:
                    mask[action_token_start:, low_base:video_seq_len] = True
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

    def _extract_observed_history_latents(
        self,
        observed_chunk_videos: Optional[list[torch.Tensor]],
        *,
        tiled: bool,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if observed_chunk_videos is None or len(observed_chunk_videos) == 0:
            return None, None

        frames: list[torch.Tensor] = []
        for idx, observed in enumerate(observed_chunk_videos):
            if observed is None:
                continue
            if observed.ndim == 3:
                if observed.shape[0] != 3:
                    raise ValueError(
                        f"Observed frame must be [3,H,W], got {tuple(observed.shape)} at index {idx}."
                    )
                frame = observed.to(device=self.device, dtype=self.torch_dtype)
                if len(frames) == 0 or not torch.equal(frames[-1], frame):
                    frames.append(frame)
                continue

            if observed.ndim == 4:
                if observed.shape[0] == 3:
                    video = observed.unsqueeze(0)  # [1,3,T,H,W]
                elif observed.shape[1] == 3:
                    video = observed.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,H,W]
                else:
                    raise ValueError(
                        f"Observed 4D tensor must be [3,T,H,W] or [T,3,H,W], got {tuple(observed.shape)} at index {idx}."
                    )
            elif observed.ndim == 5:
                video = observed
            else:
                raise ValueError(
                    f"Observed item must be [3,H,W], [3,T,H,W], [T,3,H,W], or [1,3,T,H,W], got {tuple(observed.shape)} at index {idx}."
                )

            if video.shape[0] != 1 or video.shape[1] != 3:
                raise ValueError(
                    f"Observed video must be [1,3,T,H,W], got {tuple(video.shape)} at index {idx}."
                )
            video = video.to(device=self.device, dtype=self.torch_dtype)
            for t in range(int(video.shape[2])):
                frame = video[0, :, t]
                if len(frames) == 0 or not torch.equal(frames[-1], frame):
                    frames.append(frame)

        if len(frames) == 0:
            return None, None

        observed_video = torch.stack(frames, dim=1).unsqueeze(0)  # [1,3,T,H,W]
        high_prefix_latents = self._encode_video_latents(observed_video, tiled=tiled)
        last_frame = observed_video[:, :, -1]
        low_init_latent = self._encode_input_image_latents_tensor(last_frame, tiled=tiled)
        return high_prefix_latents, low_init_latent

    @staticmethod
    def _build_action_from_low_attention_mask(
        *,
        high_frames: int = 0,
        low_frames: int,
        tokens_per_frame: int,
        action_seq_len: int,
        has_proprio_token: bool,
        only_first_low_frame: bool,
        device: torch.device,
        visible_high_start: Optional[int] = None,
        visible_high_end: Optional[int] = None,
    ) -> torch.Tensor:
        high_seq_len = high_frames * tokens_per_frame
        low_seq_len = low_frames * tokens_per_frame
        video_seq_len = high_seq_len + low_seq_len
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        if high_seq_len > 0:
            mask[:high_seq_len, :high_seq_len] = True
        if low_seq_len > 0:
            mask[high_seq_len:video_seq_len, high_seq_len:video_seq_len] = True
            if high_frames > 0:
                if visible_high_start is None or visible_high_end is None:
                    high_start = 0
                    high_end = high_frames - 1
                else:
                    high_start = max(0, min(int(visible_high_start), high_frames - 1))
                    high_end = max(high_start, min(int(visible_high_end), high_frames - 1))
                high_col_start = high_start * tokens_per_frame
                high_col_end = (high_end + 1) * tokens_per_frame
                mask[high_seq_len:video_seq_len, high_col_start:high_col_end] = True
            else:
                high_col_start = 0
                high_col_end = 0
        elif high_frames > 0:
            if visible_high_start is None or visible_high_end is None:
                high_start = 0
                high_end = high_frames - 1
            else:
                high_start = max(0, min(int(visible_high_start), high_frames - 1))
                high_end = max(high_start, min(int(visible_high_end), high_frames - 1))
            high_col_start = high_start * tokens_per_frame
            high_col_end = (high_end + 1) * tokens_per_frame
        else:
            high_col_start = 0
            high_col_end = 0
        if action_seq_len <= 0:
            return mask

        if has_proprio_token:
            proprio_idx = video_seq_len
            mask[proprio_idx, proprio_idx] = True
            action_begin = proprio_idx + 1
            if action_begin < total_seq_len:
                mask[action_begin:, action_begin:] = True
                mask[action_begin:, proprio_idx : proprio_idx + 1] = True
                if high_frames > 0:
                    mask[action_begin:, high_col_start:high_col_end] = True
                if only_first_low_frame:
                    mask[action_begin:, high_seq_len : high_seq_len + tokens_per_frame] = True
                else:
                    mask[action_begin:, high_seq_len:video_seq_len] = True
        else:
            mask[video_seq_len:, video_seq_len:] = True
            if high_frames > 0:
                mask[video_seq_len:, high_col_start:high_col_end] = True
            if only_first_low_frame:
                mask[video_seq_len:, high_seq_len : high_seq_len + tokens_per_frame] = True
            else:
                mask[video_seq_len:, high_seq_len:video_seq_len] = True
        return mask

    @staticmethod
    def _pad_video_latents_time(latents: torch.Tensor, target_frames: int) -> torch.Tensor:
        if latents.ndim != 5:
            raise ValueError(f"Expected [B, C, T, H, W] latents, got {tuple(latents.shape)}")
        current_frames = int(latents.shape[2])
        if current_frames > target_frames:
            raise ValueError(
                f"Cannot pad video latents from T={current_frames} down to target T={target_frames}."
            )
        if current_frames == target_frames:
            return latents
        pad_shape = list(latents.shape)
        pad_shape[2] = target_frames - current_frames
        pad = torch.zeros(pad_shape, device=latents.device, dtype=latents.dtype)
        return torch.cat([latents, pad], dim=2)

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

    def _training_loss_hierarchical(self, sample, tiled: bool = False):
        if "video" not in sample or "action" not in sample or "context" not in sample or "context_mask" not in sample:
            raise ValueError("Hierarchical training requires video/action/context/context_mask in sample.")

        video = sample["video"]
        action = sample["action"]
        proprio = sample.get("proprio", None)
        image_is_pad = sample.get("image_is_pad", None)
        action_is_pad = sample.get("action_is_pad", None)
        if video.ndim != 5 or action.ndim != 3:
            raise ValueError(
                f"Hierarchical mode expects video/action as [B,3,T,H,W]/[B,T,d], got {tuple(video.shape)} and {tuple(action.shape)}"
            )
        batch_size = int(video.shape[0])
        total_action_horizon = int(action.shape[1])
        total_video_frames = int(video.shape[2])
        if total_video_frames <= 1:
            raise ValueError(f"Hierarchical mode expects video with at least 2 frames, got {total_video_frames}")
        video_transitions = total_video_frames - 1
        if total_action_horizon % video_transitions != 0:
            raise ValueError(
                "Hierarchical mode expects action horizon divisible by video transitions; "
                f"got action={total_action_horizon}, transitions={video_transitions}."
            )
        action_per_video_step = total_action_horizon // video_transitions

        num_chunks = int(self.hierarchical_num_chunks)
        chunk_action_horizon = int(self.hierarchical_chunk_action_horizon)
        if num_chunks * chunk_action_horizon != total_action_horizon:
            if total_action_horizon % num_chunks == 0:
                chunk_action_horizon = total_action_horizon // num_chunks
            elif total_action_horizon % chunk_action_horizon == 0:
                num_chunks = total_action_horizon // chunk_action_horizon
            else:
                raise ValueError(
                    "Cannot infer chunk layout from action horizon. "
                    f"Got total_action={total_action_horizon}, num_chunks={num_chunks}, chunk_action_horizon={chunk_action_horizon}."
                )
        if chunk_action_horizon % action_per_video_step != 0:
            raise ValueError(
                "chunk_action_horizon must align with low-level video stride. "
                f"Got chunk_action_horizon={chunk_action_horizon}, action_per_video_step={action_per_video_step}."
            )
        chunk_video_steps = chunk_action_horizon // action_per_video_step
        if num_chunks * chunk_video_steps != video_transitions:
            raise ValueError(
                "Chunk video/action alignment mismatch. "
                f"num_chunks*chunk_video_steps={num_chunks * chunk_video_steps}, video_transitions={video_transitions}."
            )

        if proprio is not None and (proprio.ndim != 3 or int(proprio.shape[1]) != total_action_horizon):
            raise ValueError(
                f"Hierarchical mode expects proprio shape [B,{total_action_horizon},d], got {tuple(proprio.shape)}"
            )

        context = sample["context"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = sample["context_mask"].to(device=self.device, dtype=torch.bool, non_blocking=True)
        video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if proprio is not None:
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        # Build high-level target plan from full trajectory.
        high_video = self._temporal_downsample(video, stride=self.hierarchical_high_stride)
        high_latents = self._encode_video_latents(high_video, tiled=tiled)
        high_image_is_pad_full = self._temporal_downsample_mask(image_is_pad, stride=self.hierarchical_high_stride) if image_is_pad is not None else None
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # Chunk-parallel unified training: each chunk sample includes high+low+action.
        high_frame_count = int(high_latents.shape[2])
        if high_frame_count <= 1:
            raise ValueError(
                f"Hierarchical high-level training requires at least 2 latent frames, got {high_frame_count}."
            )

        has_proprio = self.proprio_encoder is not None
        patch_h = int(self.video_expert.patch_size[1])
        patch_w = int(self.video_expert.patch_size[2])
        if int(high_latents.shape[3]) % patch_h != 0 or int(high_latents.shape[4]) % patch_w != 0:
            raise ValueError(
                "High-level latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({high_latents.shape[3]}, {high_latents.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        tokens_per_frame = (int(high_latents.shape[3]) // patch_h) * (int(high_latents.shape[4]) // patch_w)

        chunk_high_noisy = []
        chunk_high_target = []
        chunk_high_timestep = []
        chunk_high_loss_timestep = []
        chunk_high_predict_mask = []
        chunk_high_image_pad = []

        chunk_low_noisy = []
        chunk_low_target = []
        chunk_low_timestep = []
        chunk_low_image_pad = []

        chunk_action_noisy = []
        chunk_action_target = []
        chunk_action_timestep = []
        chunk_action_is_pad = []

        chunk_proprio_token = []
        attention_masks = []
        low_latent_t = None

        for chunk_idx in range(num_chunks):
            frame_start = chunk_idx * chunk_video_steps
            frame_end = frame_start + chunk_video_steps + 1
            action_start = chunk_idx * chunk_action_horizon
            action_end = action_start + chunk_action_horizon

            low_video_chunk = video[:, :, frame_start:frame_end]
            low_plan_chunk = self._temporal_downsample(low_video_chunk, stride=self.hierarchical_low_stride)

            # Low-level branch: independent timestep/noise per chunk sample.
            low_latents_chunk = self._encode_video_latents(low_plan_chunk, tiled=tiled)
            timestep_video_chunk = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=low_latents_chunk.dtype,
            )
            noise_low_chunk = torch.randn_like(low_latents_chunk)
            noisy_low_chunk = self.train_video_scheduler.add_noise(
                low_latents_chunk,
                noise_low_chunk,
                timestep_video_chunk,
            )
            target_low_chunk = self.train_video_scheduler.training_target(
                low_latents_chunk,
                noise_low_chunk,
                timestep_video_chunk,
            )
            if fuse_flag:
                noisy_low_chunk[:, :, 0:1] = low_latents_chunk[:, :, 0:1]

            if low_latent_t is None:
                low_latent_t = int(noisy_low_chunk.shape[2])
            elif low_latent_t != int(noisy_low_chunk.shape[2]):
                raise ValueError(
                    f"All chunk low-level latent lengths must match; got {low_latent_t} and {int(noisy_low_chunk.shape[2])}."
                )

            chunk_low_noisy.append(noisy_low_chunk)
            chunk_low_target.append(target_low_chunk)
            chunk_low_timestep.append(timestep_video_chunk)

            # High-level branch: chunk i conditions on keyframes [0..i] as history,
            # and the corresponding target keyframe is i+1.
            history_frames = min(chunk_idx + 1, high_frame_count - 1)
            timestep_high_scalar = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=high_latents.dtype,
            )
            noise_high_chunk = torch.randn_like(high_latents)
            noisy_high_chunk = self.train_video_scheduler.add_noise(
                high_latents,
                noise_high_chunk,
                timestep_high_scalar,
            )
            target_high_chunk = self.train_video_scheduler.training_target(
                high_latents,
                noise_high_chunk,
                timestep_high_scalar,
            )
            timestep_high_chunk = self._build_history_predict_timestep(
                step_timestep=timestep_high_scalar,
                total_frames=high_frame_count,
                history_frames=history_frames,
                dtype=high_latents.dtype,
                device=self.device,
            )
            if fuse_flag:
                noisy_high_chunk[:, :, :history_frames] = high_latents[:, :, :history_frames]
            # Keep high-level history mostly clean, but occasionally condition on
            # scheduler-noised history with matching per-frame timesteps.
            noisy_high_chunk, timestep_high_chunk = self._perturb_history_latents_for_training(
                noisy_high_chunk,
                timestep_high_chunk,
                clean_latents=high_latents,
                history_frames=history_frames,
                preserve_initial_frame=fuse_flag,
            )

            predict_mask = torch.zeros((batch_size, high_frame_count), dtype=high_latents.dtype, device=self.device)
            predict_mask[:, history_frames:] = 1.0

            chunk_high_noisy.append(noisy_high_chunk)
            chunk_high_target.append(target_high_chunk)
            chunk_high_timestep.append(timestep_high_chunk)
            chunk_high_loss_timestep.append(timestep_high_scalar)
            chunk_high_predict_mask.append(predict_mask)
            if high_image_is_pad_full is not None:
                chunk_high_image_pad.append(high_image_is_pad_full)

            action_chunk = action[:, action_start:action_end]
            timestep_action_chunk = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=action.dtype,
            )
            noise_action = torch.randn_like(action_chunk)
            noisy_action = self.train_action_scheduler.add_noise(action_chunk, noise_action, timestep_action_chunk)
            target_action_chunk = self.train_action_scheduler.training_target(action_chunk, noise_action, timestep_action_chunk)

            chunk_action_noisy.append(noisy_action)
            chunk_action_target.append(target_action_chunk)
            chunk_action_timestep.append(timestep_action_chunk)
            chunk_action_is_pad.append(action_is_pad[:, action_start:action_end] if action_is_pad is not None else None)

            if image_is_pad is not None:
                low_image_pad_chunk = image_is_pad[:, frame_start:frame_end]
                chunk_low_image_pad.append(
                    self._temporal_downsample_mask(low_image_pad_chunk, stride=self.hierarchical_low_stride)
                )
            else:
                chunk_low_image_pad.append(None)

            if has_proprio:
                if proprio is None:
                    raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
                proprio_first = proprio[:, action_start, :]
                chunk_proprio_token.append(
                    self._encode_proprio_action_token(
                        proprio=proprio_first,
                        dtype=context.dtype,
                    )
                )

            low_visible_high_indices = self._select_high_cache_frame_indices(
                chunk_idx=chunk_idx,
                high_frames=high_frame_count,
            )
            action_seq_len = chunk_action_horizon + (1 if has_proprio else 0)
            chunk_mask = self._build_hierarchical_training_attention_mask(
                high_frames=high_frame_count,
                low_frames=int(noisy_low_chunk.shape[2]),
                tokens_per_frame=tokens_per_frame,
                action_seq_len=action_seq_len,
                has_proprio_token=has_proprio,
                history_high_frames=history_frames,
                low_visible_high_indices=low_visible_high_indices,
                device=self.device,
            )
            attention_masks.extend([chunk_mask] * batch_size)

        high_noisy_batch = torch.cat(chunk_high_noisy, dim=0)
        high_target_batch = torch.cat(chunk_high_target, dim=0)
        high_timestep_batch = torch.cat(chunk_high_timestep, dim=0)
        high_predict_mask_batch = torch.cat(chunk_high_predict_mask, dim=0)
        high_image_is_pad_batch = (
            torch.cat(chunk_high_image_pad, dim=0) if high_image_is_pad_full is not None else None
        )
        high_timestep_scalar_batch = torch.cat(chunk_high_loss_timestep, dim=0)

        noisy_low_batch = torch.cat(chunk_low_noisy, dim=0)
        target_low_batch = torch.cat(chunk_low_target, dim=0)
        timestep_video_batch = torch.cat(chunk_low_timestep, dim=0)

        chunk_action_noisy_batch = torch.cat(chunk_action_noisy, dim=0)
        chunk_action_target_batch = torch.cat(chunk_action_target, dim=0)
        timestep_action_batch = torch.cat(chunk_action_timestep, dim=0)
        chunk_action_is_pad_batch = (
            torch.cat(chunk_action_is_pad, dim=0) if action_is_pad is not None else None
        )
        chunk_low_image_pad_batch = (
            torch.cat(chunk_low_image_pad, dim=0) if image_is_pad is not None else None
        )
        proprio_token_batch = torch.cat(chunk_proprio_token, dim=0) if has_proprio else None

        low_latent_t = int(noisy_low_batch.shape[2])
        high_timestep_pre = high_timestep_batch.clone()
        low_timestep_matrix = timestep_video_batch.unsqueeze(1).expand(-1, low_latent_t).clone()
        low_timestep_matrix[:, 0] = 0
        action_joint_batch = chunk_action_noisy_batch
        timestep_action_joint = timestep_action_batch

        context_joint = context.repeat(num_chunks, 1, 1)
        context_mask_joint = context_mask.repeat(num_chunks, 1)

        high_pre = self.video_expert.pre_dit(
            x=high_noisy_batch,
            timestep=high_timestep_pre,
            context=context_joint,
            context_mask=context_mask_joint,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        low_pre = self.video_expert.pre_dit(
            x=noisy_low_batch,
            timestep=low_timestep_matrix,
            context=context_joint,
            context_mask=context_mask_joint,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_joint_batch,
            timestep=timestep_action_joint,
            context=context_joint,
            context_mask=context_mask_joint,
        )

        tokens_per_frame = int(high_pre["meta"]["tokens_per_frame"])
        high_tokens = self._add_hierarchical_level_embedding(high_pre["tokens"], "high")
        low_tokens = self._add_hierarchical_level_embedding(low_pre["tokens"], "low")
        merged_video_tokens = torch.cat([high_tokens, low_tokens], dim=1)
        # 生成每个chunk对应的索引，范围是 [0, num_chunks-1]
        # 使用 repeat_interleave(batch_size) 是因为 batch 里的每个样本都被拆成了 num_chunks 份
        # 这保证了按序排列，即 [0,0..,0, 1,1..,1, ...]
        chunk_indices_batch = torch.arange(num_chunks, device=self.device, dtype=torch.long).repeat_interleave(batch_size)
        high_frame_positions = self._global_high_frame_positions(
            high_frames=int(high_pre["meta"]["grid_size"][0]),
            block_chunks=num_chunks,
            low_frames_per_chunk=low_latent_t,
            device=high_pre["tokens"].device,
        )
        low_frame_positions = self._global_low_frame_positions(
            chunk_indices=chunk_indices_batch,
            low_frames=int(low_pre["meta"]["grid_size"][0]),
            low_frames_per_chunk=low_latent_t,
            device=low_pre["tokens"].device,
        )
        high_freqs = self._video_freqs_for_frame_positions(high_pre, high_frame_positions)
        low_freqs = self._video_freqs_for_frame_positions(low_pre, low_frame_positions)
        high_freqs = high_freqs.unsqueeze(0).expand(low_freqs.shape[0], -1, -1, -1)
        merged_video_freqs = torch.cat([high_freqs, low_freqs], dim=1)
        merged_video_t_mod = torch.cat([high_pre["t_mod"], low_pre["t_mod"]], dim=1)
        merged_video_context_mask = torch.cat([high_pre["context_mask"], low_pre["context_mask"]], dim=1)

        action_tokens, action_freqs, action_ctx, action_ctx_mask, action_t_mod = self._prepend_proprio_to_action_pre(
            action_pre,
            proprio_token_batch,
        )

        attention_mask_batch = torch.stack(attention_masks, dim=0)
        tokens_out = self.mot(
            embeds_all={
                "video": merged_video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask_batch,
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
        pred_high_all = self.video_expert.post_dit(tokens_out["video"][:, :high_seq_len], high_pre)
        pred_low_all = self.video_expert.post_dit(tokens_out["video"][:, high_seq_len:], low_pre)
        pred_low_eval = pred_low_all[:, :, 1:] if fuse_flag else pred_low_all
        target_low_eval = target_low_batch[:, :, 1:] if fuse_flag else target_low_batch

        pred_action_all_with_prop = self.action_expert.post_dit(tokens_out["action"], action_pre)
        pred_action_all = pred_action_all_with_prop[:, 1:] if has_proprio else pred_action_all_with_prop
        pred_action_chunks = pred_action_all

        pred_high_eval = pred_high_all[:, :, 1:] if fuse_flag else pred_high_all
        target_high_eval = high_target_batch[:, :, 1:] if fuse_flag else high_target_batch
        timestep_high_eval = high_timestep_pre[:, 1:] if fuse_flag else high_timestep_pre
        predict_mask_high_eval = high_predict_mask_batch[:, 1:] if fuse_flag else high_predict_mask_batch
        high_image_is_pad_eval = high_image_is_pad_batch
        high_loss_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_high_eval,
            target_video=target_high_eval,
            image_is_pad=high_image_is_pad_eval,
            include_initial_video_step=not fuse_flag,
            temporal_weights=predict_mask_high_eval.to(dtype=high_latents.dtype),
        )
        low_loss_per_chunk_sample = self._compute_video_loss_per_sample(
            pred_video=pred_low_eval,
            target_video=target_low_eval,
            image_is_pad=chunk_low_image_pad_batch,
            include_initial_video_step=not fuse_flag,
        )

        action_loss_token = F.mse_loss(
            pred_action_chunks.float(),
            chunk_action_target_batch.float(),
            reduction="none",
        ).mean(dim=2)
        if chunk_action_is_pad_batch is not None:
            valid = (~chunk_action_is_pad_batch).to(
                device=action_loss_token.device,
                dtype=action_loss_token.dtype,
            )
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_chunk_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_chunk_sample = action_loss_token.mean(dim=1)

        high_loss_chunks = high_loss_per_sample.reshape(num_chunks, batch_size).transpose(0, 1)
        low_loss_chunks = low_loss_per_chunk_sample.reshape(num_chunks, batch_size).transpose(0, 1)
        action_loss_chunks = action_loss_per_chunk_sample.reshape(num_chunks, batch_size).transpose(0, 1)

        high_weight_chunks = self.train_video_scheduler.training_weight(high_timestep_scalar_batch).to(
            device=self.device,
            dtype=high_latents.dtype,
        ).reshape(num_chunks, batch_size).transpose(0, 1)
        low_weight_chunks = self.train_video_scheduler.training_weight(timestep_video_batch).to(
            device=self.device,
            dtype=high_latents.dtype,
        ).reshape(num_chunks, batch_size).transpose(0, 1)
        action_weight_chunks = self.train_action_scheduler.training_weight(timestep_action_batch).to(
            device=self.device,
            dtype=action.dtype,
        ).reshape(num_chunks, batch_size).transpose(0, 1)

        high_loss_per_sample = (high_loss_chunks * high_weight_chunks).mean(dim=1)
        low_loss_per_sample = (low_loss_chunks * low_weight_chunks).mean(dim=1)
        action_loss_per_sample = (action_loss_chunks * action_weight_chunks).mean(dim=1)

        loss_high = high_loss_per_sample.mean()
        loss_low = low_loss_per_sample.mean()
        loss_action = action_loss_per_sample.mean()

        loss_video = loss_high + loss_low
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_video_high": self.loss_lambda_video * float(loss_high.detach().item()),
            "loss_video_low": self.loss_lambda_video * float(loss_low.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

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

    def training_loss(self, sample, tiled: bool = False):
        return self._training_loss_hierarchical(sample, tiled=tiled)

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        proprio_action_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = action_pre["tokens"]
        action_freqs = action_pre["freqs"]
        action_ctx = action_pre["context"]
        action_ctx_mask = action_pre["context_mask"]
        has_proprio_token = proprio_action_token is not None
        action_tokens, action_freqs, action_ctx, action_ctx_mask, action_t_mod = self._prepend_proprio_to_action_pre(
            action_pre,
            proprio_action_token,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context_payload={
                "context": action_ctx,
                "mask": action_ctx_mask,
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        pred_action_all = self.action_expert.post_dit(action_tokens, action_pre)
        return pred_action_all[:, 1:] if has_proprio_token else pred_action_all

    @torch.no_grad()
    def _infer_action_from_video_condition(
        self,
        *,
        latents_video_cond: torch.Tensor,
        chunk_horizon: int,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action_timesteps: torch.Tensor,
        action_deltas: torch.Tensor,
        generator: Optional[torch.Generator],
        rand_device: str,
        proprio: Optional[torch.Tensor],
        fuse_vae_embedding_in_latents: bool,
        only_first_low_frame: bool,
        low_cache_predict_timestep: Optional[torch.Tensor] = None,
        high_kv_cache_for_low: Optional[list[dict[str, torch.Tensor]]] = None,
        high_frames_for_low: Optional[int] = None,
        high_tokens_per_frame_for_low: Optional[int] = None,
        low_frame_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if only_first_low_frame:
            latents_video_cond = latents_video_cond[:, :, :1]
            if low_frame_positions is not None:
                low_frame_positions = low_frame_positions[:1] if low_frame_positions.ndim == 1 else low_frame_positions[:, :1]

        cache_timestep = None
        action_high_frames = 0
        if low_cache_predict_timestep is not None:
            cache_timestep = self._build_history_predict_timestep(
                step_timestep=low_cache_predict_timestep,
                total_frames=int(latents_video_cond.shape[2]),
                history_frames=1,
                dtype=latents_video_cond.dtype,
                device=latents_video_cond.device,
            )
        if high_kv_cache_for_low is None:
            video_kv_cache, video_seq_len, low_tokens_per_frame = self._build_video_kv_cache(
                latents_video=latents_video_cond,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                timestep_video=cache_timestep,
                video_level="low",
                frame_positions=low_frame_positions,
            )
        else:
            if high_frames_for_low is None or high_tokens_per_frame_for_low is None:
                raise ValueError(
                    "`high_frames_for_low` and `high_tokens_per_frame_for_low` are required "
                    "when `high_kv_cache_for_low` is provided."
                )
            timestep_input = (
                torch.zeros(
                    (latents_video_cond.shape[0],),
                    dtype=latents_video_cond.dtype,
                    device=latents_video_cond.device,
                )
                if cache_timestep is None
                else cache_timestep.to(device=latents_video_cond.device, dtype=latents_video_cond.dtype)
            )
            low_pre = self.video_expert.pre_dit(
                x=latents_video_cond,
                timestep=timestep_input,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            )
            low_tokens_per_frame = int(low_pre["meta"]["tokens_per_frame"])
            low_tokens = self._add_hierarchical_level_embedding(low_pre["tokens"], "low")
            low_freqs = (
                self._video_freqs_for_frame_positions(low_pre, low_frame_positions)
                if low_frame_positions is not None
                else low_pre["freqs"]
            )
            if int(high_tokens_per_frame_for_low) != low_tokens_per_frame:
                raise ValueError(
                    "Token-per-frame mismatch between high cache and low chunk: "
                    f"high={high_tokens_per_frame_for_low}, low={low_tokens_per_frame}."
                )

            low_cache_attention_mask = self._build_chunk_attention_mask(
                high_frames=int(high_frames_for_low),
                low_frames=int(latents_video_cond.shape[2]),
                tokens_per_frame=low_tokens_per_frame,
                action_seq_len=0,
                has_proprio_token=False,
                device=latents_video_cond.device,
                visible_high_start=0,
                visible_high_end=max(0, int(high_frames_for_low) - 1),
            )
            high_seq_len_for_low = int(high_frames_for_low) * low_tokens_per_frame
            low_kv_cache = self.mot.prefill_video_cache_with_prefix(
                video_tokens=low_tokens,
                video_freqs=low_freqs,
                video_t_mod=low_pre["t_mod"],
                video_context_payload={
                    "context": low_pre["context"],
                    "mask": low_pre["context_mask"],
                },
                prefix_kv_cache=high_kv_cache_for_low,
                attention_mask=low_cache_attention_mask,
                prefix_seq_len=high_seq_len_for_low,
            )
            # Action decoding should condition on the same high-history prefix
            # that the low-level video chunk used, plus the low chunk itself.
            video_kv_cache = self._concat_video_kv_caches(
                high_kv_cache_for_low,
                low_kv_cache,
            )
            video_seq_len = high_seq_len_for_low + int(low_pre["tokens"].shape[1])
            action_high_frames = int(high_frames_for_low)
        proprio_action_token = None
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`proprio` must be provided in hierarchical infer when `proprio_dim` is enabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"Expected proprio shape [1,{self.proprio_dim}] in hierarchical infer, got {tuple(proprio.shape)}"
                )
            proprio_action_token = self._encode_proprio_action_token(
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
                dtype=context.dtype,
            )

        action_seq_len = chunk_horizon + (1 if proprio_action_token is not None else 0)
        full_attention_mask = self._build_action_from_low_attention_mask(
            high_frames=action_high_frames,
            low_frames=int(latents_video_cond.shape[2]),
            tokens_per_frame=low_tokens_per_frame,
            action_seq_len=action_seq_len,
            has_proprio_token=proprio_action_token is not None,
            only_first_low_frame=only_first_low_frame,
            device=self.device,
            visible_high_start=0,
            visible_high_end=max(0, action_high_frames - 1),
        )

        latents_action = torch.randn(
            (1, chunk_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        for step_t_act, step_d_act in zip(action_timesteps, action_deltas):
            t_action = step_t_act.unsqueeze(0).to(device=self.device, dtype=latents_action.dtype)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=t_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=full_attention_mask,
                video_seq_len=video_seq_len,
                proprio_action_token=proprio_action_token,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_d_act, latents_action)
        return latents_action[0].detach().to(device="cpu", dtype=torch.float32)

    @torch.no_grad()
    def _build_video_kv_cache(
        self,
        *,
        latents_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        timestep_video: Optional[torch.Tensor] = None,
        video_level: Optional[str] = None,
        frame_positions: Optional[torch.Tensor] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], int, int]:
        if timestep_video is None:
            timestep_input = torch.zeros(
                (latents_video.shape[0],),
                dtype=latents_video.dtype,
                device=latents_video.device,
            )
        else:
            timestep_input = timestep_video.to(device=latents_video.device, dtype=latents_video.dtype)
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_input,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        video_tokens = video_pre["tokens"]
        if video_level is not None:
            video_tokens = self._add_hierarchical_level_embedding(video_tokens, video_level)
        video_freqs = (
            self._video_freqs_for_frame_positions(video_pre, frame_positions)
            if frame_positions is not None
            else video_pre["freqs"]
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
        )
        return video_kv_cache, video_seq_len, tokens_per_frame

    @staticmethod
    def _concat_chunk_video_tensors(
        video_chunks: list[torch.Tensor],
        *,
        replace_boundary_with_next_chunk_first_frame: bool = False,
    ) -> torch.Tensor:
        if len(video_chunks) == 0:
            raise ValueError("`video_chunks` must contain at least one chunk.")
        full_video = video_chunks[0]
        for chunk_idx, chunk_video in enumerate(video_chunks[1:], start=1):
            if chunk_video.ndim != 4 or chunk_video.shape[0] != 3:
                raise ValueError(
                    f"Chunk video at index {chunk_idx} must have shape [3,T,H,W], got {tuple(chunk_video.shape)}."
                )
            if replace_boundary_with_next_chunk_first_frame:
                full_video = torch.cat([full_video[:, :-1], chunk_video], dim=1)
            else:
                full_video = torch.cat([full_video, chunk_video[:, 1:]], dim=1)
        return full_video

    def _get_chunk_proprio(
        self,
        proprio: Optional[torch.Tensor],
        *,
        action_start: int,
    ) -> Optional[torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return None
        if proprio.ndim == 1:
            if proprio.shape[0] != self.proprio_dim:
                raise ValueError(
                    f"Expected proprio shape [{self.proprio_dim}] in hierarchical infer, got {tuple(proprio.shape)}"
                )
            return proprio.unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
        if proprio.ndim == 2:
            if int(proprio.shape[1]) != int(self.proprio_dim):
                raise ValueError(
                    f"Expected proprio shape [T,{self.proprio_dim}] in hierarchical infer, got {tuple(proprio.shape)}"
                )
            if int(proprio.shape[0]) == 1:
                return proprio.to(device=self.device, dtype=self.torch_dtype)
            if action_start >= int(proprio.shape[0]):
                raise ValueError(
                    f"`action_start`={action_start} exceeds proprio horizon {int(proprio.shape[0])}."
                )
            return proprio[action_start : action_start + 1].to(device=self.device, dtype=self.torch_dtype)
        if proprio.ndim == 3:
            if proprio.shape[0] != 1 or int(proprio.shape[2]) != int(self.proprio_dim):
                raise ValueError(
                    f"Expected proprio shape [1,T,{self.proprio_dim}] in hierarchical infer, got {tuple(proprio.shape)}"
                )
            if action_start >= int(proprio.shape[1]):
                raise ValueError(
                    f"`action_start`={action_start} exceeds proprio horizon {int(proprio.shape[1])}."
                )
            return proprio[:, action_start, :].to(device=self.device, dtype=self.torch_dtype)
        raise ValueError(
            f"`proprio` must have shape [D], [T,D], or [1,T,D] in hierarchical infer, got {tuple(proprio.shape)}"
        )

    @staticmethod
    def _slice_video_kv_cache(
        kv_cache: list[dict[str, torch.Tensor]],
        *,
        token_start: int,
        token_end: int,
    ) -> list[dict[str, torch.Tensor]]:
        if token_start < 0 or token_end <= token_start:
            raise ValueError(f"Invalid cache slice range [{token_start}, {token_end}).")
        sliced = []
        for layer_idx, layer_cache in enumerate(kv_cache):
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(f"`kv_cache[{layer_idx}]` must contain `k` and `v`.")
            k = layer_cache["k"]
            v = layer_cache["v"]
            if token_end > int(k.shape[1]) or token_end > int(v.shape[1]):
                raise ValueError(
                    f"Cache slice end {token_end} exceeds layer {layer_idx} seq len "
                    f"k={k.shape[1]}, v={v.shape[1]}."
                )
            sliced.append({
                "k": k[:, token_start:token_end],
                "v": v[:, token_start:token_end],
            })
        return sliced

    @staticmethod
    def _concat_video_kv_caches(
        first_cache: list[dict[str, torch.Tensor]],
        second_cache: list[dict[str, torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        if len(first_cache) != len(second_cache):
            raise ValueError(
                "Cannot concatenate KV caches with different layer counts: "
                f"{len(first_cache)} vs {len(second_cache)}."
            )
        merged = []
        for layer_idx, (first_layer, second_layer) in enumerate(zip(first_cache, second_cache)):
            if "k" not in first_layer or "v" not in first_layer or "k" not in second_layer or "v" not in second_layer:
                raise ValueError(f"Both KV caches must contain `k` and `v` at layer {layer_idx}.")
            merged.append({
                "k": torch.cat([first_layer["k"], second_layer["k"]], dim=1),
                "v": torch.cat([first_layer["v"], second_layer["v"]], dim=1),
            })
        return merged

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
    def _predict_low_video_noise_with_high_cache(
        self,
        *,
        low_chunk_latents: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        high_kv_cache: list[dict[str, torch.Tensor]],
        high_seq_len: int,
        high_frames: int,
        tokens_per_frame: int,
        fuse_vae_embedding_in_latents: bool,
        visible_high_start: Optional[int] = None,
        visible_high_end: Optional[int] = None,
        mask_low_predict: bool = False,
        low_frame_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        low_pre = self.video_expert.pre_dit(
            x=low_chunk_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        low_tokens = self._add_hierarchical_level_embedding(low_pre["tokens"], "low")
        low_freqs = (
            self._video_freqs_for_frame_positions(low_pre, low_frame_positions)
            if low_frame_positions is not None
            else low_pre["freqs"]
        )
        low_frames = int(low_chunk_latents.shape[2])
        low_tokens_per_frame = int(low_pre["meta"]["tokens_per_frame"])
        if low_tokens_per_frame != tokens_per_frame:
            raise ValueError(
                "Token-per-frame mismatch between high cache and low chunk: "
                f"high={tokens_per_frame}, low={low_tokens_per_frame}."
            )
        total_seq_len = (high_frames + low_frames) * tokens_per_frame
        attention_mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=low_tokens.device)
        high_seq = high_frames * tokens_per_frame
        low_seq = low_frames * tokens_per_frame
        if high_seq > 0:
            attention_mask[:high_seq, :high_seq] = True
            if visible_high_start is None or visible_high_end is None:
                visible_high_start = 0
                visible_high_end = high_frames - 1
            col_start = max(0, min(int(visible_high_start), high_frames - 1)) * tokens_per_frame
            col_end = (max(0, min(int(visible_high_end), high_frames - 1)) + 1) * tokens_per_frame
            attention_mask[high_seq:, col_start:col_end] = True
        if mask_low_predict:
            low_mask = self._build_first_frame_causal_mask(
                num_frames=low_frames,
                tokens_per_frame=tokens_per_frame,
                device=low_tokens.device,
            )
        else:
            low_mask = torch.ones((low_seq, low_seq), dtype=torch.bool, device=low_tokens.device)
        attention_mask[high_seq:, high_seq:] = low_mask
        low_tokens_out = self.mot.forward_video_with_video_cache(
            video_tokens=low_tokens,
            video_freqs=low_freqs,
            video_t_mod=low_pre["t_mod"],
            video_context_payload={
                "context": low_pre["context"],
                "mask": low_pre["context_mask"],
            },
            prefix_kv_cache=high_kv_cache,
            attention_mask=attention_mask,
            prefix_seq_len=high_seq_len,
        )
        return self.video_expert.post_dit(low_tokens_out, low_pre)

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
    ) -> dict[str, Any]:
        if num_video_frames is None:
            raise ValueError("`num_video_frames` is required for hierarchical `infer_action`.")
        single_chunk_horizon = int(self.hierarchical_chunk_action_horizon)
        if int(action_horizon) != single_chunk_horizon:
            logger.warning(
                "Hierarchical `infer_action` expects single-chunk horizon=%d, but got action_horizon=%d. "
                "Overriding to single-chunk horizon.",
                single_chunk_horizon,
                int(action_horizon),
            )
        del negative_prompt, text_cfg_scale
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )

        context, context_mask = self._prepare_infer_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
        )

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        action_steps = max(1, int(action_inference_steps if action_inference_steps is not None else num_inference_steps))
        action_timesteps, action_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=action_steps,
            device=self.device,
            dtype=first_frame_latents.dtype,
            shift_override=sigma_shift,
        )
        g_video = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        g_action = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)

        high_prefix_latents, observed_low_init_latent = self._extract_observed_history_latents(
            observed_chunk_videos,
            tiled=tiled,
        )
        if observed_low_init_latent is not None:
            first_frame_latents = observed_low_init_latent

        checked_h, checked_w, checked_t = self._check_resize_height_width(
            int(input_image.shape[2]),
            int(input_image.shape[3]),
            num_video_frames,
        )
        if checked_t != num_video_frames or checked_h != int(input_image.shape[2]) or checked_w != int(input_image.shape[3]):
            raise ValueError(
                f"Invalid infer shape: expected H/W multiples of 16 and T%4==1, got HxW=({input_image.shape[2]},{input_image.shape[3]}), T={num_video_frames}."
            )
        # In infer_action, `num_video_frames` is chunk-level (from deploy policy), not full-block frames.
        chunk_video_steps = int(num_video_frames) - 1
        if chunk_video_steps <= 0 or single_chunk_horizon % chunk_video_steps != 0:
            raise ValueError(
                "Hierarchical `infer_action` expects single-chunk action horizon divisible by video transitions. "
                f"Got action_horizon={single_chunk_horizon}, video_transitions={chunk_video_steps}."
            )
        action_per_video_step = single_chunk_horizon // chunk_video_steps
        low_frames = chunk_video_steps // self.hierarchical_low_stride + 1
        low_latent_t = (low_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = int(input_image.shape[2]) // self.vae.upsampling_factor
        latent_w = int(input_image.shape[3]) // self.vae.upsampling_factor
        z_dim = self.vae.model.z_dim

        # High-level planning is global across the block, not per-chunk.
        global_video_steps = chunk_video_steps * int(self.hierarchical_num_chunks)
        global_high_frames = global_video_steps // self.hierarchical_high_stride + 1
        high_latent_t = (global_high_frames - 1) // self.vae.temporal_downsample_factor + 1
        full_high_frame_positions = self._global_high_frame_positions(
            high_frames=high_latent_t,
            block_chunks=int(self.hierarchical_num_chunks),
            low_frames_per_chunk=low_latent_t,
            device=self.device,
        )
        history_high_frames = int(high_prefix_latents.shape[2]) if high_prefix_latents is not None else 1
        history_high_frames = max(1, min(history_high_frames, high_latent_t))
        current_chunk_idx = min(max(0, history_high_frames - 1), int(self.hierarchical_num_chunks) - 1)
        low_frame_positions = self._global_low_frame_positions(
            chunk_indices=current_chunk_idx,
            low_frames=low_latent_t,
            low_frames_per_chunk=low_latent_t,
            device=self.device,
        )

        if self.hierarchical_mask_high_predict:
            high_history_latents = (
                high_prefix_latents[:, :, :history_high_frames].clone()
                if high_prefix_latents is not None and int(high_prefix_latents.shape[2]) > 0
                else first_frame_latents.clone()
            )
            high_kv_cache, high_seq_len, high_tokens_per_frame = self._build_video_kv_cache(
                latents_video=high_history_latents,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                video_level="high",
                frame_positions=full_high_frame_positions[: int(high_history_latents.shape[2])],
            )
            del high_seq_len
            total_high_frames = int(high_history_latents.shape[2])
        else:
            high_latents = torch.randn(
                (1, z_dim, high_latent_t, latent_h, latent_w),
                generator=g_video,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
            high_latents[:, :, 0:1] = first_frame_latents.clone()
            fixed_high_len = 1
            if high_prefix_latents is not None:
                prefix_len = min(int(high_prefix_latents.shape[2]), int(high_latents.shape[2]))
                if prefix_len > 0:
                    high_latents[:, :, :prefix_len] = high_prefix_latents[:, :, :prefix_len]
                    fixed_high_len = prefix_len
            fixed_high_latents = high_latents[:, :, :fixed_high_len].clone()

            high_steps = max(1, int(high_video_inference_steps if high_video_inference_steps is not None else num_inference_steps // 2))
            high_timesteps_full, high_deltas_full = self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=high_steps,
                device=self.device,
                dtype=high_latents.dtype,
                shift_override=sigma_shift,
            )
            high_timesteps, high_deltas = self._truncate_inference_schedule(
                high_timesteps_full,
                high_deltas_full,
                high_denoise_step,
                name="high_denoise_step",
            )
            high_cache_predict_timestep = self._next_schedule_timestep(
                full_timesteps=high_timesteps_full,
                used_steps=int(high_timesteps.shape[0]),
                dtype=high_latents.dtype,
                device=self.device,
            )
            for step_t, step_delta in zip(high_timesteps, high_deltas):
                t_video = self._build_history_predict_timestep(
                    step_timestep=step_t,
                    total_frames=int(high_latents.shape[2]),
                    history_frames=fixed_high_len,
                    dtype=high_latents.dtype,
                    device=self.device,
                )
                pred_high = self._predict_video_only_noise(
                    latents_video=high_latents,
                    timestep_video=t_video,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    video_level="high",
                    frame_positions=full_high_frame_positions,
                )
                high_latents = self.infer_video_scheduler.step(pred_high, step_delta, high_latents)
                high_latents[:, :, :fixed_high_len] = fixed_high_latents

            high_cache_timestep = self._build_history_predict_timestep(
                step_timestep=high_cache_predict_timestep,
                total_frames=int(high_latents.shape[2]),
                history_frames=fixed_high_len,
                dtype=high_latents.dtype,
                device=self.device,
            )

            high_kv_cache, high_seq_len, high_tokens_per_frame = self._build_video_kv_cache(
                latents_video=high_latents,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                timestep_video=high_cache_timestep,
                video_level="high",
                frame_positions=full_high_frame_positions,
            )
            del high_seq_len
            total_high_frames = int(high_latents.shape[2])

        pair_high_kv_cache, high_frames_for_low, _ = self._slice_high_cache_for_chunk(
            high_kv_cache,
            chunk_idx=current_chunk_idx,
            high_frames=total_high_frames,
            tokens_per_frame=high_tokens_per_frame,
        )

        if self.hierarchical_mask_low_predict:
            low_cond_latents = first_frame_latents.clone()
            low_cache_predict_timestep = None
        else:
            latents_low = torch.randn(
                (1, z_dim, low_latent_t, latent_h, latent_w),
                generator=g_video,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
            latents_low[:, :, 0:1] = first_frame_latents.clone()
            low_steps = max(1, int(low_video_inference_steps if low_video_inference_steps is not None else num_inference_steps // 2))
            low_timesteps_full, low_deltas_full = self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=low_steps,
                device=self.device,
                dtype=latents_low.dtype,
                shift_override=sigma_shift,
            )
            low_timesteps, low_deltas = self._truncate_inference_schedule(
                low_timesteps_full,
                low_deltas_full,
                low_denoise_step,
                name="low_denoise_step",
            )
            low_cache_predict_timestep = self._next_schedule_timestep(
                full_timesteps=low_timesteps_full,
                used_steps=int(low_timesteps.shape[0]),
                dtype=latents_low.dtype,
                device=self.device,
            )
            for step_t_low, step_d_low in zip(low_timesteps, low_deltas):
                pred_low = self._predict_low_video_noise_with_high_cache(
                    low_chunk_latents=latents_low,
                    timestep_video=step_t_low.unsqueeze(0).to(device=self.device, dtype=latents_low.dtype),
                    context=context,
                    context_mask=context_mask,
                    high_kv_cache=pair_high_kv_cache,
                    high_seq_len=high_tokens_per_frame * high_frames_for_low,
                    high_frames=high_frames_for_low,
                    tokens_per_frame=high_tokens_per_frame,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    mask_low_predict=False,
                    low_frame_positions=low_frame_positions,
                )
                latents_low = self.infer_video_scheduler.step(pred_low, step_d_low, latents_low)
                latents_low[:, :, 0:1] = first_frame_latents
            low_cond_latents = latents_low

        action_out = self._infer_action_from_video_condition(
            latents_video_cond=low_cond_latents,
            chunk_horizon=single_chunk_horizon,
            context=context,
            context_mask=context_mask,
            action_timesteps=action_timesteps,
            action_deltas=action_deltas,
            generator=g_action,
            rand_device=rand_device,
            proprio=proprio,
            fuse_vae_embedding_in_latents=fuse_flag,
            only_first_low_frame=self.hierarchical_mask_low_predict,
            low_cache_predict_timestep=low_cache_predict_timestep,
            high_kv_cache_for_low=pair_high_kv_cache,
            high_frames_for_low=high_frames_for_low,
            high_tokens_per_frame_for_low=high_tokens_per_frame,
            low_frame_positions=low_frame_positions,
        )
        return {"action": action_out}

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
        action_inference_steps: Optional[int] = None,
        return_high_level_video: bool = False,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        gt_video: Optional[torch.Tensor] = None,
    ):
        del action, action_cfg_scale
        return self.infer_hierarchical(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
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
            return_high_level_video=return_high_level_video,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

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
        action_inference_steps: Optional[int] = None,
        return_high_level_video: bool = False,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        gt_video: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale
        self.eval()

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width) or checked_t != num_video_frames:
            raise ValueError(
                f"Invalid infer shape: expected H/W multiples of 16 and T%4==1, got HxW=({height},{width}), T={num_video_frames}."
            )

        chunk_h = self.hierarchical_chunk_action_horizon
        total_action = int(action_horizon) if action_horizon is not None else int(self.hierarchical_num_chunks * chunk_h)
        if total_action <= 0:
            raise ValueError(f"Invalid action horizon in hierarchical infer: {total_action}")
        video_transitions = num_video_frames - 1
        if video_transitions <= 0:
            raise ValueError(f"Invalid num_video_frames in hierarchical infer: {num_video_frames}")
        if total_action % video_transitions != 0:
            raise ValueError(
                "Hierarchical infer expects action horizon divisible by video transitions. "
                f"Got action_horizon={total_action}, video_transitions={video_transitions}."
            )
        action_per_video_step = total_action // video_transitions
        if total_action % chunk_h != 0:
            raise ValueError(
                f"Hierarchical infer expects action_horizon divisible by chunk_horizon={chunk_h}, got {total_action}."
            )
        num_chunks = total_action // chunk_h
        if chunk_h % action_per_video_step != 0:
            raise ValueError(
                "chunk_horizon must align with low-level video stride in infer. "
                f"Got chunk_horizon={chunk_h}, action_per_video_step={action_per_video_step}."
            )

        context, context_mask = self._prepare_infer_context(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
        )

        if gt_video is None:
            raise ValueError("Hierarchical infer now requires `gt_video` for teacher-forced chunk conditioning.")
        if gt_video.ndim == 4:
            gt_video_batch = gt_video.unsqueeze(0)
        elif gt_video.ndim == 5:
            gt_video_batch = gt_video
        else:
            raise ValueError(
                f"`gt_video` must have shape [3,T,H,W] or [1,3,T,H,W], got {tuple(gt_video.shape)}"
            )
        if gt_video_batch.shape[0] != 1 or gt_video_batch.shape[1] != 3:
            raise ValueError(
                f"`gt_video` must have shape [1,3,T,H,W], got {tuple(gt_video_batch.shape)}"
            )
        if int(gt_video_batch.shape[2]) != int(num_video_frames):
            raise ValueError(
                f"`gt_video` frame count mismatch: expected {num_video_frames}, got {int(gt_video_batch.shape[2])}."
            )
        gt_video_batch = gt_video_batch.to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=gt_video_batch[0, :, 0].unsqueeze(0),
            tiled=tiled,
        )
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        z_dim = self.vae.model.z_dim
        high_frames = (num_video_frames - 1) // self.hierarchical_high_stride + 1
        high_latent_t = (high_frames - 1) // self.vae.temporal_downsample_factor + 1
        chunk_video_steps = chunk_h // action_per_video_step
        low_frames = chunk_video_steps // self.hierarchical_low_stride + 1
        low_latent_t = (low_frames - 1) // self.vae.temporal_downsample_factor + 1
        full_high_frame_positions = self._global_high_frame_positions(
            high_frames=high_latent_t,
            block_chunks=num_chunks,
            low_frames_per_chunk=low_latent_t,
            device=self.device,
        )
        gt_high_video = self._temporal_downsample(gt_video_batch, stride=self.hierarchical_high_stride)
        gt_high_latents = self._encode_video_latents(gt_high_video, tiled=tiled)
        if int(gt_high_latents.shape[2]) != int(high_latent_t):
            raise ValueError(
                "GT high-level latent length mismatch: "
                f"expected {high_latent_t}, got {int(gt_high_latents.shape[2])}."
            )

        g_video = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        g_action = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)

        high_latents = torch.randn(
            (1, z_dim, high_latent_t, latent_h, latent_w),
            generator=g_video,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        high_latents[:, :, 0:1] = first_frame_latents.clone()

        if high_video_inference_steps is None:
            high_video_inference_steps = max(1, int(num_inference_steps) // 2)
        if low_video_inference_steps is None:
            low_video_inference_steps = max(1, int(num_inference_steps) // 2)
        if action_inference_steps is None:
            action_inference_steps = int(num_inference_steps)

        high_video_inference_steps = max(1, int(high_video_inference_steps))
        low_video_inference_steps = max(1, int(low_video_inference_steps))
        action_inference_steps = max(1, int(action_inference_steps))

        high_timesteps, high_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=high_video_inference_steps,
            device=self.device,
            dtype=high_latents.dtype,
            shift_override=sigma_shift,
        )
        high_cache_predict_timestep = self._next_schedule_timestep(
            full_timesteps=high_timesteps,
            used_steps=int(high_timesteps.shape[0]),
            dtype=high_latents.dtype,
            device=self.device,
        )

        low_timesteps, low_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=low_video_inference_steps,
            device=self.device,
            dtype=high_latents.dtype,
            shift_override=sigma_shift,
        )
        low_cache_predict_timestep = self._next_schedule_timestep(
            full_timesteps=low_timesteps,
            used_steps=int(low_timesteps.shape[0]),
            dtype=high_latents.dtype,
            device=self.device,
        )
        action_timesteps, action_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=action_inference_steps,
            device=self.device,
            dtype=high_latents.dtype,
            shift_override=sigma_shift,
        )

        pred_chunk_videos: list[torch.Tensor] = []
        pred_chunk_actions: list[torch.Tensor] = []
        first_chunk_high_video = None
        chunk_first_frame_latents = first_frame_latents.clone()
        total_high_frames = int(high_latents.shape[2])

        for chunk_idx in range(num_chunks):
            current_fixed_high = min(chunk_idx + 1, total_high_frames)
            fixed_high_latents = gt_high_latents[:, :, :current_fixed_high].clone()

            if current_fixed_high < total_high_frames:
                high_latents[:, :, :current_fixed_high] = fixed_high_latents[:, :, :current_fixed_high]
                for step_t, step_delta in zip(high_timesteps, high_deltas):
                    t_video = self._build_history_predict_timestep(
                        step_timestep=step_t,
                        total_frames=total_high_frames,
                        history_frames=current_fixed_high,
                        dtype=high_latents.dtype,
                        device=self.device,
                    )
                    if self.hierarchical_mask_high_predict:
                        pred_high = self._predict_high_video_noise_masked(
                            latents_video=high_latents,
                            timestep_video=t_video,
                            context=context,
                            context_mask=context_mask,
                            history_frames=current_fixed_high,
                            fuse_vae_embedding_in_latents=fuse_flag,
                            frame_positions=full_high_frame_positions,
                        )
                    else:
                        pred_high = self._predict_video_only_noise(
                            latents_video=high_latents,
                            timestep_video=t_video,
                            context=context,
                            context_mask=context_mask,
                            fuse_vae_embedding_in_latents=fuse_flag,
                            video_level="high",
                            frame_positions=full_high_frame_positions,
                        )
                    high_latents = self.infer_video_scheduler.step(pred_high, step_delta, high_latents)
                    high_latents[:, :, :current_fixed_high] = fixed_high_latents[:, :, :current_fixed_high]
            elif current_fixed_high > 0:
                high_latents[:, :, :current_fixed_high] = fixed_high_latents[:, :, :current_fixed_high]

            if chunk_idx == 0 and return_high_level_video:
                first_chunk_high_video = self._decode_latents_to_tensor(high_latents, tiled=tiled)

            if self.hierarchical_mask_high_predict:
                # Align with mask semantics: forward all available history, then slice the visible part.
                high_cache_source = high_latents[:, :, :current_fixed_high].clone()
                cache_timestep_high = self._build_history_predict_timestep(
                    step_timestep=high_cache_predict_timestep,
                    total_frames=int(high_cache_source.shape[2]),
                    history_frames=int(high_cache_source.shape[2]),
                    dtype=high_cache_source.dtype,
                    device=self.device,
                )
                source_kv_cache, _, high_tokens_per_frame = self._build_video_kv_cache(
                    latents_video=high_cache_source,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    timestep_video=cache_timestep_high,
                    video_level="high",
                    frame_positions=full_high_frame_positions[: int(high_cache_source.shape[2])],
                )
            else:
                # No high-mask: forward full high sequence, then slice the visible part.
                cache_timestep_high = self._build_history_predict_timestep(
                    step_timestep=high_cache_predict_timestep,
                    total_frames=int(high_latents.shape[2]),
                    history_frames=current_fixed_high,
                    dtype=high_latents.dtype,
                    device=self.device,
                )
                source_kv_cache, _, high_tokens_per_frame = self._build_video_kv_cache(
                    latents_video=high_latents,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    timestep_video=cache_timestep_high,
                    video_level="high",
                    frame_positions=full_high_frame_positions,
                )

            high_source_frames = current_fixed_high if self.hierarchical_mask_high_predict else total_high_frames
            high_kv_cache, high_frames_for_low, _ = self._slice_high_cache_for_chunk(
                source_kv_cache,
                chunk_idx=chunk_idx,
                high_frames=high_source_frames,
                tokens_per_frame=high_tokens_per_frame,
            )
            high_seq_len = high_frames_for_low * high_tokens_per_frame

            latents_low = torch.randn(
                (1, z_dim, low_latent_t, latent_h, latent_w),
                generator=g_video,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)
            chunk_frame_start = chunk_idx * chunk_video_steps
            chunk_first_frame_latents = self._encode_input_image_latents_tensor(
                gt_video_batch[0, :, chunk_frame_start].unsqueeze(0),
                tiled=tiled,
            )
            latents_low[:, :, 0:1] = chunk_first_frame_latents.clone()
            low_frame_positions = self._global_low_frame_positions(
                chunk_indices=chunk_idx,
                low_frames=low_latent_t,
                low_frames_per_chunk=low_latent_t,
                device=self.device,
            )

            for step_t_low, step_d_low in zip(low_timesteps, low_deltas):
                t_low = step_t_low.unsqueeze(0).to(device=self.device, dtype=latents_low.dtype)
                pred_low = self._predict_low_video_noise_with_high_cache(
                    low_chunk_latents=latents_low,
                    timestep_video=t_low,
                    context=context,
                    context_mask=context_mask,
                    high_kv_cache=high_kv_cache,
                    high_seq_len=high_seq_len,
                    high_frames=high_frames_for_low,
                    tokens_per_frame=high_tokens_per_frame,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    visible_high_start=0,
                    visible_high_end=high_frames_for_low - 1,
                    mask_low_predict=self.hierarchical_mask_low_predict,
                    low_frame_positions=low_frame_positions,
                )
                latents_low = self.infer_video_scheduler.step(pred_low, step_d_low, latents_low)
                latents_low[:, :, 0:1] = chunk_first_frame_latents

            if self.hierarchical_mask_low_predict:
                low_cache_latents = latents_low[:, :, :1].clone()
            else:
                low_cache_latents = latents_low

            chunk_action_start = chunk_idx * chunk_h
            chunk_proprio = self._get_chunk_proprio(proprio, action_start=chunk_action_start)
            pred_action_chunk = self._infer_action_from_video_condition(
                latents_video_cond=low_cache_latents,
                chunk_horizon=chunk_h,
                context=context,
                context_mask=context_mask,
                action_timesteps=action_timesteps,
                action_deltas=action_deltas,
                generator=g_action,
                rand_device=rand_device,
                proprio=chunk_proprio,
                fuse_vae_embedding_in_latents=fuse_flag,
                only_first_low_frame=self.hierarchical_mask_low_predict,
                low_cache_predict_timestep=low_cache_predict_timestep,
                high_kv_cache_for_low=high_kv_cache,
                high_frames_for_low=high_frames_for_low,
                high_tokens_per_frame_for_low=high_tokens_per_frame,
                low_frame_positions=low_frame_positions,
            )

            low_video_chunk = self._decode_latents_to_tensor(latents_low, tiled=tiled)
            pred_chunk_videos.append(low_video_chunk)
            pred_chunk_actions.append(pred_action_chunk)

        full_video = self._concat_chunk_video_tensors(
            pred_chunk_videos,
            replace_boundary_with_next_chunk_first_frame=True,
        )
        full_action = torch.cat(pred_chunk_actions, dim=0)
        out = {
            "video": self._video_tensor_to_pil(full_video),
            "action": full_action.detach().to(device="cpu", dtype=torch.float32),
        }
        if return_high_level_video:
            if first_chunk_high_video is not None:
                out["video_high_chunk0"] = self._video_tensor_to_pil(first_chunk_high_video)
        return out

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "hierarchical_level_embedding": self.hierarchical_level_embedding.state_dict(),
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

        if "hierarchical_level_embedding" in payload:
            self.hierarchical_level_embedding.load_state_dict(payload["hierarchical_level_embedding"], strict=True)
        else:
            logger.warning(
                "Checkpoint has no `hierarchical_level_embedding`; keeping current random level embeddings."
            )

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
