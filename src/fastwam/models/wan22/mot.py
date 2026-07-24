from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .wan_video_dit import flash_attention, modulate, rope_apply
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention. This will save memory but use more computation.")

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )
        
        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _normalize_slice(start: int, end: int) -> tuple[int, int]:
        start = int(max(0, start))
        end = int(max(start, end))
        return start, end

    @staticmethod
    def _slice_mask(attention_mask: torch.Tensor, query_start: int, query_end: int, key_len: int) -> torch.Tensor:
        if attention_mask.ndim == 2:
            return attention_mask[query_start:query_end, :key_len].unsqueeze(0).unsqueeze(0)
        if attention_mask.ndim == 3:
            return attention_mask[:, query_start:query_end, :key_len].unsqueeze(1)
        if attention_mask.ndim == 4:
            return attention_mask[:, :, query_start:query_end, :key_len]
        raise ValueError(
            f"`attention_mask` must be 2D/3D/4D for attention capture, got {tuple(attention_mask.shape)}"
        )

    def _summarize_attention_to_keys(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_mask: torch.Tensor,
        query_start: int,
        query_end: int,
        key_slices: dict[str, tuple[int, int]],
        query_chunk_size: int,
    ) -> dict[str, torch.Tensor]:
        query_start, query_end = self._normalize_slice(query_start, query_end)
        if query_end <= query_start:
            return {}

        bsz = int(q.shape[0])
        key_len = int(k.shape[1])
        head_dim = int(self.attn_head_dim)
        q_heads = q.reshape(bsz, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k_heads = k.reshape(bsz, -1, self.num_heads, head_dim).permute(0, 2, 1, 3)

        summaries = {
            name: torch.zeros(end - start, device=q.device, dtype=torch.float32)
            for name, (start, end) in key_slices.items()
            if end > start
        }
        chunk_size = max(1, int(query_chunk_size))
        scale = 1.0 / math.sqrt(float(head_dim))
        mask = attention_mask.to(device=q.device)

        for chunk_start in range(query_start, query_end, chunk_size):
            chunk_end = min(query_end, chunk_start + chunk_size)
            logits = torch.matmul(
                q_heads[:, :, chunk_start:chunk_end, :].float(),
                k_heads.float().transpose(-2, -1),
            ) * scale
            chunk_mask = self._slice_mask(mask, chunk_start, chunk_end, key_len)
            if chunk_mask.dtype == torch.bool:
                logits = logits.masked_fill(~chunk_mask, torch.finfo(logits.dtype).min)
            else:
                logits = logits + chunk_mask.to(dtype=logits.dtype)
            attn = torch.softmax(logits, dim=-1)
            for name, (start, end) in key_slices.items():
                if name not in summaries or end <= start:
                    continue
                key_scores = attn[..., start:end].amax(dim=(0, 1, 2)).detach()
                summaries[name] = torch.maximum(summaries[name], key_scores)

        for name in list(summaries.keys()):
            summaries[name] = summaries[name].to(device="cpu")
        return summaries

    def _maybe_capture_attention_maps(
        self,
        *,
        recorder: Optional[dict[str, Any]],
        layer_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_mask: torch.Tensor,
        relation_specs: list[dict[str, Any]],
    ) -> None:
        if not recorder or not bool(recorder.get("enabled", False)):
            return
        layers = recorder.get("layers")
        if layers is not None and int(layer_idx) not in layers:
            return
        max_records = int(recorder.get("max_records", 256))
        records = recorder.setdefault("records", [])
        if len(records) >= max_records:
            return

        query_chunk_size = int(recorder.get("query_chunk_size", 256))
        for spec in relation_specs:
            if len(records) >= max_records:
                break
            key_start, key_end = spec["key_slice"]
            query_start, query_end = spec["query_slice"]
            if key_end <= key_start or query_end <= query_start:
                continue
            summaries = self._summarize_attention_to_keys(
                q=q,
                k=k,
                attention_mask=attention_mask,
                query_start=query_start,
                query_end=query_end,
                key_slices={"target": (key_start, key_end)},
                query_chunk_size=query_chunk_size,
            )
            values = summaries.get("target")
            if values is None:
                continue
            tokens_per_frame = int(spec["tokens_per_frame"])
            if tokens_per_frame <= 0:
                continue
            usable = (int(values.numel()) // tokens_per_frame) * tokens_per_frame
            if usable <= 0:
                continue
            values = values[:usable].reshape(-1, tokens_per_frame).contiguous()
            relation = str(spec["relation"])
            target_kind = spec.get("target_kind")
            if target_kind is None:
                target_kind = "low" if relation.endswith("_to_low") else "high"
            frame_indices = spec.get("frame_indices")
            if frame_indices is None:
                frame_indices = recorder.get(f"{target_kind}_frame_indices")
            if frame_indices is None:
                frame_indices = list(range(values.shape[0]))
            records.append(
                {
                    "relation": relation,
                    "target_kind": str(target_kind),
                    "phase": str(recorder.get("phase", "")),
                    "step_idx": int(recorder.get("step_idx", -1)),
                    "total_steps": int(recorder.get("total_steps", -1)),
                    "layer_idx": int(layer_idx),
                    "tokens_per_frame": tokens_per_frame,
                    "grid_size": tuple(int(v) for v in spec["grid_size"]),
                    "frame_indices": [int(v) for v in list(frame_indices)[: values.shape[0]]],
                    "attention_score_sum": float(values.sum().item()),
                    "score_aggregation": "max_over_action_queries_heads",
                    "attention": values,
                }
            )

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill video branch once and cache per-layer K/V for action denoising.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            # Video prefill uses only video self-attention mask.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def _prefill_video_cache_inner(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Compile-friendly prefill loop without validation or dict cache output."""
        expert = self.mixtures["video"]
        x = video_tokens
        cache_k_list: list[torch.Tensor] = []
        cache_v_list: list[torch.Tensor] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                _use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            x = self._apply_expert_post_block(
                block=block,
                residual_x=residual_x,
                mixed_attn_out=mixed,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                context_payload=video_context_payload,
            )
            cache_k_list.append(k)
            cache_v_list.append(v)
        return x, cache_k_list, cache_v_list

    def _forward_action_with_video_cache_inner(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compile-friendly action loop with flat video K/V cache inputs."""
        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                _use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            k_cat = torch.cat([video_cache_k[layer_idx], k_action], dim=1)
            v_cat = torch.cat([video_cache_v[layer_idx], v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_expert_post_block(
                block=block,
                residual_x=residual_x,
                mixed_attn_out=mixed,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                context_payload=action_context_payload,
            )
        return x

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        attention_recorder: Optional[dict[str, Any]] = None,
        tokens_per_frame: Optional[int] = None,
        high_seq_len: Optional[int] = None,
        action_query_start: int = 0,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        # Use the action query rows from the joint [video+action] mask.
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            # Mixed attention: action queries attend to cached video K/V plus current action K/V.
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            if attention_recorder is not None and tokens_per_frame is not None:
                high_len = int(high_seq_len) if high_seq_len is not None else int(video_seq_len)
                high_len = max(0, min(high_len, int(video_seq_len)))
                low_len = int(video_seq_len) - high_len
                relation_specs = [
                    {
                        "relation": "action_to_high",
                        "query_slice": (int(action_query_start), action_seq_len),
                        "key_slice": (0, high_len),
                        "tokens_per_frame": int(tokens_per_frame),
                        "grid_size": (
                            int(attention_recorder.get("token_grid_h", 0)),
                            int(attention_recorder.get("token_grid_w", 0)),
                        ),
                    },
                    {
                        "relation": "action_to_low",
                        "query_slice": (int(action_query_start), action_seq_len),
                        "key_slice": (high_len, high_len + low_len),
                        "tokens_per_frame": int(tokens_per_frame),
                        "grid_size": (
                            int(attention_recorder.get("token_grid_h", 0)),
                            int(attention_recorder.get("token_grid_w", 0)),
                        ),
                    },
                ]
                self._maybe_capture_attention_maps(
                    recorder=attention_recorder,
                    layer_idx=layer_idx,
                    q=q_action,
                    k=k_cat,
                    attention_mask=action_attention_mask,
                    relation_specs=relation_specs,
                )
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x

    def forward_video_with_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        prefix_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        prefix_seq_len: int,
        attention_recorder: Optional[dict[str, Any]] = None,
        tokens_per_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """Run video branch with cached video prefix K/V.

        The queries come from `video_tokens` (usually low-level chunk tokens), and
        keys/values are concatenated from cached prefix K/V + current chunk K/V.
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `forward_video_with_video_cache`.")
        if len(prefix_kv_cache) != self.num_layers:
            raise ValueError(
                f"`prefix_kv_cache` must contain {self.num_layers} layers, got {len(prefix_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        video_seq_len = int(video_tokens.shape[1])
        total_seq_len = int(prefix_seq_len) + video_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        video_attention_mask = attention_mask[prefix_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["video"]
        x = video_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_video,
                k_video,
                v_video,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )

            layer_cache = prefix_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(f"`prefix_kv_cache[{layer_idx}]` must contain `k` and `v`.")

            k_prefix = layer_cache["k"]
            v_prefix = layer_cache["v"]
            if k_prefix.shape[1] != prefix_seq_len or v_prefix.shape[1] != prefix_seq_len:
                raise ValueError(
                    f"`prefix_kv_cache[{layer_idx}]` seq len mismatch, expected {prefix_seq_len}."
                )

            k_cat = torch.cat([k_prefix, k_video], dim=1)
            v_cat = torch.cat([v_prefix, v_video], dim=1)
            if attention_recorder is not None and tokens_per_frame is not None:
                self._maybe_capture_attention_maps(
                    recorder=attention_recorder,
                    layer_idx=layer_idx,
                    q=q_video,
                    k=k_cat,
                    attention_mask=video_attention_mask,
                    relation_specs=[
                        {
                            "relation": "low_to_high",
                            "query_slice": (0, video_seq_len),
                            "key_slice": (0, int(prefix_seq_len)),
                            "tokens_per_frame": int(tokens_per_frame),
                            "grid_size": (
                                int(attention_recorder.get("token_grid_h", 0)),
                                int(attention_recorder.get("token_grid_w", 0)),
                            ),
                        }
                    ],
                )
            mixed = self._mixed_attention(
                q_cat=q_video,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=video_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
        return x

    def forward_video_action_with_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        prefix_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        prefix_seq_len: int,
        attention_recorder: Optional[dict[str, Any]] = None,
        tokens_per_frame: Optional[int] = None,
        action_query_start: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run video and action branches jointly with cached video prefix K/V."""
        if "video" not in self.mixtures or "action" not in self.mixtures:
            raise ValueError("MoT requires both `video` and `action` experts.")
        if len(prefix_kv_cache) != self.num_layers:
            raise ValueError(
                f"`prefix_kv_cache` must contain {self.num_layers} layers, got {len(prefix_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        video_seq_len = int(video_tokens.shape[1])
        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(prefix_seq_len) + video_seq_len + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        query_attention_mask = attention_mask[prefix_seq_len:total_seq_len, :total_seq_len]

        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        x_video = video_tokens
        x_action = action_tokens
        for layer_idx in range(self.num_layers):
            video_block = video_expert.blocks[layer_idx]
            action_block = action_expert.blocks[layer_idx]

            (
                q_video,
                k_video,
                v_video,
                residual_video,
                gate_msa_video,
                shift_mlp_video,
                scale_mlp_video,
                gate_mlp_video,
                use_checkpoint_video,
            ) = self._build_expert_attention_io(
                expert=video_expert,
                block=video_block,
                x=x_video,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            (
                q_action,
                k_action,
                v_action,
                residual_action,
                gate_msa_action,
                shift_mlp_action,
                scale_mlp_action,
                gate_mlp_action,
                use_checkpoint_action,
            ) = self._build_expert_attention_io(
                expert=action_expert,
                block=action_block,
                x=x_action,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )

            layer_cache = prefix_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(f"`prefix_kv_cache[{layer_idx}]` must contain `k` and `v`.")
            k_prefix = layer_cache["k"]
            v_prefix = layer_cache["v"]
            if k_prefix.shape[1] != prefix_seq_len or v_prefix.shape[1] != prefix_seq_len:
                raise ValueError(
                    f"`prefix_kv_cache[{layer_idx}]` seq len mismatch, expected {prefix_seq_len}."
                )

            q_cat = torch.cat([q_video, q_action], dim=1)
            k_cat = torch.cat([k_prefix, k_video, k_action], dim=1)
            v_cat = torch.cat([v_prefix, v_video, v_action], dim=1)
            if attention_recorder is not None and tokens_per_frame is not None:
                relation_specs = [
                    {
                        "relation": "action_to_high",
                        "query_slice": (video_seq_len + int(action_query_start), video_seq_len + action_seq_len),
                        "key_slice": (0, int(prefix_seq_len)),
                        "tokens_per_frame": int(tokens_per_frame),
                        "grid_size": (
                            int(attention_recorder.get("token_grid_h", 0)),
                            int(attention_recorder.get("token_grid_w", 0)),
                        ),
                    },
                    {
                        "relation": "action_to_low",
                        "query_slice": (video_seq_len + int(action_query_start), video_seq_len + action_seq_len),
                        "key_slice": (int(prefix_seq_len), int(prefix_seq_len) + video_seq_len),
                        "tokens_per_frame": int(tokens_per_frame),
                        "grid_size": (
                            int(attention_recorder.get("token_grid_h", 0)),
                            int(attention_recorder.get("token_grid_w", 0)),
                        ),
                    },
                    {
                        "relation": "low_to_high",
                        "query_slice": (0, video_seq_len),
                        "key_slice": (0, int(prefix_seq_len)),
                        "tokens_per_frame": int(tokens_per_frame),
                        "grid_size": (
                            int(attention_recorder.get("token_grid_h", 0)),
                            int(attention_recorder.get("token_grid_w", 0)),
                        ),
                    },
                ]
                self._maybe_capture_attention_maps(
                    recorder=attention_recorder,
                    layer_idx=layer_idx,
                    q=q_cat,
                    k=k_cat,
                    attention_mask=query_attention_mask,
                    relation_specs=relation_specs,
                )
            mixed = self._mixed_attention(
                q_cat=q_cat,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=query_attention_mask,
            )
            mixed_video, mixed_action = mixed.split([video_seq_len, action_seq_len], dim=1)
            x_video = self._apply_post_with_optional_checkpoint(
                block=video_block,
                residual_x=residual_video,
                gate_msa=gate_msa_video,
                shift_mlp=shift_mlp_video,
                scale_mlp=scale_mlp_video,
                gate_mlp=gate_mlp_video,
                use_gradient_checkpointing=use_checkpoint_video,
                mixed_slice=mixed_video,
                context_payload=video_context_payload,
            )
            x_action = self._apply_post_with_optional_checkpoint(
                block=action_block,
                residual_x=residual_action,
                gate_msa=gate_msa_action,
                shift_mlp=shift_mlp_action,
                scale_mlp=scale_mlp_action,
                gate_mlp=gate_mlp_action,
                use_gradient_checkpointing=use_checkpoint_action,
                mixed_slice=mixed_action,
                context_payload=action_context_payload,
            )
        return x_video, x_action

    def prefill_video_cache_with_prefix(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        prefix_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        prefix_seq_len: int,
        return_tokens: bool = False,
        attention_recorder: Optional[dict[str, Any]] = None,
        tokens_per_frame: Optional[int] = None,
    ) -> list[dict[str, torch.Tensor]] | tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
        """Prefill current video-token cache while attending to cached video prefix.

        This is used when low-level video tokens must first attend corresponding
        high-level cached tokens, then expose low-only per-layer K/V for action decoding.
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache_with_prefix`.")
        if len(prefix_kv_cache) != self.num_layers:
            raise ValueError(
                f"`prefix_kv_cache` must contain {self.num_layers} layers, got {len(prefix_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        video_seq_len = int(video_tokens.shape[1])
        total_seq_len = int(prefix_seq_len) + video_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        video_attention_mask = attention_mask[prefix_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["video"]
        x = video_tokens
        low_kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_video,
                k_video,
                v_video,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )

            layer_cache = prefix_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(f"`prefix_kv_cache[{layer_idx}]` must contain `k` and `v`.")

            k_prefix = layer_cache["k"]
            v_prefix = layer_cache["v"]
            if k_prefix.shape[1] != prefix_seq_len or v_prefix.shape[1] != prefix_seq_len:
                raise ValueError(
                    f"`prefix_kv_cache[{layer_idx}]` seq len mismatch, expected {prefix_seq_len}."
                )

            k_cat = torch.cat([k_prefix, k_video], dim=1)
            v_cat = torch.cat([v_prefix, v_video], dim=1)
            if attention_recorder is not None and tokens_per_frame is not None:
                self._maybe_capture_attention_maps(
                    recorder=attention_recorder,
                    layer_idx=layer_idx,
                    q=q_video,
                    k=k_cat,
                    attention_mask=video_attention_mask,
                    relation_specs=[
                        {
                            "relation": "low_to_high",
                            "query_slice": (0, video_seq_len),
                            "key_slice": (0, int(prefix_seq_len)),
                            "tokens_per_frame": int(tokens_per_frame),
                            "grid_size": (
                                int(attention_recorder.get("token_grid_h", 0)),
                                int(attention_recorder.get("token_grid_w", 0)),
                            ),
                        }
                    ],
                )
            mixed = self._mixed_attention(
                q_cat=q_video,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=video_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            low_kv_cache.append({"k": k_video, "v": v_video})
        if return_tokens:
            return low_kv_cache, x
        return low_kv_cache

    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
        attention_recorder: Optional[dict[str, Any]] = None,
        attention_layout: Optional[dict[str, int]] = None,
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask.ndim == 2:
            if attention_mask.shape[0] != attention_mask.shape[1]:
                raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")
        elif attention_mask.ndim == 3:
            if attention_mask.shape[1] != attention_mask.shape[2]:
                raise ValueError(f"`attention_mask` must be square on last dims, got shape {tuple(attention_mask.shape)}")
        else:
            raise ValueError(
                f"`attention_mask` must be 2D [S,S] or 3D [B,S,S], got shape {tuple(attention_mask.shape)}"
            )

        tokens_all = {k: v for k, v in embeds_all.items()}

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }

            # 3. concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask.ndim == 2:
                if attention_mask.shape[0] != total_seq:
                    raise ValueError(
                        "Attention mask seq length mismatch: "
                        f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                    )
            else:
                if attention_mask.shape[1] != total_seq:
                    raise ValueError(
                        "Attention mask seq length mismatch: "
                        f"mask={attention_mask.shape[1]} vs tokens={total_seq}"
                    )
                if attention_mask.shape[0] != q_cat.shape[0]:
                    raise ValueError(
                        "3D attention mask batch mismatch: "
                        f"mask_batch={attention_mask.shape[0]} vs tokens_batch={q_cat.shape[0]}"
                    )

            mixed = self._mixed_attention(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)
            if attention_recorder is not None and attention_layout is not None:
                video_seq_len = int(tokens_all["video"].shape[1])
                action_seq_len = int(tokens_all["action"].shape[1])
                high_seq_len = int(attention_layout.get("high_seq_len", 0))
                low_seq_len = int(attention_layout.get("low_seq_len", max(0, video_seq_len - high_seq_len)))
                action_query_start = int(attention_layout.get("action_query_start", 0))
                tokens_per_frame = int(attention_layout.get("tokens_per_frame", 0))
                relation_specs = [
                    {
                        "relation": "action_to_high",
                        "query_slice": (video_seq_len + action_query_start, video_seq_len + action_seq_len),
                        "key_slice": (0, high_seq_len),
                        "tokens_per_frame": tokens_per_frame,
                        "grid_size": (
                            int(attention_recorder.get("token_grid_h", 0)),
                            int(attention_recorder.get("token_grid_w", 0)),
                        ),
                    },
                    {
                        "relation": "action_to_low",
                        "query_slice": (video_seq_len + action_query_start, video_seq_len + action_seq_len),
                        "key_slice": (high_seq_len, high_seq_len + low_seq_len),
                        "tokens_per_frame": tokens_per_frame,
                        "grid_size": (
                            int(attention_recorder.get("token_grid_h", 0)),
                            int(attention_recorder.get("token_grid_w", 0)),
                        ),
                    },
                    {
                        "relation": "low_to_high",
                        "query_slice": (high_seq_len, high_seq_len + low_seq_len),
                        "key_slice": (0, high_seq_len),
                        "tokens_per_frame": tokens_per_frame,
                        "grid_size": (
                            int(attention_recorder.get("token_grid_h", 0)),
                            int(attention_recorder.get("token_grid_w", 0)),
                        ),
                    },
                ]
                self._maybe_capture_attention_maps(
                    recorder=attention_recorder,
                    layer_idx=layer_idx,
                    q=q_cat,
                    k=k_cat,
                    attention_mask=attention_mask,
                    relation_specs=relation_specs,
                )

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all
