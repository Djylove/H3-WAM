"""Causal FACT token backbone port for the frozen-H3 D0 action carrier.

This module is deliberately a ``backbone_port``, not an official FACT
reproduction.  It preserves FACT's executable causal contract while replacing
the Wan video tokens with MiniMax-H3 K/V tokens and retaining the deployed D0
ActionDiT as the action stem.  Unlike :mod:`fact_joint_aux`, consequence targets
are represented as noisy token tracks in one shared causal transformer:

    [state/text | H3 ref | pred action | clean action |
     future state | value | future H3 representation]

The predicted-action track can only attend to the prefix and itself.  The clean
action is a K/V condition for future tracks and cannot see the predicted action.
Future representation tokens may attend to future state/value, matching FACT's
future-image placement.  The action residual decoder is zero initialized, so a
new port is exactly the D0 parent at step zero while future losses still update
the shared causal trunk and the clean-action pass through the D0 carrier.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from .dreamwam_kv_carrier import H3DreamWAMKVCarrierPolicy


FACT_PINNED_COMMIT = "618a6c16868699b6d4138941de6a863589ac00dd"
FACT_CURRENT_AUDITED_COMMIT = "9427ea451e806220742148049ef0576e43ef7382"
FACT_ACTION_WEIGHT = 10.0
FACT_FUTURE_REPRESENTATION_WEIGHT = 1.0
FACT_FUTURE_STATE_WEIGHT = 0.4
FACT_VALUE_WEIGHT = 0.4
C59_FAILURE_OVERLAY_FORMAT = "h3wam-c59-fact-failure-active-overlay-v1"
C59_VALUE_CONTRACTS = (
    "fact_code_remaining_plus_penalty",
    "fact_paper_eq6_progress_minus_penalty",
)
C60_CAUSAL_FAILURE_FORMAT = "h3wam-c60-counterfactual-failure-dataset-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class C59FailureOverlay:
    """Strict sample-label loader for the C59 failure-active contract.

    The executable FACT code and paper Eq. 6 use opposite value orientations.
    Callers must choose one explicitly.  Unannotated failures retain the base
    temporal value and observed future losses; only the failure penalty requires
    an explicit onset.  This rejects the old C48 convention that added ``+1`` to
    every failed row.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        source_dataset: Path | str,
        value_contract: str,
    ) -> None:
        self.root = Path(root).resolve()
        self.source_dataset = Path(source_dataset).resolve()
        if value_contract not in C59_VALUE_CONTRACTS:
            raise ValueError(
                f"value_contract must be one of {C59_VALUE_CONTRACTS}; "
                "FACT code and paper targets cannot be mixed"
            )
        self.value_contract = value_contract
        report_path = self.root / "COMPLETED.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("format") != C59_FAILURE_OVERLAY_FORMAT
            or report.get("status") != "PASS_C59_FAILURE_OVERLAY_CONTRACT"
        ):
            raise ValueError("C59 overlay identity/status mismatch")
        if report.get("source_dataset_sha256") != _sha256_file(self.source_dataset):
            raise ValueError("C59 source dataset SHA256 mismatch")
        label_info = report.get("files", {}).get("sample_labels", {})
        labels_path = Path(label_info.get("path", self.root / "sample_labels.jsonl"))
        if not labels_path.is_absolute():
            labels_path = self.root / labels_path
        labels_path = labels_path.resolve()
        if not labels_path.is_file() or label_info.get("sha256") != _sha256_file(
            labels_path
        ):
            raise ValueError("C59 sample-label artifact mismatch")
        self.report = report
        self.labels_path = labels_path
        self.labels: dict[int, dict] = {}
        with labels_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                required = {
                    "sample_id",
                    "failure_episode_mask",
                    "failure_active_mask",
                    "action_loss_mask",
                    "future_loss_mask",
                    "fact_code_value_raw",
                    "fact_paper_progress_target",
                }
                missing = required - set(row)
                if missing:
                    raise ValueError(
                        f"C59 labels line {line_number} missing {sorted(missing)}"
                    )
                sample_id = int(row["sample_id"])
                if sample_id in self.labels:
                    raise ValueError(f"duplicate C59 sample_id {sample_id}")
                self.labels[sample_id] = row
        if len(self.labels) != int(report.get("samples", -1)):
            raise ValueError("C59 sample count mismatch")
        episodes = self._episode_metadata()
        for sample_id, row in self.labels.items():
            episode = episodes.get(int(row["episode_id"]))
            if episode is None:
                raise ValueError(f"C59 sample {sample_id} has no episode metadata")
            failure_episode = bool(row["failure_episode_mask"])
            active = bool(row["failure_active_mask"])
            if int(row["action_loss_mask"]) != int(not failure_episode):
                raise ValueError("C59 action mask is not episode-level failure masking")
            if active and not bool(episode.get("failure_onset_available")):
                raise ValueError("C59 active failure has no explicit onset")
            code_value = float(row["fact_code_value_raw"])
            if active:
                if not 1.0 <= code_value <= 2.0:
                    raise ValueError("C59 active code value must include exactly one penalty")
            elif not 0.0 <= code_value <= 1.0:
                raise ValueError(
                    "C59 inactive code value contains a fabricated failure penalty"
                )
            paper_value = float(row["fact_paper_progress_target"])
            if not 0.0 <= paper_value <= 1.0:
                raise ValueError("C59 paper Eq.6 target is outside [0,1]")

    def for_sample(self, sample_id: int) -> dict[str, float | int | str]:
        try:
            row = self.labels[int(sample_id)]
        except KeyError as error:
            raise KeyError(f"C59 has no sample_id {int(sample_id)}") from error
        failure_episode = bool(row["failure_episode_mask"])
        failure_active = bool(row["failure_active_mask"])
        value_key = (
            "fact_code_value_raw"
            if self.value_contract == "fact_code_remaining_plus_penalty"
            else "fact_paper_progress_target"
        )
        return {
            "sample_id": int(sample_id),
            "action_loss_mask": int(row["action_loss_mask"]),
            "future_loss_mask": int(row["future_loss_mask"]),
            "value_loss_mask": 1,
            "failure_episode_mask": int(failure_episode),
            "failure_active_mask": int(failure_active),
            "value_target": float(row[value_key]),
            "value_contract": self.value_contract,
        }

    def _episode_metadata(self) -> dict[int, dict]:
        cached = getattr(self, "_episodes", None)
        if cached is not None:
            return cached
        path_info = self.report.get("files", {}).get("failure_rollouts", {})
        path = Path(path_info.get("path", self.root / "failure_rollouts.jsonl"))
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        if not path.is_file() or path_info.get("sha256") != _sha256_file(path):
            raise ValueError("C59 failure-rollout artifact mismatch")
        episodes: dict[int, dict] = {}
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    episode = int(row["episode_index"])
                    if episode in episodes:
                        raise ValueError(f"duplicate C59 episode {episode}")
                    episodes[episode] = row
        self._episodes = episodes
        return episodes


class C60CausalFailureLabels:
    """Validate C60 state-aligned counterfactual failure rows.

    C60 is the penalty-active stream: every episode branches from an exactly
    restored successful parent state, changes the first action chunk, and then
    fails.  The intervention step is therefore an explicit causal onset.  This
    class validates labels only; image/H3 K/V caches remain separate immutable
    artifacts and must be joined by ``sample_id``/observation id.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        expected_sha256: str | None = None,
        value_contract: str,
    ) -> None:
        self.path = Path(path).resolve()
        self.sha256 = _sha256_file(self.path)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("C60 dataset SHA256 mismatch")
        if value_contract not in C59_VALUE_CONTRACTS:
            raise ValueError(
                f"value_contract must be one of {C59_VALUE_CONTRACTS}; "
                "FACT code and paper targets cannot be mixed"
            )
        self.value_contract = value_contract
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        if payload.get("format") != C60_CAUSAL_FAILURE_FORMAT:
            raise ValueError("C60 dataset format mismatch")
        if payload.get("action_contract") != "all branch actions masked from imitation":
            raise ValueError("C60 action contract mismatch")
        episodes = list(payload.get("episodes", []))
        rows = list(payload.get("samples", []))
        if not episodes or not rows:
            raise ValueError("C60 dataset is empty")
        episode_by_id: dict[int, dict] = {}
        parent_sources: dict[str, set[int]] = {"train": set(), "validation": set()}
        for episode in episodes:
            episode_id = int(episode["episode_index"])
            if episode_id in episode_by_id:
                raise ValueError(f"duplicate C60 episode {episode_id}")
            if (
                not bool(episode.get("failure_episode"))
                or int(episode.get("failure_active_from_frame", -1)) != 0
                or episode.get("annotation_source")
                != "state_aligned_counterfactual_action_intervention"
            ):
                raise ValueError("C60 episode lacks explicit intervention onset")
            split = str(episode["split"])
            if split not in parent_sources:
                raise ValueError(f"invalid C60 split {split}")
            parent_sources[split].add(int(episode["source_ordinal"]))
            episode_by_id[episode_id] = episode
        if parent_sources["train"] & parent_sources["validation"]:
            raise ValueError("C60 parent-source split leakage")

        self.rows: list[dict] = []
        seen_samples: set[int] = set()
        for row in rows:
            sample_id = int(row["sample_id"])
            if sample_id in seen_samples:
                raise ValueError(f"duplicate C60 sample {sample_id}")
            seen_samples.add(sample_id)
            episode = episode_by_id.get(int(row["episode_id"]))
            if episode is None or str(row["split"]) != str(episode["split"]):
                raise ValueError("C60 sample/episode split mismatch")
            if (
                bool(row.get("success"))
                or float(row.get("action_loss_mask", -1)) != 0.0
                or float(row.get("future_loss_mask", -1)) != 1.0
                or float(row.get("value_loss_mask", -1)) != 1.0
                or not bool(row.get("failure_active"))
            ):
                raise ValueError("C60 failure/action/future/value masks are invalid")
            if int(row["current_step"]) < int(row["failure_active_from_step"]):
                raise ValueError("C60 sample precedes its causal intervention onset")
            code_value = float(row["fact_code_value_raw"])
            if not 1.0 <= code_value <= 2.0:
                raise ValueError("C60 active code value lacks exactly one penalty")
            paper_value = float(row["fact_paper_progress_target"])
            if not 0.0 <= paper_value <= 1.0:
                raise ValueError("C60 paper Eq.6 target is outside [0,1]")
            self.rows.append(row)
        counts = payload.get("counts", {})
        expected_rows = sum(int(value["samples"]) for value in counts.values())
        if expected_rows != len(self.rows):
            raise ValueError("C60 sample count mismatch")
        self.payload = payload
        self.episodes = episode_by_id
        self.parent_sources = parent_sources

    def split(self, name: str) -> list[dict]:
        if name not in {"train", "validation"}:
            raise ValueError("C60 split must be train or validation")
        return [row for row in self.rows if row["split"] == name]

    def target_for(self, row: Mapping) -> dict[str, float | int | str]:
        value_key = (
            "fact_code_value_raw"
            if self.value_contract == "fact_code_remaining_plus_penalty"
            else "fact_paper_progress_target"
        )
        return {
            "sample_id": int(row["sample_id"]),
            "action_loss_mask": 0,
            "future_loss_mask": 1,
            "value_loss_mask": 1,
            "failure_active_mask": 1,
            "value_target": float(row[value_key]),
            "value_contract": self.value_contract,
        }


@dataclass(frozen=True)
class FACTTokenLayout:
    """Exact segment lengths for the FACT teacher-forcing mask."""

    state_tokens: int
    ref_tokens: int
    pred_action_tokens: int
    clean_action_tokens: int
    future_state_tokens: int
    value_tokens: int
    future_representation_tokens: int

    def __post_init__(self) -> None:
        values = tuple(self.as_dict().values())
        if any(value < 0 for value in values):
            raise ValueError("FACT token counts cannot be negative")
        if self.state_tokens + self.ref_tokens <= 0:
            raise ValueError("FACT requires a non-empty prefix")
        if self.pred_action_tokens <= 0:
            raise ValueError("FACT requires a predicted-action track")

    @property
    def prefix_end(self) -> int:
        return self.state_tokens + self.ref_tokens

    @property
    def pred_action_end(self) -> int:
        return self.prefix_end + self.pred_action_tokens

    @property
    def clean_action_end(self) -> int:
        return self.pred_action_end + self.clean_action_tokens

    @property
    def future_state_end(self) -> int:
        return self.clean_action_end + self.future_state_tokens

    @property
    def value_end(self) -> int:
        return self.future_state_end + self.value_tokens

    @property
    def total_tokens(self) -> int:
        return self.value_end + self.future_representation_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "state_tokens": int(self.state_tokens),
            "ref_tokens": int(self.ref_tokens),
            "pred_action_tokens": int(self.pred_action_tokens),
            "clean_action_tokens": int(self.clean_action_tokens),
            "future_state_tokens": int(self.future_state_tokens),
            "value_tokens": int(self.value_tokens),
            "future_representation_tokens": int(self.future_representation_tokens),
        }


def build_fact_teacher_forcing_mask(
    layout: FACTTokenLayout,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the additive mask used by official FACT's SDPA path.

    Rows are queries and columns are keys.  ``0`` means visible and ``-inf``
    means blocked.  The future-representation segment is FACT's future-image
    segment with H3 representation tokens substituted intentionally.
    """

    length = layout.total_tokens
    mask = torch.zeros((length, length), device=device, dtype=dtype)
    p_end = layout.prefix_end
    a_end = layout.pred_action_end
    g_end = layout.clean_action_end
    sv_end = layout.value_end
    neg_inf = float("-inf")

    mask[:p_end, p_end:] = neg_inf
    mask[p_end:a_end, a_end:] = neg_inf
    mask[a_end:g_end, p_end:a_end] = neg_inf
    mask[a_end:g_end, g_end:] = neg_inf
    mask[g_end:sv_end, p_end:a_end] = neg_inf
    mask[g_end:sv_end, sv_end:] = neg_inf
    mask[sv_end:, p_end:a_end] = neg_inf
    return mask


def _vector_mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    middle = max(32, min(256, hidden_dim))
    return nn.Sequential(
        nn.Linear(input_dim, middle),
        nn.GELU(),
        nn.Linear(middle, hidden_dim),
    )


class H3FACTBackbonePort(nn.Module):
    """Full causal-token FACT port attached to a D0/H3 action carrier.

    ``forward_action`` is Stage 1.  ``forward_consequence`` is Stage 2 and keeps
    a dummy predicted-action track to preserve the training layout.  ``forward``
    executes the joint teacher-forced training graph.
    """

    segment_names = (
        "prefix",
        "pred_action",
        "clean_action",
        "future_state",
        "value",
        "future_representation",
    )

    def __init__(
        self,
        carrier: H3DreamWAMKVCarrierPolicy,
        *,
        hidden_dim: int,
        future_representation_dim: int = 256,
        future_state_dim: int = 8,
        causal_layers: int = 4,
        causal_heads: int = 8,
        causal_ffn_dim: int | None = None,
        h3_source_layer: int = 49,
        max_timestep: float = 1000.0,
    ) -> None:
        super().__init__()
        if not carrier.enabled or carrier.action_expert is None:
            raise ValueError("FACT backbone port requires an enabled D0 carrier")
        dimensions = (
            hidden_dim,
            future_representation_dim,
            future_state_dim,
            causal_layers,
            causal_heads,
        )
        if min(dimensions) <= 0:
            raise ValueError("FACT backbone dimensions must be positive")
        if hidden_dim % causal_heads:
            raise ValueError("hidden_dim must be divisible by causal_heads")
        if h3_source_layer not in carrier.carrier_layers:
            raise ValueError("h3_source_layer must be present in the carrier cache")
        if max_timestep <= 0:
            raise ValueError("max_timestep must be positive")

        self.carrier = carrier
        self.hidden_dim = int(hidden_dim)
        self.future_representation_dim = int(future_representation_dim)
        self.future_state_dim = int(future_state_dim)
        self.h3_source_layer = int(h3_source_layer)
        self.max_timestep = float(max_timestep)
        self.action_dim = int(carrier.action_dim)
        self.proprio_dim = int(carrier.proprio_dim)
        self.context_dim = int(carrier.context_dim)
        self.h3_attention_width = int(carrier.attention_width)

        self.state_encoder = _vector_mlp(self.proprio_dim, self.hidden_dim)
        self.text_encoder = nn.Linear(self.context_dim, self.hidden_dim)
        self.h3_ref_encoder = nn.Linear(self.h3_attention_width, self.hidden_dim)
        self.action_hidden_encoder = (
            nn.Identity()
            if self.hidden_dim == int(carrier.action_expert.hidden_dim)
            else nn.Linear(int(carrier.action_expert.hidden_dim), self.hidden_dim)
        )
        # FACT shares the state encoder between current and future state only
        # when dimensions match.  LIBERO current state is 8D, so this port does.
        self.future_state_encoder = (
            self.state_encoder
            if self.future_state_dim == self.proprio_dim
            else _vector_mlp(self.future_state_dim, self.hidden_dim)
        )
        self.value_encoder = _vector_mlp(1, self.hidden_dim)
        self.future_representation_encoder = _vector_mlp(
            self.future_representation_dim, self.hidden_dim
        )
        self.timestep_encoder = _vector_mlp(1, self.hidden_dim)
        self.segment_embedding = nn.Parameter(
            torch.zeros(len(self.segment_names), self.hidden_dim)
        )

        ffn_dim = int(causal_ffn_dim or (4 * self.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(causal_heads),
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.causal_backbone = nn.TransformerEncoder(
            layer, num_layers=int(causal_layers), norm=nn.LayerNorm(self.hidden_dim)
        )
        self.action_residual_decoder = nn.Linear(self.hidden_dim, self.action_dim)
        self.future_state_decoder = _vector_mlp(
            self.hidden_dim, self.future_state_dim
        )
        self.value_decoder = _vector_mlp(self.hidden_dim, 1)
        self.future_representation_decoder = _vector_mlp(
            self.hidden_dim, self.future_representation_dim
        )
        nn.init.zeros_(self.action_residual_decoder.weight)
        nn.init.zeros_(self.action_residual_decoder.bias)

    def _pooled_text(
        self, text_context: torch.Tensor, text_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if text_context.ndim != 3 or text_context.shape[-1] != self.context_dim:
            raise ValueError("text_context must be [B,L,context_dim]")
        if text_mask is None:
            return text_context.float().mean(dim=1)
        if tuple(text_mask.shape) != tuple(text_context.shape[:2]):
            raise ValueError("text_mask must match text_context")
        weights = text_mask.to(device=text_context.device, dtype=torch.float32)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (text_context.float() * weights.unsqueeze(-1)).sum(dim=1) / denominator

    def _reference_tokens(
        self,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        *,
        batch: int,
    ) -> torch.Tensor:
        # The carrier call performs the strict all-layer, no-alias cache audit.
        source = video_kv_cache[self.h3_source_layer]["v"]
        source = self.carrier._flatten_cache_tensor(
            source, name=f"layer {self.h3_source_layer} v", batch=batch
        )
        return self.h3_ref_encoder(
            source.to(
                device=self.h3_ref_encoder.weight.device,
                dtype=self.h3_ref_encoder.weight.dtype,
            )
        )

    def _time_token(self, timestep: torch.Tensor, *, batch: int) -> torch.Tensor:
        if timestep.ndim not in (1, 2) or timestep.shape[0] != batch:
            raise ValueError("timestep must be [B] or [B,1]")
        if timestep.ndim == 2 and timestep.shape[1] != 1:
            raise ValueError("rank-2 timestep must have one column")
        normalized = timestep.float().reshape(batch, 1) / self.max_timestep
        return self.timestep_encoder(
            normalized.to(
                device=self.timestep_encoder[0].weight.device,
                dtype=self.timestep_encoder[0].weight.dtype,
            )
        ).unsqueeze(1)

    @staticmethod
    def _as_track(value: torch.Tensor, *, width: int, name: str) -> torch.Tensor:
        if value.ndim == 1 and width == 1:
            value = value.reshape(-1, 1, 1)
        if value.ndim == 2:
            value = value.unsqueeze(1)
        if value.ndim != 3 or value.shape[-1] != width:
            raise ValueError(f"{name} must be [B,T,{width}] or [B,{width}]")
        return value

    def _carrier_hidden(
        self,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        text_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction, hidden = self.carrier.forward_hidden(
            actions,
            timestep,
            text_context=text_context,
            proprio=proprio,
            video_kv_cache=video_kv_cache,
            text_mask=text_mask,
        )
        return prediction, self.action_hidden_encoder(hidden)

    def _prefix(
        self,
        *,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        text_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int, int]:
        batch = int(proprio.shape[0])
        if proprio.shape != (batch, self.proprio_dim):
            raise ValueError("proprio must be [B,proprio_dim]")
        state = self.state_encoder(proprio).unsqueeze(1)
        text = self.text_encoder(self._pooled_text(text_context, text_mask)).unsqueeze(1)
        ref = self._reference_tokens(video_kv_cache, batch=batch)
        prefix = torch.cat((state, text, ref), dim=1)
        prefix = prefix + self.segment_embedding[0].view(1, 1, -1)
        return prefix, 2, int(ref.shape[1])

    def _run_causal(
        self,
        *,
        prefix: torch.Tensor,
        state_tokens: int,
        ref_tokens: int,
        pred_action_hidden: torch.Tensor,
        clean_action_hidden: torch.Tensor | None = None,
        noisy_future_state: torch.Tensor | None = None,
        noisy_value: torch.Tensor | None = None,
        noisy_future_representation: torch.Tensor | None = None,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, FACTTokenLayout]:
        batch = int(prefix.shape[0])
        time = self._time_token(timestep, batch=batch)
        pred = pred_action_hidden + time + self.segment_embedding[1].view(1, 1, -1)
        chunks = [prefix, pred]

        clean_tokens = 0
        if clean_action_hidden is not None:
            clean = clean_action_hidden + self.segment_embedding[2].view(1, 1, -1)
            chunks.append(clean)
            clean_tokens = int(clean.shape[1])

        future_state_tokens = 0
        if noisy_future_state is not None:
            future_state = self._as_track(
                noisy_future_state, width=self.future_state_dim, name="noisy_future_state"
            )
            future_state = self.future_state_encoder(future_state) + time
            future_state = future_state + self.segment_embedding[3].view(1, 1, -1)
            chunks.append(future_state)
            future_state_tokens = int(future_state.shape[1])

        value_tokens = 0
        if noisy_value is not None:
            value = self._as_track(noisy_value, width=1, name="noisy_value")
            value = self.value_encoder(value) + time
            value = value + self.segment_embedding[4].view(1, 1, -1)
            chunks.append(value)
            value_tokens = int(value.shape[1])

        representation_tokens = 0
        if noisy_future_representation is not None:
            representation = self._as_track(
                noisy_future_representation,
                width=self.future_representation_dim,
                name="noisy_future_representation",
            )
            representation = self.future_representation_encoder(representation) + time
            representation = representation + self.segment_embedding[5].view(1, 1, -1)
            chunks.append(representation)
            representation_tokens = int(representation.shape[1])

        layout = FACTTokenLayout(
            state_tokens=state_tokens,
            ref_tokens=ref_tokens,
            pred_action_tokens=int(pred.shape[1]),
            clean_action_tokens=clean_tokens,
            future_state_tokens=future_state_tokens,
            value_tokens=value_tokens,
            future_representation_tokens=representation_tokens,
        )
        sequence = torch.cat(chunks, dim=1)
        if sequence.shape[1] != layout.total_tokens:
            raise RuntimeError("FACT token layout and sequence disagree")
        mask = build_fact_teacher_forcing_mask(
            layout, device=sequence.device, dtype=sequence.dtype
        )
        return self.causal_backbone(sequence, mask=mask), layout

    def _decode(
        self,
        hidden: torch.Tensor,
        layout: FACTTokenLayout,
        *,
        base_action_velocity: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        p_end = layout.prefix_end
        a_end = layout.pred_action_end
        g_end = layout.clean_action_end
        fs_end = layout.future_state_end
        v_end = layout.value_end
        output: dict[str, torch.Tensor] = {"layout": layout}  # type: ignore[dict-item]
        if base_action_velocity is not None:
            residual = self.action_residual_decoder(hidden[:, p_end:a_end])
            output["action"] = base_action_velocity + residual
        if layout.future_state_tokens:
            output["future_state"] = self.future_state_decoder(hidden[:, g_end:fs_end])
        if layout.value_tokens:
            output["value"] = self.value_decoder(hidden[:, fs_end:v_end])
        if layout.future_representation_tokens:
            output["future_representation"] = self.future_representation_decoder(
                hidden[:, v_end:]
            )
        return output

    def forward_action(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stage 1: predict action flow without constructing future tracks."""

        base, pred_hidden = self._carrier_hidden(
            noisy_actions,
            timestep,
            text_context=text_context,
            proprio=proprio,
            video_kv_cache=video_kv_cache,
            text_mask=text_mask,
        )
        prefix, state_tokens, ref_tokens = self._prefix(
            text_context=text_context,
            proprio=proprio,
            video_kv_cache=video_kv_cache,
            text_mask=text_mask,
        )
        hidden, layout = self._run_causal(
            prefix=prefix,
            state_tokens=state_tokens,
            ref_tokens=ref_tokens,
            pred_action_hidden=pred_hidden,
            timestep=timestep,
        )
        return self._decode(hidden, layout, base_action_velocity=base)["action"]

    def forward_consequence(
        self,
        dummy_pred_actions: torch.Tensor,
        clean_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        noisy_future_state: torch.Tensor,
        noisy_value: torch.Tensor,
        noisy_future_representation: torch.Tensor,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        text_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Stage 2: predict consequences conditioned on a clean Stage-1 action."""

        _, pred_hidden = self._carrier_hidden(
            dummy_pred_actions,
            timestep,
            text_context=text_context,
            proprio=proprio,
            video_kv_cache=video_kv_cache,
            text_mask=text_mask,
        )
        clean_timestep = torch.zeros_like(timestep)
        _, clean_hidden = self._carrier_hidden(
            clean_actions,
            clean_timestep,
            text_context=text_context,
            proprio=proprio,
            video_kv_cache=video_kv_cache,
            text_mask=text_mask,
        )
        prefix, state_tokens, ref_tokens = self._prefix(
            text_context=text_context,
            proprio=proprio,
            video_kv_cache=video_kv_cache,
            text_mask=text_mask,
        )
        hidden, layout = self._run_causal(
            prefix=prefix,
            state_tokens=state_tokens,
            ref_tokens=ref_tokens,
            pred_action_hidden=pred_hidden,
            clean_action_hidden=clean_hidden,
            noisy_future_state=noisy_future_state,
            noisy_value=noisy_value,
            noisy_future_representation=noisy_future_representation,
            timestep=timestep,
        )
        result = self._decode(hidden, layout, base_action_velocity=None)
        result.pop("layout", None)
        return result

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        *,
        clean_actions: torch.Tensor,
        noisy_future_state: torch.Tensor,
        noisy_value: torch.Tensor,
        noisy_future_representation: torch.Tensor,
        text_context: torch.Tensor,
        proprio: torch.Tensor,
        video_kv_cache: Mapping[int, Mapping[str, torch.Tensor]],
        text_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Joint teacher-forced Stage-1/Stage-2 training graph."""

        if clean_actions.shape != noisy_actions.shape:
            raise ValueError("clean_actions must match noisy_actions")
        base, pred_hidden = self._carrier_hidden(
            noisy_actions,
            timestep,
            text_context=text_context,
            proprio=proprio,
            video_kv_cache=video_kv_cache,
            text_mask=text_mask,
        )
        _, clean_hidden = self._carrier_hidden(
            clean_actions,
            torch.zeros_like(timestep),
            text_context=text_context,
            proprio=proprio,
            video_kv_cache=video_kv_cache,
            text_mask=text_mask,
        )
        prefix, state_tokens, ref_tokens = self._prefix(
            text_context=text_context,
            proprio=proprio,
            video_kv_cache=video_kv_cache,
            text_mask=text_mask,
        )
        hidden, layout = self._run_causal(
            prefix=prefix,
            state_tokens=state_tokens,
            ref_tokens=ref_tokens,
            pred_action_hidden=pred_hidden,
            clean_action_hidden=clean_hidden,
            noisy_future_state=noisy_future_state,
            noisy_value=noisy_value,
            noisy_future_representation=noisy_future_representation,
            timestep=timestep,
        )
        return self._decode(hidden, layout, base_action_velocity=base)


def fact_backbone_port_losses(
    predictions: Mapping[str, torch.Tensor],
    *,
    action_target: torch.Tensor,
    future_state_target: torch.Tensor,
    value_target: torch.Tensor,
    future_representation_target: torch.Tensor,
    action_is_pad: torch.Tensor | None = None,
    action_loss_mask: torch.Tensor | None = None,
    future_loss_mask: torch.Tensor | None = None,
    future_state_loss_mask: torch.Tensor | None = None,
    value_loss_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Official-relative 10:1:0.4:0.4 flow losses with failure masking.

    ``action_loss_mask`` is one for demonstrator/success episodes and zero for
    failed rollouts.  Future state/value/representation continue to train, as in
    official FACT, only when their causal labels are valid.
    """

    action_error = (predictions["action"].float() - action_target.float()).square()
    if action_is_pad is not None:
        if tuple(action_is_pad.shape) != tuple(action_error.shape[:2]):
            raise ValueError("action_is_pad must be [B,T]")
        valid = (~action_is_pad.bool()).float().unsqueeze(-1)
        per_sample_action = (action_error * valid).sum(dim=(1, 2)) / (
            valid.sum(dim=(1, 2)) * action_error.shape[-1]
        ).clamp_min(1.0)
    else:
        per_sample_action = action_error.flatten(1).mean(dim=1)
    if action_loss_mask is not None:
        mask = action_loss_mask.float().reshape(-1).to(per_sample_action.device)
        action_loss = (per_sample_action * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        action_loss = per_sample_action.mean()

    def masked_track_loss(
        prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        per_sample = (prediction.float() - target.float()).square().flatten(1).mean(1)
        if mask is None:
            return per_sample.mean()
        weights = mask.float().reshape(-1).to(per_sample.device)
        return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)

    future_state_loss = masked_track_loss(
        predictions["future_state"],
        H3FACTBackbonePort._as_track(
            future_state_target.float(),
            width=predictions["future_state"].shape[-1],
            name="future_state_target",
        ),
        future_loss_mask
        if future_state_loss_mask is None
        else future_state_loss_mask,
    )
    value_loss = masked_track_loss(
        predictions["value"],
        H3FACTBackbonePort._as_track(value_target.float(), width=1, name="value_target"),
        value_loss_mask,
    )
    future_representation_loss = masked_track_loss(
        predictions["future_representation"],
        H3FACTBackbonePort._as_track(
            future_representation_target.float(),
            width=predictions["future_representation"].shape[-1],
            name="future_representation_target",
        ),
        future_loss_mask,
    )
    total = (
        FACT_ACTION_WEIGHT * action_loss
        + FACT_FUTURE_REPRESENTATION_WEIGHT * future_representation_loss
        + FACT_FUTURE_STATE_WEIGHT * future_state_loss
        + FACT_VALUE_WEIGHT * value_loss
    )
    return {
        "loss": total,
        "action_loss": action_loss,
        "future_representation_loss": future_representation_loss,
        "future_state_loss": future_state_loss,
        "value_loss": value_loss,
    }


__all__ = [
    "C59FailureOverlay",
    "C59_FAILURE_OVERLAY_FORMAT",
    "C59_VALUE_CONTRACTS",
    "C60CausalFailureLabels",
    "C60_CAUSAL_FAILURE_FORMAT",
    "FACT_ACTION_WEIGHT",
    "FACT_CURRENT_AUDITED_COMMIT",
    "FACT_FUTURE_REPRESENTATION_WEIGHT",
    "FACT_FUTURE_STATE_WEIGHT",
    "FACT_PINNED_COMMIT",
    "FACT_VALUE_WEIGHT",
    "FACTTokenLayout",
    "H3FACTBackbonePort",
    "build_fact_teacher_forcing_mask",
    "fact_backbone_port_losses",
]
