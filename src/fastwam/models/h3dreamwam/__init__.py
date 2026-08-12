"""DreamWAM-style world/action adaptation for the official MiniMax-H3 backbone."""

from .projection import (
    H3RGBFlowProjectionReport,
    expand_h3_rgb_flow_projections,
)
from .action_expert import (
    H3DreamActionBlock,
    H3DreamActionExpert,
    load_action_block_state,
)
from .joint_attention import (
    align_h3_action_chunk_ids,
    build_lingbot_block_causal_mask,
    four_stream_h3_action_layer,
    h3_action_temporal_positions,
    lingbot_four_stream_attention,
    paired_h3_action_layer,
    shared_h3_four_stream_layer,
)
from .model import (
    H3DreamPairedLayer,
    H3DreamWAM,
    H3DreamWAMOutput,
    apply_h3_rotary,
)
from .four_stream_model import (
    H3LingBotPairedLayer,
    H3LingBotWAM,
    H3LingBotWAMOutput,
)
from .shared_four_stream_model import (
    H3LingBotActionAdapters,
    H3LingBotSharedLayer,
    H3LingBotSharedOutput,
    H3LingBotSharedWAM,
)
from .docking import (
    H3DoTActionHead,
    H3DoTActionLayer,
    H3DoTKVFusion,
    action_rope_at_positions,
    inverse_h3_rotary,
)
from .dot_model import H3DoTHubLayer, H3DoTWAM, H3DoTWAMOutput
from .initialization import (
    H3ActionInitializationReport,
    initialize_action_expert_from_h3,
    resize_tensor,
)
from .sampling import (
    H3DreamInferenceSchedule,
    H3DreamJointSample,
    H3LingBotCausalSample,
    build_h3dream_inference_schedule,
    h3dream_flow_training_weight,
    sample_h3_lingbot_chunk_causal,
    sample_h3dream_joint_rows,
)

__all__ = [
    "H3RGBFlowProjectionReport",
    "H3DreamActionBlock",
    "H3DreamActionExpert",
    "H3DreamPairedLayer",
    "H3DreamWAM",
    "H3DreamWAMOutput",
    "H3DoTActionHead",
    "H3DoTActionLayer",
    "H3DoTKVFusion",
    "H3DoTHubLayer",
    "H3DoTWAM",
    "H3DoTWAMOutput",
    "H3LingBotPairedLayer",
    "H3LingBotWAM",
    "H3LingBotWAMOutput",
    "H3LingBotActionAdapters",
    "H3LingBotSharedLayer",
    "H3LingBotSharedOutput",
    "H3LingBotSharedWAM",
    "H3ActionInitializationReport",
    "H3DreamInferenceSchedule",
    "H3DreamJointSample",
    "H3LingBotCausalSample",
    "apply_h3_rotary",
    "action_rope_at_positions",
    "align_h3_action_chunk_ids",
    "build_h3dream_inference_schedule",
    "build_lingbot_block_causal_mask",
    "four_stream_h3_action_layer",
    "h3_action_temporal_positions",
    "lingbot_four_stream_attention",
    "shared_h3_four_stream_layer",
    "expand_h3_rgb_flow_projections",
    "initialize_action_expert_from_h3",
    "inverse_h3_rotary",
    "load_action_block_state",
    "h3dream_flow_training_weight",
    "paired_h3_action_layer",
    "resize_tensor",
    "sample_h3dream_joint_rows",
    "sample_h3_lingbot_chunk_causal",
]
