"""MiniMax H3 adapters for the H3-WAM feasibility study."""

from .action_adapter import H3ActionAdapter
from .baseline import SmallActionFlowTransformer
from .bridge import H3ActionBridge, H3ActionBridgeOutput, make_first_frame_payload
from .comfy_compat import enable_comfy_h3_autograd
from .scheduler import H3ActionFlowScheduler
from .lora import (
    DEFAULT_OFFICIAL_H3_LORA_TARGETS,
    H3LoRALinear,
    H3LoRAReport,
    h3_lora_parameters,
    h3_lora_disabled,
    h3_lora_state_dict,
    inject_h3_attention_lora,
    inject_official_h3_lora,
    load_h3_lora_state_dict,
    set_h3_lora_enabled,
)
from .inference import H3WAMSample, sample_h3wam_actions
from .deployment import (
    action_denormalization_bounds,
    ActionEnsembler,
    libero_dataset_action,
    libero_dataset_actions,
    libero_environment_actions,
    libero_observation_state,
    load_cached_task_context,
    minmax_denormalize,
    minmax_normalize,
    normalize_libero_environment_action_history,
    preprocess_libero_cameras,
    quaternion_to_axis_angle,
)
from .feature_action import (
    H3BlockFeatureCapture,
    H3FeatureActionTransformer,
    H3FeatureSwitchGate,
    H3MultiLayerActionTransformer,
    H3MixtureActionOutput,
)
from .training import (
    H3WAMFlowBatch,
    H3WAMLoss,
    h3wam_action_training_step,
    h3wam_joint_training_step,
    prepare_h3wam_flow_batch,
)
from .temporal import (
    H3WindowPlan,
    align_h3_frame_count,
    h3_video_latent_frames,
    plan_h3_window,
    resample_video_nearest,
    h3_latent_is_pad,
)
from .tail_overlay import (
    H3DenseDeltaLinear,
    H3TailDeltaReport,
    load_h3_comfy_feature_delta,
)
from .official_joint import (
    H3BlockAttentionMask,
    H3OfficialFeatureCapture,
    build_h3_observation_attention_mask,
)
__all__ = [
    "ActionEnsembler",
    "action_denormalization_bounds",
    "H3ActionAdapter",
    "H3ActionBridge",
    "H3ActionBridgeOutput",
    "H3ActionFlowScheduler",
    "H3LoRALinear",
    "H3LoRAReport",
    "H3WAMFlowBatch",
    "H3WAMLoss",
    "H3WAMSample",
    "H3WindowPlan",
    "SmallActionFlowTransformer",
    "align_h3_frame_count",
    "enable_comfy_h3_autograd",
    "h3wam_action_training_step",
    "h3wam_joint_training_step",
    "h3_video_latent_frames",
    "h3_lora_parameters",
    "h3_lora_disabled",
    "h3_lora_state_dict",
    "load_h3_lora_state_dict",
    "set_h3_lora_enabled",
    "make_first_frame_payload",
    "inject_h3_attention_lora",
    "libero_dataset_action",
    "libero_dataset_actions",
    "libero_environment_actions",
    "libero_observation_state",
    "H3BlockFeatureCapture",
    "H3FeatureActionTransformer",
    "H3FeatureSwitchGate",
    "H3MultiLayerActionTransformer",
    "H3MixtureActionOutput",
    "H3DenseDeltaLinear",
    "H3TailDeltaReport",
    "H3BlockAttentionMask",
    "H3OfficialFeatureCapture",
    "build_h3_observation_attention_mask",
    "DEFAULT_OFFICIAL_H3_LORA_TARGETS",
    "inject_official_h3_lora",
    "load_cached_task_context",
    "load_h3_comfy_feature_delta",
    "minmax_denormalize",
    "minmax_normalize",
    "normalize_libero_environment_action_history",
    "plan_h3_window",
    "prepare_h3wam_flow_batch",
    "preprocess_libero_cameras",
    "quaternion_to_axis_angle",
    "resample_video_nearest",
    "h3_latent_is_pad",
    "sample_h3wam_actions",
]
