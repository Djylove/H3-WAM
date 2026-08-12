#!/usr/bin/env python3
"""Train an independent ActionDiT on cached layerwise H3 video tokens."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="+",
        help="One or more manifests; later manifests can add weighted corrective states.",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Fine-tune an existing compatible H3 feature-action checkpoint.",
    )
    parser.add_argument("--feature-subdir", default="h3_video_features")
    parser.add_argument(
        "--preload-cache",
        action="store_true",
        help=(
            "Preload actions/state and H3 features into host RAM once. This avoids "
            "per-step torch.load overhead without duplicating cache files on disk."
        ),
    )
    parser.add_argument(
        "--feature-tail-subdir",
        help=(
            "Optional cache containing a subset of replacement H3 layers. "
            "This avoids duplicating unchanged early-block features when only the tail was adapted."
        ),
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=32,
        help="Number of leading cached actions to train. Use 1 for fully closed-loop BC.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--validation-windows", type=int, default=32)
    parser.add_argument("--val-episodes-per-task", type=int, default=1)
    parser.add_argument("--train-episode", type=int, action="append")
    parser.add_argument(
        "--use-proprio",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Condition on normalized 8-D proprio plus trajectory phase.",
    )
    parser.add_argument(
        "--use-previous-action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Condition on the previous normalized 7-D dataset action.",
    )
    parser.add_argument(
        "--task-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append an explicit one-hot task identity to every action token.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--teacher-targets",
        type=Path,
        help="Optional normalized action predictions exported by a teacher policy.",
    )
    parser.add_argument(
        "--teacher-mix-weight",
        type=float,
        default=1.0,
        help="Blend teacher targets with normalized dataset actions in [0,1].",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--num-action-modes", type=int, default=1)
    parser.add_argument(
        "--mode-labeling",
        choices=("episode_cluster", "event_stage", "task"),
        default="episode_cluster",
        help=(
            "Label mixture branches by demonstrator trajectory clusters or by "
            "four observable manipulation stages derived from gripper events, "
            "or dedicate one hard-routed output expert to each task."
        ),
    )
    parser.add_argument("--mode-loss-weight", type=float, default=0.1)
    parser.add_argument(
        "--gripper-loss-weight",
        type=float,
        default=1.0,
        help="Relative regression weight for the final gripper dimension.",
    )
    parser.add_argument("--objective", choices=("regression", "flow"), default="regression")
    parser.add_argument(
        "--include-phase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append trajectory phase; FastWAM-style flow training disables this.",
    )
    parser.add_argument("--flow-shift", type=float, default=5.0)
    parser.add_argument(
        "--repeated-flow-samples",
        type=int,
        default=1,
        help="Independent flow-noise draws per observation, as used by DiT4DiT.",
    )
    parser.add_argument("--flow-beta-alpha", type=float, default=1.0)
    parser.add_argument("--flow-beta-beta", type=float, default=1.0)
    parser.add_argument("--flow-inference-steps", type=int, default=20)
    return parser.parse_args()


def load_helpers():
    path = Path(__file__).with_name("train_libero_h3_action.py")
    spec = importlib.util.spec_from_file_location("train_libero_h3_action", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def event_stage_boundaries(
    gripper: torch.Tensor, action_horizon: int = 32
) -> tuple[int, int, int]:
    """Return reach/grasp, transport and release-stage starts for a demo."""

    values = gripper.float().reshape(-1)
    if len(values) < 4:
        raise ValueError("event-stage labeling requires at least four actions")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    closed = values < 0.5
    close_events = torch.nonzero((~closed[:-1]) & closed[1:]).flatten() + 1
    if not len(close_events):
        raise ValueError("episode has no open-to-close gripper event")
    grasp_start = int(close_events[0].item())
    open_events = torch.nonzero(closed[:-1] & (~closed[1:])).flatten() + 1
    open_after_grasp = open_events[open_events > grasp_start]
    # Some successful LIBERO demonstrations satisfy the placement predicate
    # while still holding the object.  They have no final close-to-open event;
    # use the episode end as the terminal placement boundary without changing
    # the demonstrated gripper target.
    final_open = (
        int(open_after_grasp[-1].item())
        if len(open_after_grasp)
        else len(values) - 1
    )
    # Label release planning as soon as the predicted chunk contains the final
    # open event. Horizon1 therefore changes stage only at the release itself,
    # while Horizon32 can plan it up to 31 actions earlier.
    release_start = max(
        grasp_start + 18,
        final_open - (action_horizon - 1),
    )
    transport_start = min(grasp_start + 16, release_start - 1)
    transport_start = max(grasp_start + 1, transport_start)
    # Before the first close event, demonstrations first manipulate the drawer
    # and then move toward the bowl. Splitting this interval prevents one mode
    # from averaging two motions with opposite directions.
    reach_start = max(1, round(grasp_start * 0.55))
    return reach_start, transport_start, release_start


def event_stage_for_start(start: int, boundaries: tuple[int, int, int]) -> int:
    return sum(int(start) >= boundary for boundary in boundaries)


def build_event_stage_labels(
    items: list[dict], cache_root: Path, action_horizon: int = 32
) -> tuple[dict[str, int], dict[int, tuple[int, int, int]]]:
    grouped: dict[int, list[dict]] = {}
    for item in items:
        grouped.setdefault(int(item["episode"]), []).append(item)
    labels: dict[str, int] = {}
    episode_boundaries = {}
    for episode, episode_items in grouped.items():
        ordered = sorted(episode_items, key=lambda item: int(item["start"]))
        gripper_by_step: dict[int, float] = {}
        for item in ordered:
            window = torch.load(
                cache_root / "windows" / f"{item['id']}.pt",
                map_location="cpu",
                weights_only=False,
            )
            start = int(item["start"])
            gripper_by_step[start] = float(window["actions"][0, -1].item())
            if item is ordered[-1]:
                for offset, value in enumerate(window["actions"][:, -1]):
                    gripper_by_step[start + offset] = float(value.item())
        gripper = torch.tensor(
            [gripper_by_step[index] for index in range(max(gripper_by_step) + 1)]
        )
        boundaries = event_stage_boundaries(gripper, action_horizon)
        episode_boundaries[episode] = boundaries
        for item in ordered:
            labels[str(item["id"])] = event_stage_for_start(
                int(item["start"]), boundaries
            )
    return labels, episode_boundaries


def main() -> None:
    args = parse_args()
    helpers = load_helpers()
    from fastwam.models.h3wam import (
        H3FeatureActionTransformer,
        H3MixtureActionOutput,
    )
    from fastwam.models.wan22.schedulers.scheduler_continuous import (
        WanContinuousFlowMatchScheduler,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    if args.objective == "flow" and args.num_action_modes != 1:
        raise ValueError("FastWAM-style flow replaces discrete action modes; use one mode")
    if args.mode_labeling == "event_stage" and args.num_action_modes != 4:
        raise ValueError("event-stage labeling requires exactly four action modes")
    if args.gripper_loss_weight <= 0:
        raise ValueError("gripper-loss-weight must be positive")
    if args.repeated_flow_samples <= 0:
        raise ValueError("repeated-flow-samples must be positive")
    if args.flow_beta_alpha <= 0 or args.flow_beta_beta <= 0:
        raise ValueError("flow beta parameters must be positive")
    if not 1 <= args.action_horizon <= 32:
        raise ValueError("action-horizon must be between 1 and the cached horizon 32")
    items = []
    for manifest in args.manifest:
        items.extend(helpers.read_manifest(manifest.resolve()))
    init_checkpoint = None
    if args.init_checkpoint is not None:
        init_checkpoint = torch.load(
            args.init_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
    checkpoint_task_to_index = (
        init_checkpoint.get("task_to_index", {}) if init_checkpoint is not None else {}
    )
    task_to_index = (
        {str(task): int(index) for task, index in checkpoint_task_to_index.items()}
        if args.task_conditioning and checkpoint_task_to_index
        else {
            task: index
            for index, task in enumerate(sorted({str(item["task"]) for item in items}))
        }
    )
    missing_tasks = sorted(
        {str(item["task"]) for item in items} - set(task_to_index)
    )
    if missing_tasks:
        raise ValueError(f"training tasks are absent from init task vocabulary: {missing_tasks}")
    if args.mode_labeling == "task" and args.num_action_modes != len(task_to_index):
        raise ValueError(
            "task labeling requires exactly one action mode per task: "
            f"expected {len(task_to_index)}, got {args.num_action_modes}"
        )
    train_items, validation_items = helpers.split_by_episode(
        items, args.val_episodes_per_task
    )
    if args.train_episode is not None:
        selected = set(args.train_episode)
        train_items = [
            item for item in train_items if int(item["episode"]) in selected
        ]
        if not train_items:
            raise ValueError("train-episode filters removed every training window")
    training_episodes = sorted({int(item["episode"]) for item in train_items})
    if len(validation_items) > args.validation_windows:
        indices = torch.linspace(
            0, len(validation_items) - 1, args.validation_windows
        ).round().long()
        validation_items = [validation_items[int(index)] for index in indices]
    stats = torch.load(args.cache_root / "stats.pt", map_location="cpu", weights_only=False)
    teacher_targets = None
    if args.teacher_targets is not None:
        if not 0.0 <= args.teacher_mix_weight <= 1.0:
            raise ValueError("teacher-mix-weight must be in [0,1]")
        teacher_payload = torch.load(
            args.teacher_targets.resolve(), map_location="cpu", weights_only=False
        )
        if teacher_payload.get("format") != "h3-feature-action-teacher-targets-v1":
            raise ValueError("unsupported teacher target format")
        if int(teacher_payload["action_horizon"]) != args.action_horizon:
            raise ValueError("teacher target action horizon mismatch")
        teacher_targets = teacher_payload["targets"]
        missing_teacher = [str(item["id"]) for item in items if str(item["id"]) not in teacher_targets]
        if missing_teacher:
            raise ValueError(f"teacher targets miss {len(missing_teacher)} manifest items")
    item_modes: dict[str, int] | None = None
    stage_boundaries: dict[int, tuple[int, int, int]] | None = None
    if args.mode_labeling == "task":
        item_modes = {
            str(item["id"]): task_to_index[str(item["task"])] for item in items
        }
        stage_boundaries = None
        episode_modes = {}
    elif args.mode_labeling == "event_stage":
        item_modes, stage_boundaries = build_event_stage_labels(
            items, args.cache_root.resolve(), args.action_horizon
        )
        episode_modes = {}
    elif args.num_action_modes == 1:
        episode_modes = {episode: 0 for episode in training_episodes}
    elif args.num_action_modes >= len(training_episodes):
        episode_modes = {
            episode: index for index, episode in enumerate(training_episodes)
        }
    else:
        # Cluster complete trajectory intentions by their first action chunk.
        # The visual gate then learns the cluster, while each action mode avoids
        # averaging trajectories with substantially different approaches.
        first_items = {
            episode: min(
                (item for item in train_items if int(item["episode"]) == episode),
                key=lambda item: int(item["start"]),
            )
            for episode in training_episodes
        }
        signatures = []
        for episode in training_episodes:
            window = torch.load(
                args.cache_root
                / "windows"
                / f"{first_items[episode]['id']}.pt",
                map_location="cpu",
                weights_only=False,
            )
            signature = helpers.minmax_normalize(
                window["actions"].float(),
                stats["action_min"],
                stats["action_max"],
            )
            signatures.append(signature.flatten())
        signatures = torch.stack(signatures)
        center_indices = [0]
        for _ in range(1, args.num_action_modes):
            distances = torch.cdist(
                signatures, signatures[center_indices]
            ).amin(dim=1)
            center_indices.append(int(distances.argmax().item()))
        centers = signatures[center_indices].clone()
        assignments = torch.zeros(len(signatures), dtype=torch.long)
        for _ in range(50):
            updated = torch.cdist(signatures, centers).argmin(dim=1)
            if torch.equal(updated, assignments):
                break
            assignments = updated
            for mode in range(args.num_action_modes):
                members = signatures[assignments == mode]
                if len(members):
                    centers[mode] = members.mean(dim=0)
        episode_modes = {
            episode: int(assignments[index].item())
            for index, episode in enumerate(training_episodes)
        }
    if item_modes is None:
        mode_counts = {
            str(mode): sum(value == mode for value in episode_modes.values())
            for mode in sorted(set(episode_modes.values()))
        }
    else:
        mode_counts = {
            str(mode): sum(
                item_modes[str(item["id"])] == mode for item in train_items
            )
            for mode in range(args.num_action_modes)
        }
    print(
        json.dumps(
            {
                "mode_labeling": args.mode_labeling,
                "mode_counts": mode_counts,
            }
        ),
        flush=True,
    )
    sampling_weights = [float(item.get("sample_weight", 1.0)) for item in train_items]
    if any(weight <= 0 for weight in sampling_weights):
        raise ValueError("manifest sample_weight values must be positive")
    if item_modes is not None:
        numeric_counts = {int(mode): count for mode, count in mode_counts.items()}
        empty_modes = [mode for mode, count in numeric_counts.items() if count == 0]
        if empty_modes and not (
            args.mode_labeling == "task" and init_checkpoint is not None
        ):
            raise ValueError(f"mode labeling produced empty modes {empty_modes}")
        # Task-balanced manifests already encode inverse task frequency. Dividing
        # again would make the per-task probability depend on dataset size twice.
        if args.mode_labeling != "task":
            sampling_weights = [
                weight / numeric_counts[item_modes[str(item["id"])]]
                for weight, item in zip(sampling_weights, train_items)
            ]
    if all(weight == 1.0 for weight in sampling_weights):
        sampling_weights = None
    phase_items = train_items if args.train_episode is not None else items
    phase_length = int(
        round(statistics.median(int(item["length"]) for item in phase_items))
    )
    phase_lengths_by_task = {
        task: int(
            round(
                statistics.median(
                    int(item["length"])
                    for item in phase_items
                    if str(item["task"]) == task
                )
            )
        )
        for task in sorted({str(item["task"]) for item in phase_items})
    }
    first_feature = torch.load(
        args.cache_root / args.feature_subdir / f"{items[0]['id']}.pt",
        map_location="cpu",
        weights_only=False,
    )
    replacement_positions: dict[int, int] = {}
    if args.feature_tail_subdir is not None:
        first_tail_feature = torch.load(
            args.cache_root / args.feature_tail_subdir / f"{items[0]['id']}.pt",
            map_location="cpu",
            weights_only=False,
        )
        base_layers = tuple(int(layer) for layer in first_feature["layers"])
        tail_layers = tuple(int(layer) for layer in first_tail_feature["layers"])
        if not set(tail_layers).issubset(base_layers):
            raise ValueError("feature-tail layers must be a subset of feature-subdir layers")
        if tuple(first_tail_feature["features"].shape[1:]) != tuple(
            first_feature["features"].shape[1:]
        ):
            raise ValueError("feature-tail token/hidden shape differs from the base cache")
        replacement_positions = {
            base_layers.index(layer): tail_layers.index(layer) for layer in tail_layers
        }
    feature_shape = tuple(first_feature["features"].shape)
    preloaded_windows: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
    preloaded_features: dict[str, torch.Tensor] | None = None
    preloaded_tail_features: dict[str, torch.Tensor] | None = None
    if args.preload_cache:
        preload_started = time.perf_counter()
        preloaded_windows = {}
        preloaded_features = {}
        preloaded_tail_features = {} if args.feature_tail_subdir is not None else None
        for index, item in enumerate(items, start=1):
            item_id = str(item["id"])
            window = torch.load(
                args.cache_root / "windows" / f"{item_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            preloaded_windows[item_id] = (
                window["actions"].float(),
                window["state"].float(),
            )
            preloaded_features[item_id] = torch.load(
                args.cache_root / args.feature_subdir / f"{item_id}.pt",
                map_location="cpu",
                weights_only=False,
            )["features"]
            if preloaded_tail_features is not None:
                assert args.feature_tail_subdir is not None
                preloaded_tail_features[item_id] = torch.load(
                    args.cache_root / args.feature_tail_subdir / f"{item_id}.pt",
                    map_location="cpu",
                    weights_only=False,
                )["features"]
            if index % 1000 == 0 or index == len(items):
                print(
                    json.dumps(
                        {
                            "event": "preload",
                            "complete": index,
                            "total": len(items),
                            "elapsed_seconds": round(
                                time.perf_counter() - preload_started, 2
                            ),
                        }
                    ),
                    flush=True,
                )
    state_dim = (
        (8 if args.use_proprio else 0)
        + (7 if args.use_previous_action else 0)
        + int(args.include_phase)
        + (len(task_to_index) if args.task_conditioning else 0)
    )
    if state_dim == 0:
        state_dim = 1
    model = H3FeatureActionTransformer(
        action_dim=7,
        state_dim=state_dim,
        h3_feature_dim=feature_shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        num_action_modes=args.num_action_modes,
    ).to(device)
    if args.init_checkpoint is not None:
        assert init_checkpoint is not None
        if init_checkpoint.get("policy_type") != "h3_feature_action":
            raise ValueError("init-checkpoint is not an H3 feature-action policy")
        expected_contract = {
            "action_horizon": args.action_horizon,
            "feature_shape": feature_shape,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "ffn_dim": args.ffn_dim,
            "num_action_modes": args.num_action_modes,
            "state_dim": state_dim,
        }
        for key, expected in expected_contract.items():
            actual = init_checkpoint.get(key)
            if key == "feature_shape" and actual is not None:
                actual = tuple(actual)
            if actual != expected:
                raise ValueError(
                    f"init-checkpoint {key} mismatch: expected {expected}, got {actual}"
                )
        model.load_state_dict(init_checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    flow_scheduler = WanContinuousFlowMatchScheduler(shift=args.flow_shift)
    action_dimension_weights = torch.ones(7, device=device)
    action_dimension_weights[-1] = args.gripper_loss_weight

    def per_sample_action_mse(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        per_dimension = (prediction - target).square()
        weighted = (
            per_dimension * action_dimension_weights.reshape(1, 1, -1)
        ).sum(dim=-1) / action_dimension_weights.sum()
        return weighted.mean(dim=-1)

    def load_batch(batch_items: list[dict]):
        actions, states, features, modes = [], [], [], []
        for item in batch_items:
            item_id = str(item["id"])
            if preloaded_windows is None:
                window = torch.load(
                    args.cache_root / "windows" / f"{item_id}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                window_actions = window["actions"].float()
                window_state = window["state"].float()
            else:
                window_actions, window_state = preloaded_windows[item_id]
            feature = (
                torch.load(
                    args.cache_root / args.feature_subdir / f"{item_id}.pt",
                    map_location="cpu",
                    weights_only=False,
                )["features"]
                if preloaded_features is None
                else preloaded_features[item_id]
            )
            if args.feature_tail_subdir is not None:
                tail_feature = (
                    torch.load(
                        args.cache_root
                        / args.feature_tail_subdir
                        / f"{item_id}.pt",
                        map_location="cpu",
                        weights_only=False,
                    )["features"]
                    if preloaded_tail_features is None
                    else preloaded_tail_features[item_id]
                )
                feature = feature.clone()
                for base_position, tail_position in replacement_positions.items():
                    feature[base_position] = tail_feature[tail_position]
            normalized_actions = helpers.minmax_normalize(
                window_actions, stats["action_min"], stats["action_max"]
            )[: args.action_horizon]
            if teacher_targets is not None:
                teacher = teacher_targets[str(item["id"])].float()
                if tuple(teacher.shape) != tuple(normalized_actions.shape):
                    raise ValueError(f"teacher target shape mismatch for {item['id']}")
                normalized_actions = torch.lerp(
                    normalized_actions, teacher, args.teacher_mix_weight
                )
            state_parts = []
            if args.use_proprio:
                proprio = helpers.minmax_normalize(
                    window_state, stats["state_min"], stats["state_max"]
                )
                state_parts.append(
                    proprio.reshape(1, 8).expand(
                        normalized_actions.shape[0], -1
                    ).clone()
                )
            if args.use_previous_action:
                start = int(item["start"])
                if start == 0:
                    previous_action = torch.tensor(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                    )
                else:
                    previous_id = f"ep{int(item['episode']):06d}_s{start - 1:06d}"
                    if preloaded_windows is None:
                        previous_window = torch.load(
                            args.cache_root / "windows" / f"{previous_id}.pt",
                            map_location="cpu",
                            weights_only=False,
                        )
                        previous_action = previous_window["actions"][0].float()
                    else:
                        previous_action = preloaded_windows[previous_id][0][0]
                normalized_previous = helpers.minmax_normalize(
                    previous_action, stats["action_min"], stats["action_max"]
                )
                state_parts.append(
                    normalized_previous.reshape(1, 7).expand(
                        normalized_actions.shape[0], -1
                    ).clone()
                )
            if args.include_phase:
                phase = torch.zeros(normalized_actions.shape[0], 1)
                phase_steps = torch.arange(normalized_actions.shape[0], dtype=torch.float32)
                phase_steps.add_(int(item["start"])).clamp_max_(int(item["length"]) - 1)
                phase[:, 0] = (
                    2.0 * phase_steps / max(int(item["length"]) - 1, 1) - 1.0
                )
                state_parts.append(phase)
            if args.task_conditioning:
                task_one_hot = torch.zeros(
                    normalized_actions.shape[0], len(task_to_index)
                )
                task_one_hot[:, task_to_index[str(item["task"])]] = 1.0
                state_parts.append(task_one_hot)
            if state_parts:
                state = torch.cat(state_parts, dim=-1)
            else:
                state = torch.zeros(
                    normalized_actions.shape[0], 1, dtype=torch.float32
                )
            actions.append(normalized_actions)
            states.append(state)
            features.append(feature)
            if item_modes is None:
                modes.append(episode_modes.get(int(item["episode"]), -1))
            else:
                modes.append(item_modes[str(item["id"])])
        return (
            torch.stack(actions).to(device),
            torch.stack(states).to(device),
            torch.stack(features).to(device),
            torch.tensor(modes, device=device, dtype=torch.long),
        )

    @torch.inference_mode()
    def validation_loss() -> float:
        model.eval()
        losses = []
        for offset in range(0, len(validation_items), args.batch_size):
            action, state, feature, target_mode = load_batch(
                validation_items[offset : offset + args.batch_size]
            )
            if args.objective == "flow":
                generator = torch.Generator(device=device).manual_seed(args.seed + offset)
                noise = torch.randn(
                    action.shape, device=device, dtype=action.dtype, generator=generator
                )
                # Cover the shifted flow-time distribution deterministically so
                # checkpoint selection is comparable across validation passes.
                sample_indices = torch.arange(
                    offset,
                    offset + action.shape[0],
                    device=device,
                    dtype=torch.float32,
                )
                u = (sample_indices + 0.5) / max(len(validation_items), 1)
                timestep = flow_scheduler._phi(u, args.flow_shift).to(
                    dtype=action.dtype
                ) * float(flow_scheduler.num_train_timesteps)
                model_input = flow_scheduler.add_noise(action, noise, timestep)
                target = flow_scheduler.training_target(action, noise, timestep)
            else:
                model_input = torch.zeros_like(action)
                timestep = torch.zeros(action.shape[0], device=device)
                target = action
            output = model(
                model_input,
                state=state,
                h3_features=feature,
                video_sigma=timestep,
            )
            if isinstance(output, H3MixtureActionOutput):
                selected = (
                    target_mode
                    if args.mode_labeling == "task"
                    else output.mode_logits.argmax(dim=-1)
                )
                prediction = output.actions[
                    torch.arange(action.shape[0], device=device), selected
                ]
            else:
                prediction = output
            per_sample = per_sample_action_mse(prediction, target)
            if args.objective == "flow":
                per_sample = per_sample * flow_scheduler.training_weight(timestep)
            losses.append(float(per_sample.mean().item()))
        model.train()
        return sum(losses) / len(losses)

    def checkpoint(step: int, value: float, best_value: float | None = None) -> dict:
        return {
            "model": model.state_dict(),
            "normalization": stats,
            "policy_type": "h3_feature_action",
            "objective": args.objective,
            "phase_only": not args.use_proprio and args.include_phase,
            "phase_sequence": args.include_phase,
            "include_phase": args.include_phase,
            "use_proprio": args.use_proprio,
            "use_previous_action": args.use_previous_action,
            "task_conditioning": args.task_conditioning,
            "task_to_index": task_to_index if args.task_conditioning else {},
            "phase_length": phase_length,
            "phase_lengths_by_task": phase_lengths_by_task,
            "state_dim": state_dim,
            "action_dim": 7,
            "action_horizon": args.action_horizon,
            "training_tasks": sorted({str(item["task"]) for item in items}),
            "feature_shape": feature_shape,
            "feature_layers": tuple(first_feature["layers"]),
            "feature_subdir": args.feature_subdir,
            "feature_tail_subdir": args.feature_tail_subdir,
            "feature_timestep": float(first_feature.get("timestep", 1000.0)),
            "feature_context_id": first_feature.get("context_id"),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "ffn_dim": args.ffn_dim,
            "num_action_modes": args.num_action_modes,
            "action_mode_labeling": args.mode_labeling,
            "gripper_loss_weight": args.gripper_loss_weight,
            "episode_modes": episode_modes,
            "event_stage_boundaries": stage_boundaries,
            "flow_shift": args.flow_shift,
            "repeated_flow_samples": args.repeated_flow_samples,
            "flow_beta_alpha": args.flow_beta_alpha,
            "flow_beta_beta": args.flow_beta_beta,
            "flow_inference_steps": args.flow_inference_steps,
            "step": step,
            "validation_loss": value,
            "best_validation_loss": value if best_value is None else best_value,
            "init_checkpoint": (
                None
                if args.init_checkpoint is None
                else str(args.init_checkpoint.resolve())
            ),
            "teacher_targets": (
                None if args.teacher_targets is None else str(args.teacher_targets.resolve())
            ),
            "teacher_mix_weight": (
                None if args.teacher_targets is None else args.teacher_mix_weight
            ),
        }

    started = time.perf_counter()
    best_validation = float("inf")
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for step in range(1, args.steps + 1):
        batch = random.choices(
            train_items, weights=sampling_weights, k=args.batch_size
        )
        action, state, feature, target_mode = load_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        if args.objective == "flow":
            if args.repeated_flow_samples > 1:
                repeats = args.repeated_flow_samples
                action = action.repeat_interleave(repeats, dim=0)
                state = state.repeat_interleave(repeats, dim=0)
                feature = feature.repeat_interleave(repeats, dim=0)
                target_mode = target_mode.repeat_interleave(repeats, dim=0)
            noise = torch.randn_like(action)
            if args.flow_beta_alpha == 1.0 and args.flow_beta_beta == 1.0:
                timestep = flow_scheduler.sample_training_t(
                    action.shape[0], device=device, dtype=action.dtype
                )
            else:
                base = torch.distributions.Beta(
                    args.flow_beta_alpha, args.flow_beta_beta
                ).sample((action.shape[0],)).to(device=device, dtype=action.dtype)
                timestep = flow_scheduler._phi(base, args.flow_shift) * float(
                    flow_scheduler.num_train_timesteps
                )
            model_input = flow_scheduler.add_noise(action, noise, timestep)
            target = flow_scheduler.training_target(action, noise, timestep)
        else:
            model_input = torch.zeros_like(action)
            timestep = torch.zeros(action.shape[0], device=device)
            target = action
        output = model(
            model_input,
            state=state,
            h3_features=feature,
            video_sigma=timestep,
        )
        if isinstance(output, H3MixtureActionOutput):
            prediction = output.actions[
                torch.arange(action.shape[0], device=device), target_mode
            ]
            action_loss = per_sample_action_mse(prediction, target).mean()
            mode_loss = torch.nn.functional.cross_entropy(
                output.mode_logits, target_mode
            )
            loss = action_loss + args.mode_loss_weight * mode_loss
            mode_accuracy = float(
                (output.mode_logits.argmax(dim=-1) == target_mode)
                .float()
                .mean()
                .item()
            )
        else:
            prediction = output
            per_sample = per_sample_action_mse(prediction, target)
            if args.objective == "flow":
                per_sample = per_sample * flow_scheduler.training_weight(timestep)
            action_loss = per_sample.mean()
            mode_loss = torch.zeros((), device=device)
            loss = action_loss
            mode_accuracy = 1.0
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.validation_every == 0 or step == args.steps:
            value = validation_loss()
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "step": step,
                        "train_loss": float(loss.detach().item()),
                        "action_loss": float(action_loss.detach().item()),
                        "mode_loss": float(mode_loss.detach().item()),
                        "mode_accuracy": mode_accuracy,
                        "validation_loss": value,
                        "gradient_norm": float(gradient_norm),
                        "elapsed_seconds": elapsed,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    }
                ),
                flush=True,
            )
            if value < best_validation:
                best_validation = value
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    checkpoint(step, value, best_validation),
                    args.output.with_name(args.output.stem + "_best" + args.output.suffix),
                )
        if step % args.checkpoint_every == 0 and step != args.steps:
            torch.save(
                checkpoint(
                    step,
                    value if "value" in locals() else float("nan"),
                    best_validation,
                ),
                args.output.with_name(
                    args.output.stem + f"_step{step:05d}" + args.output.suffix
                ),
            )
    # ``step == args.steps`` always runs validation above. Keep the actual final
    # loss distinct from the best loss so downstream checkpoint comparisons do
    # not mistake an overfit final model for the selected best model.
    torch.save(checkpoint(args.steps, value, best_validation), args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "train_windows": len(train_items),
                "validation_windows": len(validation_items),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "best_validation_loss": best_validation,
                "total_seconds": time.perf_counter() - started,
            }
        )
    )


if __name__ == "__main__":
    main()
