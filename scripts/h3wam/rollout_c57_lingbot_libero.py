#!/usr/bin/env python3
"""Run real LIBERO rollouts with C57's persistent feedback lifecycle.

The mature LIBERO benchmark runner remains the source of task/reset/result
semantics.  This entry point only replaces the policy server and wraps
``env.step`` so the server receives real observations at action 4 and action 8.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fastwam.models.h3wam.c57_lingbot_libero_trace import (  # noqa: E402
    C57FeedbackEnv,
    C57PersistentFeedbackClient,
)


def _load_rollout_module():
    path = Path(__file__).with_name("rollout_libero.py")
    spec = importlib.util.spec_from_file_location("_c57_libero_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load LIBERO runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_rollout_module()
_base_policy_command = BASE.policy_command
_base_predict = BASE.predict
_base_run_episode = BASE.run_episode
_active_clients: dict[int, C57PersistentFeedbackClient] = {}


def policy_command(args, port: int, ready_file: Path) -> list[str]:
    if args.policy != "h3_dreamwam_kv_int8":
        raise ValueError("C57 rollout requires --policy h3_dreamwam_kv_int8")
    command = _base_policy_command(args, port, ready_file)
    expected = str(Path(BASE.__file__).with_name("serve_rollout_policy.py"))
    replacement = str(Path(__file__).with_name("serve_c57_lingbot_policy.py"))
    matches = [index for index, value in enumerate(command) if value == expected]
    if matches != [1]:
        raise RuntimeError("could not replace the audited policy server entry point")
    command[1] = replacement
    return command


def predict(connection, *args, **kwargs):
    result = _base_predict(connection, *args, **kwargs)
    try:
        client = _active_clients[id(connection)]
    except KeyError as error:
        raise RuntimeError("C57 predict has no episode feedback client") from error
    client.mark_prediction()
    return result


def run_episode(env, initial_state, task: str, connection, **kwargs):
    if kwargs.get("wait_steps") != 0:
        raise ValueError("C57 feedback audit requires --wait-steps 0")
    if kwargs.get("replan_steps") != 8:
        raise ValueError("C57 feedback audit requires --replan-steps 8")
    if kwargs.get("first_replan_steps") not in (None, 8):
        raise ValueError("C57 forbids a different first replan horizon")
    if kwargs.get("scheduled_long_replan_step") is not None:
        raise ValueError("C57 forbids scheduled long replans")
    if kwargs.get("use_action_ensembler"):
        raise ValueError("C57 lifecycle canary forbids temporal action ensembling")
    episode_key = str(kwargs["episode_key"])
    client = C57PersistentFeedbackClient(
        connection, episode_key=episode_key, task=task
    )
    client.reset()
    key = id(connection)
    if key in _active_clients:
        raise RuntimeError("C57 connection already owns an episode")
    _active_clients[key] = client
    try:
        return _base_run_episode(
            C57FeedbackEnv(env, client),
            initial_state,
            task,
            connection,
            **kwargs,
        )
    finally:
        client.discard_terminal_tail()
        _active_clients.pop(key, None)


def main() -> None:
    BASE.policy_command = policy_command
    BASE.predict = predict
    BASE.run_episode = run_episode
    BASE.main()


if __name__ == "__main__":
    main()
