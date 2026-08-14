"""Environment isolation helpers for simulator/policy split processes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def standalone_policy_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy *source* and remove the exact simulator wheel path from PYTHONPATH."""

    environment = dict(os.environ if source is None else source)
    simulator_site = environment.get("SIM_SITE_PACKAGES")
    python_path = environment.get("PYTHONPATH")
    if simulator_site and python_path:
        environment["PYTHONPATH"] = os.pathsep.join(
            entry
            for entry in python_path.split(os.pathsep)
            if entry and Path(entry) != Path(simulator_site)
        )
    return environment
