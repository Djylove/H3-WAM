#!/usr/bin/env python3
"""Bootstrap the official FastWAM evaluator with the local LIBERO install.

The FastWAM and LIBERO environments intentionally use different dependency
sets on this workstation.  Run this file with the FastWAM Python interpreter;
it appends only the LIBERO environment after FastWAM's own site-packages so the
correct Torch build remains authoritative.
"""

from __future__ import annotations

import argparse
import os
import runpy
import site
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--libero-site-packages", type=Path, required=True)
    parser.add_argument("--model-base-path", type=Path, required=True)
    args, hydra_overrides = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parents[2]
    libero_root = args.libero_root.resolve()
    libero_site = args.libero_site_packages.resolve()
    if not (libero_root / "libero").is_dir():
        raise FileNotFoundError(f"LIBERO package not found below {libero_root}")
    if not libero_site.is_dir():
        raise FileNotFoundError(f"LIBERO site-packages not found: {libero_site}")

    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(1, str(libero_root))
    site.addsitedir(str(libero_site))
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(args.model_base_path.resolve())

    evaluator = repo_root / "experiments" / "libero" / "eval_libero_single.py"
    # Running through runpy does not add the target script's directory to
    # sys.path like ``python path/to/script.py`` does.  The official evaluator
    # imports its sibling ``action_ensembler.py`` as a top-level module.
    sys.path.insert(0, str(evaluator.parent))
    sys.argv = [str(evaluator), *hydra_overrides]
    runpy.run_path(str(evaluator), run_name="__main__")


if __name__ == "__main__":
    main()
