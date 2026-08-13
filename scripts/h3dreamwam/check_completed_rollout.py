#!/usr/bin/env python3
"""Exit zero only for an atomically completed rollout result."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.h3dreamwam.result_contract import completed_rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    completed_rollout(args.result)


if __name__ == "__main__":
    main()
