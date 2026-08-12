#!/usr/bin/env python3
"""Rotate a JSONL sampling schedule without changing its row population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--offset", type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite rotated manifest: {output}")
    rows = [line for line in args.input.resolve().read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("input manifest is empty")
    offset = args.offset % len(rows)
    rotated = rows[offset:] + rows[:offset]
    input_ids = [str(json.loads(line)["id"]) for line in rows]
    output_ids = [str(json.loads(line)["id"]) for line in rotated]
    if len(set(input_ids)) != len(input_ids) or set(input_ids) != set(output_ids):
        raise ValueError("rotation did not preserve a unique manifest population")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rotated) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "manifest_rotation_complete",
                "rows": len(rows),
                "offset": offset,
                "first_id": output_ids[0],
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
