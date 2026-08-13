#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

pattern='wandb_v1_[A-Za-z0-9_-]+|WANDB_API_KEY[[:space:]]*=[[:space:]]*[A-Za-z0-9_-]{24,}|--wandb-key[[:space:]]+[0-9a-f]{32,}'
if git grep -n -E "${pattern}" -- ':!third_party/**'; then
  echo "tracked credential-like literal detected" >&2
  exit 1
fi
echo "tracked credential scan: PASS"
