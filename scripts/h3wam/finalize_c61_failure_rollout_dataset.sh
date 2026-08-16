#!/usr/bin/env bash
set -euo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
source_root="${C61_SOURCE_ROOT:-${workspace}/eval/c61-failure-rollout-expansion-v1}"
output_root="${C61_FINAL_ROOT:-${workspace}/eval/c61-finalized-fact-failure-dataset-v1}"
python="${workspace}/runtime/h3-int8-native/bin/python"

export PYTHONPATH="${project}/src:${project}"
exec "${python}" "${project}/scripts/h3wam/finalize_c61_failure_rollout_dataset.py" \
  --c61-root "${source_root}" \
  --c48-dataset "${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt" \
  --c48-observations "${workspace}/eval/c48-fact-dense-value-dataset-v1/observations.jsonl" \
  --output-root "${output_root}" \
  --expected-frozen-sha256 17f39b26cd033171bc9cc0afad819ea6b2794f24d1e03d4218c988969a76f6b4 \
  --expected-jobs-sha256 6e2d76e60cdf2a514cb5378c9b0cccf56a97df7b7698b74d1729a28dfa487666 \
  --num-nodes 1
