#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
BASE="${H3_WORKSPACE}/outputs/h3dotwam-dense/m13_dense_full_head_gb128_s1569"
EVAL="${PROJECT_ROOT}/scripts/h3dreamwam/eval_h3dotwam_dense_head_checkpoint.sh"

wait_gpu_idle() {
  until [[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) -eq 0 ]]; do
    sleep 30
  done
}

wait_checkpoint_ready() {
  local checkpoint="$1"
  local size_before size_after
  while true; do
    if [[ ! -s "${checkpoint}" ]]; then
      sleep 30
      continue
    fi
    size_before=$(stat -c %s "${checkpoint}")
    sleep 15
    size_after=$(stat -c %s "${checkpoint}")
    if [[ "${size_before}" != "${size_after}" ]]; then
      continue
    fi
    if "${H3_WORKSPACE}/runtime/conda-py311/bin/python" - "${checkpoint}" <<'PY'
import sys
import torch

stage = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert isinstance(stage, dict)
assert stage.get("format") == "h3dotwam_stage_v2"
assert isinstance(stage.get("action_head"), dict) and stage["action_head"]
assert isinstance(stage.get("kv_fusion"), dict) and stage["kv_fusion"]
assert isinstance(stage.get("state_embedding"), dict) and stage["state_embedding"]
PY
    then
      return 0
    fi
    sleep 30
  done
}

for step in 200 400 800 1200; do
  checkpoint="${BASE}_step$(printf '%06d' "${step}").pt"
  wait_checkpoint_ready "${checkpoint}"
  wait_gpu_idle
  bash "${EVAL}" "${checkpoint}" "m13_step$(printf '%04d' "${step}")"
done
