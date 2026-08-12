#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
BASE="${H3_WORKSPACE}/outputs/h3dotwam-frameindexed/m11_frameindexed_head_gb128_s2170"
EVAL="${PROJECT_ROOT}/scripts/h3dreamwam/eval_h3dotwam_frameindexed_checkpoint.sh"

wait_gpu_idle() {
  until [[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) -eq 0 ]]; do
    sleep 30
  done
}

for step in 200 400 800 1600; do
  checkpoint="${BASE}_step$(printf '%06d' "${step}").pt"
  until [[ -s "${checkpoint}" ]]; do sleep 30; done
  wait_gpu_idle
  bash "${EVAL}" "${checkpoint}" "step$(printf '%04d' "${step}")"
done
until [[ -s "${BASE}.pt" ]]; do sleep 30; done
wait_gpu_idle
bash "${EVAL}" "${BASE}.pt" final2170
