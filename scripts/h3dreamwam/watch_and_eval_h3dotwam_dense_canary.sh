#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project}"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3dotwam-dense"
EVAL="${PROJECT_ROOT}/scripts/h3dreamwam/eval_h3dotwam_dense_canary_checkpoint.sh"
BASELINE="${H3_WORKSPACE}/outputs/h3dotwam/m0v2_h32_gb128_s150_step000125.pt"
STEP40="${OUTPUT_ROOT}/m10_dense_canary_head_gb128_s80_step000040.pt"
FINAL="${OUTPUT_ROOT}/m10_dense_canary_head_gb128_s80.pt"

until [[ -s "${H3_WORKSPACE}/data/v7_dense_canary_candidate/candidate_report.json" ]]; do
  sleep 15
done
bash "${EVAL}" "${BASELINE}" baseline_sparse_head
until [[ -s "${STEP40}" ]]; do sleep 15; done
bash "${EVAL}" "${STEP40}" dense_step40
until [[ -s "${FINAL}" ]]; do sleep 15; done
bash "${EVAL}" "${FINAL}" dense_step80
