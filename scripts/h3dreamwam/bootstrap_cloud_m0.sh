#!/usr/bin/env bash
set -Eeuo pipefail

H3_ROOT=${H3_ROOT:-/mnt/h3-wam}
PROJECT=${H3_ROOT}/project
PYTHON=${H3_ROOT}/.venv/bin/python
MODEL=${H3_ROOT}/models/MiniMax-H3
EXTRACTED=${H3_ROOT}/data/libero_fastwam_extracted
BASE=${H3_ROOT}/data/v4_multisuite_base
CACHE=${H3_ROOT}/data/v2_full_cache
CANDIDATE=${H3_ROOT}/data/v4_multisuite_uniform_candidate
STAGING=${CANDIDATE}.staging
OUTPUTS=${H3_ROOT}/outputs/h3dotwam
LOGS=${H3_ROOT}/logs/pipeline
RUN=m0v2_h32_gb128_s150

mkdir -p "${BASE}" "${CACHE}/contexts" "${CACHE}/windows" "${OUTPUTS}" "${LOGS}"
export HOME=${H3_ROOT}
export XDG_CACHE_HOME=${H3_ROOT}/cache
export HF_HOME=${H3_ROOT}/cache/huggingface
export TMPDIR=${H3_ROOT}/tmp
export PYTHONPATH=${PROJECT}/src:${PROJECT}

wait_download() {
  local label=$1
  local pid_file=$2
  local log_file=$3
  local pid
  pid=$(<"${pid_file}")
  echo "WAIT ${label} pid=${pid}"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 30
  done
  if ! grep -q '"event": "complete"' "${log_file}"; then
    echo "FAILED ${label}: completion record missing" >&2
    tail -n 40 "${log_file}" >&2
    return 1
  fi
  echo "READY ${label}"
}

wait_download \
  libero \
  "${H3_ROOT}/logs/download/libero_fastwam_segmented.pid" \
  "${H3_ROOT}/logs/download/libero_fastwam_segmented.log"
wait_download \
  vae \
  "${H3_ROOT}/logs/download/h3_vae.pid" \
  "${H3_ROOT}/logs/download/h3_vae.log"

if [[ ! -s "${BASE}/candidate_report.json" ]]; then
  "${PYTHON}" "${PROJECT}/scripts/h3wam/prepare_libero_full_candidate.py" \
    --dataset "libero_10=${EXTRACTED}/libero_10_no_noops_lerobot" \
    --dataset "libero_goal=${EXTRACTED}/libero_goal_no_noops_lerobot" \
    --dataset "libero_object=${EXTRACTED}/libero_object_no_noops_lerobot" \
    --dataset "libero_spatial=${EXTRACTED}/libero_spatial_no_noops_lerobot" \
    --output-dir "${BASE}"
fi

"${PYTHON}" - "${BASE}/candidate_report.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = {"tasks": 40, "episodes": 1712, "windows": 8560,
            "train_windows": 7710, "validation_windows": 850}
actual = {key: report.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"candidate contract mismatch: {actual} != {expected}")
print("CANDIDATE_OK", actual)
PY

cp -n "${H3_ROOT}"/data/v2_task_contexts_nvfp4/*.pt "${CACHE}/contexts/"
"${PYTHON}" - "${BASE}/task_contexts.json" "${CACHE}/contexts" <<'PY'
import json
import pathlib
import sys

expected = set(json.loads(pathlib.Path(sys.argv[1]).read_text()))
available = {path.stem for path in pathlib.Path(sys.argv[2]).glob("*.pt")}
if expected != available:
    raise SystemExit(
        f"context contract mismatch: missing={sorted(expected-available)}, "
        f"extra={sorted(available-expected)}"
    )
print(f"CONTEXTS_OK count={len(expected)}")
PY

expected_windows=$(wc -l < "${BASE}/manifest_all.jsonl")
available_windows=$(find "${CACHE}/windows" -maxdepth 1 -type f -name '*.pt' 2>/dev/null | wc -l)
if [[ "${available_windows}" -ne "${expected_windows}" ]]; then
  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "${PROJECT}/scripts/h3wam/precompute_libero_official_h3.py" \
    vae "${BASE}/manifest_all.jsonl" \
    --cache-root "${CACHE}" --model "${MODEL}" \
    --world-size 8 --progress-every 50
fi

available_windows=$(find "${CACHE}/windows" -maxdepth 1 -type f -name '*.pt' | wc -l)
if [[ "${available_windows}" -ne "${expected_windows}" ]]; then
  echo "VAE cache incomplete: ${available_windows}/${expected_windows}" >&2
  exit 1
fi
echo "VAE_CACHE_OK windows=${available_windows}"

"${PYTHON}" "${PROJECT}/scripts/h3wam/precompute_libero_official_h3.py" \
  stats "${BASE}/manifest_train.jsonl" --cache-root "${CACHE}"

if [[ ! -s "${CANDIDATE}/candidate_report.json" ]]; then
  "${PYTHON}" "${PROJECT}/scripts/h3dreamwam/prepare_multisuite_training_candidate.py" \
    --base-candidate "${BASE}" --cache-root "${CACHE}" \
    --output-dir "${STAGING}" --target-total-repeats 1
  mv "${STAGING}" "${CANDIDATE}"
fi
if [[ ! -s "${CANDIDATE}/manifest_val_stratified40.jsonl" ]]; then
  "${PYTHON}" "${PROJECT}/scripts/h3dreamwam/build_stratified_eval_manifest.py" \
    "${CANDIDATE}/manifest_val.jsonl" \
    "${CANDIDATE}/manifest_val_stratified40.jsonl" --per-task 1
fi

wait_download \
  transformer \
  "${H3_ROOT}/logs/download/h3_transformer.pid" \
  "${H3_ROOT}/logs/download/h3_transformer.log"

if [[ ! -s "${OUTPUTS}/${RUN}.json" || ! -s "${OUTPUTS}/${RUN}.pt" ]]; then
  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=8 \
    "${PROJECT}/scripts/h3dreamwam/train_h3dotwam_fsdp.py" \
    --model "${MODEL}" --data-root "${CACHE}" \
    --manifest "${CANDIDATE}/manifest_train_uniform.jsonl" \
    --output "${OUTPUTS}/${RUN}.json" \
    --save-stage "${OUTPUTS}/${RUN}.pt" \
    --steps 150 --gradient-accumulation-steps 16 \
    --checkpoint-every 25 --lr-schedule cosine \
    --action-horizon 32 --learning-rate 1e-4 --last-h3-blocks 0 \
    --require-text-only-context --log-every 1 \
    > "${LOGS}/${RUN}.log" 2>&1
fi

echo "M0_READY report=${OUTPUTS}/${RUN}.json stage=${OUTPUTS}/${RUN}.pt"
