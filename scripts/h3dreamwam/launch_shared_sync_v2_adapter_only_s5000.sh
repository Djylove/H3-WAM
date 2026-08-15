#!/usr/bin/env bash
set -Eeuo pipefail

H3_WORKSPACE="${H3_WORKSPACE:-/mnt/h3-wam}"
PROJECT_ROOT="${PROJECT_ROOT:-${H3_WORKSPACE}/project-adapter-sync-v2}"
PYTHON_BIN="${PYTHON_BIN:-${H3_WORKSPACE}/runtime/conda-py311/bin/python}"
OUTPUT_ROOT="${H3_WORKSPACE}/outputs/h3-lingbot-shared-sync-v2-adapter-only-s5000-fresh"
RUN_NAME="shared_sync_v2_adapter_only_s5000_fresh"
TRAINER="scripts/h3dreamwam/verify_h3_lingbot_four_stream_fsdp.py"
EXPECTED_TRAINER_SHA256="bf086d452959875d391beefa77e21e7d1456e5f2f5ab810ec74c73dcdc8a7ff4"
TRAIN_MANIFEST="${H3_WORKSPACE}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
BUNDLED_CU13="${H3_WORKSPACE}/runtime/h3-int8-native/lib/python3.11/site-packages/nvidia/cu13/lib"

cd "${PROJECT_ROOT}"
actual_sha256="$(sha256sum "${TRAINER}" | awk '{print $1}')"
[[ "${actual_sha256}" == "${EXPECTED_TRAINER_SHA256}" ]] || { echo "trainer identity mismatch" >&2; exit 2; }
[[ $(wc -l < "${TRAIN_MANIFEST}") -eq 200779 ]] || { echo "train manifest row count changed" >&2; exit 2; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "refusing overwrite: ${OUTPUT_ROOT}" >&2; exit 2; }
[[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d' | wc -l) -eq 0 ]] || { echo "GPUs not idle" >&2; exit 2; }

mkdir -p "${OUTPUT_ROOT}" "${H3_WORKSPACE}/tmp/h3-lingbot-adapter-sync-v2-s5000"
export PYTHONPATH="${H3_WORKSPACE}/project/third_party/diffusers_h3/src:${PROJECT_ROOT}/src:${PROJECT_ROOT}:${H3_WORKSPACE}/.venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="${BUNDLED_CU13}:${H3_WORKSPACE}/runtime/gl_root/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TMPDIR="${H3_WORKSPACE}/tmp/h3-lingbot-adapter-sync-v2-s5000"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# Start from the same deterministic seed/data order as the s1000 arm. The v2
# stage format intentionally does not contain optimizer or scheduler state, so
# restarting from clean initialization is the only strict way to call this a
# continuous 5000-step learning curve.
exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  "${TRAINER}" --shared-backbone --freeze-shared-blocks \
  --model "${H3_WORKSPACE}/models/MiniMax-H3" \
  --data-root "${H3_WORKSPACE}/data/v7_dense_h3_cache" --manifest "${TRAIN_MANIFEST}" \
  --action-normalization quantile --action-stats-json experiments/data/libero_v7_action_quantiles.json \
  --flow-match-loss-weighting --rotate-windows --random-timesteps \
  --output "${OUTPUT_ROOT}/${RUN_NAME}_train.json" --save-stage "${OUTPUT_ROOT}/${RUN_NAME}.pt" \
  --checkpoint-every 500 --steps 5000 --warmup-steps 10 \
  --last-trainable-layers 2 --action-horizon 32 --learning-rate 1e-5 --weight-decay 0.01 \
  --seed 2026 --sample-offset 0
