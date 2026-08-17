#!/usr/bin/env bash
set -Eeuo pipefail

workspace=${H3_WORKSPACE:-/mnt/h3-wam}
project=${PROJECT_ROOT:?set PROJECT_ROOT to an immutable C62 snapshot}
python_bin=${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}
output_root=${OUTPUT_ROOT:-${workspace}/outputs/c62-miniworld-c58-rolling-context/causal-canary100}
data_root=${DATA_ROOT:-${output_root}/frozen-data}
dense_manifest=${DENSE_MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl}
source_manifest=${SOURCE_MANIFEST:-${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl}
cache_root=${CACHE_ROOT:-${workspace}/data/v7_dense_h3_cache}
h3_checkpoint=${H3_CHECKPOINT:-${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}
parent_checkpoint=${PARENT_CHECKPOINT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt}

for path in "${python_bin}" "${project}/scripts/h3wam/train_c62_miniworld_context_canary.py" \
  "${dense_manifest}" "${source_manifest}" "${cache_root}/stats.pt" \
  "${h3_checkpoint}" "${parent_checkpoint}"; do
  [[ -e "${path}" ]] || { echo "missing C62 canary input: ${path}" >&2; exit 2; }
done
[[ ! -e "${output_root}" ]] || {
  echo "refusing existing C62 canary output: ${output_root}" >&2
  exit 2
}
mkdir -p "${output_root}"
cuda13_lib=$(${python_bin} -c 'import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()["purelib"])/"nvidia"/"cu13"/"lib")')
export LD_LIBRARY_PATH="${cuda13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${project}/src"
cd "${project}"

"${python_bin}" scripts/h3wam/build_c62_miniworld_sequence_manifest.py \
  "${dense_manifest}" "${data_root}" \
  --history-chunks 3 --train-per-suite 200 --heldout-per-suite 16 \
  --heldout-episodes-per-suite 8 --seed 62017 \
  > "${output_root}/freeze.log"

"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node=8 \
  scripts/h3wam/train_c62_miniworld_context_canary.py \
  "${data_root}/train.jsonl" \
  --heldout-manifest "${data_root}/heldout.jsonl" \
  --plan "${data_root}/PLAN.json" \
  --dense-manifest "${dense_manifest}" \
  --source-manifest "${source_manifest}" \
  --cache-root "${cache_root}" \
  --h3-checkpoint "${h3_checkpoint}" \
  --parent-checkpoint "${parent_checkpoint}" \
  --steps 100 --learning-rate 1e-4 --weight-decay 0.01 --warmup-steps 10 \
  --seed 62017 --num-workers 0 \
  --save-checkpoint "${output_root}/c62_bridge_s00100.pt" \
  --output "${output_root}/report.json" \
  2>&1 | tee "${output_root}/train.log"

echo "[C62] bounded causal/optimizer canary finished"
