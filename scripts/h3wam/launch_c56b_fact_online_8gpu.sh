#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
trainer="${C56B_ONLINE_TRAINER:-${project}/scripts/h3wam/train_c56b_fact_online.py}"
c58_module="${C58_ONLINE_INTERFACE_MODULE:-fastwam.models.h3wam.c58_online_training}"
c58_symbol="${C58_ONLINE_INTERFACE_SYMBOL:-C58OnlineFrozenH3Provider}"
c58_commit="${C58_ONLINE_INTERFACE_COMMIT:-}"
output_root="${OUTPUT_ROOT:-${workspace}/outputs/c56b-fact-online-v1}"
c60_dataset="${workspace}/eval/c60-counterfactual-failure-dataset-v1/dataset.pt"
c60_observations="${workspace}/eval/c60-counterfactual-failure-dataset-v1/observations.jsonl"
c60_sha="1abeee1ef4e5e71f66b656c9920124086046c3e7d3b3a22b769449b72b1fc1d4"
c60_observations_sha="b9a812afe034f236181a6915369535545a997688a9dac8c351df3f51c0357a55"

if [[ -z "${c58_commit}" ]]; then
  echo "NO_GO: C58_ONLINE_INTERFACE_COMMIT is not fixed; no GPU process started." >&2
  exit 65
fi
for path in \
  "${python_bin}" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl" \
  "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
  "${workspace}/data/v7_dense_h3_cache/stats.pt" \
  "${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt" \
  "${workspace}/eval/c48-fact-dense-value-dataset-v1/observations.jsonl" \
  "${workspace}/eval/c59-fact-failure-active-overlay-v1/COMPLETED.json" \
  "${c60_dataset}" "${c60_observations}" \
  "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  "${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt"; do
  [[ -e "${path}" ]] || { echo "missing C56b online input: ${path}" >&2; exit 2; }
done
[[ "$(sha256sum "${c60_dataset}" | awk '{print $1}')" == "${c60_sha}" ]]
[[ "$(sha256sum "${c60_observations}" | awk '{print $1}')" == "${c60_observations_sha}" ]]
[[ ! -e "${output_root}" ]] || {
  echo "refusing existing C56b online output: ${output_root}" >&2
  exit 2
}

cd "${project}"
git cat-file -e "${c58_commit}^{commit}"
git merge-base --is-ancestor "${c58_commit}" HEAD || {
  echo "NO_GO: working tree does not contain fixed C58 online interface commit" >&2
  exit 65
}
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}${PYTHONPATH:+:${PYTHONPATH}}"
"${python_bin}" - "${c58_module}" "${c58_symbol}" <<'PY'
import importlib
import sys
module = importlib.import_module(sys.argv[1])
symbol = getattr(module, sys.argv[2], None)
if symbol is None:
    raise SystemExit("NO_GO: fixed C58 online provider symbol is absent")
PY
[[ -f "${trainer}" ]] || {
  echo "NO_GO: C56b online trainer is intentionally pending C58 adapter integration." >&2
  exit 65
}

# The trainer contract must consume current/future pixels or VAE latents and is
# forbidden from accepting an H3 K/V cache directory.  Its dossier must pass
# GO_LONG before this launcher can become executable.
dossier="${project}/experiments/dossiers/h3_c56b_fact_online_v1.json"
"${python_bin}" skills/wam-evidence-gated-training/scripts/validate_dossier.py \
  "${dossier}" --target long

exec "${python_bin}" -m torch.distributed.run --standalone --nproc-per-node 8 \
  "${trainer}" \
  --demo-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl" \
  --demo-source-manifest "${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl" \
  --demo-cache-root "${workspace}/data/v7_dense_h3_cache" \
  --c48-dataset "${workspace}/eval/c48-fact-dense-value-dataset-v1/dataset.pt" \
  --c48-observations "${workspace}/eval/c48-fact-dense-value-dataset-v1/observations.jsonl" \
  --c59-overlay-root "${workspace}/eval/c59-fact-failure-active-overlay-v1" \
  --c60-dataset "${c60_dataset}" --c60-observations "${c60_observations}" \
  --h3-checkpoint "${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
  --h3-model "${workspace}/models/MiniMax-H3" \
  --parent-checkpoint "${workspace}/outputs/dense-carrier-d0-h32-s20000-v1/checkpoints/d0_h32_s14000.pt" \
  --output-root "${output_root}" "$@"
