#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:?PROJECT_ROOT must be a fresh read-only snapshot}"
root="${C56B_EXPANDED_ROOT:?C56B_EXPANDED_ROOT is required}"
prepared="${root}/PREPARED.json"
checkpoint="${workspace}/outputs/c56b-fact-online-v1/online-long10000-v1/checkpoints/c56b_online_s10000.pt"
gate="${workspace}/outputs/c56b-fact-online-v1/paired-final-eval-v2/balanced80/PAIRED_BALANCED80.json"
policy_python="${workspace}/runtime/h3-int8-native/bin/python"
sim_python="${workspace}/runtime/conda-py311/bin/python"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
h3_model="${workspace}/models/MiniMax-H3"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
source_root="${workspace}/upstream-readonly/FastWAM-45d8e145/wan22"
canary="${root}/mechanical-canary"

deps=(
  "${source_root}/action_dit.py:1301d9224149de43bb701f620a5d41858ecc63c6b19a573ec32edd45a3bdb0a2"
  "${source_root}/wan_video_dit.py:d098ad77665feeefa81634f31f5bb1d5771c4556d1a67859135f0ed35f9eb6c2"
  "${source_root}/helpers/gradient.py:ba5d8f7272eb029dc6cd2849ca99b70f6ad5abb838d21c818beb0590620dc793"
  "${project}/third_party/StarWAM/starwam/modules/action_dit.py:b6cd067cac448d8f4dba20f3778bae9bef622f58bdd854b1dfccc190d9dcf8b1"
  "${project}/third_party/StarWAM/starwam/modules/wan_block.py:303344329ba63692616494e40dd3b2288945d329d905e38c1bdcc26af5467524"
  "${project}/third_party/DreamWAM/dreamwam/layers.py:3cd38ad24eff05e748d9353af3f39200e93b16b6d07d22f153ccef0f36becd96"
  "${project}/third_party/DreamWAM/dreamwam/experts.py:9ba51dbb15b8df8e4ff01c5a08acf443a950c422544e4497d13e0e2658bd489c"
  "${project}/third_party/DreamWAM/dreamwam/mot.py:5467d135287a6e77074cb653fc3d72218490fcfa40ac486b61d5cc5975ab6c01"
)
for spec in "${deps[@]}"; do
  path="${spec%%:*}"; expected="${spec##*:}"
  [[ -f "${path}" ]] || { echo "missing pinned dependency: ${path}" >&2; exit 2; }
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || {
    echo "pinned dependency hash mismatch: ${path}" >&2; exit 2;
  }
done
for path in "${prepared}" "${checkpoint}" "${gate}" "${policy_python}" \
  "${sim_python}" "${h3_checkpoint}" "${h3_model}" "${source_manifest}" \
  "${cache_root}/stats.pt"; do
  [[ -e "${path}" ]] || { echo "missing canary input: ${path}" >&2; exit 2; }
done
[[ ! -w "${project}/scripts/h3wam/rollout_libero.py" ]] || {
  echo "PROJECT_ROOT is not read-only" >&2; exit 2;
}
[[ ! -e "${canary}" ]] || { echo "refusing reused canary root" >&2; exit 2; }
mkdir -p "${canary}"

cd "${project}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
CUDA_VISIBLE_DEVICES=0 PYTHON_BIN="${sim_python}" \
SIM_SITE_PACKAGES="/tmp/h3-wam-libero-site" \
bash "${project}/scripts/h3wam/run_cloud_libero.sh" \
  "${sim_python}" "${project}/scripts/h3wam/rollout_libero.py" \
  --policy h3_fact_online_int8 --policy-python "${policy_python}" \
  --checkpoint "${checkpoint}" --c56b-paired-ready "${gate}" \
  --cache-root "${cache_root}" --h3-checkpoint "${h3_checkpoint}" \
  --h3-model "${h3_model}" --dreamwam-source-manifest "${source_manifest}" \
  --device cuda:0 --suite libero_spatial --task-ids 0 --trial-indices 34 \
  --max-steps 400 --wait-steps 30 --replan-steps 8 --action-horizon 32 \
  --h3-feature-audio-horizon 32 --target-latent-frames 12 \
  --model-evaluations 10 --seed 42 --normalized-action-pre-clamp \
  --save-trajectories --output-dir "${canary}" \
  >"${canary}/launcher.log" 2>&1

"${sim_python}" "${project}/scripts/h3wam/audit_c56b_fact_expanded_canary.py" \
  --root "${root}" --output "${canary}/CANARY_PASS.json"
