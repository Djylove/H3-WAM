#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:?PROJECT_ROOT must be a fresh read-only snapshot}"
root="${C56B_EXPANDED_ROOT:?C56B_EXPANDED_ROOT is required}"
prepared="${root}/PREPARED.json"
manifest="${root}/jobs.jsonl"
canary="${root}/mechanical-canary/CANARY_PASS.json"
checkpoint="${workspace}/outputs/c56b-fact-online-v1/online-long10000-v1/checkpoints/c56b_online_s10000.pt"
gate="${workspace}/outputs/c56b-fact-online-v1/paired-final-eval-v2/balanced80/PAIRED_BALANCED80.json"
policy_python="${workspace}/runtime/h3-int8-native/bin/python"
sim_python="${workspace}/runtime/conda-py311/bin/python"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
h3_model="${workspace}/models/MiniMax-H3"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
source_root="${workspace}/upstream-readonly/FastWAM-45d8e145/wan22"

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
for path in "${prepared}" "${manifest}" "${canary}" "${checkpoint}" "${gate}" \
  "${policy_python}" "${sim_python}" "${h3_checkpoint}" "${h3_model}" \
  "${source_manifest}" "${cache_root}/stats.pt"; do
  [[ -e "${path}" ]] || { echo "missing C60 expanded input: ${path}" >&2; exit 2; }
done
[[ "$(stat -c '%A' "${project}/scripts/h3wam/rollout_libero.py")" != *w* ]] || {
  echo "PROJECT_ROOT is not read-only" >&2; exit 2;
}
[[ ! -e "${root}/COMPLETED.json" ]] || { echo "C60 expanded rollout already complete"; exit 0; }
[[ ! -e "${root}/INVALID.json" ]] || { echo "C60 expanded root is invalid" >&2; exit 2; }
lock="${root}/.launcher.lock"
mkdir "${lock}" 2>/dev/null || { echo "another C60 launcher owns ${lock}" >&2; exit 75; }
trap 'rmdir "${lock}" 2>/dev/null || true' EXIT

"${sim_python}" - "${prepared}" "${canary}" "${manifest}" <<'PY'
import hashlib, json, sys
from pathlib import Path
p=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2])); manifest=Path(sys.argv[3])
digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
assert p["permission"] == "GO_MECHANICAL_CANARY_THEN_8GPU_640_FRESH_PROCESSES"
assert p["jobs"] == 640 and p["candidate_episodes"] == 640
assert p["one_episode_per_process"] is True and digest == p["manifest_sha256"]
assert c["permission"] == "GO_8GPU_640_FRESH_PROCESSES_NO_INTERMEDIATE_STOP"
assert c["checkpoint_sha256"] == p["candidate_checkpoint_sha256"]
PY

cd "${project}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
mkdir -p "${root}/logs"

run_worker() {
  local gpu="$1" job_id suite task trial assigned_gpu output log
  while IFS=$'\t' read -r job_id suite task trial assigned_gpu output; do
    [[ "${assigned_gpu}" == "${gpu}" ]] || continue
    [[ ! -e "${output}" ]] || {
      echo "refusing partial/reused C60 output: ${output}" >&2; return 1;
    }
    mkdir -p "${output}"
    log="${root}/logs/job$(printf '%03d' "${job_id}")_${suite}_task${task}_trial${trial}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHON_BIN="${sim_python}" \
    SIM_SITE_PACKAGES="/tmp/h3-wam-libero-site" \
    bash "${project}/scripts/h3wam/run_cloud_libero.sh" \
      "${sim_python}" "${project}/scripts/h3wam/rollout_libero.py" \
      --policy h3_fact_online_int8 --policy-python "${policy_python}" \
      --checkpoint "${checkpoint}" --c56b-paired-ready "${gate}" \
      --cache-root "${cache_root}" --h3-checkpoint "${h3_checkpoint}" \
      --h3-model "${h3_model}" --dreamwam-source-manifest "${source_manifest}" \
      --device cuda:0 --suite "${suite}" --task-ids "${task}" \
      --trial-indices "${trial}" --max-steps 400 --wait-steps 30 \
      --replan-steps 8 --action-horizon 32 --h3-feature-audio-horizon 32 \
      --target-latent-frames 12 --model-evaluations 10 --seed 42 \
      --normalized-action-pre-clamp --save-trajectories --output-dir "${output}" \
      >"${log}" 2>&1
  done < <("${sim_python}" - "${manifest}" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    row=json.loads(line)
    print("\t".join(map(str, (
        row["job_id"], row["suite"], row["tasks"][0], row["trials"][0],
        row["gpu"], row["output"],
    ))))
PY
)
}

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  run_worker "${gpu}" & pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "one or more C60 expanded workers failed" >&2; exit 1; }

"${sim_python}" - "${prepared}" "${manifest}" "${root}/COMPLETED.json" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
p=json.load(open(sys.argv[1])); rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
if len(rows) != 640: raise ValueError("C60 isolated job count mismatch")
episodes=0
for row in rows:
    path=Path(row["output"])/"results.json"; data=json.load(open(path))
    found=[e for task in data.get("tasks",[]) for e in task.get("episodes",[])]
    if len(found) != 1 or data.get("task_ids") != row["tasks"] or data.get("trial_indices") != row["trials"]:
        raise ValueError(f"invalid isolated C60 job: {path}")
    episodes += 1
if episodes != 640: raise ValueError("C60 candidate episode count mismatch")
out=Path(sys.argv[3]); tmp=out.with_name(f".{out.name}.{os.getpid()}.partial")
tmp.write_text(json.dumps({
  "format":"h3wam-c56b-fact-expanded-isolated-complete-v1", "status":"COMPLETE",
  "episodes":episodes, "manifest_sha256":p["manifest_sha256"],
},indent=2)+"\n")
os.replace(tmp,out)
PY
