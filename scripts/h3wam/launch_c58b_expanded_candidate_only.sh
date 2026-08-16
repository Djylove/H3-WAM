#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
root="${C58B_EXPANDED_ROOT:-${workspace}/outputs/c58b-fastwam-layerwise-v1/expanded-paired-trials34-49-v1}"
prepared="${root}/PREPARED.json"
manifest="${root}/jobs.jsonl"
checkpoint="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/checkpoints/c58b_online_s10000.pt"
gate="${workspace}/outputs/c58b-fastwam-layerwise-v1/online-final-eval-v1/balanced80/BALANCED80_READY.14fa645.json"
d0_ready="${workspace}/eval/c58b-expanded-d0-control-v1/READY.json"
policy_python="${workspace}/runtime/h3-int8-native/bin/python"
sim_python="${workspace}/runtime/conda-py311/bin/python"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
h3_model="${workspace}/models/MiniMax-H3"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
source_root="${H3WAM_FASTWAM_SOURCE_ROOT:-${workspace}/upstream-readonly/FastWAM-45d8e145/wan22}"

for path in "${prepared}" "${manifest}" "${checkpoint}" "${gate}" "${d0_ready}" \
  "${policy_python}" "${sim_python}" "${h3_checkpoint}" "${h3_model}" \
  "${source_manifest}" "${cache_root}/stats.pt" "${source_root}/action_dit.py"; do
  [[ -e "${path}" ]] || { echo "missing C58b expanded input: ${path}" >&2; exit 2; }
done
[[ ! -e "${root}/COMPLETED.json" ]] || { echo "expanded candidate rollout already complete"; exit 0; }
lock="${root}/.launcher.lock"
mkdir "${lock}" 2>/dev/null || { echo "another expanded launcher owns ${lock}" >&2; exit 75; }
trap 'rmdir "${lock}" 2>/dev/null || true' EXIT

"${sim_python}" - "${prepared}" "${d0_ready}" <<'PY'
import json, sys
p=json.load(open(sys.argv[1])); d=json.load(open(sys.argv[2]))
assert p["permission"] == "GO_8GPU_CANDIDATE_ONLY_NO_INTERMEDIATE_STOP"
assert p["jobs"] == 8 and p["candidate_episodes"] == 640
assert d["permission"] == "GO_C58B_CANDIDATE_ONLY_TRIALS34_49"
PY

cd "${project}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
mkdir -p "${root}/logs"

run_job() {
  local index="$1"
  local line suite gpu trials_text output
  line="$(${sim_python} - "${manifest}" "${index}" <<'PY'
import json, sys
row=json.loads(open(sys.argv[1]).read().splitlines()[int(sys.argv[2])])
print("\t".join((row["suite"],str(row["gpu"]),",".join(map(str,row["trials"])),row["output"])))
PY
)"
  IFS=$'\t' read -r suite gpu trials_text output <<<"${line}"
  [[ ! -e "${output}" ]] || { echo "refusing partial/reused candidate output: ${output}" >&2; return 1; }
  IFS=',' read -r -a trials <<<"${trials_text}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHON_BIN="${sim_python}" \
  SIM_SITE_PACKAGES="/tmp/h3-wam-libero-site" \
  bash "${project}/scripts/h3wam/run_cloud_libero.sh" \
    "${sim_python}" "${project}/scripts/h3wam/rollout_libero.py" \
    --policy h3_fastwam_online_int8 --policy-python "${policy_python}" \
    --checkpoint "${checkpoint}" --c58b-balanced80-ready "${gate}" \
    --cache-root "${cache_root}" --h3-checkpoint "${h3_checkpoint}" \
    --h3-model "${h3_model}" --dreamwam-source-manifest "${source_manifest}" \
    --device cuda:0 --suite "${suite}" --task-ids 0 1 2 3 4 5 6 7 8 9 \
    --trial-indices "${trials[@]}" --max-steps 400 --wait-steps 30 \
    --replan-steps 8 --action-horizon 32 --h3-feature-audio-horizon 32 \
    --target-latent-frames 12 --model-evaluations 10 --seed 42 \
    --normalized-action-pre-clamp --save-trajectories --output-dir "${output}" \
    >"${root}/logs/job$(printf '%02d' "${index}")_${suite}_trials${trials[0]}-${trials[7]}.log" 2>&1
}

pids=()
for index in 0 1 2 3 4 5 6 7; do
  run_job "${index}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
(( status == 0 )) || { echo "one or more C58b expanded jobs failed" >&2; exit 1; }

"${sim_python}" - "${prepared}" "${manifest}" "${root}/COMPLETED.json" <<'PY'
import json, os, sys
from pathlib import Path
prepared=json.load(open(sys.argv[1])); rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
episodes=0
for row in rows:
    path=Path(row["output"])/"results.json"; data=json.load(open(path))
    found=[e for task in data.get("tasks",[]) for e in task.get("episodes",[])]
    if len(found)!=80 or data.get("successes")!=sum(bool(e.get("success")) for e in found):
        raise ValueError(f"incomplete C58b expanded job: {path}")
    episodes += len(found)
if episodes != 640: raise ValueError("expanded candidate episode count mismatch")
out=Path(sys.argv[3]); tmp=out.with_name(f".{out.name}.{os.getpid()}.partial")
tmp.write_text(json.dumps({"format":"h3wam-c58b-expanded-candidate-complete-v1","status":"COMPLETE","episodes":episodes,"manifest_sha256":prepared["manifest_sha256"]},indent=2)+"\n")
os.replace(tmp,out)
PY
