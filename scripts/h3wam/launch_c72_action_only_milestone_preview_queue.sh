#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C72_PREVIEW_SOURCE_SNAPSHOT:?C72 preview requires a reviewed immutable snapshot}"
freeze_sha="${C72_PREVIEW_SOURCE_FREEZE_SHA256:?Set reviewed C72 preview SOURCE_FREEZE SHA256}"
train_root="${C72_TRAIN_ROOT:-${workspace}/outputs/c72-action-only-one-expert-epoch-v1/online-long30195-v1}"
root="${C72_PREVIEW_ROOT:?C72_PREVIEW_ROOT is required}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
train_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
val_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_val.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
source_root="${project}/third_party/FastWAM/src/fastwam/models/wan22"
verifier="${project}/scripts/h3wam/freeze_c67_rollout_source.py"
auditor="${project}/scripts/h3wam/prepare_c72_milestone_preview_audit.py"
evaluator="${project}/scripts/h3wam/evaluate_c67_fact_milestone_balanced80.py"

[[ ! -e "${root}" ]] || { echo "refusing existing C72 preview root: ${root}" >&2; exit 2; }
for path in "${python_bin}" "${project}/SOURCE_FREEZE.json" "${verifier}" \
  "${auditor}" "${evaluator}" "${h3_checkpoint}" "${source_manifest}" \
  "${train_manifest}" "${val_manifest}" "${cache_root}/stats.pt" \
  "${source_root}/action_dit.py" "${source_root}/wan_video_dit.py" \
  "${source_root}/helpers/gradient.py"; do
  [[ -e "${path}" ]] || { echo "missing C72 preview input: ${path}" >&2; exit 2; }
done
"${python_bin}" "${verifier}" --verify --snapshot "${project}" \
  --expected-manifest-sha256 "${freeze_sha}"

mkdir -p "${root}/preview-audit" "${root}/reports" "${root}/logs"
cd "${project}"
export PYTHONPATH="${project}/third_party/diffusers_h3/src:${project}/src:${project}"
export H3WAM_FASTWAM_SOURCE_ROOT="${source_root}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cu13_lib="$(${python_bin} - <<'PY'
import sysconfig
from pathlib import Path
print(Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib")
PY
)"
export LD_LIBRARY_PATH="${cu13_lib}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export TMPDIR="${workspace}/tmp/c72-balanced80-preview"
mkdir -p "${TMPDIR}"
milestones=($(seq 1000 1000 30000) 30195)

run_gpu() {
  local gpu="$1" index milestone checkpoint audit output
  for index in "${!milestones[@]}"; do
    (( index % 8 == gpu )) || continue
    milestone="${milestones[$index]}"
    checkpoint="${train_root}/checkpoints/c72_action_only_s${milestone}.pt"
    audit="${root}/preview-audit/s${milestone}.json"
    output="${root}/reports/s${milestone}.json"
    while [[ ! -s "${checkpoint}" || ! -s "${train_root}/reports/train_s${milestone}.json" \
      || ! -s "${train_root}/restore/restore_s${milestone}.json" ]]; do
      sleep 30
    done
    "${python_bin}" "${auditor}" --train-root "${train_root}" \
      --milestone "${milestone}" --output "${audit}" \
      >"${root}/logs/audit_s${milestone}.log" 2>&1
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" "${evaluator}" \
      --variant c72 --checkpoint "${checkpoint}" --preview-audit "${audit}" \
      --milestone "${milestone}" --h3-checkpoint "${h3_checkpoint}" \
      --source-manifest "${source_manifest}" --train-manifest "${train_manifest}" \
      --val-manifest "${val_manifest}" --cache-root "${cache_root}" \
      --device cuda:0 --output "${output}" \
      >"${root}/logs/eval_s${milestone}.log" 2>&1
  done
}

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  run_gpu "${gpu}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "one or more C72 preview workers failed" >&2; exit 1; }

"${python_bin}" - "${root}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

root = Path(sys.argv[1])
milestones = list(range(1000, 30001, 1000)) + [30195]
reports = {}
for step in milestones:
    path = root / f"reports/s{step}.json"
    value = json.loads(path.read_text())
    if (
        value.get("format") != "h3wam-c72-action-only-milestone-balanced80-v1"
        or value.get("variant") != "c72"
        or value.get("milestone") != step
        or value.get("permission") != "PREVIEW_ONLY_PENDING_TRAINING_COMPLETE_REBIND"
    ):
        raise SystemExit(f"invalid C72 preview report: s{step}")
    reports[str(step)] = hashlib.sha256(path.read_bytes()).hexdigest()
marker = {
    "format": "h3wam-c72-action-only-preview-complete-v1",
    "status": "PASS_C72_ALL_31_PREVIEWS_COMPLETE",
    "permission": "WAIT_FOR_FIXED_C72_BUDGET_AGGREGATION",
    "effect_status": "NOT_EVIDENCE_READY",
    "reports_sha256": reports,
    "claim_boundary": "No checkpoint selection, rollout, or budget-effect claim.",
}
output = root / "PREVIEWS_COMPLETE.json"
temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(marker, indent=2) + "\n")
os.replace(temporary, output)
PY
