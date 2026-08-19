#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C73_FOLLOWUP_SOURCE_SNAPSHOT:?C73 followup requires a reviewed immutable snapshot}"
freeze_sha="${C73_FOLLOWUP_SOURCE_FREEZE_SHA256:?Set reviewed C73 followup SOURCE_FREEZE SHA256}"
train_root="${C73_TRAIN_ROOT:-${workspace}/outputs/c73-action-only-three-expert-epoch-v1/online-long130585-v1}"
root="${C73_FOLLOWUP_ROOT:?C73_FOLLOWUP_ROOT is required}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
train_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
val_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_val.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
source_root="${project}/third_party/FastWAM/src/fastwam/models/wan22"
verifier="${project}/scripts/h3wam/freeze_c67_rollout_source.py"
auditor="${project}/scripts/h3wam/prepare_c73_milestone_preview_audit.py"
evaluator="${project}/scripts/h3wam/evaluate_c67_fact_milestone_balanced80.py"
milestones=(38000 42000)

[[ ! -e "${root}" ]] || { echo "refusing existing C73 followup root: ${root}" >&2; exit 2; }
for path in "${python_bin}" "${project}/SOURCE_FREEZE.json" "${verifier}" \
  "${auditor}" "${evaluator}" "${h3_checkpoint}" "${source_manifest}" \
  "${train_manifest}" "${val_manifest}" "${cache_root}/stats.pt" \
  "${source_root}/action_dit.py" "${source_root}/wan_video_dit.py" \
  "${source_root}/helpers/gradient.py"; do
  [[ -e "${path}" ]] || { echo "missing C73 followup input: ${path}" >&2; exit 2; }
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
export TMPDIR="${workspace}/tmp/c73-balanced80-followup-preview"
mkdir -p "${TMPDIR}"

run_one() {
  local gpu="$1" milestone="$2"
  local checkpoint="${train_root}/checkpoints/c73_action_only_s${milestone}.pt"
  local audit="${root}/preview-audit/s${milestone}.json"
  local output="${root}/reports/s${milestone}.json"
  while [[ ! -s "${checkpoint}" \
    || ! -s "${train_root}/reports/train_s${milestone}.json" \
    || ! -s "${train_root}/restore/restore_s${milestone}.json" ]]; do
    sleep 30
  done
  "${python_bin}" "${auditor}" --train-root "${train_root}" \
    --milestone "${milestone}" --output "${audit}" \
    >"${root}/logs/audit_s${milestone}.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" "${evaluator}" \
    --variant c73 --checkpoint "${checkpoint}" --preview-audit "${audit}" \
    --milestone "${milestone}" --h3-checkpoint "${h3_checkpoint}" \
    --source-manifest "${source_manifest}" --train-manifest "${train_manifest}" \
    --val-manifest "${val_manifest}" --cache-root "${cache_root}" \
    --device cuda:0 --output "${output}" \
    >"${root}/logs/eval_s${milestone}.log" 2>&1
}

run_one 0 38000 & p0="$!"
run_one 1 42000 & p1="$!"
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
(( status == 0 )) || { echo "one or more C73 followup workers failed" >&2; exit 1; }

"${python_bin}" - "${root}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

root = Path(sys.argv[1])
reports = {}
for step in (38000, 42000):
    path = root / f"reports/s{step}.json"
    value = json.loads(path.read_text())
    if (
        value.get("format") != "h3wam-c73-action-only-milestone-balanced80-v1"
        or value.get("variant") != "c73"
        or value.get("milestone") != step
        or value.get("permission") != "PREVIEW_ONLY_PENDING_TRAINING_COMPLETE_REBIND"
    ):
        raise SystemExit(f"invalid C73 followup report: s{step}")
    reports[str(step)] = hashlib.sha256(path.read_bytes()).hexdigest()
marker = {
    "format": "h3wam-c73-action-only-followup-preview-complete-v1",
    "status": "PASS_C73_S38000_S42000_PREVIEWS_COMPLETE",
    "permission": "DIAGNOSTIC_ONLY_CONTINUE_FIXED_TRAJECTORY",
    "effect_status": "NOT_EVIDENCE_READY",
    "reports_sha256": reports,
    "claim_boundary": "No checkpoint selection, rollout, early stop, or final budget-effect claim.",
}
output = root / "PREVIEWS_COMPLETE.json"
temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(marker, indent=2) + "\n")
os.replace(temporary, output)
PY
