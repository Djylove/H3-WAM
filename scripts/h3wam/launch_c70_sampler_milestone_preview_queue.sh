#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${C70_PREVIEW_SOURCE_SNAPSHOT:?C70 preview requires a reviewed immutable snapshot}"
freeze_sha="${C70_PREVIEW_SOURCE_FREEZE_SHA256:?Set reviewed C70 preview SOURCE_FREEZE SHA256}"
train_root="${C70_TRAIN_ROOT:-${workspace}/outputs/c70-sampler-coverage-v1/online-long20000-v1}"
root="${C70_PREVIEW_ROOT:?C70_PREVIEW_ROOT is required}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
h3_checkpoint="${workspace}/int8-action/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
source_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_all.jsonl"
train_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_train_uniform.jsonl"
val_manifest="${workspace}/data/v7_multisuite_dense_candidate/manifest_val.jsonl"
cache_root="${workspace}/data/v7_dense_h3_cache"
source_root="${project}/third_party/FastWAM/src/fastwam/models/wan22"
verifier="${project}/scripts/h3wam/freeze_c67_rollout_source.py"
auditor="${project}/scripts/h3wam/prepare_c70_milestone_preview_audit.py"
evaluator="${project}/scripts/h3wam/evaluate_c67_fact_milestone_balanced80.py"

[[ ! -e "${root}" ]] || { echo "refusing existing C70 preview root: ${root}" >&2; exit 2; }
for path in "${python_bin}" "${project}/SOURCE_FREEZE.json" "${verifier}" \
  "${auditor}" "${evaluator}" "${h3_checkpoint}" "${source_manifest}" \
  "${train_manifest}" "${val_manifest}" "${cache_root}/stats.pt" \
  "${source_root}/action_dit.py" "${source_root}/wan_video_dit.py" \
  "${source_root}/helpers/gradient.py"; do
  [[ -e "${path}" ]] || { echo "missing C70 preview input: ${path}" >&2; exit 2; }
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
export TMPDIR="${workspace}/tmp/c70-balanced80-preview"
mkdir -p "${TMPDIR}"

run_gpu() {
  local gpu="$1" milestone checkpoint train_report restore_report audit output
  for milestone in $(seq 1000 1000 20000); do
    (( (milestone / 1000 - 1) % 8 == gpu )) || continue
    checkpoint="${train_root}/checkpoints/c70_sampler_s${milestone}.pt"
    train_report="${train_root}/reports/train_s${milestone}.json"
    restore_report="${train_root}/restore/restore_s${milestone}.json"
    audit="${root}/preview-audit/s${milestone}.json"
    output="${root}/reports/s${milestone}.json"
    while [[ ! -s "${checkpoint}" || ! -s "${train_report}" || ! -s "${restore_report}" ]]; do
      sleep 30
    done
    "${python_bin}" "${auditor}" --train-root "${train_root}" \
      --milestone "${milestone}" --output "${audit}" \
      >"${root}/logs/audit_s${milestone}.log" 2>&1
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" "${evaluator}" \
      --variant c70 --checkpoint "${checkpoint}" --preview-audit "${audit}" \
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
(( status == 0 )) || { echo "one or more C70 preview workers failed" >&2; exit 1; }

"${python_bin}" - "${root}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
reports = {}
for step in range(1000, 20001, 1000):
    path = root / f"reports/s{step}.json"
    value = json.loads(path.read_text())
    if (
        value.get("format") != "h3wam-c70-sampler-coverage-milestone-balanced80-v1"
        or value.get("variant") != "c70"
        or value.get("milestone") != step
        or value.get("permission") != "PREVIEW_ONLY_PENDING_TRAINING_COMPLETE_REBIND"
    ):
        raise SystemExit(f"invalid C70 preview report: s{step}")
    reports[str(step)] = hashlib.sha256(path.read_bytes()).hexdigest()
marker = {
    "format": "h3wam-c70-sampler-coverage-preview-complete-v1",
    "status": "PASS_C70_ALL_20_PREVIEWS_COMPLETE",
    "permission": "WAIT_FOR_FIXED_C67_VS_C70_AGGREGATION",
    "effect_status": "NOT_EVIDENCE_READY",
    "reports_sha256": reports,
    "claim_boundary": "No checkpoint selection, rollout, or mechanism claim.",
}
output = root / "PREVIEWS_COMPLETE.json"
temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(marker, indent=2) + "\n")
os.replace(temporary, output)
PY
