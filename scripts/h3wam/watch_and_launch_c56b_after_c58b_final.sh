#!/usr/bin/env bash
set -Eeuo pipefail

workspace="${H3_WORKSPACE:-/mnt/h3-wam}"
project="${PROJECT_ROOT:-${workspace}/candidate-d0-rollout-96976ce/project}"
python_bin="${PYTHON_BIN:-${workspace}/runtime/h3-int8-native/bin/python}"
c58_ready="${C58_PARENT_READY:-${workspace}/outputs/c58b-fastwam-layerwise-v1/online-long10000/READY.json}"
c56_canary="${CANARY_MARKER:-${workspace}/outputs/c56b-fact-online-v1/optimizer-canary10-v1/GO_LONG.json}"
c56_output="${OUTPUT_ROOT:-${workspace}/outputs/c56b-fact-online-v1/online-long10000-v1}"
arm_root="${ARM_ROOT:-${workspace}/outputs/c56b-fact-online-v1/arm-after-c58b-final-v1}"
causal_ready="${CAUSAL_FAILURE_READY:-}"
poll_seconds="${POLL_SECONDS:-30}"
idle_confirm_seconds="${IDLE_CONFIRM_SECONDS:-15}"

[[ "${poll_seconds}" =~ ^[0-9]+$ && "${poll_seconds}" -gt 0 ]] || {
  echo "POLL_SECONDS must be positive" >&2; exit 2;
}
[[ "${idle_confirm_seconds}" =~ ^[0-9]+$ && "${idle_confirm_seconds}" -gt 0 ]] || {
  echo "IDLE_CONFIRM_SECONDS must be positive" >&2; exit 2;
}
mkdir -p "$(dirname "${arm_root}")"
if ! mkdir "${arm_root}" 2>/dev/null; then
  echo "C56b arming state already exists; refusing duplicate launch: ${arm_root}" >&2
  exit 2
fi
printf '%s\n' "$$" > "${arm_root}/PID"
exec >> "${arm_root}/watcher.log" 2>&1
trap 'code=$?; printf "{\"status\":\"FAILED\",\"exit_code\":%d}\n" "${code}" > "${arm_root}/FAILED.json"; exit "${code}"' ERR INT TERM

if [[ -z "${causal_ready}" && ( -n "${CAUSAL_FAILURE_DATASET:-}" \
    || -n "${CAUSAL_FAILURE_OBSERVATIONS:-}" \
    || -n "${EXPECTED_CAUSAL_DATASET_SHA256:-}" \
    || -n "${EXPECTED_CAUSAL_OBSERVATIONS_SHA256:-}" ) ]]; then
  echo "custom causal data requires CAUSAL_FAILURE_READY" >&2
  exit 2
fi
while [[ ! -s "${c58_ready}" || ! -s "${c56_canary}" \
         || ( -n "${causal_ready}" && ! -s "${causal_ready}" ) ]]; do
  sleep "${poll_seconds}"
done

c58_parent="$(${python_bin} - "${c58_ready}" "${c56_canary}" <<'PY'
import hashlib, json, sys
from pathlib import Path
ready = json.loads(Path(sys.argv[1]).read_text())
canary = json.loads(Path(sys.argv[2]).read_text())
checkpoint = Path(ready.get("checkpoint", "")).resolve()
checks = {
    "c56_canary_status": canary.get("status") == "GO_LONG",
    "c56_canary_effect_boundary": canary.get("effect_status") == "NOT_EVIDENCE_READY",
    "c56_canary_gate": all(canary.get("gate", {}).values()),
    "status": ready.get("status") == "PASS_C58B_ONLINE_LONG10000_STRICT_RESTORE",
    "permission": ready.get("permission") == "READY_FOR_CHILD_BRANCH_AND_LIBERO_EVAL",
    "steps": int(ready.get("completed_steps", -1)) == 10000,
    "checkpoint": checkpoint.is_file(),
    "size": checkpoint.is_file() and checkpoint.stat().st_size == int(ready.get("checkpoint_size_bytes", -1)),
}
if all(checks.values()):
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    checks["sha256"] = digest.hexdigest() == ready.get("checkpoint_sha256")
failed = [key for key, value in checks.items() if not value]
if failed:
    raise SystemExit("C58b READY identity failed: " + ",".join(failed))
print(checkpoint)
PY
)"

launch_env=(
  C58_PARENT_CHECKPOINT="${c58_parent}"
  C58_PARENT_READY="${c58_ready}"
  OUTPUT_ROOT="${c56_output}"
)
if [[ -n "${causal_ready}" ]]; then
  causal_identity="$(${python_bin} - "${causal_ready}" "${arm_root}/C61_DATA_READY.json" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
completed_path = Path(sys.argv[1]).resolve()
audit_path = Path(sys.argv[2]).resolve()
completed = json.loads(completed_path.read_text())
root = completed_path.parent
dataset, observations = root / "dataset.pt", root / "observations.jsonl"
checks = {
    "format": completed.get("format") == "h3wam-c61-finalized-fact-failure-dataset-v1",
    "status": completed.get("status") == "PASS_C61_FINALIZED_FACT_FAILURE_DATASET",
    "all_gates": completed.get("gates")
        and all(value == "PASS" for value in completed["gates"].values()),
    "failed_rows": int(completed.get("retained_failed_jobs", 0)) > 0,
    "dataset": dataset.is_file(),
    "observations": observations.is_file(),
}
hashes = {}
for name, path in (("dataset", dataset), ("observations", observations)):
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(16 * 1024 * 1024):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
checks["dataset_sha256"] = hashes.get("dataset") == completed.get("dataset_sha256")
checks["observations_sha256"] = hashes.get("observations") == completed.get("observations_sha256")
failed = [key for key, value in checks.items() if not value]
if failed:
    raise SystemExit("C61 finalized causal-data gate failed: " + ",".join(failed))
completed_digest = hashlib.sha256(completed_path.read_bytes()).hexdigest()
audit = {
    "format": "h3wam-c56b-c61-matched-data-gate-v1",
    "status": "PASS_C61_MATCHED_DATA_GATE",
    "effect_status": "NOT_EVIDENCE_READY",
    "single_variable": "C60 causal_failure pool replaced by finalized C61 train split",
    "fixed_contract": {
        "global_batch": 8,
        "rank_mixture": ["expert_demo"] * 4 + ["success_rollout"] * 2
            + ["observational_failure", "causal_failure"],
        "optimizer_steps": 10000,
        "seed": 20260816,
        "loss_weights": [10.0, 1.0, 0.4, 0.4],
    },
    "completed": str(completed_path),
    "completed_sha256": completed_digest,
    "dataset": str(dataset),
    "dataset_sha256": hashes["dataset"],
    "observations": str(observations),
    "observations_sha256": hashes["observations"],
    "train_counts": completed.get("counts", {}).get("train"),
    "validation_counts": completed.get("counts", {}).get("validation"),
    "claim_boundary": "Finalized data/paired-contract gate only; no training or effect claim.",
}
temporary = audit_path.with_name(f".{audit_path.name}.{os.getpid()}.partial")
temporary.write_text(json.dumps(audit, indent=2) + "\n")
os.replace(temporary, audit_path)
print("\t".join((str(dataset), str(observations), hashes["dataset"], hashes["observations"])))
PY
)"
  IFS=$'\t' read -r causal_dataset causal_observations causal_dataset_sha causal_observations_sha <<< "${causal_identity}"
  launch_env+=(
    CAUSAL_FAILURE_READY="${causal_ready}"
    CAUSAL_FAILURE_DATASET="${causal_dataset}"
    CAUSAL_FAILURE_OBSERVATIONS="${causal_observations}"
    EXPECTED_CAUSAL_DATASET_SHA256="${causal_dataset_sha}"
    EXPECTED_CAUSAL_OBSERVATIONS_SHA256="${causal_observations_sha}"
  )
fi

[[ ! -e "${c56_output}" ]] || {
  echo "refusing existing C56b long output: ${c56_output}" >&2; exit 2;
}
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"
[[ "${gpu_count}" -eq 8 ]] || { echo "C56b requires exactly eight visible GPUs" >&2; exit 2; }
compute_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | grep -E '^[[:space:]]*[0-9]+[[:space:]]*$' || true
}
while true; do
  if [[ -z "$(compute_pids)" ]]; then
    sleep "${idle_confirm_seconds}"
    [[ -z "$(compute_pids)" ]] && break
  else
    sleep "${poll_seconds}"
  fi
done

printf '{"status":"LAUNCHING","c58_parent":"%s","output":"%s"}\n' \
  "${c58_parent}" "${c56_output}" > "${arm_root}/LAUNCHING.json"
cd "${project}"
env "${launch_env[@]}" bash scripts/h3wam/launch_c56b_fact_online_long10000_8gpu.sh
printf '{"status":"COMPLETE","output":"%s"}\n' "${c56_output}" > "${arm_root}/COMPLETE.json"
