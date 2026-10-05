#!/usr/bin/env bash
# Run both decision-boundary-complexity experiments (PCA, then DT depth sweep) unattended.
#
# Launch detached so it survives logout:
#   nohup ./run_overnight.sh > logs/overnight.out 2>&1 &
#
# Options:
#   --n_jobs N        parallel datasets per experiment (default 12)
#   --max_depth D     deepest DT to train (default 40)
#   --resume          DT sweep: skip finished datasets, continue partial ones
#   --only pca|dt     run a single experiment
#   --output_root DIR write results to DIR/{pca,smallest_p95_dt} instead of the default results/ dirs
#
# Each step logs to logs/<step>_<timestamp>.log; one failing step does not stop the other.

set -uo pipefail

N_JOBS=12
MAX_DEPTH=40
RESUME=""
ONLY=""
OUTPUT_ROOT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --n_jobs) N_JOBS="$2"; shift 2 ;;
        --max_depth) MAX_DEPTH="$2"; shift 2 ;;
        --resume) RESUME="-resume"; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
export PYTHONUNBUFFERED=1

PCA_OUT=()
DT_OUT=()
if [ -n "${OUTPUT_ROOT}" ]; then
    PCA_OUT=(-output_dir "${OUTPUT_ROOT}/pca")
    DT_OUT=(-output_dir "${OUTPUT_ROOT}/smallest_p95_dt")
fi

SUMMARY=()

run_step() {
    local name="$1"; shift
    local log="${LOG_DIR}/${name}_$(date +%Y%m%d_%H%M%S).log"
    local start=$(date +%s)
    echo "=== [$(date '+%F %T')] START ${name} -> ${log}"
    echo "    $*"
    "$@" 2>&1 | tee "${log}"
    local status=${PIPESTATUS[0]}
    local dur=$(( $(date +%s) - start ))
    local result="OK"
    [ "${status}" -ne 0 ] && result="FAILED (exit ${status})"
    echo "=== [$(date '+%F %T')] END ${name}: ${result} after $((dur / 3600))h$(( (dur % 3600) / 60 ))m$((dur % 60))s"
    SUMMARY+=("$(printf '%-16s %-18s %dh%02dm%02ds  %s' "${name}" "${result}" $((dur / 3600)) $(( (dur % 3600) / 60 )) $((dur % 60)) "${log}")")
}

echo "Decision boundary complexity overnight run started at $(date '+%F %T') on $(hostname)"
echo "n_jobs=${N_JOBS} max_depth=${MAX_DEPTH} resume=${RESUME:-no} only=${ONLY:-all} output_root=${OUTPUT_ROOT:-default}"

if [ -z "${ONLY}" ] || [ "${ONLY}" = "pca" ]; then
    run_step pca python3 "${SCRIPT_DIR}/pca/run_pca.py" -n_jobs "${N_JOBS}" "${PCA_OUT[@]}"
fi

if [ -z "${ONLY}" ] || [ "${ONLY}" = "dt" ]; then
    run_step dt_depth_sweep python3 "${SCRIPT_DIR}/smallest_p95_dt/run_depth_sweep.py" \
        -n_jobs "${N_JOBS}" -max_depth "${MAX_DEPTH}" ${RESUME} "${DT_OUT[@]}"
fi

echo
echo "================ Summary ($(date '+%F %T')) ================"
printf '%s\n' "${SUMMARY[@]}"

for line in "${SUMMARY[@]}"; do
    [[ "${line}" == *FAILED* ]] && exit 1
done
exit 0
