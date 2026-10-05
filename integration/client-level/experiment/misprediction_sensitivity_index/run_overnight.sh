#!/usr/bin/env bash
# Misprediction Sensitivity Index: replay FlashNet with injected flips, then analyze.
# Meant to run inside tmux:
#   tmux new -s msi
#   ./run_overnight.sh                      # 1 run of every p (~9 h, ~10.5 h if p=0 is added)
#   ./run_overnight.sh --num_runs 3         # later night: adds runs 1-2 (finished runs are skipped)
#
# Options:
#   --p0 auto|robust|single
#                         where the p=0 point comes from (default auto):
#                           robust: the existing robust FlashNet runs (flashnet/run_*), required for every trace
#                           single: add a p=0 replay (flips off, same decisions as the original FlashNet)
#                                   to this run, written to flashnet_flip_p00 (--p0_runs runs, default 1)
#                           auto:   robust if every trace has flashnet/run_*, otherwise single
#   --p0_runs N           number of p=0 replays when they are added (default 1)
#   --num_runs N          total runs per p>0 and trace, existing ones included (default 1)
#   --flip_probs "..."    flip probabilities as fractions (default "0.01 0.02 0.03 0.04 0.05 0.10")
#   --with_p0_reference   in robust mode, also add the p=0 replay as the reference for decision drift
#                         (the original FlashNet replayer doesn't log decisions; single mode always has it)
#   --devices "..."       default "/dev/nvme0n1 /dev/nvme1n1"
#   --data_root DIR       default $HEIMDALL/integration/client-level/data
#   --analyze_only        skip replays, only run the analysis
#
# Logs: logs/<step>_<timestamp>.log

set -uo pipefail

NUM_RUNS=1
P0_MODE=auto
P0_RUNS=1
WITH_P0_REFERENCE=0
FLIP_PROBS="0.01 0.02 0.03 0.04 0.05 0.10"
DEVICES="/dev/nvme0n1 /dev/nvme1n1"
DATA_ROOT="${HEIMDALL:-/mnt/heimdall-exp/Heimdall}/integration/client-level/data"
ANALYZE_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --num_runs) NUM_RUNS="$2"; shift 2 ;;
        --flip_probs) FLIP_PROBS="$2"; shift 2 ;;
        --p0) P0_MODE="$2"; shift 2 ;;
        --p0_runs) P0_RUNS="$2"; shift 2 ;;
        --with_p0_reference) WITH_P0_REFERENCE=1; shift ;;
        --devices) DEVICES="$2"; shift 2 ;;
        --data_root) DATA_ROOT="$2"; shift 2 ;;
        --analyze_only) ANALYZE_ONLY=1; shift ;;
        -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
export PYTHONUNBUFFERED=1

read -r -a DEV_ARR <<< "${DEVICES}"
PAIR="$(basename "${DEV_ARR[0]}")...$(basename "${DEV_ARR[1]}")"

# Trace dirs that have results for this device pair
TRACE_DIRS=()
for d in "${DATA_ROOT}"/*/*/*; do
    [ -d "${d}/${PAIR}" ] && TRACE_DIRS+=("${d}")
done

# Pre-flight: baseline must exist for every trace (FlashNet weights are checked by run_flashnet_flip.py)
MISSING=0
ROBUST_ALL=1
for d in "${TRACE_DIRS[@]}"; do
    [ -f "${d}/${PAIR}/baseline/trace_1.trace" ] || { echo "  [MISSING] baseline: ${d}/${PAIR}/baseline"; MISSING=1; }
    ls -d "${d}/${PAIR}"/flashnet/run_* >/dev/null 2>&1 || ROBUST_ALL=0
done
if [ "${#TRACE_DIRS[@]}" -eq 0 ] || [ "${MISSING}" -ne 0 ]; then
    echo "Pre-flight check failed (need a baseline replay for every trace). Aborting."
    exit 1
fi

# Where does p=0 come from?
case "${P0_MODE}" in
    auto) [ "${ROBUST_ALL}" -eq 1 ] && P0_MODE=robust || P0_MODE=single ;;
    robust)
        if [ "${ROBUST_ALL}" -ne 1 ]; then
            echo "--p0 robust needs flashnet/run_* for every trace; missing for:"
            for d in "${TRACE_DIRS[@]}"; do
                ls -d "${d}/${PAIR}"/flashnet/run_* >/dev/null 2>&1 || echo "  ${d}/${PAIR}/flashnet"
            done
            echo "Use --p0 single to add a p=0 replay instead. Aborting."
            exit 1
        fi ;;
    single) ;;
    *) echo "--p0 must be auto, robust or single"; exit 1 ;;
esac
if [ "${P0_MODE}" = "single" ]; then
    P0_SOURCE=flip
    FLIP_PROBS="0.0 ${FLIP_PROBS}"
    P0_DESC="${P0_RUNS} added p=0 replay(s) in flashnet_flip_p00 (flips off)"
else
    P0_SOURCE=robust
    [ "${WITH_P0_REFERENCE}" -eq 1 ] && FLIP_PROBS="0.0 ${FLIP_PROBS}"
    P0_DESC="existing flashnet/run_*"
fi

echo "MSI overnight run started at $(date '+%F %T') on $(hostname)"
echo "devices=${DEVICES} pair=${PAIR} traces=${#TRACE_DIRS[@]} num_runs=${NUM_RUNS} flip_probs=${FLIP_PROBS}"
echo "p=0 source: ${P0_DESC}"

SUMMARY=()
run_step() {
    local name="$1"; shift
    local log="${LOG_DIR}/${name}_$(date +%Y%m%d_%H%M%S).log"
    local start=$(date +%s)
    echo "=== [$(date '+%F %T')] START ${name} -> ${log}"
    "$@" 2>&1 | tee "${log}"
    local status=${PIPESTATUS[0]}
    local dur=$(( $(date +%s) - start ))
    local result="OK"
    [ "${status}" -ne 0 ] && result="FAILED (exit ${status})"
    echo "=== [$(date '+%F %T')] END ${name}: ${result}"
    SUMMARY+=("$(printf '%-10s %-18s %dh%02dm%02ds  %s' "${name}" "${result}" $((dur / 3600)) $(( (dur % 3600) / 60 )) $((dur % 60)) "${log}")")
}

if [ "${ANALYZE_ONLY}" -eq 0 ]; then
    sudo -n true || { echo "sudo (without a password prompt) is required for replays. Aborting."; exit 1; }
    # Keep the sudo timestamp fresh for the whole night
    ( while true; do sudo -n true; sleep 240; done ) 2>/dev/null &
    SUDO_KEEPALIVE=$!
    trap 'kill ${SUDO_KEEPALIVE} 2>/dev/null' EXIT

    read -r -a PROBS_ARR <<< "${FLIP_PROBS}"
    run_step replay python3 "${SCRIPT_DIR}/run_flashnet_flip.py" -devices "${DEV_ARR[@]}" \
        -trace_dirs "${TRACE_DIRS[@]}" -flip_probs "${PROBS_ARR[@]}" -num_runs "${NUM_RUNS}" \
        -p0_runs "${P0_RUNS}" -resume
fi

run_step analyze python3 "${SCRIPT_DIR}/analyze_msi.py" -data_root "${DATA_ROOT}" -dev_pair "${PAIR}" \
    -trace_dirs "${TRACE_DIRS[@]}" -p0_source "${P0_SOURCE}"

echo
echo "================ Summary ($(date '+%F %T')) ================"
printf '%s\n' "${SUMMARY[@]}"
for line in "${SUMMARY[@]}"; do
    [[ "${line}" == *FAILED* ]] && exit 1
done
exit 0
