#!/usr/bin/env bash
# FlashNet surrogate DT of a fixed depth: train once, replay N times, then generate its latency stats.
# The algorithm name is this script's directory name (surrogate_dt_depth<N>), and it calls the
# run_<dirname>.py wrapper next to it, so the identical file is used in every surrogate_dt_depth*/ dir.
# Meant to run inside tmux:
#   tmux new -s dt
#   ./run_overnight.sh                  # train + 1 replay per trace (~3.5 min per replay => ~20 min for 6 traces)
#   ./run_overnight.sh --num_runs 10    # robust: 10 repeats (~3.5 h); a later larger N only adds the missing runs
#
# Options:
#   --num_runs N        replay repetitions per trace (default 1)
#   --only_training     train only
#   --only_replaying    skip training (headers must already exist)
#   --skip_stats        don't run the latency-stats step at the end
#   --devices "..."     default "/dev/nvme0n1 /dev/nvme1n1"
#   --data_root DIR     default $HEIMDALL/integration/client-level/data
#
# Steps are logged to logs/<step>_<timestamp>.log; a failed step is reported, the next one still runs.

set -uo pipefail

NUM_RUNS=1
MODE_FLAGS=()
SKIP_STATS=0
DEVICES="/dev/nvme0n1 /dev/nvme1n1"
DATA_ROOT="${HEIMDALL:-/mnt/heimdall-exp/Heimdall}/integration/client-level/data"

while [ $# -gt 0 ]; do
    case "$1" in
        --num_runs) NUM_RUNS="$2"; shift 2 ;;
        --only_training) MODE_FLAGS+=(-only_training); shift ;;
        --only_replaying) MODE_FLAGS+=(-only_replaying); shift ;;
        --skip_stats) SKIP_STATS=1; shift ;;
        --devices) DEVICES="$2"; shift 2 ;;
        --data_root) DATA_ROOT="$2"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_LEVEL_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
ALGO_NAME="$(basename "${SCRIPT_DIR}")"            # e.g. surrogate_dt_depth11
WRAPPER="${SCRIPT_DIR}/run_${ALGO_NAME}.py"
if [[ ! "${ALGO_NAME}" =~ ^surrogate_dt_depth[0-9]+$ ]] || [ ! -f "${WRAPPER}" ]; then
    echo "This script must live in a surrogate_dt_depth<N>/ directory next to run_surrogate_dt_depth<N>.py (got: ${SCRIPT_DIR})"
    exit 1
fi
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

# Pre-flight: the surrogate is distilled from FlashNet's predictions, so each trace needs the
# baseline replay (trace stats are read for the replay duration) and FlashNet's training results:
# the dataset mldrive<i>.csv plus the exported teacher weights mldrive<i>.csv.{weight,bias}_{0..3}.csv.
MISSING=0
for d in "${TRACE_DIRS[@]}"; do
    p="${d}/${PAIR}"
    [ -f "${p}/baseline/trace_1.trace" ] || { echo "  [MISSING] baseline replay: ${p}/baseline"; MISSING=1; }
    for i in 0 1; do
        ds="${p}/flashnet/training_results/mldrive${i}.csv"
        [ -f "${ds}" ] || { echo "  [MISSING] FlashNet dataset: ${ds}"; MISSING=1; }
        for k in 0 1 2 3; do
            [ -f "${ds}.weight_${k}.csv" ] && [ -f "${ds}.bias_${k}.csv" ] || { echo "  [MISSING] FlashNet teacher weights: ${ds}.{weight,bias}_${k}.csv"; MISSING=1; }
        done
    done
done
if [ "${#TRACE_DIRS[@]}" -eq 0 ] || [ "${MISSING}" -ne 0 ]; then
    echo "Pre-flight check failed. Aborting."
    exit 1
fi
if [[ ! " ${MODE_FLAGS[*]} " =~ " -only_training " ]]; then
    sudo -n true || { echo "sudo (without a password prompt) is required for replays. Aborting."; exit 1; }
fi

echo "${ALGO_NAME} overnight run started at $(date '+%F %T') on $(hostname)"
echo "devices=${DEVICES} pair=${PAIR} traces=${#TRACE_DIRS[@]} num_runs=${NUM_RUNS} mode=${MODE_FLAGS[*]:-train+replay}"

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
    SUMMARY+=("$(printf '%-14s %-18s %dh%02dm%02ds  %s' "${name}" "${result}" $((dur / 3600)) $(( (dur % 3600) / 60 )) $((dur % 60)) "${log}")")
}

run_step train_replay python3 "${WRAPPER}" -devices "${DEV_ARR[@]}" \
    -trace_dirs "${TRACE_DIRS[@]}" -num_runs "${NUM_RUNS}" -resume "${MODE_FLAGS[@]}"

# Latency stats for this algorithm only (mean/std across run_* dirs, or the single replay when
# num_runs=1, which the replayer writes directly into <algo>/trace_{1,2}.trace).
if [ "${SKIP_STATS}" -eq 0 ] && [[ ! " ${MODE_FLAGS[*]} " =~ " -only_training " ]]; then
    ALGO_DIRS=()
    for d in "${TRACE_DIRS[@]}"; do
        a="${d}/${PAIR}/${ALGO_NAME}"
        if ls -d "${a}"/run_* >/dev/null 2>&1 || [ -f "${a}/trace_1.trace.stats" ]; then
            ALGO_DIRS+=("${a}")
        fi
    done
    run_step stats python3 -c '
import sys
sys.path.insert(0, sys.argv[1])
from generate_latency_stats_with_dt import start_process_per_algo
for algo_dir in sys.argv[2:]:
    print("Processing " + algo_dir, flush=True)
    start_process_per_algo(algo_dir)
' "${CLIENT_LEVEL_DIR}/algo_analysis" "${ALGO_DIRS[@]}"
fi

echo
echo "================ Summary ($(date '+%F %T')) ================"
printf '%s\n' "${SUMMARY[@]}"
for line in "${SUMMARY[@]}"; do
    [[ "${line}" == *FAILED* ]] && exit 1
done
exit 0
