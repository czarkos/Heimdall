#!/usr/bin/env bash
# Train (if needed) and replay the FlashNet surrogate DTs of several depths, one after another
# (they share the SSDs, so never in parallel), then generate each one's latency stats.
# Each depth lives in experiment/surrogate_dt_depth<N>/ and is driven by its own run_overnight.sh.
#
#   tmux new -s sweep
#   ./run_surrogate_depth_sweep.sh                      # depths 6 7 8 11, 1 replay per trace (~20 min per depth)
#   ./run_surrogate_depth_sweep.sh --depths "6 8"       # a subset
#   ./run_surrogate_depth_sweep.sh --num_runs 10        # robust runs instead
#   ./run_surrogate_depth_sweep.sh --only_training      # just train every depth (CPU only, no SSD access)
#
# All other options (--only_replaying, --skip_stats, --devices, --data_root) are passed through to
# each depth's run_overnight.sh. Everything is resumable: finished training and replays are skipped.

set -uo pipefail

DEPTHS="6 7 8 11"
PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --depths) DEPTHS="$2"; shift 2 ;;
        --num_runs|--devices|--data_root) PASS+=("$1" "$2"); shift 2 ;;
        --only_training|--only_replaying|--skip_stats) PASS+=("$1"); shift ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs_surrogate_depth_sweep"
mkdir -p "${LOG_DIR}"

for N in ${DEPTHS}; do
    [ -x "${SCRIPT_DIR}/surrogate_dt_depth${N}/run_overnight.sh" ] || { echo "Missing ${SCRIPT_DIR}/surrogate_dt_depth${N}/run_overnight.sh"; exit 1; }
done

echo "Surrogate depth sweep started at $(date '+%F %T'): depths = ${DEPTHS}, options = ${PASS[*]:-defaults}"
SUMMARY=()
for N in ${DEPTHS}; do
    log="${LOG_DIR}/depth${N}_$(date +%Y%m%d_%H%M%S).log"
    start=$(date +%s)
    echo
    echo "##### [$(date '+%F %T')] depth ${N} -> ${log}"
    "${SCRIPT_DIR}/surrogate_dt_depth${N}/run_overnight.sh" "${PASS[@]}" 2>&1 | tee "${log}"
    status=${PIPESTATUS[0]}
    dur=$(( $(date +%s) - start ))
    result="OK"; [ "${status}" -ne 0 ] && result="FAILED (exit ${status})"
    SUMMARY+=("$(printf 'depth %-3s %-18s %dh%02dm%02ds  %s' "${N}" "${result}" $((dur / 3600)) $(( (dur % 3600) / 60 )) $((dur % 60)) "${log}")")
done

echo
echo "================ Sweep summary ($(date '+%F %T')) ================"
printf '%s\n' "${SUMMARY[@]}"
for line in "${SUMMARY[@]}"; do
    [[ "${line}" == *FAILED* ]] && exit 1
done
exit 0
