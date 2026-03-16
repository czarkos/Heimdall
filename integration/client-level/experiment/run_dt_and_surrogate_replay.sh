#!/bin/bash

set -euo pipefail

# Wrapper to run replay for BOTH:
#   1) dt (existing pipeline)
#   2) surrogate_dt (fidelity-based surrogate pipeline)
#
# It first validates that both pipelines write to distinct directories:
#   replay outputs:
#     <trace_dir>/<dev0>...<dev1>/dt
#     <trace_dir>/<dev0>...<dev1>/surrogate_dt
#   model artifacts:
#     <trace_dir>/<dev0>...<dev1>/dt/training_results/dt_weights_header
#     <trace_dir>/<dev0>...<dev1>/surrogate_dt/training_results/surrogate_headers

if [ $# -lt 3 ]; then
    echo "Usage: $0 /dev/dev0 /dev/dev1 trace_dir1 [trace_dir2 ...]"
    echo "Example: $0 /dev/nvme0n1 /dev/nvme2n1 /mnt/.../data/*/*/*"
    exit 1
fi

DEV0="$1"
DEV1="$2"
shift 2
TRACE_DIRS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DEV0_NAME="$(basename "${DEV0}")"
DEV1_NAME="$(basename "${DEV1}")"
DEV_PAIR="${DEV0_NAME}...${DEV1_NAME}"

echo "Devices:"
echo "  - ${DEV0}"
echo "  - ${DEV1}"
echo "Device-pair key: ${DEV_PAIR}"
echo "Trace dirs count: ${#TRACE_DIRS[@]}"

echo
echo "[1/3] Validating dt vs surrogate_dt directory separation..."
for TRACE_DIR in "${TRACE_DIRS[@]}"; do
    DT_REPLAY_DIR="${TRACE_DIR}/${DEV_PAIR}/dt"
    SUR_REPLAY_DIR="${TRACE_DIR}/${DEV_PAIR}/surrogate_dt"

    DT_HEADER_DIR="${TRACE_DIR}/${DEV_PAIR}/dt/training_results/dt_weights_header"
    SUR_HEADER_DIR="${TRACE_DIR}/${DEV_PAIR}/surrogate_dt/training_results/surrogate_headers"

    if [ "${DT_REPLAY_DIR}" = "${SUR_REPLAY_DIR}" ]; then
        echo "ERROR: Replay output directories collide for ${TRACE_DIR}"
        echo "  dt:          ${DT_REPLAY_DIR}"
        echo "  surrogate_dt:${SUR_REPLAY_DIR}"
        exit 1
    fi

    if [ "${DT_HEADER_DIR}" = "${SUR_HEADER_DIR}" ]; then
        echo "ERROR: Header directories collide for ${TRACE_DIR}"
        echo "  dt headers:          ${DT_HEADER_DIR}"
        echo "  surrogate_dt headers:${SUR_HEADER_DIR}"
        exit 1
    fi
done
echo "Separation checks passed."

echo
echo "[2/3] Running DT replay pipeline..."
python3 ./run_dt.py \
    -devices "${DEV0}" "${DEV1}" \
    -trace_dirs "${TRACE_DIRS[@]}" \
    -only_replaying

echo
echo "[3/3] Running surrogate_dt replay pipeline..."
python3 ./run_surrogate_dt.py \
    -devices "${DEV0}" "${DEV1}" \
    -trace_dirs "${TRACE_DIRS[@]}" \
    -only_replaying

echo
echo "Done. Replay completed for both 'dt' and 'surrogate_dt'."

