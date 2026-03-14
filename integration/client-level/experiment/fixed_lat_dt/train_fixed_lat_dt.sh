#!/bin/bash

set -euo pipefail

# Train a standard DT model for fixed_lat_dt pipeline.
# Output layout:
#   <trace_dir>/<dev0>...<dev1>/fixed_lat_dt/training_results/
#   <trace_dir>/<dev0>...<dev1>/fixed_lat_dt/training_results/fixed_lat_headers/

if [ $# -lt 3 ]; then
    echo "Usage: $0 device0 device1 dir_to_replayed_traces [more_dirs...]"
    echo "Example: $0 nvme0n1 nvme2n1 /mnt/.../data/*/*/*"
    exit 1
fi

DEV0="$1"
DEV1="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

for TRACE_DIR in "$@"; do
    TRAINING_RESULTS_DIR="${TRACE_DIR}/${DEV0}...${DEV1}/fixed_lat_dt/training_results"
    FLASHNET_TRAINING_RESULTS_DIR="${TRACE_DIR}/${DEV0}...${DEV1}/flashnet/training_results"

    DATASET0="${TRAINING_RESULTS_DIR}/mldrive0.csv"
    DATASET1="${TRAINING_RESULTS_DIR}/mldrive1.csv"

    echo
    echo "======================================================="
    echo "Training fixed-latency DT for workload:"
    echo "  Trace root dir    => ${TRACE_DIR}"
    echo "  Output dir        => ${TRAINING_RESULTS_DIR}"
    echo "======================================================="

    mkdir -p "${TRAINING_RESULTS_DIR}"

    # Prefer local fixed_lat_dt datasets. If absent, backfill from flashnet.
    if [ ! -f "${DATASET0}" ] && [ -f "${FLASHNET_TRAINING_RESULTS_DIR}/mldrive0.csv" ]; then
        cp "${FLASHNET_TRAINING_RESULTS_DIR}/mldrive0.csv" "${DATASET0}"
    fi
    if [ ! -f "${DATASET1}" ] && [ -f "${FLASHNET_TRAINING_RESULTS_DIR}/mldrive1.csv" ]; then
        cp "${FLASHNET_TRAINING_RESULTS_DIR}/mldrive1.csv" "${DATASET1}"
    fi

    if [ ! -f "${DATASET0}" ] || [ ! -f "${DATASET1}" ]; then
        echo "  [SKIP] Missing datasets under ${TRAINING_RESULTS_DIR}"
        continue
    fi

    OUT_DIR="${TRAINING_RESULTS_DIR}/fixed_lat_headers"
    mkdir -p "${OUT_DIR}"

    echo "  -> Training DT header for dev_0"
    python3 "${SCRIPT_DIR}/../dt/train_dt.py" \
        -dataset "${DATASET0}" \
        -workload Trace \
        -drive dev_0 \
        -output_dir "${OUT_DIR}"

    echo "  -> Training DT header for dev_1"
    python3 "${SCRIPT_DIR}/../dt/train_dt.py" \
        -dataset "${DATASET1}" \
        -workload Trace \
        -drive dev_1 \
        -output_dir "${OUT_DIR}"

    echo "  Headers generated at ${OUT_DIR}"
done

