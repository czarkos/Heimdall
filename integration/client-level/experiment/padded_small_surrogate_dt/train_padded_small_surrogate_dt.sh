#!/bin/bash

set -euo pipefail

# Train a small-depth (fidelity) surrogate DT for padded_small_surrogate_dt pipeline.
# Uses train_small_surrogate_dt_fidelity.py to distill FlashNet into a shallow tree.
#
# Output layout:
#   <trace_dir>/<dev0>...<dev1>/padded_small_surrogate_dt/training_results/
#     surrogate_headers/w_Trace_dev_{0,1}_dt.h

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

MAX_DEPTH="${PADDED_SMALL_SURROGATE_MAX_DEPTH:-5}"
ALGO_NAME="padded_small_surrogate_dt"

TRAIN_SCRIPT="${SCRIPT_DIR}/../small_surrogate_dt/train_small_surrogate_dt_fidelity.py"

for TRACE_DIR in "$@"; do
    FLASHNET_RESULTS_DIR="${TRACE_DIR}/${DEV0}...${DEV1}/flashnet/training_results"
    TRAINING_ROOT="${TRACE_DIR}/${DEV0}...${DEV1}/${ALGO_NAME}/training_results"

    DATASET0="${TRAINING_ROOT}/mldrive0.csv"
    DATASET1="${TRAINING_ROOT}/mldrive1.csv"

    echo
    echo "======================================================="
    echo "Training padded small surrogate DT (depth=${MAX_DEPTH}):"
    echo "  Trace root dir    => ${TRACE_DIR}"
    echo "  FlashNet data dir => ${FLASHNET_RESULTS_DIR}"
    echo "  Output dir        => ${TRAINING_ROOT}"
    echo "  max_depth         => ${MAX_DEPTH}"
    echo "======================================================="

    mkdir -p "${TRAINING_ROOT}"

    if [ ! -f "${DATASET0}" ] && [ -f "${FLASHNET_RESULTS_DIR}/mldrive0.csv" ]; then
        cp "${FLASHNET_RESULTS_DIR}/mldrive0.csv" "${DATASET0}"
    fi
    if [ ! -f "${DATASET1}" ] && [ -f "${FLASHNET_RESULTS_DIR}/mldrive1.csv" ]; then
        cp "${FLASHNET_RESULTS_DIR}/mldrive1.csv" "${DATASET1}"
    fi

    for DRIVE_ID in 0 1; do
        for LAYER_ID in 0 1 2 3; do
            SRC_W="${FLASHNET_RESULTS_DIR}/mldrive${DRIVE_ID}.csv.weight_${LAYER_ID}.csv"
            SRC_B="${FLASHNET_RESULTS_DIR}/mldrive${DRIVE_ID}.csv.bias_${LAYER_ID}.csv"
            DST_W="${TRAINING_ROOT}/mldrive${DRIVE_ID}.csv.weight_${LAYER_ID}.csv"
            DST_B="${TRAINING_ROOT}/mldrive${DRIVE_ID}.csv.bias_${LAYER_ID}.csv"
            if [ ! -f "${DST_W}" ] && [ -f "${SRC_W}" ]; then
                cp "${SRC_W}" "${DST_W}"
            fi
            if [ ! -f "${DST_B}" ] && [ -f "${SRC_B}" ]; then
                cp "${SRC_B}" "${DST_B}"
            fi
        done
    done

    if [ ! -f "${DATASET0}" ] || [ ! -f "${DATASET1}" ]; then
        echo "  [SKIP] Missing datasets under ${TRAINING_ROOT}"
        continue
    fi

    OUT_DIR="${TRAINING_ROOT}/surrogate_headers"
    mkdir -p "${OUT_DIR}"

    echo "  -> Training small surrogate DT header for dev_0 (depth=${MAX_DEPTH})"
    python3 "${TRAIN_SCRIPT}" \
        -target flashnet \
        -dataset "${DATASET0}" \
        -workload Trace \
        -drive dev_0 \
        -output_dir "${OUT_DIR}" \
        -max_depth "${MAX_DEPTH}"

    echo "  -> Training small surrogate DT header for dev_1 (depth=${MAX_DEPTH})"
    python3 "${TRAIN_SCRIPT}" \
        -target flashnet \
        -dataset "${DATASET1}" \
        -workload Trace \
        -drive dev_1 \
        -output_dir "${OUT_DIR}" \
        -max_depth "${MAX_DEPTH}"

    echo "  Surrogate headers generated at ${OUT_DIR}"
done
