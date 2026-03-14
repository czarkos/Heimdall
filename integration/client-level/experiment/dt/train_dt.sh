#!/bin/bash

set -euo pipefail

# Canonical entrypoint for standard DT training.
# Stores model artifacts under:
#   <trace_dir>/<dev0>...<dev1>/dt/training_results/

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
    TRAINING_RESULTS_DIR="${TRACE_DIR}/${DEV0}...${DEV1}/dt/training_results"
    FLASHNET_TRAINING_RESULTS_DIR="${TRACE_DIR}/${DEV0}...${DEV1}/flashnet/training_results"

    DATASET0="${TRAINING_RESULTS_DIR}/mldrive0.csv"
    DATASET1="${TRAINING_RESULTS_DIR}/mldrive1.csv"

    echo
    echo "======================================================="
    echo "Training standard DT for workload:"
    echo "  Trace root dir    => ${TRACE_DIR}"
    echo "  DT results dir    => ${TRAINING_RESULTS_DIR}"
    echo "  Dataset dev_0     => ${DATASET0}"
    echo "  Dataset dev_1     => ${DATASET1}"
    echo "======================================================="

    mkdir -p "${TRAINING_RESULTS_DIR}"

    # Safety guard: DT outputs must never be written under flashnet.
    case "${TRAINING_RESULTS_DIR}" in
        *"/flashnet/"*)
            echo "  [ERROR] Refusing to write DT outputs under flashnet path: ${TRAINING_RESULTS_DIR}"
            exit 1
            ;;
    esac

    # Backfill DT datasets from FlashNet if needed (read-only source).
    if [ ! -f "${DATASET0}" ] && [ -f "${FLASHNET_TRAINING_RESULTS_DIR}/mldrive0.csv" ]; then
        cp "${FLASHNET_TRAINING_RESULTS_DIR}/mldrive0.csv" "${DATASET0}"
        echo "  Copied fallback dataset: ${FLASHNET_TRAINING_RESULTS_DIR}/mldrive0.csv -> ${DATASET0}"
    fi
    if [ ! -f "${DATASET1}" ] && [ -f "${FLASHNET_TRAINING_RESULTS_DIR}/mldrive1.csv" ]; then
        cp "${FLASHNET_TRAINING_RESULTS_DIR}/mldrive1.csv" "${DATASET1}"
        echo "  Copied fallback dataset: ${FLASHNET_TRAINING_RESULTS_DIR}/mldrive1.csv -> ${DATASET1}"
    fi

    if [ ! -f "${DATASET0}" ]; then
        echo "  [SKIP] Dataset for dev_0 not found: ${DATASET0}"
        continue
    fi
    if [ ! -f "${DATASET1}" ]; then
        echo "  [SKIP] Dataset for dev_1 not found: ${DATASET1}"
        continue
    fi

    OUT_DIR="${TRAINING_RESULTS_DIR}/dt_weights_header"
    mkdir -p "${OUT_DIR}"
    case "${OUT_DIR}" in
        *"/flashnet/"*)
            echo "  [ERROR] Refusing to write DT headers under flashnet path: ${OUT_DIR}"
            exit 1
            ;;
    esac

    echo
    echo "  -> Training DT for dev_0"
    python3 "${SCRIPT_DIR}/train_dt.py" \
        -dataset "${DATASET0}" \
        -workload Trace \
        -drive dev_0 \
        -output_dir "${OUT_DIR}"

    echo
    echo "  -> Training DT for dev_1"
    python3 "${SCRIPT_DIR}/train_dt.py" \
        -dataset "${DATASET1}" \
        -workload Trace \
        -drive dev_1 \
        -output_dir "${OUT_DIR}"

    echo
    echo "  DT headers generated under:"
    echo "    ${OUT_DIR}"
    echo "    (w_Trace_dev_0_dt.h, w_Trace_dev_1_dt.h)"
done

