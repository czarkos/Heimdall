#!/bin/bash

set -euo pipefail

# Usage:
#   ./train_surrogate_dt_fidelity.sh dev0 dev1 dir1 [dir2 dir3 ...]
#
# For each dirX, expects FlashNet training outputs:
#   dirX/<dev0>...<dev1>/flashnet/training_results/mldrive0.csv
#   dirX/<dev0>...<dev1>/flashnet/training_results/mldrive1.csv
# plus CSV weight exports next to each dataset:
#   mldrive*.csv.weight_{0,1,2,3}.csv and mldrive*.csv.bias_{0,1,2,3}.csv
#
# Output headers:
#   dirX/<dev0>...<dev1>/surrogate_dt/training_results/surrogate_headers/
#     w_Trace_dev_0_dt.h
#     w_Trace_dev_1_dt.h

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
    TRAINING_RESULTS_DIR="${TRACE_DIR}/${DEV0}...${DEV1}/flashnet/training_results"
    DATASET0="${TRAINING_RESULTS_DIR}/mldrive0.csv"
    DATASET1="${TRAINING_RESULTS_DIR}/mldrive1.csv"

    echo
    echo "======================================================="
    echo "Training fidelity surrogate DT for workload:"
    echo "  Trace root dir    => ${TRACE_DIR}"
    echo "  Training results  => ${TRAINING_RESULTS_DIR}"
    echo "  Dataset dev_0     => ${DATASET0}"
    echo "  Dataset dev_1     => ${DATASET1}"
    echo "======================================================="

    if [ ! -f "${DATASET0}" ]; then
        echo "  [SKIP] Dataset for dev_0 not found: ${DATASET0}"
        continue
    fi
    if [ ! -f "${DATASET1}" ]; then
        echo "  [SKIP] Dataset for dev_1 not found: ${DATASET1}"
        continue
    fi

    OUT_DIR="${TRACE_DIR}/${DEV0}...${DEV1}/surrogate_dt/training_results/surrogate_headers"
    mkdir -p "${OUT_DIR}"

    echo
    echo "  -> Training fidelity surrogate for dev_0"
    python3 "${SCRIPT_DIR}/train_surrogate_dt_fidelity.py" \
        -target flashnet \
        -dataset "${DATASET0}" \
        -workload Trace \
        -drive dev_0 \
        -output_dir "${OUT_DIR}"

    echo
    echo "  -> Training fidelity surrogate for dev_1"
    python3 "${SCRIPT_DIR}/train_surrogate_dt_fidelity.py" \
        -target flashnet \
        -dataset "${DATASET1}" \
        -workload Trace \
        -drive dev_1 \
        -output_dir "${OUT_DIR}"

    echo
    echo "  Fidelity surrogate headers generated under:"
    echo "    ${OUT_DIR}"
    echo "    (w_Trace_dev_0_dt.h, w_Trace_dev_1_dt.h)"
done

