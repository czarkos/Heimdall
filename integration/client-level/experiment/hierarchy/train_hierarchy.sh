#!/bin/bash

set -euo pipefail

# Train hierarchy artifacts:
#  1) Surrogate DT headers (trained to mimic FlashNet outputs)
#  2) uncertainty headers (kNN-distance in DT feature space)
#
# Output layout:
#   <trace_dir>/<dev0>...<dev1>/hierarchy/training_results/
#     - hierarchy_headers/w_Trace_dev_{0,1}_dt.h
#     - uncertainty_headers/u_Trace_dev_{0,1}_uncert.h
#     - hierarchy_training.stats (includes surrogate fidelity summary)

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

REF_SIZE="${HIERARCHY_REF_SIZE:-512}"
TAU_PERCENTILE="${HIERARCHY_TAU_PERCENTILE:-95.0}"
SEED="${HIERARCHY_SEED:-42}"

for TRACE_DIR in "$@"; do
    TRAINING_RESULTS_DIR="${TRACE_DIR}/${DEV0}...${DEV1}/hierarchy/training_results"
    FLASHNET_TRAINING_RESULTS_DIR="${TRACE_DIR}/${DEV0}...${DEV1}/flashnet/training_results"

    DATASET0="${TRAINING_RESULTS_DIR}/mldrive0.csv"
    DATASET1="${TRAINING_RESULTS_DIR}/mldrive1.csv"

    echo
    echo "======================================================="
    echo "Training hierarchy model artifacts:"
    echo "  Trace root dir    => ${TRACE_DIR}"
    echo "  Output dir        => ${TRAINING_RESULTS_DIR}"
    echo "  ref_size          => ${REF_SIZE}"
    echo "  tau_percentile    => ${TAU_PERCENTILE}"
    echo "======================================================="

    mkdir -p "${TRAINING_RESULTS_DIR}"

    # Prefer local hierarchy datasets. If absent, backfill from flashnet.
    if [ ! -f "${DATASET0}" ] && [ -f "${FLASHNET_TRAINING_RESULTS_DIR}/mldrive0.csv" ]; then
        cp "${FLASHNET_TRAINING_RESULTS_DIR}/mldrive0.csv" "${DATASET0}"
    fi
    if [ ! -f "${DATASET1}" ] && [ -f "${FLASHNET_TRAINING_RESULTS_DIR}/mldrive1.csv" ]; then
        cp "${FLASHNET_TRAINING_RESULTS_DIR}/mldrive1.csv" "${DATASET1}"
    fi

    # Surrogate fidelity training requires FlashNet exported weights/biases
    # next to each mldrive*.csv file:
    #   mldriveX.csv.weight_{0..3}.csv
    #   mldriveX.csv.bias_{0..3}.csv
    for DRIVE_ID in 0 1; do
        for LAYER_ID in 0 1 2 3; do
            SRC_W="${FLASHNET_TRAINING_RESULTS_DIR}/mldrive${DRIVE_ID}.csv.weight_${LAYER_ID}.csv"
            SRC_B="${FLASHNET_TRAINING_RESULTS_DIR}/mldrive${DRIVE_ID}.csv.bias_${LAYER_ID}.csv"
            DST_W="${TRAINING_RESULTS_DIR}/mldrive${DRIVE_ID}.csv.weight_${LAYER_ID}.csv"
            DST_B="${TRAINING_RESULTS_DIR}/mldrive${DRIVE_ID}.csv.bias_${LAYER_ID}.csv"

            if [ ! -f "${DST_W}" ] && [ -f "${SRC_W}" ]; then
                cp "${SRC_W}" "${DST_W}"
            fi
            if [ ! -f "${DST_B}" ] && [ -f "${SRC_B}" ]; then
                cp "${SRC_B}" "${DST_B}"
            fi
        done
    done

    if [ ! -f "${DATASET0}" ] || [ ! -f "${DATASET1}" ]; then
        echo "  [SKIP] Missing datasets under ${TRAINING_RESULTS_DIR}"
        continue
    fi

    HIER_DT_DIR="${TRAINING_RESULTS_DIR}/hierarchy_headers"
    UNCERT_DIR="${TRAINING_RESULTS_DIR}/uncertainty_headers"
    mkdir -p "${HIER_DT_DIR}" "${UNCERT_DIR}"

    METRICS0_JSON="${TRAINING_RESULTS_DIR}/hierarchy_dev_0_fidelity.json"
    METRICS1_JSON="${TRAINING_RESULTS_DIR}/hierarchy_dev_1_fidelity.json"
    STATS_PATH="${TRAINING_RESULTS_DIR}/hierarchy_training.stats"

    echo "  -> Training surrogate DT header for dev_0 (target=flashnet)"
    python3 "${SCRIPT_DIR}/../surrogate_dt/train_surrogate_dt_fidelity.py" \
        -target flashnet \
        -dataset "${DATASET0}" \
        -workload Trace \
        -drive dev_0 \
        -output_dir "${HIER_DT_DIR}" \
        -metrics_output "${METRICS0_JSON}"

    echo "  -> Training surrogate DT header for dev_1 (target=flashnet)"
    python3 "${SCRIPT_DIR}/../surrogate_dt/train_surrogate_dt_fidelity.py" \
        -target flashnet \
        -dataset "${DATASET1}" \
        -workload Trace \
        -drive dev_1 \
        -output_dir "${HIER_DT_DIR}" \
        -metrics_output "${METRICS1_JSON}"

    echo "  -> Exporting uncertainty header for dev_0"
    python3 "${SCRIPT_DIR}/train_uncertainty_header.py" \
        -dataset "${DATASET0}" \
        -drive dev_0 \
        -output_dir "${UNCERT_DIR}" \
        -ref_size "${REF_SIZE}" \
        -tau_percentile "${TAU_PERCENTILE}" \
        -seed "${SEED}"

    echo "  -> Exporting uncertainty header for dev_1"
    python3 "${SCRIPT_DIR}/train_uncertainty_header.py" \
        -dataset "${DATASET1}" \
        -drive dev_1 \
        -output_dir "${UNCERT_DIR}" \
        -ref_size "${REF_SIZE}" \
        -tau_percentile "${TAU_PERCENTILE}" \
        -seed "${SEED}"

    python3 - "${METRICS0_JSON}" "${METRICS1_JSON}" "${STATS_PATH}" "${TRACE_DIR}" "${REF_SIZE}" "${TAU_PERCENTILE}" "${SEED}" <<'PY'
import json
import os
import sys
from datetime import datetime

metrics0_path, metrics1_path, stats_path, trace_dir, ref_size, tau_percentile, seed = sys.argv[1:]

with open(metrics0_path, "r") as f:
    m0 = json.load(f)
with open(metrics1_path, "r") as f:
    m1 = json.load(f)

lines = [
    "========== Hierarchy Surrogate Training Stats ==========",
    f"generated_on = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"trace_dir = {trace_dir}",
    f"uncert_ref_size = {ref_size}",
    f"uncert_tau_percentile = {tau_percentile}",
    f"uncert_seed = {seed}",
    "",
    "[dev_0 surrogate fidelity]",
    f"train_fidelity = {m0.get('train_fidelity', float('nan')):.6f}",
    f"test_fidelity = {m0.get('test_fidelity', float('nan')):.6f}",
    f"surrogate_train_acc_gt = {m0.get('dt_train_acc_gt', float('nan')):.6f}",
    f"surrogate_test_acc_gt = {m0.get('dt_test_acc_gt', float('nan')):.6f}",
    "",
    "[dev_1 surrogate fidelity]",
    f"train_fidelity = {m1.get('train_fidelity', float('nan')):.6f}",
    f"test_fidelity = {m1.get('test_fidelity', float('nan')):.6f}",
    f"surrogate_train_acc_gt = {m1.get('dt_train_acc_gt', float('nan')):.6f}",
    f"surrogate_test_acc_gt = {m1.get('dt_test_acc_gt', float('nan')):.6f}",
    "========================================================",
]

os.makedirs(os.path.dirname(stats_path), exist_ok=True)
with open(stats_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"  hierarchy training stats written to: {stats_path}")
PY

    echo "  hierarchy headers:   ${HIER_DT_DIR}"
    echo "  uncertainty headers: ${UNCERT_DIR}"
    echo "  training stats:      ${STATS_PATH}"
done

