#!/bin/bash

set -euo pipefail

# Usage:
#   ./train_small_hierarchy.sh dev0 dev1 dir1 [dir2 ...]
#
# Environment overrides:
#   SMALL_HIERARCHY_ALGO=small_hierarchy_p95|small_hierarchy_p97|small_hierarchy_p98
#   SMALL_HIERARCHY_TAU_PERCENTILE=95.0|97.0|98.0
#   SMALL_HIERARCHY_REF_SIZE=512
#   SMALL_HIERARCHY_SEED=42
#   SMALL_SURROGATE_MAX_DEPTH=10

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

ALGO_NAME="${SMALL_HIERARCHY_ALGO:-small_hierarchy_p95}"
TAU_PERCENTILE="${SMALL_HIERARCHY_TAU_PERCENTILE:-95.0}"
REF_SIZE="${SMALL_HIERARCHY_REF_SIZE:-512}"
SEED="${SMALL_HIERARCHY_SEED:-42}"
MAX_DEPTH="${SMALL_SURROGATE_MAX_DEPTH:-10}"

for TRACE_DIR in "$@"; do
    FLASHNET_RESULTS_DIR="${TRACE_DIR}/${DEV0}...${DEV1}/flashnet/training_results"
    TRAINING_ROOT="${TRACE_DIR}/${DEV0}...${DEV1}/${ALGO_NAME}/training_results"

    DATASET0="${TRAINING_ROOT}/mldrive0.csv"
    DATASET1="${TRAINING_ROOT}/mldrive1.csv"

    HIER_DT_DIR="${TRAINING_ROOT}/hierarchy_headers"
    UNCERT_DIR="${TRAINING_ROOT}/uncertainty_headers"
    mkdir -p "${HIER_DT_DIR}" "${UNCERT_DIR}"

    METRICS0_JSON="${TRAINING_ROOT}/${ALGO_NAME}_dev_0_fidelity.json"
    METRICS1_JSON="${TRAINING_ROOT}/${ALGO_NAME}_dev_1_fidelity.json"
    STATS0_PATH="${TRAINING_ROOT}/${ALGO_NAME}_dev_0_training.stats"
    STATS1_PATH="${TRAINING_ROOT}/${ALGO_NAME}_dev_1_training.stats"
    STATS_PATH="${TRAINING_ROOT}/${ALGO_NAME}_training.stats"

    echo
    echo "======================================================="
    echo "Training ${ALGO_NAME} artifacts:"
    echo "  Trace root dir    => ${TRACE_DIR}"
    echo "  FlashNet data dir => ${FLASHNET_RESULTS_DIR}"
    echo "  Output dir        => ${TRAINING_ROOT}"
    echo "  max_depth         => ${MAX_DEPTH}"
    echo "  ref_size          => ${REF_SIZE}"
    echo "  tau_percentile    => ${TAU_PERCENTILE}"
    echo "======================================================="

    mkdir -p "${TRAINING_ROOT}"

    # Keep local training snapshot under this algorithm's training_results.
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

    echo "  -> Training small-surrogate hierarchy tree for dev_0"
    python3 "${SCRIPT_DIR}/train_small_surrogate_dt_fidelity.py" \
        -target flashnet \
        -dataset "${DATASET0}" \
        -workload Trace \
        -drive dev_0 \
        -output_dir "${HIER_DT_DIR}" \
        -max_depth "${MAX_DEPTH}" \
        -metrics_output "${METRICS0_JSON}" \
        -stats_output "${STATS0_PATH}"

    echo "  -> Training small-surrogate hierarchy tree for dev_1"
    python3 "${SCRIPT_DIR}/train_small_surrogate_dt_fidelity.py" \
        -target flashnet \
        -dataset "${DATASET1}" \
        -workload Trace \
        -drive dev_1 \
        -output_dir "${HIER_DT_DIR}" \
        -max_depth "${MAX_DEPTH}" \
        -metrics_output "${METRICS1_JSON}" \
        -stats_output "${STATS1_PATH}"

    echo "  -> Exporting uncertainty header for dev_0"
    python3 "${SCRIPT_DIR}/small_hierarchy/train_uncertainty_header.py" \
        -dataset "${DATASET0}" \
        -drive dev_0 \
        -output_dir "${UNCERT_DIR}" \
        -ref_size "${REF_SIZE}" \
        -tau_percentile "${TAU_PERCENTILE}" \
        -seed "${SEED}"

    echo "  -> Exporting uncertainty header for dev_1"
    python3 "${SCRIPT_DIR}/small_hierarchy/train_uncertainty_header.py" \
        -dataset "${DATASET1}" \
        -drive dev_1 \
        -output_dir "${UNCERT_DIR}" \
        -ref_size "${REF_SIZE}" \
        -tau_percentile "${TAU_PERCENTILE}" \
        -seed "${SEED}"

    python3 - "${METRICS0_JSON}" "${METRICS1_JSON}" "${STATS_PATH}" "${TRACE_DIR}" "${ALGO_NAME}" "${MAX_DEPTH}" "${REF_SIZE}" "${TAU_PERCENTILE}" "${SEED}" <<'PY'
import json
import os
import sys
from datetime import datetime

(metrics0_path, metrics1_path, stats_path, trace_dir, algo_name,
 max_depth, ref_size, tau_percentile, seed) = sys.argv[1:]

with open(metrics0_path, "r") as f:
    m0 = json.load(f)
with open(metrics1_path, "r") as f:
    m1 = json.load(f)

lines = [
    "========== Small Hierarchy Training Stats ==========",
    f"generated_on = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"trace_dir = {trace_dir}",
    f"algo_name = {algo_name}",
    f"small_surrogate_max_depth = {max_depth}",
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
    "====================================================",
]

os.makedirs(os.path.dirname(stats_path), exist_ok=True)
with open(stats_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"  small hierarchy training stats written to: {stats_path}")
PY

    echo "  hierarchy headers:   ${HIER_DT_DIR}"
    echo "  uncertainty headers: ${UNCERT_DIR}"
    echo "  training stats:      ${STATS_PATH}"
done

