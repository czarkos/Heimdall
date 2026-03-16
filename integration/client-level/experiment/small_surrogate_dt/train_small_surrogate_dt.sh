#!/bin/bash

set -euo pipefail

# Usage:
#   ./train_small_surrogate_dt.sh dev0 dev1 dir1 [dir2 dir3 ...]
#
# Output (separate from surrogate_dt):
#   dirX/<dev0>...<dev1>/small_surrogate_dt/training_results/
#     - surrogate_headers/w_Trace_dev_{0,1}_dt.h
#     - small_surrogate_dev_{0,1}_metrics.json
#     - small_surrogate_dev_{0,1}_training.stats
#     - small_surrogate_training.stats

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

MAX_DEPTH="${SMALL_SURROGATE_MAX_DEPTH:-15}"
ALGO_NAME="${SMALL_SURROGATE_ALGO_NAME:-small_surrogate_dt}"

for TRACE_DIR in "$@"; do
    FLASHNET_RESULTS_DIR="${TRACE_DIR}/${DEV0}...${DEV1}/flashnet/training_results"
    DATASET0="${FLASHNET_RESULTS_DIR}/mldrive0.csv"
    DATASET1="${FLASHNET_RESULTS_DIR}/mldrive1.csv"

    TRAINING_ROOT="${TRACE_DIR}/${DEV0}...${DEV1}/${ALGO_NAME}/training_results"
    OUT_DIR="${TRAINING_ROOT}/surrogate_headers"
    METRICS0_JSON="${TRAINING_ROOT}/small_surrogate_dev_0_metrics.json"
    METRICS1_JSON="${TRAINING_ROOT}/small_surrogate_dev_1_metrics.json"
    STATS0_PATH="${TRAINING_ROOT}/small_surrogate_dev_0_training.stats"
    STATS1_PATH="${TRAINING_ROOT}/small_surrogate_dev_1_training.stats"
    COMBINED_STATS_PATH="${TRAINING_ROOT}/small_surrogate_training.stats"

    echo
    echo "======================================================="
    echo "Training small surrogate DT for workload:"
    echo "  Trace root dir    => ${TRACE_DIR}"
    echo "  FlashNet data dir => ${FLASHNET_RESULTS_DIR}"
    echo "  Output dir        => ${TRAINING_ROOT}"
    echo "  max_depth         => ${MAX_DEPTH}"
    echo "======================================================="

    if [ ! -f "${DATASET0}" ]; then
        echo "  [SKIP] Dataset for dev_0 not found: ${DATASET0}"
        continue
    fi
    if [ ! -f "${DATASET1}" ]; then
        echo "  [SKIP] Dataset for dev_1 not found: ${DATASET1}"
        continue
    fi

    mkdir -p "${OUT_DIR}"

    echo
    echo "  -> Training small surrogate for dev_0"
    python3 "${SCRIPT_DIR}/train_small_surrogate_dt_fidelity.py" \
        -target flashnet \
        -dataset "${DATASET0}" \
        -workload Trace \
        -drive dev_0 \
        -output_dir "${OUT_DIR}" \
        -max_depth "${MAX_DEPTH}" \
        -metrics_output "${METRICS0_JSON}" \
        -stats_output "${STATS0_PATH}"

    echo
    echo "  -> Training small surrogate for dev_1"
    python3 "${SCRIPT_DIR}/train_small_surrogate_dt_fidelity.py" \
        -target flashnet \
        -dataset "${DATASET1}" \
        -workload Trace \
        -drive dev_1 \
        -output_dir "${OUT_DIR}" \
        -max_depth "${MAX_DEPTH}" \
        -metrics_output "${METRICS1_JSON}" \
        -stats_output "${STATS1_PATH}"

    python3 - "${METRICS0_JSON}" "${METRICS1_JSON}" "${COMBINED_STATS_PATH}" "${TRACE_DIR}" "${MAX_DEPTH}" <<'PY'
import json
import os
import sys
from datetime import datetime

metrics0_path, metrics1_path, out_path, trace_dir, max_depth = sys.argv[1:]
with open(metrics0_path, "r") as f:
    m0 = json.load(f)
with open(metrics1_path, "r") as f:
    m1 = json.load(f)

def getv(d, key):
    return d.get(key, float("nan"))

lines = [
    "========== Small Surrogate DT Training Stats ==========",
    f"generated_on = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"trace_dir = {trace_dir}",
    f"configured_max_depth = {max_depth}",
    "",
    "[dev_0]",
    f"train_fidelity = {getv(m0, 'train_fidelity'):.6f}",
    f"test_fidelity = {getv(m0, 'test_fidelity'):.6f}",
    f"teacher_train_acc_gt = {getv(m0, 'teacher_train_acc_gt'):.6f}",
    f"teacher_test_acc_gt = {getv(m0, 'teacher_test_acc_gt'):.6f}",
    f"small_surrogate_train_acc_gt = {getv(m0, 'dt_train_acc_gt'):.6f}",
    f"small_surrogate_test_acc_gt = {getv(m0, 'dt_test_acc_gt'):.6f}",
    "",
    "[dev_1]",
    f"train_fidelity = {getv(m1, 'train_fidelity'):.6f}",
    f"test_fidelity = {getv(m1, 'test_fidelity'):.6f}",
    f"teacher_train_acc_gt = {getv(m1, 'teacher_train_acc_gt'):.6f}",
    f"teacher_test_acc_gt = {getv(m1, 'teacher_test_acc_gt'):.6f}",
    f"small_surrogate_train_acc_gt = {getv(m1, 'dt_train_acc_gt'):.6f}",
    f"small_surrogate_test_acc_gt = {getv(m1, 'dt_test_acc_gt'):.6f}",
    "=======================================================",
]

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"  small surrogate training stats written to: {out_path}")
PY

    echo "  Small surrogate headers:"
    echo "    ${OUT_DIR}"
    echo "    (w_Trace_dev_0_dt.h, w_Trace_dev_1_dt.h)"
    echo "  Small surrogate metrics:"
    echo "    ${METRICS0_JSON}"
    echo "    ${METRICS1_JSON}"
    echo "  Small surrogate training stats:"
    echo "    ${STATS0_PATH}"
    echo "    ${STATS1_PATH}"
    echo "    ${COMBINED_STATS_PATH}"
done
