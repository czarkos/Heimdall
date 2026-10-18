#!/bin/bash

set -euo pipefail

# Usage:
#   ./train_surrogate_lr.sh dev0 dev1 dir1 [dir2 dir3 ...]
#
# For each dirX, expects FlashNet's training outputs (datasets + exported teacher weights):
#   dirX/<dev0>...<dev1>/flashnet/training_results/mldrive{0,1}.csv
#   dirX/<dev0>...<dev1>/flashnet/training_results/mldrive{0,1}.csv.{weight,bias}_{0..3}.csv
#
# Output:
#   dirX/<dev0>...<dev1>/<ALGO_NAME>/training_results/
#     - lr_headers/w_Trace_dev_{0,1}_lr.h
#     - surrogate_lr_dev_{0,1}_metrics.json
#     - surrogate_lr_dev_{0,1}_training.stats
#     - surrogate_lr_training.stats
#
# Environment overrides:
#   SURROGATE_LR_ALGO_NAME  output directory name (default: surrogate_lr)
#   SURROGATE_LR_C          inverse L2 regularization strength (default: 1.0)
#   SURROGATE_LR_TARGET     flashnet (default) or label

if [ $# -lt 3 ]; then
    echo "Usage: $0 device0 device1 dir_to_replayed_traces [more_dirs...]"
    echo "Example: $0 nvme0n1 nvme1n1 /mnt/.../data/*/*/*"
    exit 1
fi

DEV0="$1"
DEV1="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ALGO_NAME="${SURROGATE_LR_ALGO_NAME:-surrogate_lr}"
LR_C="${SURROGATE_LR_C:-1.0}"
LR_TARGET="${SURROGATE_LR_TARGET:-flashnet}"

for TRACE_DIR in "$@"; do
    FLASHNET_RESULTS_DIR="${TRACE_DIR}/${DEV0}...${DEV1}/flashnet/training_results"
    DATASET0="${FLASHNET_RESULTS_DIR}/mldrive0.csv"
    DATASET1="${FLASHNET_RESULTS_DIR}/mldrive1.csv"

    TRAINING_ROOT="${TRACE_DIR}/${DEV0}...${DEV1}/${ALGO_NAME}/training_results"
    OUT_DIR="${TRAINING_ROOT}/lr_headers"
    METRICS0_JSON="${TRAINING_ROOT}/surrogate_lr_dev_0_metrics.json"
    METRICS1_JSON="${TRAINING_ROOT}/surrogate_lr_dev_1_metrics.json"
    STATS0_PATH="${TRAINING_ROOT}/surrogate_lr_dev_0_training.stats"
    STATS1_PATH="${TRAINING_ROOT}/surrogate_lr_dev_1_training.stats"
    COMBINED_STATS_PATH="${TRAINING_ROOT}/surrogate_lr_training.stats"

    echo
    echo "======================================================="
    echo "Training surrogate logistic regression for workload:"
    echo "  Trace root dir    => ${TRACE_DIR}"
    echo "  FlashNet data dir => ${FLASHNET_RESULTS_DIR}"
    echo "  Output dir        => ${TRAINING_ROOT}"
    echo "  target / C        => ${LR_TARGET} / ${LR_C}"
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
    echo "  -> Training surrogate LR for dev_0"
    python3 "${SCRIPT_DIR}/train_surrogate_lr_fidelity.py" \
        -target "${LR_TARGET}" \
        -dataset "${DATASET0}" \
        -workload Trace \
        -drive dev_0 \
        -output_dir "${OUT_DIR}" \
        -C "${LR_C}" \
        -metrics_output "${METRICS0_JSON}" \
        -stats_output "${STATS0_PATH}"

    echo
    echo "  -> Training surrogate LR for dev_1"
    python3 "${SCRIPT_DIR}/train_surrogate_lr_fidelity.py" \
        -target "${LR_TARGET}" \
        -dataset "${DATASET1}" \
        -workload Trace \
        -drive dev_1 \
        -output_dir "${OUT_DIR}" \
        -C "${LR_C}" \
        -metrics_output "${METRICS1_JSON}" \
        -stats_output "${STATS1_PATH}"

    python3 - "${METRICS0_JSON}" "${METRICS1_JSON}" "${COMBINED_STATS_PATH}" "${TRACE_DIR}" "${LR_C}" "${LR_TARGET}" <<'PY'
import json
import os
import sys
from datetime import datetime

metrics0_path, metrics1_path, out_path, trace_dir, lr_c, lr_target = sys.argv[1:]
with open(metrics0_path, "r") as f:
    m0 = json.load(f)
with open(metrics1_path, "r") as f:
    m1 = json.load(f)

def getv(d, key):
    return d.get(key, float("nan"))

def dev_block(name, m):
    return [
        f"[{name}]",
        f"train_fidelity = {getv(m, 'train_fidelity'):.6f}",
        f"test_fidelity = {getv(m, 'test_fidelity'):.6f}",
        f"teacher_train_acc_gt = {getv(m, 'teacher_train_acc_gt'):.6f}",
        f"teacher_test_acc_gt = {getv(m, 'teacher_test_acc_gt'):.6f}",
        f"surrogate_lr_train_acc_gt = {getv(m, 'lr_train_acc_gt'):.6f}",
        f"surrogate_lr_test_acc_gt = {getv(m, 'lr_test_acc_gt'):.6f}",
        f"header_match_rate = {getv(m, 'header_match_rate'):.6f}",
        f"lr_reject_rate_reads = {getv(m, 'lr_reject_rate_reads'):.6f}",
        f"n_iter = {m.get('n_iter', 'nan')}, converged = {m.get('converged', 'nan')}",
    ]

lines = [
    "========== Surrogate Logistic Regression Training Stats ==========",
    f"generated_on = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"trace_dir = {trace_dir}",
    f"target = {lr_target}",
    f"C = {lr_c}",
    "",
] + dev_block("dev_0", m0) + [""] + dev_block("dev_1", m1) + [
    "==================================================================",
]

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"  surrogate LR training stats written to: {out_path}")
PY

    echo "  Surrogate LR headers:"
    echo "    ${OUT_DIR}"
    echo "    (w_Trace_dev_0_lr.h, w_Trace_dev_1_lr.h)"
    echo "  Surrogate LR metrics:"
    echo "    ${METRICS0_JSON}"
    echo "    ${METRICS1_JSON}"
    echo "  Surrogate LR training stats:"
    echo "    ${STATS0_PATH}"
    echo "    ${STATS1_PATH}"
    echo "    ${COMBINED_STATS_PATH}"
done
