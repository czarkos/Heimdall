#!/usr/bin/env python3
"""
Train a logistic-regression surrogate of FlashNet and export it as a C header.

Same protocol as the surrogate decision trees (small_surrogate_dt/train_small_surrogate_dt_fidelity.py):
  - inputs: the 12 FlashNet features of mldrive<dev>.csv, in the order the C replayer fills them
  - target: FlashNet's own predictions, reconstructed from the exported weight CSVs
            (-target flashnet), or the ground-truth 'reject' labels (-target label)
  - 50/50 train/test split with random_state=42

The model is StandardScaler + LogisticRegression. The scaler is folded into the exported
weights, so the C side (lr_algo.c) computes a single dot product on the raw long features:
    z = bias_eff + sum_j weight_eff[j] * x[j],   reject iff z > 0
with  weight_eff = coef / scale,  bias_eff = intercept - sum(coef * mean / scale).

The trainer re-computes z in exactly the C accumulation order and reports header_match_rate,
the fraction of rows where that decision equals sklearn's predict(); it fails if it is not ~1.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reuse the dataset preparation and the FlashNet teacher reconstruction of the DT surrogates.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), "small_surrogate_dt"))
from train_small_surrogate_dt_fidelity import (  # noqa: E402
    get_flashnet_teacher_predictions,
    prepare_dataset,
)

# Order the C replayer expects in input_vec_i (lr_algo.h). prepare_dataset() yields the CSV
# columns minus 'reject' and 'latency', which is this order for the FlashNet datasets; the
# check below makes that explicit instead of trusting it.
EXPECTED_INPUT_FEATURES = [
    "io_type",
    "size",
    "cur_queue_len",
    "prev_queue_len_1",
    "prev_queue_len_2",
    "prev_queue_len_3",
    "prev_latency_1",
    "prev_latency_2",
    "prev_latency_3",
    "prev_throughput_1",
    "prev_throughput_2",
    "prev_throughput_3",
]
FEATURE_ALIASES = {"queue_len": "cur_queue_len"}

HEADER_MATCH_TOLERANCE = 1e-6  # allowed fraction of rows whose sign flips from float rounding


def check_feature_order(feature_names: List[str]) -> List[str]:
    canonical = [FEATURE_ALIASES.get(f, f) for f in feature_names]
    if canonical != EXPECTED_INPUT_FEATURES:
        raise ValueError(
            "Dataset feature order does not match the C feature vector.\n"
            "  got:      {}\n  expected: {}".format(canonical, EXPECTED_INPUT_FEATURES)
        )
    return canonical


def c_decision(x_raw: np.ndarray, weight_eff: np.ndarray, bias_eff: float) -> np.ndarray:
    """z computed in the same order as lr_algo.c: bias first, then features 0..11."""
    acc = np.full(x_raw.shape[0], bias_eff, dtype=np.float64)
    for j in range(x_raw.shape[1]):
        acc = acc + weight_eff[j] * x_raw[:, j].astype(np.float64)
    return acc


def train_surrogate_lr(
    dataset_path: str,
    target: str = "flashnet",
    C: float = 1.0,
    max_iter: int = 2000,
) -> Tuple[Dict[str, object], str]:
    x, y_gt, feature_names = prepare_dataset(dataset_path)
    feature_names = check_feature_order(feature_names)
    x_raw = x.to_numpy(dtype=np.float64)

    if target == "flashnet":
        y_teacher = get_flashnet_teacher_predictions(dataset_path, x)
    elif target == "label":
        y_teacher = None
    else:
        raise ValueError("Unsupported target: {}".format(target))

    x_train, x_test, y_train_gt, y_test_gt = train_test_split(
        x_raw, y_gt.to_numpy(), test_size=0.5, random_state=42
    )
    if y_teacher is not None:
        y_train_target, y_test_target = train_test_split(y_teacher, test_size=0.5, random_state=42)
    else:
        y_train_target, y_test_target = y_train_gt, y_test_gt

    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, random_state=42)
    clf.fit(scaler.transform(x_train), y_train_target)
    n_iter = int(np.asarray(clf.n_iter_).max())
    converged = n_iter < max_iter

    y_train_pred = clf.predict(scaler.transform(x_train))
    y_test_pred = clf.predict(scaler.transform(x_test))

    # Fold the scaler into the weights (what the C header holds).
    coef = clf.coef_.reshape(-1).astype(np.float64)
    intercept = float(clf.intercept_[0])
    mean = scaler.mean_.astype(np.float64)
    scale = scaler.scale_.astype(np.float64)   # sklearn sets scale_=1 for constant features
    weight_eff = coef / scale
    bias_eff = intercept - float(np.sum(coef * mean / scale))

    # Header fidelity: the C-order decision must reproduce sklearn's predict() on every row.
    z_all = c_decision(x_raw, weight_eff, bias_eff)
    c_pred_all = (z_all > 0.0).astype(int)
    sk_pred_all = clf.predict(scaler.transform(x_raw))
    header_match_rate = float(np.mean(c_pred_all == sk_pred_all))
    if header_match_rate < 1.0 - HEADER_MATCH_TOLERANCE:
        raise RuntimeError(
            "Folded header disagrees with sklearn on {:.6f} of rows; refusing to export.".format(
                1.0 - header_match_rate
            )
        )
    is_read = x_raw[:, 0] == 1
    lr_reject_rate_all = float(np.mean(c_pred_all))
    lr_reject_rate_reads = float(np.mean(c_pred_all[is_read])) if np.any(is_read) else float("nan")

    metrics: Dict[str, object] = {
        "model": "logistic_regression",
        "target": target,
        "C": float(C),
        "max_iter": int(max_iter),
        "n_iter": n_iter,
        "converged": bool(converged),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "header_match_rate": header_match_rate,
        "lr_reject_rate_all_rows": lr_reject_rate_all,
        "lr_reject_rate_reads": lr_reject_rate_reads,
        "feature_names": feature_names,
        "coef_standardized": coef.tolist(),
        "intercept_standardized": intercept,
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "weight_eff": weight_eff.tolist(),
        "bias_eff": bias_eff,
    }

    if target == "flashnet":
        train_fidelity = accuracy_score(y_train_target, y_train_pred)
        test_fidelity = accuracy_score(y_test_target, y_test_pred)
        teacher_train_acc_gt = accuracy_score(y_train_gt, y_train_target)
        teacher_test_acc_gt = accuracy_score(y_test_gt, y_test_target)
        lr_train_acc_gt = accuracy_score(y_train_gt, y_train_pred)
        lr_test_acc_gt = accuracy_score(y_test_gt, y_test_pred)
        fidelity_report = classification_report(y_test_target, y_test_pred, digits=4)
        fidelity_confusion = confusion_matrix(y_test_target, y_test_pred)

        stats_lines = [
            "=== Surrogate Logistic Regression (mimic FlashNet outputs) ===",
            "Train fidelity (LR vs FlashNet): {:.4f}".format(train_fidelity),
            "Test  fidelity (LR vs FlashNet): {:.4f}".format(test_fidelity),
            "C = {}, lbfgs iterations = {}, converged = {}".format(C, n_iter, converged),
            "Header match rate (C-order decision vs sklearn predict): {:.6f}".format(header_match_rate),
            "",
            "Accuracy against ground-truth labels (for reference)",
            "FlashNet teacher - Train: {:.4f}, Test: {:.4f}".format(teacher_train_acc_gt, teacher_test_acc_gt),
            "Surrogate LR     - Train: {:.4f}, Test: {:.4f}".format(lr_train_acc_gt, lr_test_acc_gt),
            "",
            "Classification report (LR vs teacher target, test set):",
            str(fidelity_report),
            "Confusion matrix (LR vs teacher target, test set):",
            str(fidelity_confusion),
            "",
            "Standardized coefficients (feature: coef):",
        ] + ["  {:18s} {:+.6f}".format(f, c) for f, c in zip(feature_names, coef)] + [
            "  {:18s} {:+.6f}".format("intercept", intercept),
        ]
        metrics.update({
            "train_fidelity": float(train_fidelity),
            "test_fidelity": float(test_fidelity),
            "teacher_train_acc_gt": float(teacher_train_acc_gt),
            "teacher_test_acc_gt": float(teacher_test_acc_gt),
            "lr_train_acc_gt": float(lr_train_acc_gt),
            "lr_test_acc_gt": float(lr_test_acc_gt),
        })
    else:
        train_acc = accuracy_score(y_train_gt, y_train_pred)
        test_acc = accuracy_score(y_test_gt, y_test_pred)
        direct_report = classification_report(y_test_gt, y_test_pred, digits=4)
        direct_confusion = confusion_matrix(y_test_gt, y_test_pred)
        stats_lines = [
            "=== Surrogate Logistic Regression (direct on labels) ===",
            "Train accuracy: {:.4f}".format(train_acc),
            "Test  accuracy: {:.4f}".format(test_acc),
            "C = {}, lbfgs iterations = {}, converged = {}".format(C, n_iter, converged),
            "Header match rate (C-order decision vs sklearn predict): {:.6f}".format(header_match_rate),
            "",
            "Classification report (test set):",
            str(direct_report),
            "Confusion matrix (test set):",
            str(direct_confusion),
        ]
        metrics.update({
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
        })

    stats_text = "\n".join(stats_lines)
    print(stats_text)
    return metrics, stats_text


def export_lr_to_header(
    metrics: Dict[str, object],
    workload: str,
    drive: str,
    output_dir: str,
) -> str:
    """
    Write w_<workload>_<drive>_lr.h, consumed by lr_algo.c:
        #define LR_<DRIVE>_N_FEATURES 12
        double lr_weight_<drive>[LR_<DRIVE>_N_FEATURES] = {...};
        double lr_bias_<drive> = ...;
    Doubles are written with repr() (shortest round-trip form), so C reads them back exactly.
    """
    weight_eff = metrics["weight_eff"]
    bias_eff = metrics["bias_eff"]
    feature_names = metrics["feature_names"]

    os.makedirs(output_dir, exist_ok=True)
    header_name = "w_{}_{}_lr.h".format(workload, drive)
    header_path = os.path.join(output_dir, header_name)
    guard = "__W_{}_LR_H".format(drive.upper())
    macro = "LR_{}_N_FEATURES".format(drive.upper())

    print("Writing logistic regression header to: {}".format(header_path))
    with open(header_path, "w") as f:
        f.write("\n/* Surrogate logistic regression for {} / {} */\n".format(workload, drive))
        f.write("/**\n *  generated on {}\n".format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
        f.write(" *  target = {}, C = {}, test_fidelity = {}\n".format(
            metrics.get("target"), metrics.get("C"), metrics.get("test_fidelity", metrics.get("test_accuracy"))))
        f.write(" *  StandardScaler folded in: z = bias + sum(weight[j] * raw_feature[j]); reject iff z > 0\n */\n\n")
        f.write("#ifndef {}\n#define {}\n\n".format(guard, guard))
        f.write("#define {} {}\n\n".format(macro, len(weight_eff)))
        f.write("/* feature order: {} */\n".format(", ".join(feature_names)))
        f.write("double lr_weight_{}[{}] = {{\n".format(drive, macro))
        for name, w in zip(feature_names, weight_eff):
            f.write("  {}, /* {} */\n".format(repr(float(w)), name))
        f.write("};\n\n")
        f.write("double lr_bias_{} = {};\n\n".format(drive, repr(float(bias_eff))))
        f.write("#endif /* {} */\n".format(guard))

    print("Header written: {}".format(header_path))
    return header_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a logistic-regression surrogate of FlashNet and export a C header."
    )
    parser.add_argument(
        "-target",
        help="Training target: 'flashnet' to mimic FlashNet outputs, or 'label' for original labels",
        type=str,
        choices=["flashnet", "label"],
        default="flashnet",
    )
    parser.add_argument("-dataset", type=str, required=True, help="FlashNet dataset CSV (mldrive<dev>.csv)")
    parser.add_argument("-workload", type=str, default="Trace")
    parser.add_argument("-drive", type=str, required=True, help="dev_0 or dev_1 (used in symbol names)")
    parser.add_argument("-output_dir", type=str, default="lr_headers")
    parser.add_argument("-C", type=float, default=1.0, help="Inverse L2 regularization strength (default 1.0)")
    parser.add_argument("-max_iter", type=int, default=2000)
    parser.add_argument("-metrics_output", type=str, default="", help="Optional path to write metrics as JSON.")
    parser.add_argument("-stats_output", type=str, default="", help="Optional path to write human-readable stats.")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise SystemExit("Dataset not found: {}".format(args.dataset))

    metrics, stats_text = train_surrogate_lr(
        dataset_path=args.dataset,
        target=args.target,
        C=args.C,
        max_iter=args.max_iter,
    )
    header_path = export_lr_to_header(
        metrics=metrics,
        workload=args.workload,
        drive=args.drive,
        output_dir=args.output_dir,
    )

    print("\n=== Surrogate logistic regression training complete ===")
    print("Target mode: {}".format(args.target))
    print("Generated header: {}".format(header_path))

    if args.metrics_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_output)), exist_ok=True)
        with open(args.metrics_output, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print("Metrics JSON: {}".format(args.metrics_output))

    if args.stats_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.stats_output)), exist_ok=True)
        with open(args.stats_output, "w") as f:
            f.write(stats_text + "\n")
        print("Stats text: {}".format(args.stats_output))


if __name__ == "__main__":
    main()
