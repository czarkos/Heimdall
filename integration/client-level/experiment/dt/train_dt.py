#!/usr/bin/env python3

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree

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


def normalize_feature_aliases(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize historical column aliases to the canonical names expected by C.

    Current alias support:
      - queue_len -> cur_queue_len
    """
    columns = dataset.columns.tolist()
    has_cur = "cur_queue_len" in columns
    has_queue = "queue_len" in columns

    # If only queue_len exists, rename to canonical cur_queue_len.
    if (not has_cur) and has_queue:
        dataset = dataset.rename(columns={"queue_len": "cur_queue_len"})
        return dataset

    # If both exist, keep canonical cur_queue_len and ignore duplicate queue_len.
    if has_cur and has_queue:
        dataset = dataset.drop(columns=["queue_len"])

    return dataset


def train_decision_tree(
    dataset_path: str,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
) -> Tuple[DecisionTreeClassifier, List[str], Dict[str, float], str]:
    """
    Train a decision tree directly on the FlashNet training dataset.

    The dataset is expected to be the same CSV used by nnK.py
    (e.g., mldrive0.csv), with a 'reject' column as the label and
    optionally a 'latency' column which is not used as an input feature.
    """
    dataset = pd.read_csv(dataset_path)

    # Normalize dataset column aliases before feature selection.
    dataset = normalize_feature_aliases(dataset)

    # Move latency to the end, matching nnK.py behavior if present
    if "latency" in dataset.columns:
        reordered_cols = [col for col in dataset.columns if col != "latency"] + ["latency"]
        dataset = dataset[reordered_cols]

    if "reject" not in dataset.columns:
        raise ValueError("Dataset must contain a 'reject' column as label.")

    # Use only the exact 12-feature layout expected by C inference.
    # This avoids training on extra columns (e.g., latency) that would
    # generate out-of-range feature indices during tree traversal.
    missing_features = [f for f in EXPECTED_INPUT_FEATURES if f not in dataset.columns]
    if missing_features:
        raise ValueError(
            "Dataset is missing required DT features: {}\n"
            "Expected feature set: {}\n"
            "Available columns: {}".format(
                missing_features, EXPECTED_INPUT_FEATURES, dataset.columns.tolist()
            )
        )

    unexpected_feature_like_cols = [
        col for col in dataset.columns if col not in set(EXPECTED_INPUT_FEATURES + ["reject", "latency"])
    ]
    if unexpected_feature_like_cols:
        print(
            "[Warning] Ignoring extra dataset columns not used by DT inference: {}".format(
                unexpected_feature_like_cols
            )
        )

    X = dataset[EXPECTED_INPUT_FEATURES].copy(deep=True)
    # For the surrogate used in C, we rely on raw feature scales (no normalization)
    y = dataset["reject"].astype(int)

    feature_names = X.columns.tolist()

    # 50/50 split by default (same as nnK.py typical usage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Basic evaluation for sanity checking
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    test_report = classification_report(y_test, y_test_pred, digits=4)
    test_confusion = confusion_matrix(y_test, y_test_pred)

    stats_lines = [
        "=== Decision Tree Surrogate (direct on labels) ===",
        f"Train accuracy: {train_acc:.4f}",
        f"Test  accuracy: {test_acc:.4f}",
        "",
        "Classification report (test set):",
        str(test_report),
        "Confusion matrix (test set):",
        str(test_confusion),
    ]
    stats_text = "\n".join(stats_lines)

    print(stats_text)

    metrics = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
    }

    return clf, feature_names, metrics, stats_text


def export_tree_to_header(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    workload: str,
    drive: str,
    output_dir: str,
):
    """
    Export a trained sklearn DecisionTreeClassifier into a C header file
    containing flat arrays describing the tree.

    The header is consumed by dt_algo.c and expected to live under:
        dt_weights_header/w_<workload>_<drive>_dt.h
    """
    tree: _tree.Tree = clf.tree_
    n_nodes = tree.node_count

    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature.copy()
    threshold = tree.threshold.copy()
    value = tree.value  # shape: [n_nodes, 1, n_classes] for binary classification

    # For leaves, sklearn uses _tree.TREE_UNDEFINED (-2) as feature index.
    # We normalize this to -1 for easier checks in C (feat < 0 => leaf).
    feature[feature == _tree.TREE_UNDEFINED] = -1

    # Predicted class at each node (we only actually use it at leaves)
    # value[node, 0, class] is the count of samples in that class.
    node_value = np.argmax(value[:, 0, :], axis=1).astype(int)

    # Thresholds are floats; features are integer-like. For C, we round
    # thresholds to nearest integer and store them as long.
    thr_int = np.round(threshold).astype(np.int64)

    # Sanity checks
    if len(feature_names) == 0:
        raise ValueError("No features found when exporting tree.")
    max_internal_feature_idx = np.max(feature[feature >= 0]) if np.any(feature >= 0) else -1
    if max_internal_feature_idx >= len(feature_names):
        raise ValueError(
            "Trained tree uses feature index {} but only {} features are available.".format(
                int(max_internal_feature_idx), len(feature_names)
            )
        )

    # Prepare header path
    os.makedirs(output_dir, exist_ok=True)
    header_name = f"w_{workload}_{drive}_dt.h"
    header_path = os.path.join(output_dir, header_name)

    macro_prefix = f"DT_{drive.upper()}_NODE_COUNT"

    print(f"Writing decision tree header to: {header_path}")
    with open(header_path, "w") as f:
        f.write(f"\n/* Surrogate decision tree for {workload} / {drive} */\n")
        from datetime import datetime

        f.write("/**\n")
        f.write(f" *  generated on {datetime.now().strftime('%m/%d/%Y %H:%M:%S')}\n")
        f.write(" */\n\n")

        guard = f"__W_{drive.upper()}_DT_H"
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")

        f.write(f"#define {macro_prefix} {n_nodes}\n\n")

        # Feature indices
        f.write(f"int dt_feature_{drive}[] = {{\n  ")
        f.write(", ".join(str(int(x)) for x in feature))
        f.write("\n};\n\n")

        # Thresholds
        f.write(f"long dt_threshold_{drive}[] = {{\n  ")
        f.write(", ".join(str(int(x)) for x in thr_int))
        f.write("\n};\n\n")

        # Children indices
        f.write(f"int dt_left_{drive}[] = {{\n  ")
        f.write(", ".join(str(int(x)) for x in children_left))
        f.write("\n};\n\n")

        f.write(f"int dt_right_{drive}[] = {{\n  ")
        f.write(", ".join(str(int(x)) for x in children_right))
        f.write("\n};\n\n")

        # Predicted class at each node
        f.write(f"int dt_value_{drive}[] = {{\n  ")
        f.write(", ".join(str(int(x)) for x in node_value))
        f.write("\n};\n\n")

        f.write(f"#endif /* {guard} */\n")

    print(f"Header written: {header_path}")
    return header_path


def main():
    parser = argparse.ArgumentParser(
        description="Train a decision tree surrogate for FlashNet and export C header."
    )
    parser.add_argument(
        "-dataset",
        help="Path to the FlashNet training dataset CSV (e.g., mldrive0.csv)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-workload",
        help="Workload name used in header file name (default: Trace)",
        type=str,
        default="Trace",
    )
    parser.add_argument(
        "-drive",
        help="Drive identifier (e.g., dev_0, dev_1). Used in header symbol names.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-output_dir",
        help="Directory to place generated header (default: ./dt_weights_header)",
        type=str,
        default="dt_weights_header",
    )
    parser.add_argument(
        "-max_depth",
        help="Maximum depth of the decision tree (default: None for unbounded)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-min_samples_split",
        help="Minimum number of samples required to split an internal node (default: 2)",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-min_samples_leaf",
        help="Minimum number of samples required to be at a leaf node (default: 1)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-stats_output",
        help="Optional path to persist training stats text output.",
        type=str,
        default="",
    )

    args = parser.parse_args()

    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        raise SystemExit(f"Dataset not found: {dataset_path}")

    clf, feature_names, metrics, stats_text = train_decision_tree(
        dataset_path,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
    )

    header_path = export_tree_to_header(
        clf,
        feature_names,
        workload=args.workload,
        drive=args.drive,
        output_dir=args.output_dir,
    )

    print("\n=== Surrogate DT training complete ===")
    print(f"Generated header: {header_path}")
    if args.stats_output:
        stats_dir = os.path.dirname(os.path.abspath(args.stats_output))
        os.makedirs(stats_dir, exist_ok=True)
        with open(args.stats_output, "w") as f:
            f.write(stats_text + "\n")
            f.write("\n")
            f.write(
                "metrics_json = "
                + '{"train_accuracy": %.6f, "test_accuracy": %.6f}\n'
                % (metrics["train_accuracy"], metrics["test_accuracy"])
            )
        print(f"Training stats written: {args.stats_output}")


if __name__ == "__main__":
    main()

