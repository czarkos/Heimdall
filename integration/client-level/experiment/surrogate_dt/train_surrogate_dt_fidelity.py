#!/usr/bin/env python3

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree


def prepare_dataset(dataset_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Load a FlashNet training dataset and prepare model inputs similarly to nnK.py.
    """
    dataset = pd.read_csv(dataset_path)

    if "latency" in dataset.columns:
        reordered_cols = [col for col in dataset.columns if col != "latency"] + ["latency"]
        dataset = dataset[reordered_cols]

    if "reject" not in dataset.columns:
        raise ValueError("Dataset must contain a 'reject' column as label.")

    x = dataset.drop(columns=["reject"])
    if "latency" in x.columns:
        x = x.drop(columns=["latency"])
    y_gt = dataset["reject"].astype(int)
    feature_names = x.columns.tolist()
    return x, y_gt, feature_names


def _load_csv_array(path: str) -> np.ndarray:
    return np.asarray(np.loadtxt(path, delimiter=","))


def get_flashnet_teacher_predictions(dataset_path: str, x_raw: pd.DataFrame) -> np.ndarray:
    """
    Reconstruct FlashNet outputs from exported weights written by flashnet/training/nnK.py.

    Expected files next to dataset_path:
      <dataset>.weight_0.csv  (scaler data_min_)
      <dataset>.bias_0.csv    (scaler data_range_)
      <dataset>.weight_1.csv, <dataset>.bias_1.csv
      <dataset>.weight_2.csv, <dataset>.bias_2.csv
      <dataset>.weight_3.csv, <dataset>.bias_3.csv
    """
    p_w0 = dataset_path + ".weight_0.csv"
    p_b0 = dataset_path + ".bias_0.csv"
    p_w1 = dataset_path + ".weight_1.csv"
    p_b1 = dataset_path + ".bias_1.csv"
    p_w2 = dataset_path + ".weight_2.csv"
    p_b2 = dataset_path + ".bias_2.csv"
    p_w3 = dataset_path + ".weight_3.csv"
    p_b3 = dataset_path + ".bias_3.csv"

    required = [p_w0, p_b0, p_w1, p_b1, p_w2, p_b2, p_w3, p_b3]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing exported FlashNet weight files:\n  " + "\n  ".join(missing)
        )

    # MinMax scaler stats from nnK.py export
    data_min = _load_csv_array(p_w0).reshape(-1)
    data_range = _load_csv_array(p_b0).reshape(-1)
    safe_range = np.where(np.abs(data_range) < 1e-12, 1.0, data_range)

    # DNN weights (Dense: 12->128->16->1)
    w1 = np.atleast_2d(_load_csv_array(p_w1))
    b1 = _load_csv_array(p_b1).reshape(-1)
    w2 = np.atleast_2d(_load_csv_array(p_w2))
    b2 = _load_csv_array(p_b2).reshape(-1)
    w3 = np.atleast_2d(_load_csv_array(p_w3))
    b3 = _load_csv_array(p_b3).reshape(-1)

    # Normalize last layer shape to (hidden, 1)
    if w3.shape[1] != 1:
        if w3.shape[0] == 1:
            w3 = w3.T
        else:
            raise ValueError("Unexpected shape for weight_3: {}".format(w3.shape))

    x = x_raw.to_numpy(dtype=float)
    x_norm = (x - data_min) / safe_range

    # FlashNet forward pass
    z1 = np.maximum(0.0, x_norm @ w1 + b1)
    z2 = np.maximum(0.0, z1 @ w2 + b2)
    logits = z2 @ w3 + b3

    # sigmoid(logit) > 0.5  <=> logit > 0
    return (logits.reshape(-1) >= 0.0).astype(int)


def train_surrogate_tree(
    dataset_path: str,
    target: str = "flashnet",
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
) -> Tuple[DecisionTreeClassifier, List[str]]:
    x, y_gt, feature_names = prepare_dataset(dataset_path)

    if target == "flashnet":
        y_teacher = get_flashnet_teacher_predictions(dataset_path, x)
    elif target == "label":
        y_teacher = None
    else:
        raise ValueError("Unsupported target: {}".format(target))

    x_train, x_test, y_train_gt, y_test_gt = train_test_split(
        x, y_gt, test_size=0.5, random_state=42
    )

    if y_teacher is not None:
        y_train_target, y_test_target = train_test_split(
            y_teacher, test_size=0.5, random_state=42
        )
    else:
        y_train_target, y_test_target = y_train_gt.to_numpy(), y_test_gt.to_numpy()

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    clf.fit(x_train, y_train_target)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    if target == "flashnet":
        train_fidelity = accuracy_score(y_train_target, y_train_pred)
        test_fidelity = accuracy_score(y_test_target, y_test_pred)
        teacher_train_acc_gt = accuracy_score(y_train_gt, y_train_target)
        teacher_test_acc_gt = accuracy_score(y_test_gt, y_test_target)
        dt_train_acc_gt = accuracy_score(y_train_gt, y_train_pred)
        dt_test_acc_gt = accuracy_score(y_test_gt, y_test_pred)

        print("=== Decision Tree Surrogate (mimic FlashNet outputs) ===")
        print("Train fidelity (DT vs FlashNet): {:.4f}".format(train_fidelity))
        print("Test  fidelity (DT vs FlashNet): {:.4f}".format(test_fidelity))
        print("")
        print("Accuracy against ground-truth labels (for reference)")
        print(
            "FlashNet teacher - Train: {:.4f}, Test: {:.4f}".format(
                teacher_train_acc_gt, teacher_test_acc_gt
            )
        )
        print(
            "Surrogate DT     - Train: {:.4f}, Test: {:.4f}".format(
                dt_train_acc_gt, dt_test_acc_gt
            )
        )
        print("\nClassification report (DT vs teacher target, test set):")
        print(classification_report(y_test_target, y_test_pred, digits=4))
        print("Confusion matrix (DT vs teacher target, test set):")
        print(confusion_matrix(y_test_target, y_test_pred))
    else:
        train_acc = accuracy_score(y_train_gt, y_train_pred)
        test_acc = accuracy_score(y_test_gt, y_test_pred)
        print("=== Decision Tree Surrogate (direct on labels) ===")
        print("Train accuracy: {:.4f}".format(train_acc))
        print("Test  accuracy: {:.4f}".format(test_acc))
        print("\nClassification report (test set):")
        print(classification_report(y_test_gt, y_test_pred, digits=4))
        print("Confusion matrix (test set):")
        print(confusion_matrix(y_test_gt, y_test_pred))

    return clf, feature_names


def export_tree_to_header(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    workload: str,
    drive: str,
    output_dir: str,
) -> str:
    tree = clf.tree_
    n_nodes = tree.node_count

    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature.copy()
    threshold = tree.threshold.copy()
    value = tree.value

    feature[feature == _tree.TREE_UNDEFINED] = -1
    node_value = np.argmax(value[:, 0, :], axis=1).astype(int)
    thr_int = np.round(threshold).astype(np.int64)

    if len(feature_names) == 0:
        raise ValueError("No features found when exporting tree.")

    os.makedirs(output_dir, exist_ok=True)
    header_name = "w_{}_{}_dt.h".format(workload, drive)
    header_path = os.path.join(output_dir, header_name)
    macro_prefix = "DT_{}_NODE_COUNT".format(drive.upper())

    print("Writing decision tree header to: {}".format(header_path))
    with open(header_path, "w") as f:
        f.write("\n/* Surrogate decision tree for {} / {} */\n".format(workload, drive))
        from datetime import datetime

        f.write("/**\n")
        f.write(" *  generated on {}\n".format(datetime.now().strftime("%m/%d/%Y %H:%M:%S")))
        f.write(" */\n\n")

        guard = "__W_{}_DT_H".format(drive.upper())
        f.write("#ifndef {}\n".format(guard))
        f.write("#define {}\n\n".format(guard))

        f.write("#define {} {}\n\n".format(macro_prefix, n_nodes))

        f.write("int dt_feature_{}[] = {{\n  ".format(drive))
        f.write(", ".join(str(int(x)) for x in feature))
        f.write("\n};\n\n")

        f.write("long dt_threshold_{}[] = {{\n  ".format(drive))
        f.write(", ".join(str(int(x)) for x in thr_int))
        f.write("\n};\n\n")

        f.write("int dt_left_{}[] = {{\n  ".format(drive))
        f.write(", ".join(str(int(x)) for x in children_left))
        f.write("\n};\n\n")

        f.write("int dt_right_{}[] = {{\n  ".format(drive))
        f.write(", ".join(str(int(x)) for x in children_right))
        f.write("\n};\n\n")

        f.write("int dt_value_{}[] = {{\n  ".format(drive))
        f.write(", ".join(str(int(x)) for x in node_value))
        f.write("\n};\n\n")

        f.write("#endif /* {} */\n".format(guard))

    print("Header written: {}".format(header_path))
    return header_path


def main():
    parser = argparse.ArgumentParser(
        description="Train a DT surrogate and export C headers for client-level integration."
    )
    parser.add_argument(
        "-target",
        help="Training target: 'flashnet' to mimic FlashNet outputs, or 'label' for original labels",
        type=str,
        choices=["flashnet", "label"],
        default="flashnet",
    )
    parser.add_argument(
        "-dataset",
        help="Path to FlashNet training dataset CSV (e.g., mldrive0.csv)",
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

    args = parser.parse_args()
    if not os.path.exists(args.dataset):
        raise SystemExit("Dataset not found: {}".format(args.dataset))

    clf, feature_names = train_surrogate_tree(
        dataset_path=args.dataset,
        target=args.target,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
    )

    header_path = export_tree_to_header(
        clf=clf,
        feature_names=feature_names,
        workload=args.workload,
        drive=args.drive,
        output_dir=args.output_dir,
    )

    print("\n=== Surrogate DT training complete ===")
    print("Target mode: {}".format(args.target))
    print("Generated header: {}".format(header_path))


if __name__ == "__main__":
    main()

