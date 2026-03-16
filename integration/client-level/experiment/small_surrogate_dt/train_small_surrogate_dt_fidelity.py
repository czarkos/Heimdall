#!/usr/bin/env python3

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree


def prepare_dataset(dataset_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
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

    data_min = _load_csv_array(p_w0).reshape(-1)
    data_range = _load_csv_array(p_b0).reshape(-1)
    safe_range = np.where(np.abs(data_range) < 1e-12, 1.0, data_range)

    w1 = np.atleast_2d(_load_csv_array(p_w1))
    b1 = _load_csv_array(p_b1).reshape(-1)
    w2 = np.atleast_2d(_load_csv_array(p_w2))
    b2 = _load_csv_array(p_b2).reshape(-1)
    w3 = np.atleast_2d(_load_csv_array(p_w3))
    b3 = _load_csv_array(p_b3).reshape(-1)

    if w3.shape[1] != 1:
        if w3.shape[0] == 1:
            w3 = w3.T
        else:
            raise ValueError("Unexpected shape for weight_3: {}".format(w3.shape))

    x = x_raw.to_numpy(dtype=float)
    x_norm = (x - data_min) / safe_range

    z1 = np.maximum(0.0, x_norm @ w1 + b1)
    z2 = np.maximum(0.0, z1 @ w2 + b2)
    logits = z2 @ w3 + b3
    return (logits.reshape(-1) >= 0.0).astype(int)


def train_surrogate_tree(
    dataset_path: str,
    target: str = "flashnet",
    max_depth: Optional[int] = 15,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
) -> Tuple[DecisionTreeClassifier, List[str], Dict[str, float], str]:
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
        fidelity_report = classification_report(y_test_target, y_test_pred, digits=4)
        fidelity_confusion = confusion_matrix(y_test_target, y_test_pred)

        stats_lines = [
            "=== Small Surrogate DT (mimic FlashNet outputs) ===",
            "Train fidelity (DT vs FlashNet): {:.4f}".format(train_fidelity),
            "Test  fidelity (DT vs FlashNet): {:.4f}".format(test_fidelity),
            "Configured max_depth: {}".format(max_depth),
            "",
            "Accuracy against ground-truth labels (for reference)",
            "FlashNet teacher - Train: {:.4f}, Test: {:.4f}".format(
                teacher_train_acc_gt, teacher_test_acc_gt
            ),
            "Surrogate DT     - Train: {:.4f}, Test: {:.4f}".format(
                dt_train_acc_gt, dt_test_acc_gt
            ),
            "",
            "Classification report (DT vs teacher target, test set):",
            str(fidelity_report),
            "Confusion matrix (DT vs teacher target, test set):",
            str(fidelity_confusion),
        ]
        stats_text = "\n".join(stats_lines)
        print(stats_text)
        metrics: Dict[str, float] = {
            "max_depth": float(max_depth) if max_depth is not None else -1.0,
            "train_fidelity": float(train_fidelity),
            "test_fidelity": float(test_fidelity),
            "teacher_train_acc_gt": float(teacher_train_acc_gt),
            "teacher_test_acc_gt": float(teacher_test_acc_gt),
            "dt_train_acc_gt": float(dt_train_acc_gt),
            "dt_test_acc_gt": float(dt_test_acc_gt),
        }
    else:
        train_acc = accuracy_score(y_train_gt, y_train_pred)
        test_acc = accuracy_score(y_test_gt, y_test_pred)
        direct_report = classification_report(y_test_gt, y_test_pred, digits=4)
        direct_confusion = confusion_matrix(y_test_gt, y_test_pred)
        stats_lines = [
            "=== Small Surrogate DT (direct on labels) ===",
            "Train accuracy: {:.4f}".format(train_acc),
            "Test  accuracy: {:.4f}".format(test_acc),
            "Configured max_depth: {}".format(max_depth),
            "",
            "Classification report (test set):",
            str(direct_report),
            "Confusion matrix (test set):",
            str(direct_confusion),
        ]
        stats_text = "\n".join(stats_lines)
        print(stats_text)
        metrics = {
            "max_depth": float(max_depth) if max_depth is not None else -1.0,
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
        f.write("\n/* Small surrogate decision tree for {} / {} */\n".format(workload, drive))
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a small-depth surrogate DT and export C headers."
    )
    parser.add_argument(
        "-target",
        help="Training target: 'flashnet' to mimic FlashNet outputs, or 'label' for original labels",
        type=str,
        choices=["flashnet", "label"],
        default="flashnet",
    )
    parser.add_argument("-dataset", type=str, required=True)
    parser.add_argument("-workload", type=str, default="Trace")
    parser.add_argument("-drive", type=str, required=True)
    parser.add_argument("-output_dir", type=str, default="surrogate_headers")
    parser.add_argument(
        "-max_depth",
        type=int,
        default=15,
        help="Maximum depth of the decision tree (default: 15).",
    )
    parser.add_argument("-min_samples_split", type=int, default=2)
    parser.add_argument("-min_samples_leaf", type=int, default=1)
    parser.add_argument(
        "-metrics_output",
        help="Optional path to write metrics as JSON.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-stats_output",
        help="Optional path to write human-readable training stats.",
        type=str,
        default="",
    )

    args = parser.parse_args()
    if not os.path.exists(args.dataset):
        raise SystemExit("Dataset not found: {}".format(args.dataset))

    clf, feature_names, metrics, stats_text = train_surrogate_tree(
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

    print("\n=== Small surrogate DT training complete ===")
    print("Target mode: {}".format(args.target))
    print("Configured max_depth: {}".format(args.max_depth))
    print("Generated header: {}".format(header_path))

    if args.metrics_output:
        metrics_dir = os.path.dirname(os.path.abspath(args.metrics_output))
        os.makedirs(metrics_dir, exist_ok=True)
        with open(args.metrics_output, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print("Metrics JSON: {}".format(args.metrics_output))

    if args.stats_output:
        stats_dir = os.path.dirname(os.path.abspath(args.stats_output))
        os.makedirs(stats_dir, exist_ok=True)
        with open(args.stats_output, "w") as f:
            f.write(stats_text + "\n")
        print("Stats text: {}".format(args.stats_output))


if __name__ == "__main__":
    main()
