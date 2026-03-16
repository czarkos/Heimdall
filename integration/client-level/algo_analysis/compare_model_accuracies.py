#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


FLASHNET_ACCURACY_RE = re.compile(
    r"accuracy:\s*([0-9]*\.?[0-9]+)[^\n\r]*val_accuracy:\s*([0-9]*\.?[0-9]+)"
)
MLDRIVE_ID_RE = re.compile(r"mldrive(\d+)results\.txt$")
DEV_ID_RE = re.compile(r"dev_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FlashNet, surrogate_dt, small_surrogate_dt, and "
            "small_surrogate_dt_depth5 accuracies for corresponding models and "
            "write a CSV report."
        )
    )
    parser.add_argument(
        "--root",
        default="/mnt/heimdall-exp/Heimdall/integration/client-level/data",
        help="Root directory to scan recursively.",
    )
    parser.add_argument(
        "--csv-out",
        default="model_accuracy_comparison.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def parse_flashnet_final_acc(log_path: Path) -> Optional[Tuple[float, float]]:
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return None

    normalized = text.replace("\r", "\n")
    matches = FLASHNET_ACCURACY_RE.findall(normalized)
    if not matches:
        return None
    train_acc_str, val_acc_str = matches[-1]
    return float(train_acc_str), float(val_acc_str)


def extract_model_idx_from_flashnet(path: Path) -> str:
    m = MLDRIVE_ID_RE.search(path.name)
    return m.group(1) if m else ""


def extract_model_idx_from_dev_name(path: Path) -> str:
    m = DEV_ID_RE.search(path.name)
    return m.group(1) if m else ""


def extract_context(path: Path, algo_name: str) -> Tuple[str, str]:
    # Expected shape:
    # .../<trace_dir>/<device_pair>/<algo_name>/training_results/<file>
    parts = path.parts
    try:
        algo_idx = parts.index(algo_name)
    except ValueError:
        return "", ""

    device_pair = parts[algo_idx - 1] if algo_idx >= 1 else ""
    trace_dir = str(Path(*parts[: algo_idx - 1])) if algo_idx >= 2 else ""
    return trace_dir, device_pair


def set_metric(
    merged: Dict[
        Tuple[str, str, str], Dict[str, object]
    ],
    key: Tuple[str, str, str],
    metric_key: str,
    value: object,
) -> None:
    if key not in merged:
        merged[key] = {
            "trace_dir": key[0],
            "device_pair": key[1],
            "model_idx": key[2],
            "flashnet_train_accuracy": "",
            "flashnet_val_accuracy": "",
            "flashnet_fidelity_accuracy": "",
            "surrogate_dt_train_accuracy": "",
            "surrogate_dt_val_accuracy": "",
            "surrogate_dt_fidelity_accuracy": "",
            "small_surrogate_dt_train_accuracy": "",
            "small_surrogate_dt_val_accuracy": "",
            "small_surrogate_dt_fidelity_accuracy": "",
            "small_surrogate_dt_depth5_train_accuracy": "",
            "small_surrogate_dt_depth5_val_accuracy": "",
            "small_surrogate_dt_depth5_fidelity_accuracy": "",
            "delta_surrogate_dt_vs_flashnet_train": "",
            "delta_surrogate_dt_vs_flashnet_val": "",
            "delta_surrogate_dt_vs_flashnet_fidelity": "",
            "delta_small_surrogate_dt_vs_flashnet_train": "",
            "delta_small_surrogate_dt_vs_flashnet_val": "",
            "delta_small_surrogate_dt_vs_flashnet_fidelity": "",
            "delta_small_surrogate_dt_depth5_vs_flashnet_train": "",
            "delta_small_surrogate_dt_depth5_vs_flashnet_val": "",
            "delta_small_surrogate_dt_depth5_vs_flashnet_fidelity": "",
            "flashnet_log_path": "",
            "surrogate_dt_metrics_path": "",
            "small_surrogate_dt_metrics_path": "",
            "small_surrogate_dt_depth5_metrics_path": "",
            "status": "",
        }
    merged[key][metric_key] = value


def as_float(v: object) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def compute_delta(base: object, other: object) -> object:
    base_f = as_float(base)
    other_f = as_float(other)
    if base_f is None or other_f is None:
        return ""
    return round(other_f - base_f, 6)


def round_or_blank(v: Optional[float]) -> object:
    if v is None:
        return ""
    return round(v, 6)


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    merged: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    flashnet_logs = sorted(root.rglob("mldrive*results.txt"))
    flashnet_logs = [p for p in flashnet_logs if "/flashnet/training_results/" in p.as_posix()]
    flashnet_ok = 0
    flashnet_failed = 0

    for path in flashnet_logs:
        trace_dir, device_pair = extract_context(path, "flashnet")
        model_idx = extract_model_idx_from_flashnet(path)
        key = (trace_dir, device_pair, model_idx)

        parsed = parse_flashnet_final_acc(path)
        if parsed is None:
            flashnet_failed += 1
            set_metric(merged, key, "status", "flashnet_parse_failed")
            set_metric(merged, key, "flashnet_log_path", str(path))
            continue

        train_acc, val_acc = parsed
        flashnet_ok += 1
        set_metric(merged, key, "flashnet_train_accuracy", round(train_acc, 6))
        set_metric(merged, key, "flashnet_val_accuracy", round(val_acc, 6))
        # Fidelity for teacher model against itself is 1.0 by definition.
        set_metric(merged, key, "flashnet_fidelity_accuracy", 1.0)
        set_metric(merged, key, "flashnet_log_path", str(path))

    surrogate_metrics = sorted(root.rglob("surrogate_dev_*_metrics.json"))
    surrogate_metrics = [
        p for p in surrogate_metrics if "/surrogate_dt/training_results/" in p.as_posix()
    ]
    surrogate_ok = 0
    surrogate_failed = 0

    for path in surrogate_metrics:
        trace_dir, device_pair = extract_context(path, "surrogate_dt")
        model_idx = extract_model_idx_from_dev_name(path)
        key = (trace_dir, device_pair, model_idx)

        try:
            with path.open("r") as f:
                data = json.load(f)
        except Exception:
            surrogate_failed += 1
            set_metric(merged, key, "status", "surrogate_dt_parse_failed")
            set_metric(merged, key, "surrogate_dt_metrics_path", str(path))
            continue

        surrogate_ok += 1
        set_metric(
            merged,
            key,
            "surrogate_dt_train_accuracy",
            round_or_blank(as_float(data.get("dt_train_acc_gt"))),
        )
        set_metric(
            merged,
            key,
            "surrogate_dt_val_accuracy",
            round_or_blank(as_float(data.get("dt_test_acc_gt"))),
        )
        set_metric(
            merged,
            key,
            "surrogate_dt_fidelity_accuracy",
            round_or_blank(as_float(data.get("test_fidelity"))),
        )
        set_metric(merged, key, "surrogate_dt_metrics_path", str(path))

    small_surrogate_metrics = sorted(root.rglob("small_surrogate_dev_*_metrics.json"))
    small_surrogate_metrics = [
        p
        for p in small_surrogate_metrics
        if "/small_surrogate_dt/training_results/" in p.as_posix()
    ]
    small_surrogate_ok = 0
    small_surrogate_failed = 0

    for path in small_surrogate_metrics:
        trace_dir, device_pair = extract_context(path, "small_surrogate_dt")
        model_idx = extract_model_idx_from_dev_name(path)
        key = (trace_dir, device_pair, model_idx)

        try:
            with path.open("r") as f:
                data = json.load(f)
        except Exception:
            small_surrogate_failed += 1
            set_metric(merged, key, "status", "small_surrogate_dt_parse_failed")
            set_metric(merged, key, "small_surrogate_dt_metrics_path", str(path))
            continue

        small_surrogate_ok += 1
        set_metric(
            merged,
            key,
            "small_surrogate_dt_train_accuracy",
            round_or_blank(as_float(data.get("dt_train_acc_gt"))),
        )
        set_metric(
            merged,
            key,
            "small_surrogate_dt_val_accuracy",
            round_or_blank(as_float(data.get("dt_test_acc_gt"))),
        )
        set_metric(
            merged,
            key,
            "small_surrogate_dt_fidelity_accuracy",
            round_or_blank(as_float(data.get("test_fidelity"))),
        )
        set_metric(merged, key, "small_surrogate_dt_metrics_path", str(path))

    small_surrogate_depth5_metrics = sorted(root.rglob("small_surrogate_dev_*_metrics.json"))
    small_surrogate_depth5_metrics = [
        p
        for p in small_surrogate_depth5_metrics
        if "/small_surrogate_dt_depth5/training_results/" in p.as_posix()
    ]
    small_surrogate_depth5_ok = 0
    small_surrogate_depth5_failed = 0

    for path in small_surrogate_depth5_metrics:
        trace_dir, device_pair = extract_context(path, "small_surrogate_dt_depth5")
        model_idx = extract_model_idx_from_dev_name(path)
        key = (trace_dir, device_pair, model_idx)

        try:
            with path.open("r") as f:
                data = json.load(f)
        except Exception:
            small_surrogate_depth5_failed += 1
            set_metric(merged, key, "status", "small_surrogate_dt_depth5_parse_failed")
            set_metric(merged, key, "small_surrogate_dt_depth5_metrics_path", str(path))
            continue

        small_surrogate_depth5_ok += 1
        set_metric(
            merged,
            key,
            "small_surrogate_dt_depth5_train_accuracy",
            round_or_blank(as_float(data.get("dt_train_acc_gt"))),
        )
        set_metric(
            merged,
            key,
            "small_surrogate_dt_depth5_val_accuracy",
            round_or_blank(as_float(data.get("dt_test_acc_gt"))),
        )
        set_metric(
            merged,
            key,
            "small_surrogate_dt_depth5_fidelity_accuracy",
            round_or_blank(as_float(data.get("test_fidelity"))),
        )
        set_metric(merged, key, "small_surrogate_dt_depth5_metrics_path", str(path))

    rows: List[Dict[str, object]] = []
    for key in sorted(merged.keys()):
        row = merged[key]

        row["delta_surrogate_dt_vs_flashnet_train"] = compute_delta(
            row["flashnet_train_accuracy"], row["surrogate_dt_train_accuracy"]
        )
        row["delta_surrogate_dt_vs_flashnet_val"] = compute_delta(
            row["flashnet_val_accuracy"], row["surrogate_dt_val_accuracy"]
        )
        row["delta_surrogate_dt_vs_flashnet_fidelity"] = compute_delta(
            row["flashnet_fidelity_accuracy"], row["surrogate_dt_fidelity_accuracy"]
        )
        row["delta_small_surrogate_dt_vs_flashnet_train"] = compute_delta(
            row["flashnet_train_accuracy"], row["small_surrogate_dt_train_accuracy"]
        )
        row["delta_small_surrogate_dt_vs_flashnet_val"] = compute_delta(
            row["flashnet_val_accuracy"], row["small_surrogate_dt_val_accuracy"]
        )
        row["delta_small_surrogate_dt_vs_flashnet_fidelity"] = compute_delta(
            row["flashnet_fidelity_accuracy"], row["small_surrogate_dt_fidelity_accuracy"]
        )
        row["delta_small_surrogate_dt_depth5_vs_flashnet_train"] = compute_delta(
            row["flashnet_train_accuracy"], row["small_surrogate_dt_depth5_train_accuracy"]
        )
        row["delta_small_surrogate_dt_depth5_vs_flashnet_val"] = compute_delta(
            row["flashnet_val_accuracy"], row["small_surrogate_dt_depth5_val_accuracy"]
        )
        row["delta_small_surrogate_dt_depth5_vs_flashnet_fidelity"] = compute_delta(
            row["flashnet_fidelity_accuracy"],
            row["small_surrogate_dt_depth5_fidelity_accuracy"],
        )

        missing = []
        if row["flashnet_train_accuracy"] == "":
            missing.append("flashnet")
        if row["surrogate_dt_train_accuracy"] == "":
            missing.append("surrogate_dt")
        if row["small_surrogate_dt_train_accuracy"] == "":
            missing.append("small_surrogate_dt")
        if row["small_surrogate_dt_depth5_train_accuracy"] == "":
            missing.append("small_surrogate_dt_depth5")
        if not row["status"]:
            row["status"] = "ok" if not missing else "missing:" + "|".join(missing)
        rows.append(row)

    out_path = Path(args.csv_out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trace_dir",
                "device_pair",
                "model_idx",
                "flashnet_train_accuracy",
                "flashnet_val_accuracy",
                "flashnet_fidelity_accuracy",
                "surrogate_dt_train_accuracy",
                "surrogate_dt_val_accuracy",
                "surrogate_dt_fidelity_accuracy",
                "small_surrogate_dt_train_accuracy",
                "small_surrogate_dt_val_accuracy",
                "small_surrogate_dt_fidelity_accuracy",
                "small_surrogate_dt_depth5_train_accuracy",
                "small_surrogate_dt_depth5_val_accuracy",
                "small_surrogate_dt_depth5_fidelity_accuracy",
                "delta_surrogate_dt_vs_flashnet_train",
                "delta_surrogate_dt_vs_flashnet_val",
                "delta_surrogate_dt_vs_flashnet_fidelity",
                "delta_small_surrogate_dt_vs_flashnet_train",
                "delta_small_surrogate_dt_vs_flashnet_val",
                "delta_small_surrogate_dt_vs_flashnet_fidelity",
                "delta_small_surrogate_dt_depth5_vs_flashnet_train",
                "delta_small_surrogate_dt_depth5_vs_flashnet_val",
                "delta_small_surrogate_dt_depth5_vs_flashnet_fidelity",
                "flashnet_log_path",
                "surrogate_dt_metrics_path",
                "small_surrogate_dt_metrics_path",
                "small_surrogate_dt_depth5_metrics_path",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"FlashNet logs: discovered={len(flashnet_logs)} parsed={flashnet_ok} failed={flashnet_failed}")
    print(
        "surrogate_dt metrics: discovered={} parsed={} failed={}".format(
            len(surrogate_metrics), surrogate_ok, surrogate_failed
        )
    )
    print(
        "small_surrogate_dt metrics: discovered={} parsed={} failed={}".format(
            len(small_surrogate_metrics), small_surrogate_ok, small_surrogate_failed
        )
    )
    print(
        "small_surrogate_dt_depth5 metrics: discovered={} parsed={} failed={}".format(
            len(small_surrogate_depth5_metrics),
            small_surrogate_depth5_ok,
            small_surrogate_depth5_failed,
        )
    )
    print(f"Rows written: {len(rows)}")
    print(f"CSV written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
