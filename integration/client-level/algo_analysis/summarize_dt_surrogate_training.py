#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DT_TRAIN_RE = re.compile(r"Train accuracy:\s*([0-9]*\.?[0-9]+)")
DT_TEST_RE = re.compile(r"Test\s+accuracy:\s*([0-9]*\.?[0-9]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize DT and surrogate_dt training metrics across trace results "
            "and write a unified CSV report."
        )
    )
    parser.add_argument(
        "--root",
        default="/mnt/heimdall-exp/Heimdall/integration/client-level/data",
        help="Root directory to scan recursively.",
    )
    parser.add_argument(
        "--csv-out",
        default="dt_surrogate_training_summary.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def aggregate(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def path_context(path: Path, algo_name: str) -> Tuple[str, str]:
    # Expected path shape:
    # .../<trace_dir>/<device_pair>/<algo_name>/training_results/<file>
    parts = path.parts
    try:
        algo_idx = parts.index(algo_name)
        device_pair = parts[algo_idx - 1] if algo_idx >= 1 else ""
        trace_dir = str(Path(*parts[: algo_idx - 1])) if algo_idx >= 2 else ""
        return trace_dir, device_pair
    except ValueError:
        return "", ""


def infer_drive_id_from_name(path: Path) -> str:
    m = re.search(r"dev_(\d+)", path.name)
    return f"dev_{m.group(1)}" if m else ""


def parse_dt_stats_file(path: Path) -> Optional[Tuple[float, float]]:
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None
    train_match = DT_TRAIN_RE.search(text)
    test_match = DT_TEST_RE.search(text)
    if not train_match or not test_match:
        return None
    return float(train_match.group(1)), float(test_match.group(1))


def append_dt_rows(rows: List[Dict[str, object]], root: Path) -> Dict[str, object]:
    dt_files = sorted(root.rglob("dt_dev_*_training.stats"))
    dt_files = [p for p in dt_files if "/dt/training_results/" in p.as_posix()]

    parsed = 0
    failed = 0
    train_vals: List[float] = []
    test_vals: List[float] = []

    for path in dt_files:
        parsed_vals = parse_dt_stats_file(path)
        trace_dir, device_pair = path_context(path, "dt")
        drive_id = infer_drive_id_from_name(path)
        if parsed_vals is None:
            failed += 1
            rows.append(
                {
                    "level": "per_file",
                    "algo": "dt",
                    "file_path": str(path),
                    "trace_dir": trace_dir,
                    "device_pair": device_pair,
                    "drive_id": drive_id,
                    "train_accuracy": "",
                    "test_accuracy": "",
                    "train_fidelity": "",
                    "test_fidelity": "",
                    "teacher_train_acc_gt": "",
                    "teacher_test_acc_gt": "",
                    "surrogate_train_acc_gt": "",
                    "surrogate_test_acc_gt": "",
                    "status": "parse_failed",
                }
            )
            continue

        train_acc, test_acc = parsed_vals
        parsed += 1
        train_vals.append(train_acc)
        test_vals.append(test_acc)
        rows.append(
            {
                "level": "per_file",
                "algo": "dt",
                "file_path": str(path),
                "trace_dir": trace_dir,
                "device_pair": device_pair,
                "drive_id": drive_id,
                "train_accuracy": round(train_acc, 6),
                "test_accuracy": round(test_acc, 6),
                "train_fidelity": "",
                "test_fidelity": "",
                "teacher_train_acc_gt": "",
                "teacher_test_acc_gt": "",
                "surrogate_train_acc_gt": "",
                "surrogate_test_acc_gt": "",
                "status": "ok",
            }
        )

    train_stats = aggregate(train_vals)
    test_stats = aggregate(test_vals)
    rows.append(
        {
            "level": "algo_summary",
            "algo": "dt",
            "file_path": str(root),
            "trace_dir": "",
            "device_pair": "",
            "drive_id": "",
            "train_accuracy": round(train_stats["mean"], 6),
            "test_accuracy": round(test_stats["mean"], 6),
            "train_fidelity": "",
            "test_fidelity": "",
            "teacher_train_acc_gt": "",
            "teacher_test_acc_gt": "",
            "surrogate_train_acc_gt": "",
            "surrogate_test_acc_gt": "",
            "status": (
                "parsed={} failed={} "
                "train[min,max]=[{:.6f},{:.6f}] "
                "test[min,max]=[{:.6f},{:.6f}]"
            ).format(
                parsed,
                failed,
                train_stats["min"],
                train_stats["max"],
                test_stats["min"],
                test_stats["max"],
            ),
        }
    )

    return {
        "files": len(dt_files),
        "parsed": parsed,
        "failed": failed,
        "train_stats": train_stats,
        "test_stats": test_stats,
    }


def append_surrogate_rows(rows: List[Dict[str, object]], root: Path) -> Dict[str, object]:
    metrics_files = sorted(root.rglob("surrogate_dev_*_metrics.json"))
    metrics_files = [
        p for p in metrics_files if "/surrogate_dt/training_results/" in p.as_posix()
    ]

    parsed = 0
    failed = 0
    train_fidelity_vals: List[float] = []
    test_fidelity_vals: List[float] = []
    train_gt_vals: List[float] = []
    test_gt_vals: List[float] = []

    for path in metrics_files:
        trace_dir, device_pair = path_context(path, "surrogate_dt")
        drive_id = infer_drive_id_from_name(path)
        try:
            with path.open("r") as f:
                data = json.load(f)
        except Exception:
            failed += 1
            rows.append(
                {
                    "level": "per_file",
                    "algo": "surrogate_dt",
                    "file_path": str(path),
                    "trace_dir": trace_dir,
                    "device_pair": device_pair,
                    "drive_id": drive_id,
                    "train_accuracy": "",
                    "test_accuracy": "",
                    "train_fidelity": "",
                    "test_fidelity": "",
                    "teacher_train_acc_gt": "",
                    "teacher_test_acc_gt": "",
                    "surrogate_train_acc_gt": "",
                    "surrogate_test_acc_gt": "",
                    "status": "parse_failed",
                }
            )
            continue

        parsed += 1
        train_fidelity = data.get("train_fidelity")
        test_fidelity = data.get("test_fidelity")
        teacher_train = data.get("teacher_train_acc_gt")
        teacher_test = data.get("teacher_test_acc_gt")
        surrogate_train = data.get("dt_train_acc_gt")
        surrogate_test = data.get("dt_test_acc_gt")

        if isinstance(train_fidelity, (int, float)):
            train_fidelity_vals.append(float(train_fidelity))
        if isinstance(test_fidelity, (int, float)):
            test_fidelity_vals.append(float(test_fidelity))
        if isinstance(surrogate_train, (int, float)):
            train_gt_vals.append(float(surrogate_train))
        if isinstance(surrogate_test, (int, float)):
            test_gt_vals.append(float(surrogate_test))

        rows.append(
            {
                "level": "per_file",
                "algo": "surrogate_dt",
                "file_path": str(path),
                "trace_dir": trace_dir,
                "device_pair": device_pair,
                "drive_id": drive_id,
                "train_accuracy": "",
                "test_accuracy": "",
                "train_fidelity": round(float(train_fidelity), 6)
                if isinstance(train_fidelity, (int, float))
                else "",
                "test_fidelity": round(float(test_fidelity), 6)
                if isinstance(test_fidelity, (int, float))
                else "",
                "teacher_train_acc_gt": round(float(teacher_train), 6)
                if isinstance(teacher_train, (int, float))
                else "",
                "teacher_test_acc_gt": round(float(teacher_test), 6)
                if isinstance(teacher_test, (int, float))
                else "",
                "surrogate_train_acc_gt": round(float(surrogate_train), 6)
                if isinstance(surrogate_train, (int, float))
                else "",
                "surrogate_test_acc_gt": round(float(surrogate_test), 6)
                if isinstance(surrogate_test, (int, float))
                else "",
                "status": "ok",
            }
        )

    tr_fid_stats = aggregate(train_fidelity_vals)
    te_fid_stats = aggregate(test_fidelity_vals)
    tr_gt_stats = aggregate(train_gt_vals)
    te_gt_stats = aggregate(test_gt_vals)
    rows.append(
        {
            "level": "algo_summary",
            "algo": "surrogate_dt",
            "file_path": str(root),
            "trace_dir": "",
            "device_pair": "",
            "drive_id": "",
            "train_accuracy": "",
            "test_accuracy": "",
            "train_fidelity": round(tr_fid_stats["mean"], 6),
            "test_fidelity": round(te_fid_stats["mean"], 6),
            "teacher_train_acc_gt": "",
            "teacher_test_acc_gt": "",
            "surrogate_train_acc_gt": round(tr_gt_stats["mean"], 6),
            "surrogate_test_acc_gt": round(te_gt_stats["mean"], 6),
            "status": (
                "parsed={} failed={} "
                "train_fidelity[min,max]=[{:.6f},{:.6f}] "
                "test_fidelity[min,max]=[{:.6f},{:.6f}] "
                "surrogate_train_gt[min,max]=[{:.6f},{:.6f}] "
                "surrogate_test_gt[min,max]=[{:.6f},{:.6f}]"
            ).format(
                parsed,
                failed,
                tr_fid_stats["min"],
                tr_fid_stats["max"],
                te_fid_stats["min"],
                te_fid_stats["max"],
                tr_gt_stats["min"],
                tr_gt_stats["max"],
                te_gt_stats["min"],
                te_gt_stats["max"],
            ),
        }
    )

    return {
        "files": len(metrics_files),
        "parsed": parsed,
        "failed": failed,
        "train_fidelity_stats": tr_fid_stats,
        "test_fidelity_stats": te_fid_stats,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows: List[Dict[str, object]] = []

    dt_info = append_dt_rows(rows, root)
    surrogate_info = append_surrogate_rows(rows, root)

    out_path = Path(args.csv_out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "algo",
                "file_path",
                "trace_dir",
                "device_pair",
                "drive_id",
                "train_accuracy",
                "test_accuracy",
                "train_fidelity",
                "test_fidelity",
                "teacher_train_acc_gt",
                "teacher_test_acc_gt",
                "surrogate_train_acc_gt",
                "surrogate_test_acc_gt",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        "DT logs: discovered={files}, parsed={parsed}, failed={failed}".format(
            **dt_info
        )
    )
    print(
        "Surrogate logs: discovered={files}, parsed={parsed}, failed={failed}".format(
            **surrogate_info
        )
    )
    print(f"CSV written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
