#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ACCURACY_LINE_RE = re.compile(
    r"accuracy:\s*([0-9]*\.?[0-9]+)[^\n\r]*val_accuracy:\s*([0-9]*\.?[0-9]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize FlashNet training accuracy from mldrive*results.txt logs "
            "and write a CSV report."
        )
    )
    parser.add_argument(
        "--root",
        default="/mnt/heimdall-exp/Heimdall/integration/client-level/data",
        help="Root directory to scan recursively.",
    )
    parser.add_argument(
        "--csv-out",
        default="flashnet_training_accuracy_summary.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def infer_device_pair(log_path: Path) -> str:
    # Expected shape includes:
    # .../<trace>/<device_pair>/flashnet/training_results/mldriveXresults.txt
    parts = log_path.parts
    try:
        flashnet_idx = parts.index("flashnet")
        if flashnet_idx >= 1:
            return parts[flashnet_idx - 1]
    except ValueError:
        pass
    return ""


def infer_mldrive_id(log_path: Path) -> str:
    m = re.search(r"(mldrive\d+)results\.txt$", log_path.name)
    return m.group(1) if m else ""


def parse_accuracy_from_log(log_path: Path) -> Optional[Tuple[float, float, int]]:
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return None

    # Keras verbose output often contains carriage-return updates.
    normalized = text.replace("\r", "\n")
    matches = ACCURACY_LINE_RE.findall(normalized)
    if not matches:
        return None

    train_acc_str, val_acc_str = matches[-1]
    return float(train_acc_str), float(val_acc_str), len(matches)


def aggregate(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    log_files = sorted(root.rglob("mldrive*results.txt"))
    log_files = [p for p in log_files if "flashnet/training_results" in p.as_posix()]

    if not log_files:
        print(f"No FlashNet training logs found under: {root}")
        return

    rows: List[Dict[str, object]] = []
    train_vals: List[float] = []
    val_vals: List[float] = []
    parsed_count = 0
    skipped_count = 0

    for log_path in log_files:
        parsed = parse_accuracy_from_log(log_path)
        if parsed is None:
            skipped_count += 1
            rows.append(
                {
                    "level": "per_file",
                    "file_path": str(log_path),
                    "device_pair": infer_device_pair(log_path),
                    "mldrive_id": infer_mldrive_id(log_path),
                    "epochs_detected": 0,
                    "final_train_accuracy": "",
                    "final_val_accuracy": "",
                    "train_minus_val": "",
                    "status": "parse_failed",
                }
            )
            continue

        train_acc, val_acc, epochs = parsed
        parsed_count += 1
        train_vals.append(train_acc)
        val_vals.append(val_acc)

        rows.append(
            {
                "level": "per_file",
                "file_path": str(log_path),
                "device_pair": infer_device_pair(log_path),
                "mldrive_id": infer_mldrive_id(log_path),
                "epochs_detected": epochs,
                "final_train_accuracy": round(train_acc, 6),
                "final_val_accuracy": round(val_acc, 6),
                "train_minus_val": round(train_acc - val_acc, 6),
                "status": "ok",
            }
        )

    train_stats = aggregate(train_vals)
    val_stats = aggregate(val_vals)
    gap_stats = aggregate([t - v for t, v in zip(train_vals, val_vals)])

    rows.append(
        {
            "level": "global_summary",
            "file_path": str(root),
            "device_pair": "",
            "mldrive_id": "",
            "epochs_detected": "",
            "final_train_accuracy": round(train_stats["mean"], 6),
            "final_val_accuracy": round(val_stats["mean"], 6),
            "train_minus_val": round(gap_stats["mean"], 6),
            "status": (
                "parsed={} skipped={} "
                "train[min,max]=[{:.6f},{:.6f}] "
                "val[min,max]=[{:.6f},{:.6f}] "
                "gap[min,max]=[{:.6f},{:.6f}]"
            ).format(
                parsed_count,
                skipped_count,
                train_stats["min"],
                train_stats["max"],
                val_stats["min"],
                val_stats["max"],
                gap_stats["min"],
                gap_stats["max"],
            ),
        }
    )

    out_path = Path(args.csv_out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "file_path",
                "device_pair",
                "mldrive_id",
                "epochs_detected",
                "final_train_accuracy",
                "final_val_accuracy",
                "train_minus_val",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Discovered log files: {len(log_files)}")
    print(f"Parsed successfully:  {parsed_count}")
    print(f"Parse failed:         {skipped_count}")
    print(
        "Train accuracy min/max/mean: "
        f"{train_stats['min']:.6f}/{train_stats['max']:.6f}/{train_stats['mean']:.6f}"
    )
    print(
        "Val accuracy min/max/mean:   "
        f"{val_stats['min']:.6f}/{val_stats['max']:.6f}/{val_stats['mean']:.6f}"
    )
    print(f"CSV written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
