#!/usr/bin/env python3
"""
Produce two summary CSVs by scanning the raw trace data:

  1. accuracy_flashnet_vs_dt_depth5.csv
     FlashNet accuracy vs surrogate DT (depth 5) accuracy, plus fidelity.

  2. hierarchy_fallback_rates.csv
     Fallback rates for the small_hierarchy_p95 model (aggregated across runs).

Usage:
    python3 summarize_models.py [--root <data_dir>]
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DATA_ROOT = "/mnt/heimdall-exp/Heimdall/integration/client-level/data"

FLASHNET_ACCURACY_RE = re.compile(
    r"accuracy:\s*([0-9]*\.?[0-9]+)[^\n\r]*val_accuracy:\s*([0-9]*\.?[0-9]+)"
)
MLDRIVE_ID_RE = re.compile(r"mldrive(\d+)results\.txt$")
DEV_ID_RE = re.compile(r"dev_(\d+)")
INF_RE = re.compile(r"Hierarchy inference count\s*=\s*(\d+)")
FALLBACK_RE = re.compile(r"Hierarchy flashnet fallback count\s*=\s*(\d+)")
RATE_RE = re.compile(r"Hierarchy flashnet fallback rate\s*=\s*([0-9.]+)\s*%")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    p.add_argument(
        "--root",
        default=DATA_ROOT,
        help="Root directory containing the trace data.",
    )
    p.add_argument(
        "--out-accuracy",
        default=str(here / "accuracy_flashnet_vs_dt_depth5.csv"),
    )
    p.add_argument(
        "--out-fallback",
        default=str(here / "hierarchy_fallback_rates.csv"),
    )
    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────

def shorten_path(full: str, root: str) -> str:
    prefix = root.rstrip("/") + "/"
    return full[len(prefix):] if full.startswith(prefix) else full


def as_float(v: object) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def round_or_blank(v: Optional[float]) -> object:
    return "" if v is None else round(v, 6)


def compute_delta(base: object, other: object) -> object:
    bf = as_float(base)
    of = as_float(other)
    if bf is None or of is None:
        return ""
    return round(of - bf, 6)


def extract_context(path: Path, algo_name: str) -> Tuple[str, str]:
    parts = path.parts
    try:
        algo_idx = parts.index(algo_name)
    except ValueError:
        return "", ""
    device_pair = parts[algo_idx - 1] if algo_idx >= 1 else ""
    trace_dir = str(Path(*parts[: algo_idx - 1])) if algo_idx >= 2 else ""
    return trace_dir, device_pair


# ── CSV 1: accuracy comparison ───────────────────────────────────────────────

def parse_flashnet_final_acc(log_path: Path) -> Optional[Tuple[float, float]]:
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return None
    normalized = text.replace("\r", "\n")
    matches = FLASHNET_ACCURACY_RE.findall(normalized)
    if not matches:
        return None
    train_str, val_str = matches[-1]
    return float(train_str), float(val_str)


ACC_FIELDS = [
    "trace_pair",
    "device_pair",
    "model_idx",
    "flashnet_train_acc",
    "flashnet_val_acc",
    "dt_depth5_train_acc",
    "dt_depth5_val_acc",
    "dt_depth5_fidelity",
    "delta_train_acc",
    "delta_val_acc",
]


def build_accuracy_csv(root: Path, dst_path: str) -> None:
    merged: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    def ensure_key(k: Tuple[str, str, str]) -> Dict[str, object]:
        if k not in merged:
            merged[k] = {f: "" for f in ACC_FIELDS}
            merged[k]["trace_pair"] = shorten_path(k[0], str(root))
            merged[k]["device_pair"] = k[1]
            merged[k]["model_idx"] = k[2]
        return merged[k]

    flashnet_logs = sorted(
        p for p in root.rglob("mldrive*results.txt")
        if "/flashnet/training_results/" in p.as_posix()
    )
    fn_ok = 0
    for path in flashnet_logs:
        trace_dir, device_pair = extract_context(path, "flashnet")
        m = MLDRIVE_ID_RE.search(path.name)
        model_idx = m.group(1) if m else ""
        key = (trace_dir, device_pair, model_idx)

        parsed = parse_flashnet_final_acc(path)
        if parsed is None:
            continue
        fn_ok += 1
        row = ensure_key(key)
        row["flashnet_train_acc"] = round(parsed[0], 6)
        row["flashnet_val_acc"] = round(parsed[1], 6)

    dt5_files = sorted(
        p for p in root.rglob("small_surrogate_dev_*_metrics.json")
        if "/small_surrogate_dt_depth5/training_results/" in p.as_posix()
    )
    dt5_ok = 0
    for path in dt5_files:
        trace_dir, device_pair = extract_context(path, "small_surrogate_dt_depth5")
        m = DEV_ID_RE.search(path.name)
        model_idx = m.group(1) if m else ""
        key = (trace_dir, device_pair, model_idx)

        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        dt5_ok += 1
        row = ensure_key(key)
        row["dt_depth5_train_acc"] = round_or_blank(as_float(data.get("dt_train_acc_gt")))
        row["dt_depth5_val_acc"] = round_or_blank(as_float(data.get("dt_test_acc_gt")))
        row["dt_depth5_fidelity"] = round_or_blank(as_float(data.get("test_fidelity")))

    rows: List[Dict[str, object]] = []
    for key in sorted(merged):
        row = merged[key]
        if not row["flashnet_train_acc"] and not row["dt_depth5_train_acc"]:
            continue
        row["delta_train_acc"] = compute_delta(row["flashnet_train_acc"], row["dt_depth5_train_acc"])
        row["delta_val_acc"] = compute_delta(row["flashnet_val_acc"], row["dt_depth5_val_acc"])
        rows.append(row)

    with open(dst_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ACC_FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_both = sum(1 for r in rows if r["dt_depth5_train_acc"])
    print(f"[accuracy] flashnet logs parsed: {fn_ok}, dt_depth5 metrics parsed: {dt5_ok}")
    print(f"[accuracy] {len(rows)} rows written ({n_both} with both models) -> {dst_path}")


# ── CSV 2: hierarchy fallback rates ──────────────────────────────────────────

def parse_stat_file(path: Path) -> Optional[Tuple[int, int, float]]:
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None
    m_inf = INF_RE.search(text)
    m_fb = FALLBACK_RE.search(text)
    if not m_inf or not m_fb:
        return None
    inference = int(m_inf.group(1))
    fallback = int(m_fb.group(1))
    m_rate = RATE_RE.search(text)
    rate = float(m_rate.group(1)) if m_rate else (fallback / inference * 100.0 if inference else 0.0)
    return inference, fallback, rate


FB_FIELDS = [
    "trace_pair",
    "inference_count",
    "fallback_count",
    "fallback_rate_pct",
]


def build_fallback_csv(root: Path, dst_path: str) -> None:
    stat_files = sorted(root.rglob("small_hierarchy_p95/run_*/trace_*.trace.stats"))
    if not stat_files:
        stat_files = sorted(root.rglob("small_hierarchy_p95/trace_*.trace.stats"))
    if not stat_files:
        print("[fallback] No small_hierarchy_p95 trace stats found, skipping.")
        return

    # Group by the small_hierarchy_p95 directory (above run_*)
    group_totals: Dict[str, Dict[str, int]] = {}
    global_inf = 0
    global_fb = 0
    parsed_count = 0

    for sf in stat_files:
        parsed = parse_stat_file(sf)
        if parsed is None:
            continue
        parsed_count += 1
        inf, fb, _ = parsed
        global_inf += inf
        global_fb += fb

        # Walk up to the small_hierarchy_p95 directory
        hierarchy_dir = sf.parent
        while hierarchy_dir.name != "small_hierarchy_p95" and hierarchy_dir != root:
            hierarchy_dir = hierarchy_dir.parent
        label = shorten_path(str(hierarchy_dir), str(root))
        if label.endswith("/small_hierarchy_p95"):
            label = label[: -len("/small_hierarchy_p95")]

        if label not in group_totals:
            group_totals[label] = {"inference": 0, "fallback": 0}
        group_totals[label]["inference"] += inf
        group_totals[label]["fallback"] += fb

    rows: List[Dict[str, object]] = []
    for label in sorted(group_totals):
        inf = group_totals[label]["inference"]
        fb = group_totals[label]["fallback"]
        rate = (fb / inf * 100.0) if inf > 0 else 0.0
        rows.append({
            "trace_pair": label,
            "inference_count": inf,
            "fallback_count": fb,
            "fallback_rate_pct": round(rate, 4),
        })

    global_rate = (global_fb / global_inf * 100.0) if global_inf > 0 else 0.0
    rows.append({
        "trace_pair": "GLOBAL",
        "inference_count": global_inf,
        "fallback_count": global_fb,
        "fallback_rate_pct": round(global_rate, 4),
    })

    with open(dst_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FB_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"[fallback] Parsed {parsed_count} stats files across {len(group_totals)} trace pairs.")
    print(f"[fallback] Global: {global_inf} inferences, {global_fb} fallbacks, {global_rate:.2f}%")
    print(f"[fallback] {len(rows)} rows written -> {dst_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    root = Path(args.root)
    build_accuracy_csv(root, args.out_accuracy)
    build_fallback_csv(root, args.out_fallback)


if __name__ == "__main__":
    main()
