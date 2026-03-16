#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


RE_NODE_COUNT = re.compile(r"#define\s+DT_[A-Z0-9_]+_NODE_COUNT\s+(\d+)")
RE_FEATURE = re.compile(r"int\s+dt_feature_[A-Za-z0-9_]+\[\]\s*=\s*\{([^}]*)\};", re.S)
RE_LEFT = re.compile(r"int\s+dt_left_[A-Za-z0-9_]+\[\]\s*=\s*\{([^}]*)\};", re.S)
RE_RIGHT = re.compile(r"int\s+dt_right_[A-Za-z0-9_]+\[\]\s*=\s*\{([^}]*)\};", re.S)


def parse_int_array(blob: str) -> List[int]:
    vals = []
    for tok in blob.replace("\n", " ").split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    return vals


def compute_max_depth(left: List[int], right: List[int], root: int = 0) -> int:
    if not left or not right:
        return 0
    n = len(left)
    if root < 0 or root >= n:
        return 0

    # Iterative DFS: depth(root) = 0
    max_d = 0
    stack = [(root, 0)]
    visited_guard = 0
    while stack:
        node, depth = stack.pop()
        max_d = max(max_d, depth)
        visited_guard += 1
        if visited_guard > 10_000_000:  # safety guard against malformed cycles
            break
        l = left[node]
        r = right[node]
        if l >= 0:
            stack.append((l, depth + 1))
        if r >= 0:
            stack.append((r, depth + 1))
    return max_d


def infer_algo(path: Path) -> str:
    p = path.as_posix()
    for algo in ["surrogate_dt", "fixed_lat_dt", "padded_lat_dt", "hierarchy", "dt"]:
        if f"/{algo}/" in p:
            return algo
    return "unknown"


def infer_drive(path: Path) -> str:
    m = re.search(r"w_Trace_(dev_\d+)_dt\.h$", path.name)
    return m.group(1) if m else ""


def parse_header(path: Path) -> Optional[Dict[str, object]]:
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None

    m_count = RE_NODE_COUNT.search(text)
    m_feat = RE_FEATURE.search(text)
    m_left = RE_LEFT.search(text)
    m_right = RE_RIGHT.search(text)

    if not (m_count and m_feat and m_left and m_right):
        return None

    node_count = int(m_count.group(1))
    feature = parse_int_array(m_feat.group(1))
    left = parse_int_array(m_left.group(1))
    right = parse_int_array(m_right.group(1))

    # Prefer parsed length if macro is inconsistent.
    n = min(node_count, len(feature), len(left), len(right))
    feature = feature[:n]
    left = left[:n]
    right = right[:n]
    node_count = n

    leaf_count = sum(1 for f in feature if f < 0)
    internal_count = node_count - leaf_count
    max_depth = compute_max_depth(left, right)

    # Arrays stored in headers: feature, threshold, left, right, value
    stored_array_entries = 5 * node_count

    return {
        "node_count": node_count,
        "internal_node_count": internal_count,
        "leaf_count": leaf_count,
        "max_depth": max_depth,
        "stored_array_entries": stored_array_entries,
    }


def aggregate(vals: List[float]) -> Tuple[float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0
    return min(vals), max(vals), sum(vals) / len(vals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize structural characteristics of DT header files."
    )
    parser.add_argument(
        "--root",
        default="/mnt/heimdall-exp/Heimdall/integration/client-level/data",
        help="Root directory to scan recursively.",
    )
    parser.add_argument(
        "--csv-out",
        default="decision_tree_characteristics.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    headers = sorted(root.rglob("w_Trace_dev_*_dt.h"))
    rows: List[Dict[str, object]] = []

    for header in headers:
        parsed = parse_header(header)
        if parsed is None:
            rows.append(
                {
                    "level": "per_header",
                    "algo": infer_algo(header),
                    "drive_id": infer_drive(header),
                    "node_count": "",
                    "internal_node_count": "",
                    "leaf_count": "",
                    "max_depth": "",
                    "stored_array_entries": "",
                    "header_path": str(header),
                    "status": "parse_failed",
                }
            )
            continue

        rows.append(
            {
                "level": "per_header",
                "algo": infer_algo(header),
                "drive_id": infer_drive(header),
                "node_count": parsed["node_count"],
                "internal_node_count": parsed["internal_node_count"],
                "leaf_count": parsed["leaf_count"],
                "max_depth": parsed["max_depth"],
                "stored_array_entries": parsed["stored_array_entries"],
                "header_path": str(header),
                "status": "ok",
            }
        )

    # Per-algorithm summary rows
    algos = sorted({row["algo"] for row in rows if row.get("algo")})
    for algo in algos:
        ok_rows = [r for r in rows if r["level"] == "per_header" and r["algo"] == algo and r["status"] == "ok"]
        if not ok_rows:
            continue

        node_vals = [float(r["node_count"]) for r in ok_rows]
        depth_vals = [float(r["max_depth"]) for r in ok_rows]
        entry_vals = [float(r["stored_array_entries"]) for r in ok_rows]
        nmin, nmax, nmean = aggregate(node_vals)
        dmin, dmax, dmean = aggregate(depth_vals)
        emin, emax, emean = aggregate(entry_vals)

        rows.append(
            {
                "level": "algo_summary",
                "algo": algo,
                "drive_id": "",
                "node_count": round(nmean, 3),
                "internal_node_count": "",
                "leaf_count": "",
                "max_depth": round(dmean, 3),
                "stored_array_entries": round(emean, 3),
                "header_path": str(root),
                "status": (
                    f"count={len(ok_rows)} "
                    f"nodes[min,max]=[{int(nmin)},{int(nmax)}] "
                    f"depth[min,max]=[{int(dmin)},{int(dmax)}] "
                    f"entries[min,max]=[{int(emin)},{int(emax)}]"
                ),
            }
        )

    out_path = Path(args.csv_out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "algo",
                "drive_id",
                "node_count",
                "internal_node_count",
                "leaf_count",
                "max_depth",
                "stored_array_entries",
                "header_path",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    ok_count = len([r for r in rows if r["level"] == "per_header" and r["status"] == "ok"])
    fail_count = len([r for r in rows if r["level"] == "per_header" and r["status"] != "ok"])
    print(f"Headers discovered: {len(headers)}")
    print(f"Parsed successfully: {ok_count}")
    print(f"Parse failed: {fail_count}")
    print(f"CSV written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
