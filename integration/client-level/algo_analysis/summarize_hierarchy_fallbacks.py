#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

INF_RE = re.compile(r"Hierarchy inference count\s*=\s*(\d+)")
FALLBACK_RE = re.compile(r"Hierarchy flashnet fallback count\s*=\s*(\d+)")
RATE_RE = re.compile(r"Hierarchy flashnet fallback rate\s*=\s*([0-9.]+)\s*%")


def parse_stat_file(path: Path) -> Optional[Tuple[int, int, float]]:
    inference = None
    fallback = None
    rate = None

    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None

    m = INF_RE.search(text)
    if m:
        inference = int(m.group(1))
    m = FALLBACK_RE.search(text)
    if m:
        fallback = int(m.group(1))
    m = RATE_RE.search(text)
    if m:
        rate = float(m.group(1))

    if inference is None or fallback is None:
        return None

    if rate is None:
        rate = (fallback / inference * 100.0) if inference > 0 else 0.0

    return inference, fallback, rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize hierarchy fallback stats across all traces."
    )
    parser.add_argument(
        "--root",
        default="/mnt/heimdall-exp/Heimdall/integration/client-level/data",
        help="Root directory to scan recursively.",
    )
    parser.add_argument(
        "--csv-out",
        default="hierarchy_fallback_summary.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    stat_files = sorted(root.rglob("hierarchy/trace_*.trace.stats"))

    if not stat_files:
        print(f"No hierarchy trace stats found under: {root}")
        return

    rows: List[Dict[str, object]] = []
    group_totals: Dict[Path, Dict[str, int]] = {}

    global_inf = 0
    global_fb = 0

    for sf in stat_files:
        parsed = parse_stat_file(sf)
        if parsed is None:
            continue

        inf, fb, rate = parsed
        global_inf += inf
        global_fb += fb

        hierarchy_dir = sf.parent
        if hierarchy_dir not in group_totals:
            group_totals[hierarchy_dir] = {"inference": 0, "fallback": 0}
        group_totals[hierarchy_dir]["inference"] += inf
        group_totals[hierarchy_dir]["fallback"] += fb

        rows.append(
            {
                "level": "per_trace",
                "hierarchy_dir": str(hierarchy_dir),
                "stats_file": str(sf),
                "inference_count": inf,
                "fallback_count": fb,
                "fallback_rate_pct": round(rate, 4),
            }
        )

    # Per hierarchy folder combined summary
    for hdir, vals in sorted(group_totals.items(), key=lambda x: str(x[0])):
        inf = vals["inference"]
        fb = vals["fallback"]
        rate = (fb / inf * 100.0) if inf > 0 else 0.0
        rows.append(
            {
                "level": "per_hierarchy_dir",
                "hierarchy_dir": str(hdir),
                "stats_file": "",
                "inference_count": inf,
                "fallback_count": fb,
                "fallback_rate_pct": round(rate, 4),
            }
        )

    global_rate = (global_fb / global_inf * 100.0) if global_inf > 0 else 0.0
    rows.append(
        {
            "level": "global",
            "hierarchy_dir": str(root),
            "stats_file": "",
            "inference_count": global_inf,
            "fallback_count": global_fb,
            "fallback_rate_pct": round(global_rate, 4),
        }
    )

    out_path = Path(args.csv_out)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "hierarchy_dir",
                "stats_file",
                "inference_count",
                "fallback_count",
                "fallback_rate_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    parsed_count = len([r for r in rows if r["level"] == "per_trace"])
    print(f"Parsed {parsed_count} trace stats files.")
    print(f"Global inference count: {global_inf}")
    print(f"Global fallback count:  {global_fb}")
    print(f"Global fallback rate:   {global_rate:.2f}%")
    print(f"CSV written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()

