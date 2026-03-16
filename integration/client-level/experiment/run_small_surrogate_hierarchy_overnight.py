#!/usr/bin/env python3

import argparse
import subprocess
from typing import List, Tuple


def run_step(command: List[str], title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("Command:", " ".join(command))
    print("=" * 80)
    subprocess.run(command, check=True)


def parse_trace_args(args: argparse.Namespace) -> List[str]:
    out: List[str] = []
    if args.trace_dirs:
        out.extend(["-trace_dirs"] + args.trace_dirs)
    elif args.trace_dir:
        out.extend(["-trace_dir", args.trace_dir])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run small_surrogate_dt and small_hierarchy threshold variants "
            "(p95/p97/p98) overnight."
        )
    )
    parser.add_argument("-devices", nargs="+", type=str, required=True)
    parser.add_argument("-trace_dir", type=str)
    parser.add_argument("-trace_dirs", nargs="+", type=str)
    parser.add_argument("-resume", action="store_true")
    parser.add_argument("-timer_mins", type=int, default=0)
    parser.add_argument("-only_training", action="store_true", default=False)
    parser.add_argument("-only_replaying", action="store_true", default=False)
    parser.add_argument("-if_model_updated", action="store_true", default=False)
    parser.add_argument("-reverse", action="store_true")
    parser.add_argument("-max_depth", type=int, default=10)
    parser.add_argument("-ref_size", type=int, default=512)
    parser.add_argument("-seed", type=int, default=42)
    args = parser.parse_args()

    if not (args.trace_dir or args.trace_dirs):
        raise SystemExit("ERROR: provide -trace_dir or -trace_dirs")

    base_common = ["-devices"] + args.devices + parse_trace_args(args)

    optional_flags: List[str] = []
    if args.resume:
        optional_flags.append("-resume")
    if args.only_training:
        optional_flags.append("-only_training")
    if args.only_replaying:
        optional_flags.append("-only_replaying")
    if args.if_model_updated:
        optional_flags.append("-if_model_updated")
    if args.reverse:
        optional_flags.append("-reverse")
    if args.timer_mins > 0:
        optional_flags += ["-timer_mins", str(args.timer_mins)]

    # 1) Small surrogate DT replay/training baseline.
    cmd_small_surrogate = [
        "python3",
        "small_surrogate_dt/run_small_surrogate_dt.py",
    ] + base_common + optional_flags + ["-max_depth", str(args.max_depth)]
    run_step(cmd_small_surrogate, "Step 1/4: small_surrogate_dt")

    # 2-4) Small hierarchy variants with different uncertainty thresholds.
    variants: List[Tuple[str, float]] = [
        ("small_hierarchy_p95", 95.0),
        ("small_hierarchy_p97", 97.0),
        ("small_hierarchy_p98", 98.0),
    ]
    for idx, (algo_name, tau_pct) in enumerate(variants, start=2):
        cmd_variant = [
            "python3",
            "small_surrogate_dt/run_small_hierarchy.py",
        ] + base_common + optional_flags + [
            "-algorithm_name",
            algo_name,
            "-tau_percentile",
            str(tau_pct),
            "-max_depth",
            str(args.max_depth),
            "-ref_size",
            str(args.ref_size),
            "-seed",
            str(args.seed),
        ]
        run_step(cmd_variant, f"Step {idx}/4: {algo_name} (tau p{tau_pct})")

    print("\nAll overnight steps completed successfully.")


if __name__ == "__main__":
    main()

