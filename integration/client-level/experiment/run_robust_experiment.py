#!/usr/bin/env python3
"""
Master orchestration script for statistically robust experiments.

Runs training (once) and then 10x replay for each of the four target algorithms:
  1. flashnet
  2. small_surrogate_dt_depth5
  3. small_hierarchy_p95  (depth-5 surrogate + FlashNet fallback, tau=p95)
  4. padded_small_surrogate_dt  (depth-5 surrogate + shadow FlashNet padding)

Also runs baseline once (needed as the control group and by hedging).

Usage examples:
  # Full run: train + 10x replay
  python3 run_robust_experiment.py \
      -devices /dev/nvme0n1 /dev/nvme2n1 \
      -trace_dirs /path/to/data/*/*/* \
      -num_runs 10

  # Replay only (training already done)
  python3 run_robust_experiment.py \
      -devices /dev/nvme0n1 /dev/nvme2n1 \
      -trace_dirs /path/to/data/*/*/* \
      -num_runs 10 \
      -only_replaying

  # Training only (no replays)
  python3 run_robust_experiment.py \
      -devices /dev/nvme0n1 /dev/nvme2n1 \
      -trace_dirs /path/to/data/*/*/* \
      -only_training
"""

import argparse
import subprocess
import sys
from typing import List


def run_step(command: List[str], title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("Command:", " ".join(command))
    print("=" * 80, flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"[WARNING] Step failed with exit code {result.returncode}: {title}")
        print("Continuing to next step...")


def parse_trace_args(args: argparse.Namespace) -> List[str]:
    out: List[str] = []
    if args.trace_dirs:
        out.extend(["-trace_dirs"] + args.trace_dirs)
    elif args.trace_dir:
        out.extend(["-trace_dir", args.trace_dir])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master script for statistically robust experiments (train + Nx replay)."
    )
    parser.add_argument("-devices", nargs="+", type=str, required=True)
    parser.add_argument("-trace_dir", type=str)
    parser.add_argument("-trace_dirs", nargs="+", type=str)
    parser.add_argument("-resume", action="store_true")
    parser.add_argument("-reverse", action="store_true")
    parser.add_argument(
        "-num_runs",
        type=int,
        default=10,
        help="Number of replay repetitions per trace (default: 10).",
    )
    parser.add_argument(
        "-only_training",
        action="store_true",
        default=False,
        help="Only run training, skip all replays.",
    )
    parser.add_argument(
        "-only_replaying",
        action="store_true",
        default=False,
        help="Skip training, only run replays (assumes models are already trained).",
    )
    parser.add_argument(
        "-skip_baseline",
        action="store_true",
        default=False,
        help="Skip baseline replay (if already done separately).",
    )
    args = parser.parse_args()

    if not (args.trace_dir or args.trace_dirs):
        print("ERROR: provide -trace_dir or -trace_dirs")
        raise SystemExit(1)

    base_args = ["-devices"] + args.devices + parse_trace_args(args)
    optional_flags: List[str] = []
    if args.resume:
        optional_flags.append("-resume")
    if args.reverse:
        optional_flags.append("-reverse")

    # ======================================================================
    # PHASE 1: TRAINING
    # ======================================================================
    if not args.only_replaying:
        print("\n" + "#" * 80)
        print("# PHASE 1: TRAINING")
        print("#" * 80)

        # 1a. Train FlashNet (prerequisite for all surrogate/padded variants)
        run_step(
            ["python3", "run_flashnet.py"]
            + base_args
            + optional_flags
            + ["-only_training"],
            "Train FlashNet",
        )

        # 1b. Train small surrogate DT depth-5
        run_step(
            [
                "python3",
                "small_surrogate_dt/run_small_surrogate_dt.py",
            ]
            + base_args
            + optional_flags
            + ["-only_training", "-max_depth", "5", "-algo_name", "small_surrogate_dt_depth5"],
            "Train small surrogate DT (depth=5)",
        )

        # 1c. Train small hierarchy p95 (depth-5 DT + uncertainty headers)
        run_step(
            [
                "python3",
                "small_surrogate_dt/run_small_hierarchy.py",
            ]
            + base_args
            + optional_flags
            + [
                "-only_training",
                "-algorithm_name",
                "small_hierarchy_p95",
                "-tau_percentile",
                "95.0",
                "-max_depth",
                "5",
            ],
            "Train small hierarchy p95 (depth=5)",
        )

        # 1d. Train padded small surrogate DT (depth-5)
        run_step(
            ["python3", "run_padded_small_surrogate_dt.py"]
            + base_args
            + optional_flags
            + ["-only_training"],
            "Train padded small surrogate DT (depth=5)",
        )

    if args.only_training:
        print("\n[Training-only mode] All training complete.")
        return

    # ======================================================================
    # PHASE 2: REPLAY (Nx)
    # ======================================================================
    print("\n" + "#" * 80)
    print(f"# PHASE 2: REPLAY ({args.num_runs}x)")
    print("#" * 80)

    num_runs_args = ["-num_runs", str(args.num_runs)]

    # 2a. Baseline (single run, no multi-run needed since it's the control)
    if not args.skip_baseline:
        run_step(
            ["python3", "run_baseline.py"]
            + base_args
            + optional_flags,
            "Replay baseline (1x)",
        )

    # 2b. FlashNet replay (Nx)
    run_step(
        ["python3", "run_flashnet.py"]
        + base_args
        + optional_flags
        + ["-only_replaying"]
        + num_runs_args,
        f"Replay FlashNet ({args.num_runs}x)",
    )

    # 2c. Small surrogate DT depth-5 replay (Nx)
    run_step(
        [
            "python3",
            "small_surrogate_dt/run_small_surrogate_dt.py",
        ]
        + base_args
        + optional_flags
        + [
            "-only_replaying",
            "-max_depth",
            "5",
            "-algo_name",
            "small_surrogate_dt_depth5",
        ]
        + num_runs_args,
        f"Replay small surrogate DT depth-5 ({args.num_runs}x)",
    )

    # 2d. Small hierarchy p95 (depth-5) replay (Nx)
    run_step(
        [
            "python3",
            "small_surrogate_dt/run_small_hierarchy.py",
        ]
        + base_args
        + optional_flags
        + [
            "-only_replaying",
            "-algorithm_name",
            "small_hierarchy_p95",
            "-tau_percentile",
            "95.0",
            "-max_depth",
            "5",
        ]
        + num_runs_args,
        f"Replay small hierarchy p95 depth-5 ({args.num_runs}x)",
    )

    # 2e. Padded small surrogate DT (depth-5) replay (Nx)
    run_step(
        ["python3", "run_padded_small_surrogate_dt.py"]
        + base_args
        + optional_flags
        + ["-only_replaying"]
        + num_runs_args,
        f"Replay padded small surrogate DT depth-5 ({args.num_runs}x)",
    )

    print("\n" + "=" * 80)
    print("All steps completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
