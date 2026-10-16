#!/usr/bin/env python3
"""
Train and replay the depth-8 FlashNet surrogate decision tree.

Thin wrapper: the training (train_small_surrogate_dt_fidelity.py, distilled from
FlashNet's predictions) and the replay (the surrogate_dt C replayer, which walks a
tree of any depth) are the same as for the depth-5 surrogate, so this just runs
small_surrogate_dt/run_small_surrogate_dt.py with

    -max_depth 8 -algo_name surrogate_dt_depth8

Results land next to the other models:
    <trace>/<dev0>...<dev1>/surrogate_dt_depth8/training_results/   (headers, fidelity stats)
    <trace>/<dev0>...<dev1>/surrogate_dt_depth8/run_<i>/             (replays)

Usage:
    python3 run_surrogate_dt_depth8.py -devices /dev/nvme0n1 /dev/nvme1n1 \\
        -trace_dirs $HEIMDALL/integration/client-level/data/*/*/* -num_runs 10 -resume
    python3 run_surrogate_dt_depth8.py ... -only_training
    python3 run_surrogate_dt_depth8.py ... -only_replaying
"""

import argparse
import os
import subprocess
import sys

MAX_DEPTH = 8
ALGO_NAME = "surrogate_dt_depth8"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
RUNNER = os.path.join(EXPERIMENT_DIR, "small_surrogate_dt", "run_small_surrogate_dt.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and replay the depth-{} FlashNet surrogate DT.".format(MAX_DEPTH))
    parser.add_argument("-devices", nargs="+", type=str, required=True, help="e.g. /dev/nvme0n1 /dev/nvme1n1")
    parser.add_argument("-trace_dir", type=str)
    parser.add_argument("-trace_dirs", nargs="+", type=str)
    parser.add_argument("-num_runs", type=int, default=1, help="Replay repetitions per trace (default 1; use 10 for robust results)")
    parser.add_argument("-resume", action="store_true", help="Skip training/runs that already exist")
    parser.add_argument("-only_training", action="store_true")
    parser.add_argument("-only_replaying", action="store_true")
    parser.add_argument("-reverse", action="store_true", help="Process trace dirs in reverse order")
    args = parser.parse_args()

    if not (args.trace_dir or args.trace_dirs):
        sys.exit("ERROR: provide -trace_dir or -trace_dirs")
    if args.only_training and args.only_replaying:
        sys.exit("ERROR: -only_training and -only_replaying are mutually exclusive")

    cmd = ["python3", RUNNER, "-devices"] + args.devices
    if args.trace_dirs:
        cmd += ["-trace_dirs"] + args.trace_dirs
    else:
        cmd += ["-trace_dir", args.trace_dir]
    cmd += ["-max_depth", str(MAX_DEPTH), "-algo_name", ALGO_NAME, "-num_runs", str(args.num_runs)]
    for flag in ("resume", "only_training", "only_replaying", "reverse"):
        if getattr(args, flag):
            cmd.append("-" + flag)

    print("[{}] max_depth={} num_runs={}".format(ALGO_NAME, MAX_DEPTH, args.num_runs))
    print("[{}] running: {}".format(ALGO_NAME, " ".join(cmd)), flush=True)
    # Same working directory the robust experiment uses for this runner.
    result = subprocess.run(cmd, cwd=EXPERIMENT_DIR)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
