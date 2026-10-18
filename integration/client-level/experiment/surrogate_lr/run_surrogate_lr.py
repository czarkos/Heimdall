#!/usr/bin/env python3
"""
Train and replay the logistic-regression surrogate of FlashNet (surrogate_lr).

Mirrors small_surrogate_dt/run_small_surrogate_dt.py:
  1. train_surrogate_lr.sh writes the per-device headers under
         <trace>/<dev0>...<dev1>/<algo_name>/training_results/lr_headers/
  2. this directory (lr_algo.c + io_replayer.c) is copied to experiment/tmp_running/,
     the headers are dropped into lr_weights_header/, the replayer is built with gcc
  3. replay_lr.sh runs both clients concurrently, -num_runs times.

Outputs:
  <trace>/<dev0>...<dev1>/<algo_name>/trace_{1,2}.trace[.stats]      with -num_runs 1
  <trace>/<dev0>...<dev1>/<algo_name>/run_<i>/trace_{1,2}.trace[.stats]  with -num_runs > 1

Usage:
  python3 run_surrogate_lr.py -devices /dev/nvme0n1 /dev/nvme1n1 \
      -trace_dirs $HEIMDALL/integration/client-level/data/*/*/* -resume
  python3 run_surrogate_lr.py ... -num_runs 10 -resume      # robust runs; finished runs are skipped
  python3 run_surrogate_lr.py ... -only_training
  python3 run_surrogate_lr.py ... -only_replaying
"""

import argparse
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import pandas as pd

DEFAULT_ALGORITHM = "surrogate_lr"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ROOT = os.path.dirname(SCRIPT_DIR)
HEADER_FILES = ["w_Trace_dev_0_lr.h", "w_Trace_dev_1_lr.h"]
COMPILE_CMD = ["gcc", "lr_algo.c", "io_replayer.c", "-o", "io_replayer_lr", "-lpthread"]


def get_output_dir(trace_dir: str, devices: List[str], algo_name: str, run_idx: int = -1) -> str:
    dev_names = [os.path.basename(d) for d in devices]
    base = os.path.join(str(trace_dir), "...".join(dev_names), algo_name)
    if run_idx >= 0:
        return os.path.join(base, f"run_{run_idx}")
    return base


def get_header_dir(trace_dir: str, devices: List[str], algo_name: str) -> str:
    dev_names = [os.path.basename(d) for d in devices]
    return os.path.join(str(trace_dir), "...".join(dev_names), algo_name, "training_results", "lr_headers")


def run_command(command: str) -> None:
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")


def get_duration_from_trace(trace_path: str) -> str:
    with open(trace_path) as f:
        for line in f:
            if "Duration" in line:
                value_raw = line.split("=")[2]
                if "." in value_raw:
                    return re.findall(r"-?\d+\.\d+", value_raw)[0]
                return re.findall(r"-?\d+", value_raw)[0]
    raise RuntimeError(f"Duration line not found in {trace_path}")


def start_processing(trace_dir: str, args, specific_workplace: str, run_idx: int = -1) -> None:
    output_dir = get_output_dir(trace_dir, args.devices, args.algo_name, run_idx=run_idx)
    devices_list_str = "-".join(args.devices)

    commands = []
    for idx, _device in enumerate(args.devices):
        trace_path = os.path.join(trace_dir, f"trace_{idx+1}.trace")
        stats_path = os.path.join(trace_dir, f"trace_{idx+1}.stats")
        duration = get_duration_from_trace(stats_path)
        cmd = (
            f"cd {specific_workplace}/; "
            f"sudo ./replay_lr.sh -user $USER -original_device_index {idx} "
            f"-devices_list {devices_list_str} -trace {trace_path} "
            f"-output_dir {output_dir} -duration {duration}; exit"
        )
        commands.append(cmd)

    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        for command in commands:
            executor.submit(run_command, command)

    print("Output dir = " + output_dir)
    subprocess.run("stty sane", shell=True, check=True)


def delete_dir(path: str) -> bool:
    try:
        shutil.rmtree(path)
    except OSError as e:
        print(f"Error: {e}")
        return False
    return True


def copy_lr_headers(trace_dir: str, devices: List[str], specific_workplace: str, algo_name: str) -> bool:
    src_dir = get_header_dir(trace_dir, devices, algo_name)
    for name in HEADER_FILES:
        if not os.path.exists(os.path.join(src_dir, name)):
            print(f"header file: {os.path.join(src_dir, name)} not exist.")
            return False
    dst_dir = os.path.join(specific_workplace, "lr_weights_header")
    os.makedirs(dst_dir, exist_ok=True)
    for name in HEADER_FILES:
        shutil.copy(os.path.join(src_dir, name), dst_dir)
    return True


def train_surrogate_lr(trace_dir: str, devices: List[str], algo_name: str, C: float) -> bool:
    dev_names = [os.path.basename(d) for d in devices]
    train_command = [os.path.join(SCRIPT_DIR, "train_surrogate_lr.sh")] + dev_names
    if not os.path.exists(trace_dir):
        print(f"[Error] trace_dir not exist: {trace_dir}")
        return False
    train_command.append(trace_dir)

    env = os.environ.copy()
    env["SURROGATE_LR_ALGO_NAME"] = algo_name
    env["SURROGATE_LR_C"] = str(C)
    try:
        subprocess.run(train_command, check=True, env=env)
    except subprocess.CalledProcessError as train_e:
        print(f"Error running surrogate LR training: {train_e}")
        return False
    return True


def build_workplace(specific_workplace: str) -> None:
    original_directory = os.getcwd()
    try:
        os.chdir(specific_workplace)
        try:
            subprocess.run(COMPILE_CMD, check=True)
        except subprocess.CalledProcessError:
            for file in Path("/tmp").glob("*.o"):
                file.unlink()
            subprocess.run(COMPILE_CMD, check=True)
    finally:
        os.chdir(original_directory)


def do_sleep(timer_mins: int, time_start: pd.Timestamp) -> None:
    time_elapsed = (pd.Timestamp.now() - time_start).seconds
    if time_elapsed < timer_mins * 60:
        import time as _time
        _time.sleep(timer_mins * 60 - time_elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-devices", nargs="+", type=str)
    parser.add_argument("-trace_dir", type=str)
    parser.add_argument("-trace_dirs", nargs="+", type=str)
    parser.add_argument("-resume", action="store_true", help="Skip training/runs that already exist")
    parser.add_argument("-timer_mins", type=int, default=0)
    parser.add_argument("-only_training", action="store_true", default=False)
    parser.add_argument("-only_replaying", action="store_true", default=False)
    parser.add_argument("-if_model_updated", action="store_true", default=False)
    parser.add_argument("-reverse", action="store_true")
    parser.add_argument("-C", type=float, default=1.0, help="Inverse L2 regularization strength for training")
    parser.add_argument("-algo_name", type=str, default=DEFAULT_ALGORITHM,
                        help="Output subdirectory name under <trace>/<dev0...dev1>/")
    parser.add_argument("-num_runs", type=int, default=1,
                        help="Replay repetitions per trace (default 1; use 10 for robust results)")
    args = parser.parse_args()

    if (not args.devices) or (not (args.trace_dir or args.trace_dirs)):
        print("ERROR: provide -devices and -trace_dir/-trace_dirs")
        raise SystemExit(1)
    if args.only_training and args.only_replaying:
        raise SystemExit("ERROR: -only_training and -only_replaying are mutually exclusive")

    trace_dirs: List[str] = []
    if args.trace_dirs:
        trace_dirs += args.trace_dirs
    elif args.trace_dir:
        trace_dirs.append(args.trace_dir)
    if args.reverse:
        trace_dirs = trace_dirs[::-1]

    print("algo = " + args.algo_name)
    print("devices = " + str(args.devices))
    print("Found " + str(len(trace_dirs)) + " trace dirs")

    for idx, trace_dir in enumerate(trace_dirs):
        print(f"\nProcessing trace dir {idx + 1} out of {len(trace_dirs)}: {trace_dir}")
        output_dir = get_output_dir(trace_dir, args.devices, args.algo_name)
        output_stat_path = os.path.join(output_dir, "trace_1.trace.stats")
        header_dir = get_header_dir(trace_dir, args.devices, args.algo_name)
        header_paths = [os.path.join(header_dir, name) for name in HEADER_FILES]
        headers_exist = all(os.path.isfile(p) for p in header_paths)

        time_start = pd.Timestamp.now()

        if not args.only_replaying:
            if args.resume and headers_exist:
                print("     The surrogate LR model is trained, skipping training")
            elif not train_surrogate_lr(trace_dir, args.devices, args.algo_name, args.C):
                raise SystemExit(1)
            headers_exist = all(os.path.isfile(p) for p in header_paths)

        if args.timer_mins > 0:
            do_sleep(args.timer_mins, time_start)
        if args.only_training:
            continue

        if not headers_exist:
            print("     WARNING: surrogate LR headers are not ready, skipping replay")
            continue

        if os.path.isfile(output_stat_path) and args.if_model_updated:
            if os.path.getmtime(header_paths[0]) < os.path.getmtime(output_stat_path):
                print("     The model is OLDER than the replayed traces, skipping")
                continue

        use_run_dirs = args.num_runs > 1
        for run_i in range(args.num_runs):
            effective_run_idx = run_i if use_run_dirs else -1
            run_output_dir = get_output_dir(trace_dir, args.devices, args.algo_name, run_idx=effective_run_idx)
            run_stat_path = os.path.join(run_output_dir, "trace_1.trace.stats")
            if args.resume and os.path.isfile(run_stat_path):
                print(f"     Run {run_i}: already replayed, skipping")
                continue
            if use_run_dirs:
                print(f"     Run {run_i}/{args.num_runs}")

            specific_workplace = os.path.join(
                EXPERIMENT_ROOT, "tmp_running",
                "{}_{}...{}".format(args.algo_name, args.devices[0].split("/")[2], args.devices[1].split("/")[2]),
            )
            if os.path.exists(specific_workplace) and not delete_dir(specific_workplace):
                raise SystemExit(1)
            try:
                shutil.copytree(SCRIPT_DIR, specific_workplace, ignore=shutil.ignore_patterns("logs", "__pycache__"))
            except Exception as e:
                print(f"Failed to copy replay source: {e}")
                raise SystemExit(1)

            if not copy_lr_headers(trace_dir, args.devices, specific_workplace, args.algo_name):
                raise SystemExit(1)
            build_workplace(specific_workplace)

            subprocess.run("stty sane", shell=True, check=True)
            start_processing(trace_dir, args, specific_workplace, run_idx=effective_run_idx)

            if not delete_dir(specific_workplace):
                raise SystemExit(1)
