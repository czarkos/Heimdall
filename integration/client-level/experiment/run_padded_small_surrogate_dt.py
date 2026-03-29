#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import pandas as pd

ALGORITHM = "padded_small_surrogate_dt"


def get_output_dir(trace_dir: str, devices: List[str], run_idx: int = -1) -> str:
    dev_names = [os.path.basename(d) for d in devices]
    base = os.path.join(str(trace_dir), "...".join(dev_names), ALGORITHM)
    if run_idx >= 0:
        return os.path.join(base, f"run_{run_idx}")
    return base


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


def start_processing(
    trace_dir: str, args, specific_workplace: str, run_idx: int = -1
) -> None:
    output_dir = get_output_dir(trace_dir, args.devices, run_idx=run_idx)
    devices_list_str = "-".join(args.devices)

    commands = []
    for idx, _device in enumerate(args.devices):
        trace_name = f"trace_{idx+1}.trace"
        stats_name = f"trace_{idx+1}.stats"
        trace_path = os.path.join(trace_dir, trace_name)
        stats_path = os.path.join(trace_dir, stats_name)
        duration = get_duration_from_trace(stats_path)
        cmd = (
            f"cd {specific_workplace}/; "
            f"sudo ./replay_dt.sh -user $USER -original_device_index {idx} "
            f"-devices_list {devices_list_str} -trace {trace_path} "
            f"-output_dir {output_dir} -duration {duration}; exit"
        )
        commands.append(cmd)

    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        for command in commands:
            executor.submit(run_command, command)

    subprocess.run("stty sane", shell=True, check=True)


def delete_dir(path: str) -> bool:
    try:
        shutil.rmtree(path)
    except OSError as e:
        print(f"Error: {e}")
        return False
    return True


def ensure_dt_shared_source(specific_workplace: str) -> bool:
    tmp_parent = os.path.dirname(specific_workplace)
    dt_shared_dir = os.path.join(tmp_parent, "dt")
    if os.path.exists(dt_shared_dir):
        if not delete_dir(dt_shared_dir):
            return False
    try:
        subprocess.run(["cp", "-r", "./dt", dt_shared_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error copying dt shared source: {e}")
        return False
    return True


def copy_surrogate_weights(
    trace_dir: str, devices_list: List[str], specific_workplace: str
) -> bool:
    dev_names = [os.path.basename(d) for d in devices_list]
    dev_pair_dir = os.path.join(str(trace_dir), "...".join(dev_names))
    path_to_weights = os.path.join(
        dev_pair_dir, ALGORITHM, "training_results", "surrogate_headers"
    )

    for dev_id, _ in enumerate(dev_names):
        header_file_path = os.path.join(path_to_weights, f"w_Trace_dev_{dev_id}_dt.h")
        if not os.path.exists(header_file_path):
            print(f"header file: {header_file_path} not exist.")
            return False

    dst_dir = os.path.join(specific_workplace, "dt_weights_header")
    os.makedirs(dst_dir, exist_ok=True)
    for dev_id, _ in enumerate(dev_names):
        header_file_path = os.path.join(path_to_weights, f"w_Trace_dev_{dev_id}_dt.h")
        subprocess.run(["cp", header_file_path, dst_dir], check=True)
    return True


def copy_flashnet_headers(
    trace_dir: str, devices_list: List[str], specific_workplace: str
) -> bool:
    dev_names = [os.path.basename(d) for d in devices_list]
    dev_pair_dir = os.path.join(str(trace_dir), "...".join(dev_names))
    src_dir = os.path.join(
        dev_pair_dir, "flashnet", "training_results", "weights_header_2ssds"
    )
    dst_dir = os.path.join(specific_workplace, "2ssds_weights_header")
    os.makedirs(dst_dir, exist_ok=True)
    for dev_id, _ in enumerate(dev_names):
        src = os.path.join(src_dir, f"w_Trace_dev_{dev_id}.h")
        if not os.path.exists(src):
            print(f"flashnet header missing: {src}")
            return False
        subprocess.run(["cp", src, dst_dir], check=True)
    return True


def train_padded_small_surrogate_dt(trace_dir: str, devices: List[str]) -> bool:
    dev_names = [os.path.basename(d) for d in devices]
    train_command = ["./train_padded_small_surrogate_dt.sh"] + dev_names

    if os.path.exists(trace_dir):
        train_command.append(trace_dir)
    else:
        print(f"[Error] trace_dir not exist: {trace_dir}")
        return False

    print(f"training command: {train_command}")
    original_directory = os.getcwd()
    try:
        os.chdir(f"./{ALGORITHM}")
        subprocess.run(train_command, check=True)
    except subprocess.CalledProcessError as train_e:
        print(f"Error running training: {train_e}")
        os.chdir(original_directory)
        return False

    os.chdir(original_directory)
    return True


def run_single_replay(
    trace_dir: str, args, run_idx: int = -1
) -> None:
    specific_workplace = "./tmp_running/{}_{}...{}".format(
        ALGORITHM, args.devices[0].split("/")[2], args.devices[1].split("/")[2]
    )
    if os.path.exists(specific_workplace):
        if not delete_dir(specific_workplace):
            raise SystemExit(1)

    try:
        subprocess.run(
            ["cp", "-r", f"./{ALGORITHM}", specific_workplace], check=True
        )
    except subprocess.CalledProcessError:
        raise SystemExit(1)

    if not ensure_dt_shared_source(specific_workplace):
        raise SystemExit(1)
    if not copy_surrogate_weights(trace_dir, args.devices, specific_workplace):
        raise SystemExit(1)
    if not copy_flashnet_headers(trace_dir, args.devices, specific_workplace):
        raise SystemExit(1)

    original_directory = os.getcwd()
    try:
        os.chdir(specific_workplace)
        try:
            subprocess.run(["make"], check=True)
        except subprocess.CalledProcessError:
            for file in Path("/tmp").glob("*.o"):
                file.unlink()
            subprocess.run(["make"], check=True)
    finally:
        os.chdir(original_directory)

    subprocess.run("stty sane", shell=True, check=True)
    start_processing(trace_dir, args, specific_workplace, run_idx=run_idx)

    if not delete_dir(specific_workplace):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-devices", nargs="+", type=str)
    parser.add_argument("-trace_dir", type=str)
    parser.add_argument("-trace_dirs", nargs="+", type=str)
    parser.add_argument("-resume", action="store_true")
    parser.add_argument("-only_training", action="store_true", default=False)
    parser.add_argument("-only_replaying", action="store_true", default=False)
    parser.add_argument("-reverse", action="store_true")
    parser.add_argument(
        "-num_runs",
        type=int,
        default=1,
        help="Number of replay repetitions per trace for statistical robustness.",
    )
    args = parser.parse_args()

    if (not args.devices) or (not (args.trace_dir or args.trace_dirs)):
        print("ERROR: provide -devices and -trace_dir/-trace_dirs")
        raise SystemExit(1)

    trace_dirs: List[str] = []
    if args.trace_dirs:
        trace_dirs += args.trace_dirs
    elif args.trace_dir:
        trace_dirs.append(args.trace_dir)

    if args.reverse:
        trace_dirs = trace_dirs[::-1]

    dev_names = [os.path.basename(d) for d in args.devices]

    for idx, trace_dir in enumerate(trace_dirs):
        print(f"\nProcessing trace dir {idx+1} out of {len(trace_dirs)}")

        dev_pair_dir = os.path.join(str(trace_dir), "...".join(dev_names))
        weights_dir = os.path.join(
            dev_pair_dir, ALGORITHM, "training_results", "surrogate_headers"
        )
        output_weights_0 = os.path.join(weights_dir, "w_Trace_dev_0_dt.h")
        output_weights_1 = os.path.join(weights_dir, "w_Trace_dev_1_dt.h")
        flashnet_header_dir = os.path.join(
            dev_pair_dir, "flashnet", "training_results", "weights_header_2ssds"
        )
        fn_header_0 = os.path.join(flashnet_header_dir, "w_Trace_dev_0.h")
        fn_header_1 = os.path.join(flashnet_header_dir, "w_Trace_dev_1.h")

        weights_ready = os.path.isfile(output_weights_0) and os.path.isfile(
            output_weights_1
        )

        # Training phase (once per trace)
        if not args.only_replaying:
            if not (args.resume and weights_ready):
                if not train_padded_small_surrogate_dt(trace_dir, args.devices):
                    raise SystemExit(1)
                weights_ready = os.path.isfile(output_weights_0) and os.path.isfile(
                    output_weights_1
                )

        if args.only_training:
            continue

        if not weights_ready:
            print("     surrogate weights not ready, skipping")
            continue
        if not os.path.isfile(fn_header_0) or not os.path.isfile(fn_header_1):
            print("     flashnet headers not ready, skipping")
            continue

        # Replay phase (N runs)
        use_run_dirs = args.num_runs > 1
        for run_idx in range(args.num_runs):
            effective_run_idx = run_idx if use_run_dirs else -1
            output_dir = get_output_dir(
                trace_dir, args.devices, run_idx=effective_run_idx
            )
            output_stat_path = os.path.join(output_dir, "trace_1.trace.stats")

            if args.resume and os.path.isfile(output_stat_path):
                print(
                    f"     Run {run_idx}: already replayed, skipping"
                )
                continue

            print(f"     Run {run_idx}/{args.num_runs} for {trace_dir}")
            subprocess.run("stty sane", shell=True, check=True)
            run_single_replay(trace_dir, args, run_idx=effective_run_idx)
