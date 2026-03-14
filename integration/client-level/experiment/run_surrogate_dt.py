#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import pandas as pd
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List

ALGORITHM = "surrogate_dt"


def get_output_dir(trace_dir, devices):
    dev_names = [os.path.basename(d) for d in devices]
    return os.path.join(str(trace_dir), "...".join(dev_names), ALGORITHM)


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


def start_processing(trace_dir: str, args, specific_workplace: str) -> None:
    output_dir = get_output_dir(trace_dir, args.devices)
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
        import shutil
        shutil.rmtree(path)
    except OSError as e:
        print(f"Error: {e}")
        return False
    return True


def copy_surrogate_weights(trace_dir: str, devices_list: List[str], specific_workplace: str) -> bool:
    dev_names = [os.path.basename(d) for d in devices_list]
    dev_pair_dir = os.path.join(str(trace_dir), "...".join(dev_names))
    path_to_weights = os.path.join(dev_pair_dir, "surrogate_dt", "training_results", "surrogate_headers")

    for dev_id, _ in enumerate(dev_names):
        header_file_path = os.path.join(path_to_weights, f"w_Trace_dev_{dev_id}_dt.h")
        if not os.path.exists(header_file_path):
            print(f"header file: {header_file_path} not exist.")
            return False

    tmp_weights_header_path = os.path.join(specific_workplace, "dt_weights_header")
    os.makedirs(tmp_weights_header_path, exist_ok=True)

    for dev_id, _ in enumerate(dev_names):
        header_file_path = os.path.join(path_to_weights, f"w_Trace_dev_{dev_id}_dt.h")
        subprocess.run(["cp", header_file_path, tmp_weights_header_path], check=True)
    return True


def train_surrogate_dt(trace_dir: str, devices: List[str]) -> bool:
    dev_names = [os.path.basename(d) for d in devices]
    train_command = ["./train_surrogate_dt_fidelity.sh"] + dev_names
    if os.path.exists(trace_dir):
        train_command.append(trace_dir)
    else:
        print(f"[Error] trace_dir not exist: {trace_dir}")
        return False

    original_directory = os.getcwd()
    try:
        os.chdir("./surrogate_dt")
        subprocess.run(train_command, check=True)
    except subprocess.CalledProcessError as train_e:
        print(f"Error running surrogate DT training: {train_e}")
        os.chdir(original_directory)
        return False

    os.chdir(original_directory)
    return True


def do_sleep(timer_mins: int, time_start: pd.Timestamp) -> None:
    time_elapsed = (pd.Timestamp.now() - time_start).seconds
    if time_elapsed < timer_mins * 60:
        import time as _time
        _time.sleep(timer_mins * 60 - time_elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-devices", nargs="+", type=str)
    parser.add_argument("-trace_dir", type=str)
    parser.add_argument("-trace_dirs", nargs="+", type=str)
    parser.add_argument("-resume", action="store_true")
    parser.add_argument("-timer_mins", type=int, default=0)
    parser.add_argument("-only_training", action="store_true", default=False)
    parser.add_argument("-only_replaying", action="store_true", default=False)
    parser.add_argument("-if_model_updated", action="store_true", default=False)
    parser.add_argument("-reverse", action="store_true")
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

    for trace_dir in trace_dirs:
        output_dir = get_output_dir(trace_dir, args.devices)
        output_stat_path = os.path.join(output_dir, "trace_1.trace.stats")
        dev_pair_dir = os.path.join(str(trace_dir), "...".join(dev_names))
        dt_weights_dir = os.path.join(dev_pair_dir, "surrogate_dt", "training_results", "surrogate_headers")
        output_weights_path_0 = os.path.join(dt_weights_dir, "w_Trace_dev_0_dt.h")
        output_weights_path_1 = os.path.join(dt_weights_dir, "w_Trace_dev_1_dt.h")

        if args.resume and os.path.isfile(output_stat_path) and os.path.isfile(output_weights_path_0) and os.path.isfile(output_weights_path_1):
            continue

        time_start = pd.Timestamp.now()

        if not args.only_replaying:
            if not (args.resume and os.path.isfile(output_weights_path_0) and os.path.isfile(output_weights_path_1)):
                if not train_surrogate_dt(trace_dir, args.devices):
                    raise SystemExit(1)

        if args.timer_mins > 0:
            do_sleep(args.timer_mins, time_start)
        if args.only_training:
            continue

        if not os.path.isfile(output_weights_path_0) or not os.path.isfile(output_weights_path_1):
            continue

        if os.path.isfile(output_stat_path) and args.if_model_updated:
            weight_modified_time = os.path.getmtime(output_weights_path_0)
            stat_modified_time = os.path.getmtime(output_stat_path)
            if weight_modified_time < stat_modified_time:
                continue

        specific_workplace = "./tmp_running/{}_{}...{}".format(
            ALGORITHM, args.devices[0].split("/")[2], args.devices[1].split("/")[2]
        )
        if os.path.exists(specific_workplace):
            if not delete_dir(specific_workplace):
                raise SystemExit(1)

        try:
            subprocess.run(["cp", "-r", "./{}".format(ALGORITHM), specific_workplace], check=True)
        except subprocess.CalledProcessError:
            raise SystemExit(1)

        if not copy_surrogate_weights(trace_dir, args.devices, specific_workplace):
            raise SystemExit(1)

        # Build surrogate replayer directly with gcc to avoid makefile
        # formatting issues (e.g., missing tab separators).
        original_directory = os.getcwd()
        try:
            os.chdir(specific_workplace)
            compile_cmd = [
                "gcc",
                "dt_algo.c",
                "io_replayer.c",
                "-o",
                "io_replayer_dt",
                "-lpthread",
            ]
            try:
                subprocess.run(compile_cmd, check=True)
            except subprocess.CalledProcessError:
                # Retry once after cleaning any stale object files in /tmp.
                for file in Path("/tmp").glob("*.o"):
                    file.unlink()
                subprocess.run(compile_cmd, check=True)
        finally:
            os.chdir(original_directory)

        subprocess.run("stty sane", shell=True, check=True)
        start_processing(trace_dir, args, specific_workplace)

        if not delete_dir(specific_workplace):
            raise SystemExit(1)

