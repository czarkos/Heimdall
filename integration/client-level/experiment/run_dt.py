#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import pandas as pd
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List

ALGORITHM = "dt"


def assert_not_flashnet_output(path: str) -> None:
    normalized = os.path.abspath(path)
    if f"{os.sep}flashnet{os.sep}" in normalized:
        raise RuntimeError(
            f"Unsafe output path for dt pipeline (points to flashnet): {normalized}"
        )


def get_output_dir(trace_dir, devices):
    dev_names = [os.path.basename(d) for d in devices]
    output_dir = os.path.join(str(trace_dir), "...".join(dev_names), ALGORITHM)
    assert_not_flashnet_output(output_dir)
    return output_dir


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
                    duration = re.findall(r"-?\d+\.\d+", value_raw)[0]
                else:
                    duration = re.findall(r"-?\d+", value_raw)[0]
                return duration
    raise RuntimeError(f"Duration line not found in {trace_path}")


def start_processing(trace_dir: str, args, specific_workplace: str) -> None:
    print("Processing " + str(trace_dir))

    output_dir = get_output_dir(trace_dir, args.devices)
    assert_not_flashnet_output(output_dir)

    # Build devices_list string (e.g., "/dev/nvme0n1-/dev/nvme2n1")
    devices_list_str = "-".join(args.devices)
    print(f"The devices_list_str is {devices_list_str}")

    commands = []
    for idx, _device in enumerate(args.devices):
        cmd = "echo 'Starting client ' " + str(idx) + "; "
        cmd += "cd " + specific_workplace + "/; "

        trace_name = f"trace_{idx+1}.trace"
        stats_name = f"trace_{idx+1}.stats"
        trace_path = os.path.join(trace_dir, trace_name)
        stats_path = os.path.join(trace_dir, stats_name)
        duration = get_duration_from_trace(stats_path)

        cmd += (
            "sudo ./replay_dt.sh "
            f"-user $USER "
            f"-original_device_index {idx} "
            f"-devices_list {devices_list_str} "
            f"-trace {trace_path} "
            f"-output_dir {output_dir} "
            f"-duration {duration}; exit"
        )
        commands.append(cmd)

    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        for command in commands:
            executor.submit(run_command, command)

    print("Output dir = " + output_dir)
    subprocess.run("stty sane", shell=True, check=True)


def delete_dir(path: str) -> bool:
    try:
        import shutil

        shutil.rmtree(path)
        print(f"Directory '{path}' has been deleted.")
    except OSError as e:
        print(f"Error: {e}")
        return False
    return True


def copy_dt_weights(trace_dir: str, devices_list: List[str], specific_workplace: str) -> bool:
    """
    Copy per-workload standard DT headers from the DT training directory
    into the temporary dt workplace.

    Expected source:
      <trace_dir>/<dev0>...<dev1>/dt/training_results/dt_weights_header/w_Trace_dev_*.h

    Destination:
      <specific_workplace>/dt_weights_header/w_Trace_dev_*.h
    """
    dev_names = [os.path.basename(d) for d in devices_list]
    dev_pair_dir = os.path.join(str(trace_dir), "...".join(dev_names))
    path_to_weights = os.path.join(
        dev_pair_dir, "dt", "training_results", "dt_weights_header"
    )

    # Check existence of source headers
    for dev_id, _dev_name in enumerate(dev_names):
        header_file_path = os.path.join(path_to_weights, f"w_Trace_dev_{dev_id}_dt.h")
        print("DT header file path = {}".format(header_file_path))
        if not os.path.exists(header_file_path):
            print(f"header file: {header_file_path} not exist.")
            return False

    # Destination directory in temporary workplace
    tmp_weights_header_path = os.path.join(specific_workplace, "dt_weights_header")
    os.makedirs(tmp_weights_header_path, exist_ok=True)

    for dev_id, _dev_name in enumerate(dev_names):
        header_file_path = os.path.join(path_to_weights, f"w_Trace_dev_{dev_id}_dt.h")
        subprocess.run(["cp", header_file_path, tmp_weights_header_path], check=True)
        print(f"File {header_file_path} copied to {tmp_weights_header_path}")

    return True


def train_dt(trace_dir: str, devices: List[str]) -> bool:
    """
    Train standard DT model artifacts associated with trace_dir.
    """
    dev_names = [os.path.basename(d) for d in devices]

    train_command = ["./train_dt.sh"]
    train_command.extend(dev_names)

    if os.path.exists(trace_dir):
        train_command.append(trace_dir)
    else:
        print(f"[Error!!] trace_dir not exist: {trace_dir}")
        return False

    print("DT training command: {}".format(train_command))

    original_directory = os.getcwd()
    try:
        os.chdir("./dt")
        subprocess.run(train_command, check=True)
    except subprocess.CalledProcessError as train_e:
        print(f"Error running DT training: {train_e}")
        os.chdir(original_directory)
        return False

    os.chdir(original_directory)
    return True


def do_sleep(timer_mins: int, time_start: pd.Timestamp) -> None:
    time_end = pd.Timestamp.now()
    time_elapsed = (time_end - time_start).seconds
    print("time_elapsed = " + str(time_elapsed))
    if time_elapsed < timer_mins * 60:
        print("Sleeping for " + str(timer_mins * 60 - time_elapsed) + " seconds")
        import time as _time

        _time.sleep(timer_mins * 60 - time_elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-devices",
        help="The array of storage devices (separated by space)",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "-trace_dir", help="File path to the trace sections description", type=str
    )
    parser.add_argument(
        "-trace_dirs",
        help="Arr of file path to the trace sections description",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "-resume",
        help="Resume the processing, not replaying the replayed traces",
        action="store_true",
    )
    parser.add_argument(
        "-timer_mins",
        help="Will not replay the traces until it reaches this timer target",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-only_training", help="Only do training", action="store_true", default=False
    )
    parser.add_argument(
        "-only_replaying",
        help="Only do replaying, assuming that it's already trained",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-if_model_updated",
        help="Check if surrogate weights are newer than the replayed traces",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-reverse", help="Will start from the last combination", action="store_true"
    )

    args = parser.parse_args()
    if (not args.devices) or (not (args.trace_dir or args.trace_dirs)):
        print(
            "    ERROR: You must provide these arguments: "
            "-devices <the array of storage devices> "
            "-trace_dir <the trace dir> or -trace_dirs <the array of trace dirs"
        )
        raise SystemExit(1)

    trace_dirs: List[str] = []
    if args.trace_dirs:
        trace_dirs += args.trace_dirs
    elif args.trace_dir:
        trace_dirs.append(args.trace_dir)

    print("trace_paths = " + str(trace_dirs))
    print("devices = " + str(args.devices))
    print("algo = " + ALGORITHM)
    print("Found " + str(len(trace_dirs)) + " trace dirs")

    if args.reverse:
        trace_dirs = trace_dirs[::-1]
        print("Reversed the order of the trace dirs")

    dev_names = [os.path.basename(d) for d in args.devices]

    for idx, trace_dir in enumerate(trace_dirs):
        print(
            "\nProcessing trace dir "
            + str(idx + 1)
            + " out of "
            + str(len(trace_dirs))
        )

        # Paths for replayed stats and DT headers
        output_dir = get_output_dir(trace_dir, args.devices)
        output_stat_path = os.path.join(output_dir, "trace_1.trace.stats")

        dev_pair_dir = os.path.join(str(trace_dir), "...".join(dev_names))
        dt_weights_dir = os.path.join(
            dev_pair_dir, "dt", "training_results", "dt_weights_header"
        )
        assert_not_flashnet_output(dt_weights_dir)
        output_weights_path_0 = os.path.join(dt_weights_dir, "w_Trace_dev_0_dt.h")
        output_weights_path_1 = os.path.join(dt_weights_dir, "w_Trace_dev_1_dt.h")

        if args.resume:
            # If we already have both stats and surrogate weights, skip
            if (
                os.path.isfile(output_stat_path)
                and os.path.isfile(output_weights_path_0)
                and os.path.isfile(output_weights_path_1)
            ):
                print("     The trace is already trained and replayed (dt), skipping")
                continue

        time_start = pd.Timestamp.now()

        if not args.only_replaying:
            if args.resume:
                if os.path.isfile(output_weights_path_0) and os.path.isfile(
                    output_weights_path_1
                ):
                    print("     The dt model is trained, skipping training\n\n")
                else:
                    train_result = train_dt(trace_dir, args.devices)
                    if train_result is False:
                        print(
                            "\n[Train DT Error], dir: {}".format(trace_dir)
                        )
                        raise SystemExit(1)
            else:
                train_result = train_dt(trace_dir, args.devices)
                if train_result is False:
                    print("\n[Train DT Error], dir: {}".format(trace_dir))
                    raise SystemExit(1)

        if args.timer_mins > 0:
            do_sleep(args.timer_mins, time_start)

        if args.only_training:
            continue

        # Ensure DT weights exist
        if not os.path.isfile(output_weights_path_0) or not os.path.isfile(
            output_weights_path_1
        ):
            print(
                "     WARNING: The dt model weights are not ready, skipping\n\n"
            )
            continue

        # Check if model is newer than replayed traces (optional)
        if os.path.isfile(output_stat_path) and args.if_model_updated:
            weight_modified_time = os.path.getmtime(output_weights_path_0)
            stat_modified_time = os.path.getmtime(output_stat_path)
            if weight_modified_time < stat_modified_time:
                print(
                    "     The dt model is OLDER than the replayed traces, skipping\n\n"
                )
                continue

        # Prepare temporary workplace for this device pair (avoid conflicting builds)
        specific_workplace = "./tmp_running/{}_{}...{}".format(
            ALGORITHM, args.devices[0].split("/")[2], args.devices[1].split("/")[2]
        )
        if os.path.exists(specific_workplace):
            if delete_dir(specific_workplace) is False:
                print("Workplace delete error: {}".format(specific_workplace))
                raise SystemExit(1)

        print("The specific workplace is {}".format(specific_workplace))
        try:
            subprocess.run(
                ["cp", "-r", "./{}".format(ALGORITHM), specific_workplace], check=True
            )
        except subprocess.CalledProcessError:
            print("cp workplace wrong!")
            raise SystemExit(1)

        # Copy per-workload DT headers into this workplace
        if not copy_dt_weights(trace_dir, args.devices, specific_workplace):
            print(
                "\n[ERROR] DT weights are not complete. "
                "Please (re)train dt on this trace."
            )
            raise SystemExit(1)

        # Build DT replayer in the workplace
        original_directory = os.getcwd()
        try:
            os.chdir(specific_workplace)
            try:
                subprocess.run(["make"], check=True)
            except subprocess.CalledProcessError as make_error:
                print(f"Error running 'make' in dt: {make_error}")
                # Retry once after cleaning /tmp/*.o, mimicking run_flashnet
                print("Removing any .o files in the /tmp folder and retrying...")
                for file in Path("/tmp").glob("*.o"):
                    file.unlink()
                subprocess.run(["make"], check=True)
        finally:
            os.chdir(original_directory)

        # Replaying traces with dt
        subprocess.run("stty sane", shell=True, check=True)
        start_processing(trace_dir, args, specific_workplace)

        # Clean up workplace
        if delete_dir(specific_workplace) is False:
            print("Workplace delete error: {}".format(specific_workplace))
            raise SystemExit(1)

