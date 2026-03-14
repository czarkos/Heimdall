#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import pandas as pd
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List

ALGORITHM = "hierarchy"


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
                    duration = re.findall(r"-?\d+\.\d+", value_raw)[0]
                else:
                    duration = re.findall(r"-?\d+", value_raw)[0]
                return duration
    raise RuntimeError(f"Duration line not found in {trace_path}")


def start_processing(trace_dir: str, args, specific_workplace: str) -> None:
    print("Processing " + str(trace_dir))
    output_dir = get_output_dir(trace_dir, args.devices)
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
        shutil.rmtree(path)
        print(f"Directory '{path}' has been deleted.")
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


def resolve_dt_header_source_dir(trace_dir: str, devices_list: List[str]) -> str:
    """
    Resolve DT header source directory with fallback:
      1) hierarchy/training_results/hierarchy_headers
      2) dt/training_results/dt_weights_header
    """
    dev_names = [os.path.basename(d) for d in devices_list]
    dev_pair_dir = os.path.join(str(trace_dir), "...".join(dev_names))

    primary = os.path.join(
        dev_pair_dir, "hierarchy", "training_results", "hierarchy_headers"
    )
    fallback = os.path.join(
        dev_pair_dir, "dt", "training_results", "dt_weights_header"
    )

    primary_ok = (
        os.path.isfile(os.path.join(primary, "w_Trace_dev_0_dt.h"))
        and os.path.isfile(os.path.join(primary, "w_Trace_dev_1_dt.h"))
    )
    if primary_ok:
        print(f"[hierarchy] Using primary DT headers: {primary}")
        return primary

    fallback_ok = (
        os.path.isfile(os.path.join(fallback, "w_Trace_dev_0_dt.h"))
        and os.path.isfile(os.path.join(fallback, "w_Trace_dev_1_dt.h"))
    )
    if fallback_ok:
        print(f"[hierarchy] Using fallback DT headers from dt: {fallback}")
        return fallback

    return ""


def copy_hierarchy_dt_headers(trace_dir: str, devices_list: List[str], specific_workplace: str) -> bool:
    dev_names = [os.path.basename(d) for d in devices_list]
    src_dir = resolve_dt_header_source_dir(trace_dir, devices_list)
    if src_dir == "":
        print(
            "DT headers missing in both hierarchy and dt locations for trace: {}".format(
                trace_dir
            )
        )
        return False

    dst_dir = os.path.join(specific_workplace, "dt_weights_header")
    os.makedirs(dst_dir, exist_ok=True)

    for dev_id in range(len(dev_names)):
        src = os.path.join(src_dir, f"w_Trace_dev_{dev_id}_dt.h")
        if not os.path.exists(src):
            print(f"hierarchy dt header missing: {src}")
            return False
        subprocess.run(["cp", src, dst_dir], check=True)
    return True


def copy_uncertainty_headers(trace_dir: str, devices_list: List[str], specific_workplace: str) -> bool:
    dev_names = [os.path.basename(d) for d in devices_list]
    dev_pair_dir = os.path.join(str(trace_dir), "...".join(dev_names))
    src_dir = os.path.join(dev_pair_dir, "hierarchy", "training_results", "uncertainty_headers")
    dst_dir = os.path.join(specific_workplace, "uncertainty_headers")
    os.makedirs(dst_dir, exist_ok=True)

    for dev_id in range(len(dev_names)):
        src = os.path.join(src_dir, f"u_Trace_dev_{dev_id}_uncert.h")
        if not os.path.exists(src):
            print(f"uncertainty header missing: {src}")
            return False
        subprocess.run(["cp", src, dst_dir], check=True)
    return True


def copy_flashnet_headers(trace_dir: str, devices_list: List[str], specific_workplace: str) -> bool:
    dev_names = [os.path.basename(d) for d in devices_list]
    dev_pair_dir = os.path.join(str(trace_dir), "...".join(dev_names))
    src_dir = os.path.join(dev_pair_dir, "flashnet", "training_results", "weights_header_2ssds")
    dst_dir = os.path.join(specific_workplace, "2ssds_weights_header")
    os.makedirs(dst_dir, exist_ok=True)

    for dev_id in range(len(dev_names)):
        src = os.path.join(src_dir, f"w_Trace_dev_{dev_id}.h")
        if not os.path.exists(src):
            print(f"flashnet header missing: {src}")
            return False
        subprocess.run(["cp", src, dst_dir], check=True)
    return True


def train_hierarchy(trace_dir: str, devices: List[str]) -> bool:
    dev_names = [os.path.basename(d) for d in devices]
    train_command = ["./train_hierarchy.sh"] + dev_names

    if os.path.exists(trace_dir):
        train_command.append(trace_dir)
    else:
        print(f"[Error!!] trace_dir not exist: {trace_dir}")
        return False

    print("hierarchy training command: {}".format(train_command))

    original_directory = os.getcwd()
    try:
        os.chdir("./hierarchy")
        subprocess.run(train_command, check=True)
    except subprocess.CalledProcessError as train_e:
        print(f"Error running hierarchy training: {train_e}")
        os.chdir(original_directory)
        return False

    os.chdir(original_directory)
    return True


def do_sleep(timer_mins: int, time_start: pd.Timestamp) -> None:
    time_end = pd.Timestamp.now()
    time_elapsed = (time_end - time_start).seconds
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
        dt_header_dir_primary = os.path.join(
            dev_pair_dir, "hierarchy", "training_results", "hierarchy_headers"
        )
        dt_header_dir_fallback = os.path.join(
            dev_pair_dir, "dt", "training_results", "dt_weights_header"
        )
        uncertainty_header_dir = os.path.join(
            dev_pair_dir, "hierarchy", "training_results", "uncertainty_headers"
        )
        flashnet_header_dir = os.path.join(
            dev_pair_dir, "flashnet", "training_results", "weights_header_2ssds"
        )

        output_dt_header_primary_0 = os.path.join(dt_header_dir_primary, "w_Trace_dev_0_dt.h")
        output_dt_header_primary_1 = os.path.join(dt_header_dir_primary, "w_Trace_dev_1_dt.h")
        output_dt_header_fallback_0 = os.path.join(dt_header_dir_fallback, "w_Trace_dev_0_dt.h")
        output_dt_header_fallback_1 = os.path.join(dt_header_dir_fallback, "w_Trace_dev_1_dt.h")
        output_uncert_0 = os.path.join(uncertainty_header_dir, "u_Trace_dev_0_uncert.h")
        output_uncert_1 = os.path.join(uncertainty_header_dir, "u_Trace_dev_1_uncert.h")
        output_fn_header_0 = os.path.join(flashnet_header_dir, "w_Trace_dev_0.h")
        output_fn_header_1 = os.path.join(flashnet_header_dir, "w_Trace_dev_1.h")

        dt_headers_ready = (
            os.path.isfile(output_dt_header_primary_0)
            and os.path.isfile(output_dt_header_primary_1)
        ) or (
            os.path.isfile(output_dt_header_fallback_0)
            and os.path.isfile(output_dt_header_fallback_1)
        )
        uncert_headers_ready = os.path.isfile(output_uncert_0) and os.path.isfile(output_uncert_1)

        if args.resume and os.path.isfile(output_stat_path) and dt_headers_ready and uncert_headers_ready:
            print("Trace already trained and replayed for hierarchy, skipping")
            continue

        time_start = pd.Timestamp.now()

        if not args.only_replaying:
            if not (args.resume and dt_headers_ready and uncert_headers_ready):
                if not train_hierarchy(trace_dir, args.devices):
                    raise SystemExit(1)
                dt_headers_ready = (
                    os.path.isfile(output_dt_header_primary_0)
                    and os.path.isfile(output_dt_header_primary_1)
                ) or (
                    os.path.isfile(output_dt_header_fallback_0)
                    and os.path.isfile(output_dt_header_fallback_1)
                )
                uncert_headers_ready = os.path.isfile(output_uncert_0) and os.path.isfile(output_uncert_1)

        if args.timer_mins > 0:
            do_sleep(args.timer_mins, time_start)
        if args.only_training:
            continue

        if not dt_headers_ready:
            print("hierarchy DT headers are not ready, skipping")
            continue
        if not uncert_headers_ready:
            print("hierarchy uncertainty headers are not ready, skipping")
            continue
        if not os.path.isfile(output_fn_header_0) or not os.path.isfile(output_fn_header_1):
            print("flashnet headers are not ready. Train flashnet first, skipping")
            continue

        if os.path.isfile(output_stat_path) and args.if_model_updated:
            if os.path.isfile(output_dt_header_primary_0):
                weight_modified_time = os.path.getmtime(output_dt_header_primary_0)
            else:
                weight_modified_time = os.path.getmtime(output_dt_header_fallback_0)
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

        if not ensure_dt_shared_source(specific_workplace):
            raise SystemExit(1)
        if not copy_hierarchy_dt_headers(trace_dir, args.devices, specific_workplace):
            raise SystemExit(1)
        if not copy_uncertainty_headers(trace_dir, args.devices, specific_workplace):
            raise SystemExit(1)
        if not copy_flashnet_headers(trace_dir, args.devices, specific_workplace):
            raise SystemExit(1)

        original_directory = os.getcwd()
        try:
            os.chdir(specific_workplace)
            try:
                subprocess.run(["make"], check=True)
            except subprocess.CalledProcessError as make_error:
                print(f"Error running make in hierarchy: {make_error}")
                for file in Path("/tmp").glob("*.o"):
                    file.unlink()
                subprocess.run(["make"], check=True)
        finally:
            os.chdir(original_directory)

        subprocess.run("stty sane", shell=True, check=True)
        start_processing(trace_dir, args, specific_workplace)

        if not delete_dir(specific_workplace):
            raise SystemExit(1)

