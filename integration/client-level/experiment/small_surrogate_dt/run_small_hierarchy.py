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


def get_output_dir(trace_dir: str, devices: List[str], algorithm_name: str) -> str:
    dev_names = [os.path.basename(d) for d in devices]
    return os.path.join(str(trace_dir), "...".join(dev_names), algorithm_name)


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
    output_dir = get_output_dir(trace_dir, args.devices, args.algorithm_name)
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


def copy_hierarchy_dt_headers(
    trace_dir: str, devices_list: List[str], specific_workplace: str, algorithm_name: str
) -> bool:
    dev_names = [os.path.basename(d) for d in devices_list]
    dev_pair_dir = os.path.join(str(trace_dir), "...".join(dev_names))
    src_dir = os.path.join(
        dev_pair_dir, algorithm_name, "training_results", "hierarchy_headers"
    )
    dst_dir = os.path.join(specific_workplace, "dt_weights_header")
    os.makedirs(dst_dir, exist_ok=True)

    for dev_id, _ in enumerate(dev_names):
        src = os.path.join(src_dir, f"w_Trace_dev_{dev_id}_dt.h")
        if not os.path.exists(src):
            print(f"small hierarchy dt header missing: {src}")
            return False
        subprocess.run(["cp", src, dst_dir], check=True)
    return True


def copy_uncertainty_headers(
    trace_dir: str, devices_list: List[str], specific_workplace: str, algorithm_name: str
) -> bool:
    dev_names = [os.path.basename(d) for d in devices_list]
    dev_pair_dir = os.path.join(str(trace_dir), "...".join(dev_names))
    src_dir = os.path.join(
        dev_pair_dir, algorithm_name, "training_results", "uncertainty_headers"
    )
    dst_dir = os.path.join(specific_workplace, "uncertainty_headers")
    os.makedirs(dst_dir, exist_ok=True)

    for dev_id, _ in enumerate(dev_names):
        src = os.path.join(src_dir, f"u_Trace_dev_{dev_id}_uncert.h")
        if not os.path.exists(src):
            print(f"small hierarchy uncertainty header missing: {src}")
            return False
        subprocess.run(["cp", src, dst_dir], check=True)
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


def train_small_hierarchy(
    trace_dir: str, devices: List[str], args, script_dir: str
) -> bool:
    dev_names = [os.path.basename(d) for d in devices]
    train_script = os.path.join(script_dir, "train_small_hierarchy.sh")
    train_command = [train_script] + dev_names
    if os.path.exists(trace_dir):
        train_command.append(trace_dir)
    else:
        print(f"[Error] trace_dir not exist: {trace_dir}")
        return False

    env = os.environ.copy()
    env["SMALL_HIERARCHY_ALGO"] = args.algorithm_name
    env["SMALL_HIERARCHY_TAU_PERCENTILE"] = str(args.tau_percentile)
    env["SMALL_HIERARCHY_REF_SIZE"] = str(args.ref_size)
    env["SMALL_HIERARCHY_SEED"] = str(args.seed)
    env["SMALL_SURROGATE_MAX_DEPTH"] = str(args.max_depth)

    try:
        subprocess.run(train_command, check=True, env=env)
    except subprocess.CalledProcessError as train_e:
        print(f"Error running small hierarchy training: {train_e}")
        return False
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
    parser.add_argument(
        "-algorithm_name",
        type=str,
        default="small_hierarchy_p95",
        help="Result directory name for this hierarchy variant.",
    )
    parser.add_argument(
        "-tau_percentile",
        type=float,
        default=95.0,
        help="Uncertainty percentile threshold (e.g., 95, 97, 98).",
    )
    parser.add_argument(
        "-max_depth",
        type=int,
        default=10,
        help="Max depth for small surrogate tree used by hierarchy.",
    )
    parser.add_argument("-ref_size", type=int, default=512)
    parser.add_argument("-seed", type=int, default=42)
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_root = os.path.dirname(script_dir)
    replay_source_dir = os.path.join(script_dir, "small_hierarchy")

    for trace_dir in trace_dirs:
        output_dir = get_output_dir(trace_dir, args.devices, args.algorithm_name)
        output_stat_path = os.path.join(output_dir, "trace_1.trace.stats")
        dev_pair_dir = os.path.join(str(trace_dir), "...".join(dev_names))
        dt_header_dir = os.path.join(
            dev_pair_dir, args.algorithm_name, "training_results", "hierarchy_headers"
        )
        uncertainty_header_dir = os.path.join(
            dev_pair_dir, args.algorithm_name, "training_results", "uncertainty_headers"
        )
        flashnet_header_dir = os.path.join(
            dev_pair_dir, "flashnet", "training_results", "weights_header_2ssds"
        )

        output_dt_header_0 = os.path.join(dt_header_dir, "w_Trace_dev_0_dt.h")
        output_dt_header_1 = os.path.join(dt_header_dir, "w_Trace_dev_1_dt.h")
        output_uncert_0 = os.path.join(uncertainty_header_dir, "u_Trace_dev_0_uncert.h")
        output_uncert_1 = os.path.join(uncertainty_header_dir, "u_Trace_dev_1_uncert.h")
        output_fn_header_0 = os.path.join(flashnet_header_dir, "w_Trace_dev_0.h")
        output_fn_header_1 = os.path.join(flashnet_header_dir, "w_Trace_dev_1.h")

        headers_ready = os.path.isfile(output_dt_header_0) and os.path.isfile(
            output_dt_header_1
        )
        uncert_ready = os.path.isfile(output_uncert_0) and os.path.isfile(output_uncert_1)

        if args.resume and os.path.isfile(output_stat_path) and headers_ready and uncert_ready:
            print(f"{args.algorithm_name}: already trained and replayed, skipping")
            continue

        time_start = pd.Timestamp.now()

        if not args.only_replaying:
            if not (args.resume and headers_ready and uncert_ready):
                if not train_small_hierarchy(trace_dir, args.devices, args, script_dir):
                    raise SystemExit(1)
                headers_ready = os.path.isfile(output_dt_header_0) and os.path.isfile(
                    output_dt_header_1
                )
                uncert_ready = os.path.isfile(output_uncert_0) and os.path.isfile(
                    output_uncert_1
                )

        if args.timer_mins > 0:
            do_sleep(args.timer_mins, time_start)
        if args.only_training:
            continue

        if not headers_ready:
            print(f"{args.algorithm_name}: hierarchy DT headers are not ready, skipping")
            continue
        if not uncert_ready:
            print(f"{args.algorithm_name}: uncertainty headers are not ready, skipping")
            continue
        if not os.path.isfile(output_fn_header_0) or not os.path.isfile(output_fn_header_1):
            print(
                f"{args.algorithm_name}: flashnet headers are not ready. Train flashnet first, skipping"
            )
            continue

        if os.path.isfile(output_stat_path) and args.if_model_updated:
            weight_modified_time = os.path.getmtime(output_dt_header_0)
            stat_modified_time = os.path.getmtime(output_stat_path)
            if weight_modified_time < stat_modified_time:
                continue

        specific_workplace = os.path.join(
            experiment_root,
            "tmp_running",
            "{}_{}...{}".format(
                args.algorithm_name,
                args.devices[0].split("/")[2],
                args.devices[1].split("/")[2],
            ),
        )
        if os.path.exists(specific_workplace):
            if not delete_dir(specific_workplace):
                raise SystemExit(1)

        try:
            shutil.copytree(replay_source_dir, specific_workplace)
        except Exception as e:
            print(f"Failed to copy replay source: {e}")
            raise SystemExit(1)

        if not copy_hierarchy_dt_headers(
            trace_dir, args.devices, specific_workplace, args.algorithm_name
        ):
            raise SystemExit(1)
        if not copy_uncertainty_headers(
            trace_dir, args.devices, specific_workplace, args.algorithm_name
        ):
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
        start_processing(trace_dir, args, specific_workplace)

        if not delete_dir(specific_workplace):
            raise SystemExit(1)

