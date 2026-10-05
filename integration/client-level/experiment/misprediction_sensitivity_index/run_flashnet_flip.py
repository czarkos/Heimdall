#!/usr/bin/env python3
"""
Replay FlashNet with injected mispredictions (misprediction sensitivity index).

For every flip probability p and run index, builds flashnet_flip/ with
FLIP_PROB=p (the trained FlashNet weights are reused read-only from
<trace>/<dev0>...<dev1>/flashnet/training_results/weights_header_2ssds) and
replays both clients in parallel, exactly like run_flashnet.py.

Outputs:
  <trace_dir>/<dev0>...<dev1>/flashnet_flip_pXX/run_<i>/trace_{1,2}.trace[.stats]

Runs are interleaved (run index -> p -> trace) so a partially finished night
still covers every p. With -resume, finished (p, run, trace) combinations are
skipped, so increasing -num_runs later only adds the missing repeats.

Usage:
  python3 run_flashnet_flip.py -devices /dev/nvme0n1 /dev/nvme1n1 \
      -trace_dirs $HEIMDALL/integration/client-level/data/*/*/* \
      -flip_probs 0.01 0.02 0.03 0.04 0.05 0.10 -num_runs 1 -resume
"""

import argparse
import datetime
import hashlib
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(SCRIPT_DIR, "flashnet_flip")
TMP_ROOT = os.path.join(SCRIPT_DIR, "tmp_running")


def algo_name(p: float) -> str:
    pct = round(p * 100, 4)
    if float(pct).is_integer():
        return "flashnet_flip_p{:02d}".format(int(pct))
    return "flashnet_flip_p{}".format(str(pct).replace(".", "_"))


def pair_name(devices: List[str]) -> str:
    return "...".join(os.path.basename(d) for d in devices)


def get_output_dir(trace_dir: str, devices: List[str], p: float, run_idx: int) -> str:
    out = os.path.join(trace_dir, pair_name(devices), algo_name(p), "run_{}".format(run_idx))
    for forbidden in ("{0}flashnet{0}".format(os.sep), "{0}baseline{0}".format(os.sep)):
        if forbidden in out:
            raise RuntimeError("Refusing to write misprediction runs under {}: {}".format(forbidden, out))
    return out


def weights_dir(trace_dir: str, devices: List[str]) -> str:
    return os.path.join(trace_dir, pair_name(devices), "flashnet", "training_results", "weights_header_2ssds")


def is_done(output_dir: str) -> bool:
    return all(os.path.isfile(os.path.join(output_dir, "trace_{}.trace.stats".format(i))) for i in (1, 2))


def get_duration_from_trace(stats_path: str) -> str:
    # Same parsing as run_flashnet.py
    with open(stats_path) as f:
        for line in f:
            if "Duration" in line:
                value_raw = line.split("=")[2]
                if "." in value_raw:
                    return re.findall(r"-?\d+\.\d+", value_raw)[0]
                return re.findall(r"-?\d+", value_raw)[0]
    raise RuntimeError("No Duration found in {}".format(stats_path))


def md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def check_inputs(trace_dirs: List[str], devices: List[str]) -> List[str]:
    problems = []
    for trace_dir in trace_dirs:
        for i in (1, 2):
            for name in ("trace_{}.trace".format(i), "trace_{}.stats".format(i)):
                if not os.path.isfile(os.path.join(trace_dir, name)):
                    problems.append("missing {}".format(os.path.join(trace_dir, name)))
        for dev_id in (0, 1):
            header = os.path.join(weights_dir(trace_dir, devices), "w_Trace_dev_{}.h".format(dev_id))
            if not os.path.isfile(header):
                problems.append("missing FlashNet weights {}".format(header))
    return problems


def build_workspace(trace_dir: str, devices: List[str], p: float, seed: int) -> str:
    workspace = os.path.join(TMP_ROOT, "flashnet_flip_{}".format(pair_name(devices)))
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.makedirs(TMP_ROOT, exist_ok=True)
    shutil.copytree(SOURCE_DIR, workspace)
    header_dir = os.path.join(workspace, "2ssds_weights_header")
    os.makedirs(header_dir)
    for dev_id in (0, 1):
        shutil.copy(os.path.join(weights_dir(trace_dir, devices), "w_Trace_dev_{}.h".format(dev_id)), header_dir)
    subprocess.run(["make", "FLIP_PROB={}".format(p), "FLIP_SEED={}".format(seed)], cwd=workspace, check=True)
    return workspace


def replay(trace_dir: str, devices: List[str], workspace: str, output_dir: str) -> bool:
    devices_list_str = "-".join(devices)
    commands = []
    for idx in range(len(devices)):
        trace_path = os.path.join(trace_dir, "trace_{}.trace".format(idx + 1))
        duration = get_duration_from_trace(os.path.join(trace_dir, "trace_{}.stats".format(idx + 1)))
        commands.append(
            "cd {ws} && sudo ./replay.sh -user $USER -original_device_index {idx} -devices_list {devs} "
            "-trace {trace} -output_dir {out} -duration {dur}".format(
                ws=workspace, idx=idx, devs=devices_list_str, trace=trace_path, out=output_dir, dur=duration)
        )
    # Both clients replay concurrently, as in run_flashnet.py
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        results = list(executor.map(lambda c: subprocess.run(c, shell=True).returncode, commands))
    subprocess.run("stty sane", shell=True)
    return all(rc == 0 for rc in results) and is_done(output_dir)


def write_config(output_dir: str, trace_dir: str, devices: List[str], p: float, seed: int, run_idx: int) -> None:
    lines = [
        "flip_prob = {}".format(p),
        "flip_seed = {}".format(seed),
        "run_idx = {}".format(run_idx),
        "devices = {}".format(" ".join(devices)),
        "trace_dir = {}".format(trace_dir),
        "started = {}".format(datetime.datetime.now().isoformat(timespec="seconds")),
    ]
    for dev_id in (0, 1):
        header = os.path.join(weights_dir(trace_dir, devices), "w_Trace_dev_{}.h".format(dev_id))
        lines.append("weights_dev_{} = {} (md5 {})".format(dev_id, header, md5(header)))
    with open(os.path.join(output_dir, "flip_config.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay FlashNet with injected mispredictions.")
    parser.add_argument("-devices", nargs="+", required=True, help="e.g. /dev/nvme0n1 /dev/nvme1n1")
    parser.add_argument("-trace_dirs", nargs="+", required=True)
    parser.add_argument("-flip_probs", nargs="+", type=float, default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.10])
    parser.add_argument("-num_runs", type=int, default=1, help="Total runs per (p, trace), counting existing ones")
    parser.add_argument("-p0_runs", type=int, default=None,
                        help="Total runs for p = 0 (flips off, same decisions as the original FlashNet); default -num_runs")
    parser.add_argument("-seed", type=int, default=42, help="Base seed; run i uses seed + i")
    parser.add_argument("-resume", action="store_true", help="Skip (p, run, trace) combinations already replayed")
    parser.add_argument("-dry_run", action="store_true", help="Only print what would be replayed")
    args = parser.parse_args()

    if len(args.devices) != 2:
        sys.exit("Exactly two devices are required (FlashNet uses one model per SSD).")
    trace_dirs = [os.path.abspath(t) for t in args.trace_dirs]

    problems = check_inputs(trace_dirs, args.devices)
    if problems:
        print("Pre-flight check failed:\n  " + "\n  ".join(problems))
        sys.exit(1)

    p0_runs = args.num_runs if args.p0_runs is None else args.p0_runs
    plan = []
    for run_idx in range(max(args.num_runs, p0_runs)):
        for p in args.flip_probs:
            if run_idx >= (p0_runs if p == 0 else args.num_runs):
                continue
            for trace_dir in trace_dirs:
                out = get_output_dir(trace_dir, args.devices, p, run_idx)
                if args.resume and is_done(out):
                    continue
                plan.append((run_idx, p, trace_dir, out))

    print("[flip] {} replays to do ({} traces x {} p values x {} runs, p=0 runs={}, resume={})".format(
        len(plan), len(trace_dirs), len(args.flip_probs), args.num_runs, p0_runs, args.resume), flush=True)
    if args.dry_run:
        for run_idx, p, trace_dir, out in plan:
            print("  run {} p={:.3f} -> {}".format(run_idx, p, out))
        return

    failures = []
    for n, (run_idx, p, trace_dir, out) in enumerate(plan, 1):
        seed = args.seed + run_idx
        print("\n[flip] ({}/{}) {} run {} p={} seed={}\n       {}".format(
            n, len(plan), datetime.datetime.now().strftime("%F %T"), run_idx, p, seed, trace_dir), flush=True)
        if os.path.exists(out):
            shutil.rmtree(out)  # incomplete leftovers from an interrupted replay
        os.makedirs(out)
        write_config(out, trace_dir, args.devices, p, seed, run_idx)
        workspace = build_workspace(trace_dir, args.devices, p, seed)
        ok = replay(trace_dir, args.devices, workspace, out)
        shutil.rmtree(workspace, ignore_errors=True)
        if not ok:
            print("[flip] FAILED: {}".format(out), flush=True)
            failures.append(out)

    print("\n[flip] Finished {} replays, {} failed".format(len(plan), len(failures)))
    for f in failures:
        print("  FAILED " + f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
