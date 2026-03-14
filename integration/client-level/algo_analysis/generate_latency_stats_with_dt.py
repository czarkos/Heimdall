#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# Explicitly include both dt and surrogate_dt in analysis set.
# Other known algorithms are kept for parity with the original script.
ALGO_WHITELIST = {
    "baseline",
    "flashnet",
    "random",
    "hedging",
    "linnos",
    "linnos_hedging",
    "dt",
    "surrogate_dt",
}


def write_stats(file_path: str, statistics: str) -> None:
    with open(file_path, "w") as text_file:
        text_file.write(statistics)


def get_algo_dirs(trace_dir: str) -> Tuple[List[str], List[str]]:
    algo_dirs = []
    algo_names = []
    for entry in os.listdir(trace_dir):
        path = os.path.join(trace_dir, entry)
        if not os.path.isdir(path):
            continue
        if entry not in ALGO_WHITELIST:
            continue
        algo_dirs.append(path)
        algo_names.append(entry)
    return algo_dirs, algo_names


def is_all_algo_analyzed(algo_dirs: List[str]) -> bool:
    for algo_dir in algo_dirs:
        if not os.path.exists(os.path.join(algo_dir, "latency_characteristic.stats")):
            return False
    return True


def read_replayed_file(input_file: str) -> pd.DataFrame:
    df = pd.read_csv(input_file, header=None, sep=",")
    assert df.shape[1] >= 7

    df = df.iloc[:, :7]
    df.columns = [
        "ts_record",
        "latency",
        "io_type",
        "size",
        "offset",
        "ts_submit",
        "size_after_replay",
    ]
    df = df[df["size"] == df["size_after_replay"]]
    return df


def get_per_trace_latency(algo_dir: str) -> List[float]:
    read_latencies = []
    traces_in_dir = [
        trace
        for trace in os.listdir(algo_dir)
        if os.path.isfile(os.path.join(algo_dir, trace)) and str(trace).endswith(".trace")
    ]
    for trace_name in traces_in_dir:
        input_path = os.path.join(algo_dir, trace_name)
        df = read_replayed_file(input_path)
        df = df[df["io_type"] == 1]  # read IOs only
        read_latencies += list(df["latency"])
    return read_latencies


def get_percentiles(latencies: List[float], n: int = 10000) -> np.ndarray:
    divider = n / 100
    percentiles = [x / divider for x in range(1, n + 1)]
    return np.percentile(latencies, percentiles)


def start_process_per_algo(algo_dir: str) -> None:
    latency_stats = []
    read_latencies = get_per_trace_latency(algo_dir)

    try:
        latency_stats.append("count = {} IOs".format(len(read_latencies)))
        latency_stats.append("avg = {} us".format(np.mean(read_latencies)))
        latency_stats.append("std = {} us".format(np.std(read_latencies)))
        latency_stats.append("median = {} us".format(np.median(read_latencies)))
        latency_stats.append("max = {} us".format(np.max(read_latencies)))
        latency_stats.append("min = {} us".format(np.min(read_latencies)))

        n_percentiles = 10000
        arr_percentiles = get_percentiles(read_latencies, n=n_percentiles)
        divisor = n_percentiles / 100
        for idx, lat in enumerate(arr_percentiles):
            latency_stats.append(
                "p{} = {} us".format(round(((idx + 1) / divisor), 2), lat)
            )
    except Exception:
        # Keep behavior similar to original script: best-effort stats generation.
        pass

    output_path = os.path.join(algo_dir, "latency_characteristic.stats")
    write_stats(output_path, "\n".join(latency_stats))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-trace_dir", type=str)
    parser.add_argument("-trace_dirs", nargs="+", type=str)
    parser.add_argument("-reverse", action="store_true")
    parser.add_argument("-resume", action="store_true")
    parser.add_argument("-random", action="store_true")
    args = parser.parse_args()

    if not (args.trace_dir or args.trace_dirs):
        print(
            "ERROR: provide -trace_dir <dir> or -trace_dirs <dir1 dir2 ...>"
        )
        raise SystemExit(1)

    trace_dirs = []
    if args.trace_dirs:
        trace_dirs += args.trace_dirs
    elif args.trace_dir:
        trace_dirs.append(args.trace_dir)

    print("trace_paths = " + str(trace_dirs))

    if args.reverse:
        trace_dirs = trace_dirs[::-1]
        print("Reversed the order of the trace dirs")
    if args.random:
        np.random.shuffle(trace_dirs)
        print("Randomized the order of the trace dirs")

    for idx, trace_dir in enumerate(trace_dirs):
        print("\n{}. Processing {}".format(idx + 1, trace_dir))
        algo_dirs, algo_names = get_algo_dirs(trace_dir)

        # Helpful visibility for missing DT-family results.
        if "dt" not in algo_names:
            print("    [WARN] dt directory missing in {}".format(trace_dir))
        if "surrogate_dt" not in algo_names:
            print("    [WARN] surrogate_dt directory missing in {}".format(trace_dir))

        if args.resume and is_all_algo_analyzed(algo_dirs):
            print("    Existing stats are already created, skipping")
            continue

        for algo_dir in algo_dirs:
            algo_name = os.path.basename(algo_dir)
            print("    Processing {}".format(algo_name))
            start_process_per_algo(algo_dir)


if __name__ == "__main__":
    main()

