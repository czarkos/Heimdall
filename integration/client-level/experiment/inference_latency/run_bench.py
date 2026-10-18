#!/usr/bin/env python3
"""
Inference latency, CPU cost and memory footprint of FlashNet vs. the depth-5 surrogate DT
vs. the logistic-regression surrogate.

For every trace and device (one model per SSD):
  1. Real inputs: a window of mldrive<dev>.csv (the trace's FlashNet training features) is
     converted into the replayer's history pool layout (see bench.c).
  2. For each model and optimization level, bench.c is compiled against the exact inference
     code used in the replays (flashnet/flashnet_algo.c, surrogate_dt/dt_algo.c,
     surrogate_lr/lr_algo.c) plus that
     trace's trained headers, and run pinned to one CPU.
  3. Model footprint is read from the compiled object (nm symbol sizes of the weight/tree arrays).

Raw CPU numbers per inference: wall ns, thread-CPU ns, TSC ticks, per-call p50/p90/p99/p99.9.
Derived metric: core share (%) = CPU ns per inference x the client's read IOPS (from the
baseline replay) / 1e9 x 100, i.e. the fraction of one core spent on inference while
replaying that trace.

-O0 is how the replayers are built (their makefiles pass no -O flag); -O2 is for reference.

Usage:
  python3 run_bench.py                      # all traces, both devices, both models, -O0 and -O2
  python3 run_bench.py -only tencent -n_rows 50000 -passes 3     # quick check
  python3 run_bench.py -plot_only
"""

import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_ROOT = os.path.join(
    os.environ.get("HEIMDALL", "/mnt/heimdall-exp/Heimdall"), "integration", "client-level", "data")
DEFAULT_DEV_PAIR = "nvme0n1...nvme1n1"

MODELS = {
    "flashnet": {
        "label": "FlashNet",
        "src_dir": "flashnet", "algo": "flashnet_algo.c", "header": "flashnet_algo.h", "define": "MODEL_FLASHNET",
        "weights_src": "flashnet/training_results/weights_header_2ssds/w_Trace_dev_{d}.h",
        "weights_dst": "2ssds_weights_header/w_Trace_dev_{d}.h",
        "symbol_re": r"^(weight|bias)_\d+(_T)?_dev_{d}$",
        "color": "blue",
    },
    "dt_depth5": {
        "label": "DT (depth 5)",
        "src_dir": "surrogate_dt", "algo": "dt_algo.c", "header": "dt_algo.h", "define": "MODEL_DT",
        "weights_src": "small_surrogate_dt_depth5/training_results/surrogate_headers/w_Trace_dev_{d}_dt.h",
        "weights_dst": "dt_weights_header/w_Trace_dev_{d}_dt.h",
        "symbol_re": r"^dt_(feature|threshold|left|right|value)_dev_{d}$",
        "color": "magenta",
    },
    "lr": {
        "label": "Logistic regression",
        "src_dir": "surrogate_lr", "algo": "lr_algo.c", "header": "lr_algo.h", "define": "MODEL_LR",
        "weights_src": "surrogate_lr/training_results/lr_headers/w_Trace_dev_{d}_lr.h",
        "weights_dst": "lr_weights_header/w_Trace_dev_{d}_lr.h",
        "symbol_re": r"^lr_(weight|bias)_dev_{d}$",
        "color": "teal",
    },
}


def short_trace_name(trace_dir):
    target = os.path.basename(os.path.dirname(trace_dir)).replace("per_3mins.", "")
    modification = (os.path.basename(trace_dir).replace("modified.", "")
                    .replace("original", "orig").replace("...", "-"))
    return "{}__{}".format(target, modification)


def read_iops(stats_path):
    with open(stats_path) as f:
        for line in f:
            if line.startswith("Read IOPS"):
                return float(line.split("=")[1])
    raise RuntimeError("No 'Read IOPS' in {}".format(stats_path))


def write_inputs(csv_path, out_path, skip, n_rows):
    """mldrive CSV window -> bench.c input layout. Returns (n, feature consistency check)."""
    df = pd.read_csv(csv_path, nrows=skip + n_rows + 1).iloc[skip:].reset_index(drop=True)
    if "cur_queue_len" not in df.columns:
        df = df.rename(columns={"queue_len": "cur_queue_len"})
    n = len(df) - 1
    # History entry k = (queue_len, latency, throughput) of IO k, taken from IO k+1's prev_*_1,
    # so it uses exactly the values (and throughput formula) the training features use.
    hist_q = df["prev_queue_len_1"].values[1:n + 1]
    hist_lat = df["prev_latency_1"].values[1:n + 1]
    hist_thpt = df["prev_throughput_1"].values[1:n + 1]
    arrays = [df["io_type"].values[:n], df["size"].values[:n], df["cur_queue_len"].values[:n],
              hist_q, hist_lat, hist_thpt]
    with open(out_path, "wb") as f:
        np.array([n], dtype=np.int64).tofile(f)
        for a in arrays:
            np.asarray(a, dtype=np.int64).tofile(f)
    # Sanity check: the 2nd/3rd history entries the model reads must equal the CSV's prev_*_2/3
    r = np.arange(3, n)
    match = np.mean([(df["prev_latency_2"].values[r] == hist_lat[r - 2]).mean(),
                     (df["prev_latency_3"].values[r] == hist_lat[r - 3]).mean(),
                     (df["prev_throughput_2"].values[r] == hist_thpt[r - 2]).mean(),
                     (df["prev_queue_len_3"].values[r] == hist_q[r - 3]).mean()])
    reject_label_rate = df["reject"].values[r][df["io_type"].values[r] == 1].mean()
    return n, float(match), float(reject_label_rate)


def build(model_key, opt, pair_dir, workspace):
    m = MODELS[model_key]
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.makedirs(workspace)
    src = os.path.join(EXPERIMENT_DIR, m["src_dir"])
    for name in (m["algo"], m["header"], "atomic.h"):
        shutil.copy(os.path.join(src, name), workspace)
    shutil.copy(os.path.join(SCRIPT_DIR, "bench.c"), workspace)
    for d in (0, 1):
        dst = os.path.join(workspace, m["weights_dst"].format(d=d))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(pair_dir, m["weights_src"].format(d=d)), dst)
    subprocess.run(["gcc", "-" + opt, "-D" + m["define"], "bench.c", m["algo"], "-o", "bench", "-lpthread", "-lm"],
                   cwd=workspace, check=True)
    subprocess.run(["gcc", "-" + opt, "-c", m["algo"], "-o", "algo.o"], cwd=workspace, check=True)


def model_footprint(model_key, workspace, device):
    """Bytes of the model's arrays for one device, from the compiled object's symbol sizes."""
    out = subprocess.run(["nm", "-S", "--defined-only", "algo.o"], cwd=workspace, check=True,
                         capture_output=True, text=True).stdout
    pattern = re.compile(MODELS[model_key]["symbol_re"].format(d=device))
    total, n_symbols = 0, 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and pattern.match(parts[3]):
            total += int(parts[1], 16)
            n_symbols += 1
    return total, n_symbols


def model_size_units(model_key, pair_dir, device, nbytes):
    if model_key == "flashnet":
        return {"n_params": nbytes // 8, "n_nodes": np.nan}   # all weights are int64
    if model_key == "lr":
        return {"n_params": nbytes // 8, "n_nodes": np.nan}   # 12 double weights + 1 double bias
    header = os.path.join(pair_dir, MODELS[model_key]["weights_src"].format(d=device))
    m = re.search(r"#define\s+DT_DEV_\d+_NODE_COUNT\s+(\d+)", open(header).read())
    return {"n_params": np.nan, "n_nodes": int(m.group(1)) if m else np.nan}


def run_all(args):
    trace_dirs = sorted(t for t in glob.glob(os.path.join(args.data_root, "*", "*", "*"))
                        if os.path.isdir(os.path.join(t, args.dev_pair)))
    if args.only:
        trace_dirs = [t for t in trace_dirs if any(s in short_trace_name(t) for s in args.only)]
    if not trace_dirs:
        sys.exit("No trace dirs found")
    tmp = os.path.join(SCRIPT_DIR, "tmp_build")
    rows = []
    try:
        for trace_dir in trace_dirs:
            pair_dir = os.path.join(trace_dir, args.dev_pair)
            name = short_trace_name(trace_dir)
            for d in (0, 1):
                os.makedirs(tmp, exist_ok=True)
                inputs = os.path.join(tmp, "inputs_dev{}.bin".format(d))
                csv_path = os.path.join(pair_dir, "flashnet", "training_results", "mldrive{}.csv".format(d))
                n, match, label_rate = write_inputs(csv_path, inputs, args.skip_rows, args.n_rows)
                iops = read_iops(os.path.join(pair_dir, "baseline", "trace_{}.trace.stats".format(d + 1)))
                for model_key in args.models:
                    for opt in args.opt:
                        ws = os.path.join(tmp, "{}_{}".format(model_key, opt))
                        build(model_key, opt, pair_dir, ws)
                        res = subprocess.run(["./bench", inputs, str(d), str(args.cpu), str(args.passes)],
                                             cwd=ws, check=True, capture_output=True, text=True)
                        r = json.loads(res.stdout.strip().splitlines()[-1])
                        nbytes, nsym = model_footprint(model_key, ws, d)
                        r.update({
                            "trace": name, "trace_dir": trace_dir, "device": d, "model": model_key,
                            "label": MODELS[model_key]["label"], "opt": opt, "read_iops": iops,
                            "core_share_pct": r["cpu_ns_mean"] * iops / 1e7,
                            "model_bytes": nbytes, "model_symbols": nsym,
                            "feature_match_rate": match, "reject_label_rate": label_rate,
                        })
                        r.update(model_size_units(model_key, pair_dir, d, nbytes))
                        rows.append(r)
                        print("[bench] {:45s} dev{} {:9s} -{}: cpu {:9.1f} ns/inf, p99 {:9.1f} ns, "
                              "core share {:6.3f}%, {} B, reject {:.3f}".format(
                                  name[:45], d, model_key, opt, r["cpu_ns_mean"], r["percall_ns_p99"],
                                  r["core_share_pct"], nbytes, r["reject_rate"]), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return pd.DataFrame(rows)


def summarize(df):
    g = df.groupby(["model", "label", "opt"])
    summary = g.agg(
        n=("cpu_ns_mean", "size"),
        cpu_ns_mean=("cpu_ns_mean", "mean"), cpu_ns_min=("cpu_ns_mean", "min"), cpu_ns_max=("cpu_ns_mean", "max"),
        wall_ns_mean=("wall_ns_mean", "mean"), tsc_ticks_mean=("tsc_ticks_mean", "mean"),
        percall_p50_ns=("percall_ns_p50", "mean"), percall_p99_ns=("percall_ns_p99", "mean"),
        percall_p999_ns=("percall_ns_p999", "mean"),
        loop_overhead_ns=("loop_overhead_ns", "mean"),
        core_share_pct_mean=("core_share_pct", "mean"), core_share_pct_max=("core_share_pct", "max"),
        model_bytes_per_device=("model_bytes", "mean"),
    ).reset_index()
    # Both clients run concurrently: per-trace total core share = sum over the two devices
    per_trace = df.groupby(["model", "opt", "trace"])["core_share_pct"].sum().reset_index()
    tot = per_trace.groupby(["model", "opt"])["core_share_pct"].agg(["mean", "max"]).reset_index()
    tot.columns = ["model", "opt", "core_share_pct_both_clients_mean", "core_share_pct_both_clients_max"]
    summary = summary.merge(tot, on=["model", "opt"])
    base = summary[summary["model"] == "dt_depth5"].set_index("opt")["cpu_ns_mean"]
    summary["cpu_ns_vs_dt"] = summary.apply(lambda r: r["cpu_ns_mean"] / base.get(r["opt"], np.nan), axis=1)
    return summary


def make_plots(df, output_dir):
    models = [m for m in MODELS if m in set(df["model"])]
    opts = [o for o in ["O0", "O2"] if o in set(df["opt"])]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # (a) CPU time per inference
    ax = axes[0]
    width = 0.8 / len(models)
    x = np.arange(len(opts))
    for i, m in enumerate(models):
        vals = [df[(df.model == m) & (df.opt == o)]["cpu_ns_mean"] for o in opts]
        mean = np.array([v.mean() for v in vals])
        lo, hi = mean - np.array([v.min() for v in vals]), np.array([v.max() for v in vals]) - mean
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, mean, width * 0.92, yerr=[lo, hi], capsize=4,
                      color=MODELS[m]["color"], label=MODELS[m]["label"])
        for b, v in zip(bars, mean):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.15, "{:.0f} ns".format(v) if v >= 10 else "{:.1f} ns".format(v),
                    ha="center", va="bottom", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 4)
    ax.set_xticks(x)
    ax.set_xticklabels(["-O0 (as in replayer)" if o == "O0" else "-O2" for o in opts])
    ax.set_ylabel("CPU time per inference (ns, log scale)")
    ax.set_title("(a) Inference cost (bars: mean, whiskers: min–max\nacross 6 traces × 2 devices)", fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # (b) Core share per trace, both clients, as deployed
    ax = axes[1]
    deployed = "O0" if "O0" in opts else opts[0]
    per_trace = df[df.opt == deployed].groupby(["trace", "model"])["core_share_pct"].sum().unstack("model")
    traces = list(per_trace.index)
    xt = np.arange(len(traces))
    for i, m in enumerate(models):
        ax.bar(xt + (i - (len(models) - 1) / 2) * width, per_trace[m].values, width * 0.92,
               color=MODELS[m]["color"], label=MODELS[m]["label"])
    ax.set_xticks(xt)
    # target trace + the second device's modification (unique across the 6 traces)
    ax.set_xticklabels([t.split("__")[0].replace("alibaba.", "ali.").replace("2018-10-09-04.12", "2018-10-09")
                        + "\n" + t.split("__")[1].split("-")[-1] for t in traces],
                       rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Share of one core (%)")
    ax.set_title("(b) Estimated CPU share of inference while replaying\n(both clients, -{}; = ns/inf × read IOPS)".format(deployed),
                 fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # (c) Model footprint
    ax = axes[2]
    sizes = df.groupby("model")["model_bytes"].mean().reindex(models) / 1024
    bars = ax.bar([MODELS[m]["label"] for m in models], sizes.values, color=[MODELS[m]["color"] for m in models],
                  width=0.5)
    for b, v in zip(bars, sizes.values):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02, "{:.1f} KB".format(v), ha="center", va="bottom", fontsize=10)
    ax.set_ylim(top=sizes.max() * 1.15)
    ax.set_ylabel("Model size per device (KB)")
    ax.set_title("(c) Model memory footprint\n(weights / tree arrays, from compiled object)", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, "inference_overhead.{}".format(ext)), dpi=200)
    plt.close(fig)
    print("[bench] Figure written to {}".format(os.path.join(output_dir, "inference_overhead.png")))


def write_environment(output_dir, args):
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
        except Exception:
            return "n/a"
    lines = [
        "host = {}".format(platform.node()),
        "cpu_model = {}".format(sh("lscpu | grep 'Model name' | sed 's/.*: *//'")),
        "pinned_cpu = {}".format(args.cpu),
        "governor = {}".format(sh("cat /sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor".format(args.cpu))),
        "turbo_disabled = {}".format(sh("cat /sys/devices/system/cpu/intel_pstate/no_turbo")),
        "gcc = {}".format(sh("gcc --version | head -1")),
        "n_rows_per_dataset = {}".format(args.n_rows),
        "skip_rows = {}".format(args.skip_rows),
        "timed_passes = {}".format(args.passes),
        "date = {}".format(sh("date -Is")),
    ]
    with open(os.path.join(output_dir, "environment.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inference latency / CPU / memory of FlashNet vs depth-5 DT vs LR.")
    parser.add_argument("-data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("-dev_pair", default=DEFAULT_DEV_PAIR)
    parser.add_argument("-output_dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("-models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("-opt", nargs="+", default=["O0", "O2"], choices=["O0", "O1", "O2", "O3"])
    parser.add_argument("-n_rows", type=int, default=200000, help="Rows of mldrive<dev>.csv used as inputs")
    parser.add_argument("-skip_rows", type=int, default=1000, help="Skip the first rows (warm-up of the trace)")
    parser.add_argument("-passes", type=int, default=10, help="Timed passes over all inputs")
    parser.add_argument("-cpu", type=int, default=4, help="CPU core to pin the benchmark to")
    parser.add_argument("-only", nargs="+", default=None, help="Keep traces whose name contains any of these")
    parser.add_argument("-plot_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, "inference_raw.csv")
    if args.plot_only:
        df = pd.read_csv(raw_path)
    else:
        df = run_all(args)
        df.to_csv(raw_path, index=False)
        write_environment(args.output_dir, args)

    summary = summarize(df)
    summary.to_csv(os.path.join(args.output_dir, "inference_summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(summary[["label", "opt", "n", "cpu_ns_mean", "cpu_ns_min", "cpu_ns_max", "percall_p50_ns",
                       "percall_p99_ns", "tsc_ticks_mean", "core_share_pct_mean", "core_share_pct_both_clients_max",
                       "model_bytes_per_device", "cpu_ns_vs_dt"]].to_string(index=False))
    make_plots(df, args.output_dir)


if __name__ == "__main__":
    main()
