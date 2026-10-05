#!/usr/bin/env python3
"""
Misprediction Sensitivity Index (MSI) analysis for FlashNet.

Inputs per trace dir (<trace>/<dev_pair>/):
  baseline/                    reference (single run, or run_* if present)
  flashnet/run_*               p = 0 from the robust FlashNet runs (-p0_source robust)
  flashnet_flip_pXX/run_*      injected flip probability p = XX%
  flashnet_flip_p00/run_*      p = 0 with flips off (-p0_source flip); always the decision-drift reference

For every metric (avg, p95, p99, p99.9, p99.99 read latency):
  Advantage = L_baseline - L(p=0)
  S         = least-squares slope of L vs p (p as a fraction), fitted over p <= fit_max_p
  MSI       = S / Advantage            (Pensieve's misprediction sensitivity index)
  p*        = Advantage / S = 1 / MSI  (extra error rate at which FlashNet falls back to baseline)
with 95% bootstrap CIs (runs resampled per (trace, p)).

Usage:
  python3 analyze_msi.py                       # default data root and nvme0n1...nvme1n1
  python3 analyze_msi.py -plot_only            # re-plot / re-fit from results/msi_runs.csv
"""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_LEVEL_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(CLIENT_LEVEL_DIR, "algo_analysis"))
sys.dont_write_bytecode = True  # don't leave a __pycache__ in algo_analysis/
# Same read-latency extraction as the robust results (read IOs, size == size_after_replay, both clients pooled)
from generate_latency_stats_with_dt import detect_run_dirs, get_per_trace_latency  # noqa: E402

DEFAULT_DATA_ROOT = os.path.join(
    os.environ.get("HEIMDALL", "/mnt/heimdall-exp/Heimdall"), "integration", "client-level", "data")
DEFAULT_DEV_PAIR = "nvme0n1...nvme1n1"

METRICS = ["avg", "p95", "p99", "p99.9", "p99.99"]
METRIC_LABELS = {"avg": "Average", "p95": "p95", "p99": "p99", "p99.9": "p99.9", "p99.99": "p99.99"}
PERCENTILES = {"p95": 95.0, "p99": 99.0, "p99.9": 99.9, "p99.99": 99.99}

COLOR_FLASHNET = "blue"
COLOR_BASELINE = "red"
FLIP_DIR_RE = re.compile(r"^flashnet_flip_p(\d+(?:_\d+)?)$")


def short_trace_name(trace_dir: str) -> str:
    target = os.path.basename(os.path.dirname(trace_dir)).replace("per_3mins.", "")
    modification = (os.path.basename(trace_dir).replace("modified.", "")
                    .replace("original", "orig").replace("...", "-"))
    return "{}__{}".format(target, modification)


def flip_dir_prob(name: str):
    m = FLIP_DIR_RE.match(name)
    return float(m.group(1).replace("_", ".")) / 100 if m else None


def discover_runs(trace_dirs, dev_pair):
    """Return a list of run descriptors: (trace_dir, kind, p, run_idx, run_dir)."""
    runs = []
    for trace_dir in trace_dirs:
        pair_dir = os.path.join(trace_dir, dev_pair)
        if not os.path.isdir(pair_dir):
            continue
        baseline_dir = os.path.join(pair_dir, "baseline")
        if os.path.isdir(baseline_dir):
            run_dirs = detect_run_dirs(baseline_dir) or [baseline_dir]
            for i, d in enumerate(run_dirs):
                runs.append((trace_dir, "baseline", np.nan, i, d))
        flashnet_dir = os.path.join(pair_dir, "flashnet")
        if os.path.isdir(flashnet_dir):
            run_dirs = detect_run_dirs(flashnet_dir)
            if not run_dirs and glob.glob(os.path.join(flashnet_dir, "trace_*.trace")):
                print("[WARN] {}: no flashnet/run_* dirs, using the single legacy run as p=0".format(pair_dir))
                run_dirs = [flashnet_dir]
            for i, d in enumerate(run_dirs):
                runs.append((trace_dir, "flashnet", 0.0, i, d))
        for entry in sorted(os.listdir(pair_dir)):
            p = flip_dir_prob(entry)
            if p is None:
                continue
            for d in detect_run_dirs(os.path.join(pair_dir, entry)):
                if all(os.path.isfile(os.path.join(d, "trace_{}.trace.stats".format(k))) for k in (1, 2)):
                    runs.append((trace_dir, "flip", p, int(os.path.basename(d).split("_")[1]), d))
                else:
                    print("[WARN] skipping incomplete run {}".format(d))
    return runs


def trace_files_mtime(run_dir: str) -> float:
    files = glob.glob(os.path.join(run_dir, "trace_*.trace"))
    return max(os.path.getmtime(f) for f in files) if files else 0.0


def flip_stats(run_dir: str) -> dict:
    """Per-device decision stats from the instrumented replayer's extra columns (reads only)."""
    out = {}
    for dev in (0, 1):
        path = os.path.join(run_dir, "trace_{}.trace".format(dev + 1))
        df = pd.read_csv(path, header=None, usecols=[2, 7, 8, 9])
        df.columns = ["io_type", "orig_pred", "flipped", "target_device"]
        reads = df[df["io_type"] == 1]
        out["realized_flip_rate_dev{}".format(dev)] = reads["flipped"].mean()
        out["preflip_reject_rate_dev{}".format(dev)] = (reads["orig_pred"] == 1).mean()
        out["reroute_rate_dev{}".format(dev)] = (reads["target_device"] != dev).mean()
        out["n_reads_dev{}".format(dev)] = len(reads)
    return out


def run_stats(desc) -> dict:
    trace_dir, kind, p, run_idx, run_dir = desc
    lat = np.asarray(get_per_trace_latency(run_dir), dtype=float)
    row = {
        "trace_dir": trace_dir, "trace": short_trace_name(trace_dir), "kind": kind, "p": p,
        "run": run_idx, "run_dir": run_dir, "mtime": trace_files_mtime(run_dir), "n_reads": len(lat),
        "avg": lat.mean(),
    }
    row.update({m: v for m, v in zip(PERCENTILES, np.percentile(lat, list(PERCENTILES.values())))})
    if kind == "flip":
        row.update(flip_stats(run_dir))
    print("[MSI] {} {} p={} run {}: n_reads={} p95={:.1f}us".format(
        row["trace"][:50], kind, p, run_idx, len(lat), row["p95"]), flush=True)
    return row


def collect_runs(trace_dirs, dev_pair, output_dir, n_jobs) -> pd.DataFrame:
    runs = discover_runs(trace_dirs, dev_pair)
    if not runs:
        sys.exit("No runs found for device pair {}".format(dev_pair))
    cache_path = os.path.join(output_dir, "msi_runs.csv")
    cached = {}
    if os.path.isfile(cache_path):
        prev = pd.read_csv(cache_path)
        cached = {r["run_dir"]: r for r in prev.to_dict("records")}
    todo, rows = [], []
    for desc in runs:
        prev = cached.get(desc[4])
        if prev is not None and abs(prev["mtime"] - trace_files_mtime(desc[4])) < 1e-6:
            rows.append(prev)
        else:
            todo.append(desc)
    print("[MSI] {} runs found, {} cached, {} to analyze".format(len(runs), len(rows), len(todo)), flush=True)
    if todo:
        rows += Parallel(n_jobs=min(n_jobs, len(todo)))(delayed(run_stats)(d) for d in todo)
    df = pd.DataFrame(rows).sort_values(["trace", "kind", "p", "run"]).reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    return df


def resolve_p0_source(df: pd.DataFrame, p0_source: str) -> str:
    """robust = flashnet/run_* (original FlashNet); flip = flashnet_flip_p00 (flips off). auto prefers robust."""
    traces = set(df.loc[df["kind"] == "baseline", "trace"])
    has_robust = traces <= set(df.loc[df["kind"] == "flashnet", "trace"])
    has_flip0 = traces <= set(df.loc[(df["kind"] == "flip") & (df["p"] == 0), "trace"])
    if p0_source == "auto":
        p0_source = "robust" if has_robust else "flip"
    if (p0_source == "robust" and not has_robust) or (p0_source == "flip" and not has_flip0):
        sys.exit("p = 0 source '{}' is missing for some traces (robust: flashnet/run_*, flip: flashnet_flip_p00)"
                 .format(p0_source))
    return p0_source


def latency_points(df: pd.DataFrame, p0_source: str) -> pd.DataFrame:
    """FlashNet curve points: p > 0 from flashnet_flip_pXX, p = 0 from the chosen source."""
    p0 = df["kind"] == "flashnet" if p0_source == "robust" else (df["kind"] == "flip") & (df["p"] == 0)
    return df[p0 | ((df["kind"] == "flip") & (df["p"] > 0))]


def aggregate(values_by_trace: dict) -> tuple:
    """Notebook-style aggregation: mean over traces of per-trace mean / std across runs."""
    means = [np.mean(v) for v in values_by_trace.values()]
    stds = [np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in values_by_trace.values()]
    return float(np.mean(means)), float(np.mean(stds))


def fit_msi(ps, means, baseline, fit_max_p):
    ps, means = np.asarray(ps), np.asarray(means)
    mask = ps <= fit_max_p + 1e-12
    slope = np.polyfit(ps[mask], means[mask], 1)[0] if mask.sum() >= 2 else np.nan
    advantage = baseline - means[ps == 0][0]
    msi = slope / advantage if advantage != 0 else np.nan
    p_star = advantage / slope if slope > 0 else np.inf
    return advantage, slope, msi, p_star


def summarize(df: pd.DataFrame, fit_max_p: float, n_boot: int, p0_source: str, seed: int = 0) -> tuple:
    curve = latency_points(df, p0_source)
    baseline = df[df["kind"] == "baseline"]
    traces = sorted(set(curve["trace"]) & set(baseline["trace"]))
    ps = sorted(curve["p"].unique())
    if 0.0 not in ps:
        sys.exit("No p = 0 results found")
    rng = np.random.default_rng(seed)

    point_rows, summary_rows = [], []
    for scope in ["aggregate"] + traces:
        scope_traces = traces if scope == "aggregate" else [scope]
        for metric in METRICS:
            base_vals = {t: baseline[baseline["trace"] == t][metric].values for t in scope_traces}
            base_mean = float(np.mean([np.mean(v) for v in base_vals.values()]))
            vals = {p: {t: curve[(curve["trace"] == t) & (curve["p"] == p)][metric].values for t in scope_traces}
                    for p in ps}
            vals = {p: {t: v for t, v in d.items() if len(v)} for p, d in vals.items()}
            vals = {p: d for p, d in vals.items() if len(d) == len(scope_traces)}  # p must cover every trace
            used_ps = sorted(vals)
            means, stds = zip(*[aggregate(vals[p]) for p in used_ps])
            for p, m, s in zip(used_ps, means, stds):
                point_rows.append({"scope": scope, "metric": metric, "p": p, "mean": m, "std": s,
                                   "min_runs": min(len(v) for v in vals[p].values()), "baseline": base_mean})
            adv, slope, msi, p_star = fit_msi(used_ps, means, base_mean, fit_max_p)

            boot = []
            for _ in range(n_boot):
                bm = [np.mean([np.mean(rng.choice(v, size=len(v), replace=True)) for v in vals[p].values()])
                      for p in used_ps]
                boot.append(fit_msi(used_ps, bm, base_mean, fit_max_p))
            boot = np.array(boot, dtype=float)

            def ci(col):
                finite = boot[:, col][np.isfinite(boot[:, col])]
                if len(finite) < 0.5 * n_boot:
                    return np.nan, np.nan
                return tuple(np.percentile(finite, [2.5, 97.5]))

            fit_ps = [p for p in used_ps if p <= fit_max_p + 1e-12]
            summary_rows.append({
                "scope": scope, "metric": metric, "p0_source": p0_source,
                "baseline_us": base_mean, "flashnet_p0_us": means[0],
                "advantage_us": adv, "S_us_per_unit_p": slope, "S_ci_lo": ci(1)[0], "S_ci_hi": ci(1)[1],
                "MSI": msi, "MSI_ci_lo": ci(2)[0], "MSI_ci_hi": ci(2)[1],
                "p_star": p_star, "p_star_ci_lo": ci(3)[0], "p_star_ci_hi": ci(3)[1],
                "p_star_extrapolated": bool(np.isfinite(p_star) and p_star > max(used_ps)),
                "fit_ps": " ".join("{:g}".format(p) for p in fit_ps), "min_runs_per_p": min(
                    min(len(v) for v in vals[p].values()) for p in used_ps),
            })
    return pd.DataFrame(point_rows), pd.DataFrame(summary_rows)


def fmt_pstar(row) -> str:
    if not np.isfinite(row["p_star"]):
        return "p* = inf (no degradation)"
    s = "p* = {:.2f}%".format(row["p_star"] * 100)
    if np.isfinite(row["p_star_ci_lo"]):
        s += " [{:.2f}, {:.2f}]".format(row["p_star_ci_lo"] * 100, row["p_star_ci_hi"] * 100)
    return s + (" (extrap.)" if row["p_star_extrapolated"] else "")


def plot_metric(ax, points, summ, metric, fit_max_p, big=False, generic_labels=False):
    pts = points[points["metric"] == metric].sort_values("p")
    row = summ[summ["metric"] == metric].iloc[0]
    x = pts["p"].values * 100
    y, yerr = pts["mean"].values / 1000, pts["std"].values / 1000
    ax.errorbar(x, y, yerr=yerr, fmt="o-", color=COLOR_FLASHNET, linewidth=2, markersize=7 if big else 5,
                capsize=4, label="FlashNet (injected flips)")
    base = row["baseline_us"] / 1000
    ax.axhline(base, color=COLOR_BASELINE, linestyle="--", linewidth=1.8,
               label="Baseline" if generic_labels else "Baseline ({:.3f} ms)".format(base))
    if np.isfinite(row["p_star"]) and row["p_star"] * 100 <= x.max():
        ax.axvline(row["p_star"] * 100, color=COLOR_BASELINE, linestyle=":", linewidth=1.8,
                   label="p* (FlashNet = baseline)" if generic_labels else "p* = {:.2f}%".format(row["p_star"] * 100))
    ax.set_xticks(x)
    ax.set_xlabel("Injected misprediction rate p (%)", fontsize=14 if big else 11)
    ax.set_ylabel("{} read latency (ms)".format(METRIC_LABELS[metric]), fontsize=14 if big else 11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)


def make_plots(points, summary, df, output_dir, headline_metrics, fit_max_p):
    agg_pts, agg_sum = points[points["scope"] == "aggregate"], summary[summary["scope"] == "aggregate"]

    # One dedicated figure per headline metric, titled like the Pensieve MSI plot
    for metric in headline_metrics:
        row = agg_sum[agg_sum["metric"] == metric].iloc[0]
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_metric(ax, agg_pts, agg_sum, metric, fit_max_p, big=True)
        ax.set_title("FlashNet MSI = {:.3f}   (S = {:.3f} ms {} latency per unit p)".format(
            row["MSI"], row["S_us_per_unit_p"] / 1000, METRIC_LABELS[metric]), fontsize=14)
        ax.legend(loc="best", fontsize=11)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(output_dir, "msi_{}.{}".format(metric.replace(".", "_"), ext)), dpi=200)
        plt.close(fig)

    def small_multiples(pts, summ, title, path_stem):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, metric in zip(axes.flat, METRICS):
            plot_metric(ax, pts, summ, metric, fit_max_p, generic_labels=True)
            ax.set_title("{}: {}".format(METRIC_LABELS[metric], fmt_pstar(summ[summ["metric"] == metric].iloc[0])),
                         fontsize=11)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        axes.flat[-1].axis("off")
        axes.flat[-1].legend(handles, labels, loc="center", fontsize=12)
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        for ext in ("png", "pdf") if path_stem.endswith("all_metrics") else ("png",):
            fig.savefig("{}.{}".format(path_stem, ext), dpi=150)
        plt.close(fig)

    small_multiples(agg_pts, agg_sum, "FlashNet misprediction sensitivity, all metrics (mean over traces)",
                    os.path.join(output_dir, "msi_all_metrics"))
    per_dir = os.path.join(output_dir, "per_trace")
    os.makedirs(per_dir, exist_ok=True)
    for trace in sorted(set(points["scope"]) - {"aggregate"}):
        small_multiples(points[points["scope"] == trace], summary[summary["scope"] == trace], trace,
                        os.path.join(per_dir, trace))

    flips = df[df["kind"] == "flip"]
    if len(flips):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for dev, marker in ((0, "o"), (1, "s")):
            g = flips.groupby("p")
            axes[0].errorbar(g["p"].first() * 100, g["realized_flip_rate_dev{}".format(dev)].mean() * 100,
                             yerr=g["realized_flip_rate_dev{}".format(dev)].std().fillna(0) * 100,
                             fmt=marker + "-", capsize=3, label="dev_{}".format(dev))
            axes[1].errorbar(g["p"].first() * 100, g["preflip_reject_rate_dev{}".format(dev)].mean() * 100,
                             yerr=g["preflip_reject_rate_dev{}".format(dev)].std().fillna(0) * 100,
                             fmt=marker + "-", capsize=3, label="dev_{}".format(dev))
        lim = flips["p"].max() * 100
        axes[0].plot([0, lim], [0, lim], color="gray", linestyle=":", label="realized = injected")
        axes[0].set_title("Realized vs injected flip rate (reads, mean over traces and runs)")
        axes[0].set_ylabel("Realized flip rate (%)")
        axes[1].set_title("Model's own (pre-flip) reject rate: decision drift from feedback")
        axes[1].set_ylabel("Pre-flip reject rate (%)")
        for ax in axes:
            ax.set_xlabel("Injected misprediction rate p (%)")
            ax.grid(linestyle=":", alpha=0.6)
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "msi_flip_stats.png"), dpi=150)
        plt.close(fig)
    print("[MSI] Figures written under {}".format(output_dir))


def main():
    parser = argparse.ArgumentParser(description="Misprediction Sensitivity Index (MSI) analysis for FlashNet.")
    parser.add_argument("-data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("-trace_dirs", nargs="+", default=None, help="Default: <data_root>/*/*/*")
    parser.add_argument("-dev_pair", default=DEFAULT_DEV_PAIR)
    parser.add_argument("-output_dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("-headline", nargs="+", default=["p95", "p99"], choices=METRICS,
                        help="Metrics that get a dedicated figure (msi_<metric>.png)")
    parser.add_argument("-fit_max_p", type=float, default=0.05, help="Fit S over p <= this (fraction)")
    parser.add_argument("-n_boot", type=int, default=2000)
    parser.add_argument("-n_jobs", type=int, default=12)
    parser.add_argument("-p0_source", default="auto", choices=["auto", "robust", "flip"],
                        help="p = 0 from flashnet/run_* (robust), flashnet_flip_p00 (flip), or robust if available")
    parser.add_argument("-plot_only", action="store_true", help="Use results/msi_runs.csv, don't read traces")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.plot_only:
        df = pd.read_csv(os.path.join(args.output_dir, "msi_runs.csv"))
    else:
        trace_dirs = args.trace_dirs or sorted(glob.glob(os.path.join(args.data_root, "*", "*", "*")))
        df = collect_runs(trace_dirs, args.dev_pair, args.output_dir, args.n_jobs)

    p0_source = resolve_p0_source(df, args.p0_source)
    print("[MSI] p = 0 source: {}".format(
        "flashnet/run_* (original FlashNet)" if p0_source == "robust" else "flashnet_flip_p00 (flips off)"))
    points, summary = summarize(df, args.fit_max_p, args.n_boot, p0_source)
    points.to_csv(os.path.join(args.output_dir, "msi_points.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "msi_summary.csv"), index=False)

    flips = df[df["kind"] == "flip"]
    if len(flips):
        drift = flips.groupby("p")[[c for c in flips.columns if c.startswith(("realized_flip", "preflip"))]].mean()
        drift.to_csv(os.path.join(args.output_dir, "msi_flip_stats.csv"))

    agg = summary[summary["scope"] == "aggregate"].set_index("metric")
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4g}".format):
        print(agg[["baseline_us", "flashnet_p0_us", "advantage_us", "S_us_per_unit_p", "MSI", "p_star",
                   "p_star_ci_lo", "p_star_ci_hi", "p_star_extrapolated", "min_runs_per_p"]])
    make_plots(points, summary, df, args.output_dir, args.headline, args.fit_max_p)


if __name__ == "__main__":
    main()
