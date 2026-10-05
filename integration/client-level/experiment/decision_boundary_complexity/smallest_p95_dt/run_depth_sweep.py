#!/usr/bin/env python3
"""
Decision boundary complexity (2/2): smallest decision tree reaching 95% accuracy.

For each per-device training dataset (mldrive{0,1}.csv), train a decision tree
directly on the `reject` labels (not as a FlashNet surrogate) for every
max_depth in [min_depth, max_depth] and record train / validation accuracy.
Uses the same 50/50 split (random_state=42) and raw 12-feature layout as
dt/train_dt.py and FlashNet's nnK.py.

Results are checkpointed per dataset after every depth, so a long (overnight)
run can be resumed with -resume after a crash.

Usage:
  python3 run_depth_sweep.py                          # all datasets, depth 1..40
  python3 run_depth_sweep.py -only tencent -max_depth 10
  python3 run_depth_sweep.py -resume                  # continue an interrupted run
  python3 run_depth_sweep.py -plot_only               # re-plot from existing CSVs
"""

import argparse
import glob
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DEFAULT_DATA_ROOT, DEFAULT_DEV_PAIR, discover_datasets, flashnet_val_accuracy, load_xy  # noqa: E402

TARGET = 0.95
COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"
COLOR_TARGET = "#d62728"
COLOR_FLASHNET = "#52514e"


def partial_path(output_dir, name):
    return os.path.join(output_dir, "partials", f"{name}.csv")


def sweep_one(dataset, min_depth, max_depth, output_dir, resume):
    out_csv = partial_path(output_dir, dataset.name)
    rows = []
    start_depth = min_depth
    if resume and os.path.isfile(out_csv):
        prev = pd.read_csv(out_csv)
        prev = prev[(prev["max_depth"] >= min_depth) & (prev["max_depth"] <= max_depth)].sort_values("max_depth")
        # Only resume a contiguous prefix starting at min_depth; otherwise start over.
        if len(prev) and prev["max_depth"].tolist() == list(range(min_depth, min_depth + len(prev))):
            last = prev.iloc[-1]
            if int(last["max_depth"]) == max_depth:
                pd.DataFrame(prev).to_csv(out_csv, index=False)
                print(f"[DT] {dataset.name}: already done, skipping", flush=True)
                return
            if bool(last["saturated"]) or int(last["actual_depth"]) < int(last["max_depth"]):
                # Fully grown tree: extend the saturated rows without refitting.
                rows = prev.to_dict("records")
                for d in range(int(last["max_depth"]) + 1, max_depth + 1):
                    rows.append(dict(last, max_depth=d, fit_seconds=0.0, saturated=True))
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                print(f"[DT] {dataset.name}: fully grown, extended to depth {max_depth}", flush=True)
                return
            rows = prev.to_dict("records")
            start_depth = int(last["max_depth"]) + 1
            print(f"[DT] {dataset.name}: resuming at depth {start_depth}", flush=True)

    t_load = time.time()
    X, y = load_xy(dataset.csv_path)
    X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, test_size=0.5, random_state=42)
    del X, y
    flashnet_acc = flashnet_val_accuracy(dataset)
    print(f"[DT] {dataset.name}: loaded {len(X_train) + len(X_val)} rows in {time.time() - t_load:.1f}s", flush=True)

    common_fields = {
        "dataset": dataset.name,
        "device": dataset.device,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "flashnet_val_accuracy": flashnet_acc,
    }

    for depth in range(start_depth, max_depth + 1):
        t0 = time.time()
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        row = dict(common_fields)
        row.update({
            "max_depth": depth,
            "actual_depth": clf.get_depth(),
            "n_leaves": clf.get_n_leaves(),
            "n_nodes": clf.tree_.node_count,
            "train_accuracy": clf.score(X_train, y_train),
            "val_accuracy": clf.score(X_val, y_val),
            "fit_seconds": time.time() - t0,
            "saturated": False,
        })
        rows.append(row)
        print(f"[DT] {dataset.name}: depth={depth:2d} train={row['train_accuracy']:.4f} "
              f"val={row['val_accuracy']:.4f} leaves={row['n_leaves']} ({row['fit_seconds']:.1f}s)", flush=True)

        # Tree stopped growing before hitting max_depth => fully grown; deeper limits yield the same tree.
        if row["actual_depth"] < depth:
            for d in range(depth + 1, max_depth + 1):
                rows.append(dict(row, max_depth=d, fit_seconds=0.0, saturated=True))
            print(f"[DT] {dataset.name}: tree fully grown at depth {row['actual_depth']}, "
                  f"filling depths {depth + 1}..{max_depth}", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            break

        pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[DT] {dataset.name}: done", flush=True)


def first_depth(df, column, threshold):
    hit = df.loc[df[column] >= threshold, "max_depth"]
    return int(hit.min()) if len(hit) else "not reached"


def summarize(sweep_df):
    rows = []
    for name, g in sweep_df.groupby("dataset", sort=False):
        g = g.sort_values("max_depth")
        flashnet_acc = g["flashnet_val_accuracy"].iloc[0]
        best = g.loc[g["val_accuracy"].idxmax()]
        rows.append({
            "dataset": name,
            "device": g["device"].iloc[0],
            "smallest_depth_val_ge_95": first_depth(g, "val_accuracy", TARGET),
            "smallest_depth_train_ge_95": first_depth(g, "train_accuracy", TARGET),
            "flashnet_val_accuracy": flashnet_acc,
            "smallest_depth_val_ge_flashnet": (first_depth(g, "val_accuracy", flashnet_acc)
                                               if pd.notna(flashnet_acc) else "n/a"),
            "best_val_accuracy": best["val_accuracy"],
            "best_val_depth": int(best["max_depth"]),
            "train_accuracy_at_max_depth": g["train_accuracy"].iloc[-1],
            "val_accuracy_at_max_depth": g["val_accuracy"].iloc[-1],
            "fully_grown_depth": (int(g.loc[g["saturated"], "actual_depth"].iloc[0])
                                  if g["saturated"].any() else "not reached"),
        })
    return pd.DataFrame(rows)


def style_axes(ax, depths):
    ax.axhline(TARGET * 100, color=COLOR_TARGET, linestyle=":", linewidth=2, label="95% accuracy", zorder=4)
    ax.set_xlim(depths.min() - 0.5, depths.max() + 0.5)
    ax.set_xticks(sorted({int(depths.min())} | set(range(5, int(depths.max()) + 1, 5))))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel("Decision tree depth (max_depth)", fontsize=15)
    ax.set_ylabel("Accuracy (%)", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)


def make_plots(output_dir):
    sweep_df = pd.read_csv(os.path.join(output_dir, "dt_depth_sweep.csv"))
    per_dir = os.path.join(output_dir, "per_dataset")
    os.makedirs(per_dir, exist_ok=True)

    for name, g in sweep_df.groupby("dataset"):
        g = g.sort_values("max_depth")
        depths = g["max_depth"].values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(depths, g["train_accuracy"] * 100, "-o", color=COLOR_TRAIN, linewidth=2, markersize=5, label="Train")
        ax.plot(depths, g["val_accuracy"] * 100, "-s", color=COLOR_VAL, linewidth=2, markersize=5, label="Validation")
        flashnet_acc = g["flashnet_val_accuracy"].iloc[0]
        if pd.notna(flashnet_acc):
            ax.axhline(flashnet_acc * 100, color=COLOR_FLASHNET, linestyle="--", linewidth=1.5,
                       label=f"FlashNet validation ({flashnet_acc * 100:.1f}%)", zorder=3)
        style_axes(ax, depths)
        ax.set_title(f"Decision tree accuracy vs. depth (trained on labels)\n{name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(per_dir, f"{name}.png"), dpi=150)
        plt.close(fig)

    agg = sweep_df.groupby("max_depth").agg(
        train_mean=("train_accuracy", "mean"), train_min=("train_accuracy", "min"), train_max=("train_accuracy", "max"),
        val_mean=("val_accuracy", "mean"), val_min=("val_accuracy", "min"), val_max=("val_accuracy", "max"),
    ).reset_index()
    depths = agg["max_depth"].values
    n_datasets = sweep_df["dataset"].nunique()
    flashnet_mean = sweep_df.groupby("dataset")["flashnet_val_accuracy"].first().mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    for split, color, marker, label in (("train", COLOR_TRAIN, "o", "Train"), ("val", COLOR_VAL, "s", "Validation")):
        ax.fill_between(depths, agg[f"{split}_min"] * 100, agg[f"{split}_max"] * 100,
                        color=color, alpha=0.15, linewidth=0, zorder=1)
        ax.plot(depths, agg[f"{split}_mean"] * 100, "-" + marker, color=color, linewidth=2, markersize=5,
                label=f"{label} (mean; band = min–max)", zorder=3)
    if pd.notna(flashnet_mean):
        ax.axhline(flashnet_mean * 100, color=COLOR_FLASHNET, linestyle="--", linewidth=1.5,
                   label=f"FlashNet validation (mean {flashnet_mean * 100:.1f}%)", zorder=3)
    style_axes(ax, depths)
    ax.set_title(f"Decision tree accuracy vs. depth (trained on labels, {n_datasets} datasets)", fontsize=15)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"dt_depth_sweep_summary.{ext}"), dpi=200)
    plt.close(fig)
    print(f"[DT] Figures written under {output_dir}")


def collect(output_dir):
    """Combine every dataset's partial CSV (not just this run's -only subset) into the final outputs."""
    paths = sorted(glob.glob(os.path.join(output_dir, "partials", "*.csv")))
    if not paths:
        sys.exit(f"No partial results found under {output_dir}/partials")
    sweep_df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    sweep_df.to_csv(os.path.join(output_dir, "dt_depth_sweep.csv"), index=False)
    summary_df = summarize(sweep_df)
    summary_df.to_csv(os.path.join(output_dir, "smallest_depth.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(summary_df.drop(columns=["device"]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Smallest decision tree (trained on labels) reaching 95% accuracy.")
    parser.add_argument("-data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("-dev_pair", default=DEFAULT_DEV_PAIR)
    parser.add_argument("-output_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    parser.add_argument("-min_depth", type=int, default=1)
    parser.add_argument("-max_depth", type=int, default=40)
    parser.add_argument("-n_jobs", type=int, default=12)
    parser.add_argument("-only", nargs="+", default=None, help="Keep datasets whose name contains any of these")
    parser.add_argument("-resume", action="store_true", help="Skip finished datasets, continue partial ones")
    parser.add_argument("-plot_only", action="store_true", help="Re-collect partials and re-plot, no training")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "partials"), exist_ok=True)
    if not args.plot_only:
        datasets = discover_datasets(args.data_root, args.dev_pair, args.only)
        if not datasets:
            sys.exit(f"No datasets found under {args.data_root}")
        print(f"[DT] {len(datasets)} datasets, depths {args.min_depth}..{args.max_depth}, "
              f"n_jobs={args.n_jobs}, resume={args.resume}", flush=True)
        if not args.resume:
            for d in datasets:
                if os.path.exists(partial_path(args.output_dir, d.name)):
                    os.remove(partial_path(args.output_dir, d.name))
        # Largest datasets first so the longest jobs start immediately.
        datasets.sort(key=lambda d: os.path.getsize(d.csv_path), reverse=True)
        Parallel(n_jobs=min(args.n_jobs, len(datasets)))(
            delayed(sweep_one)(d, args.min_depth, args.max_depth, args.output_dir, args.resume) for d in datasets
        )

    collect(args.output_dir)
    make_plots(args.output_dir)


if __name__ == "__main__":
    main()
