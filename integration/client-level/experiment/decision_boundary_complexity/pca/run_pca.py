#!/usr/bin/env python3
"""
Decision boundary complexity (1/2): PCA on Heimdall's input features.

For each per-device training dataset (mldrive{0,1}.csv), standardize the 12
input features and find how many principal components explain 95% of the
variance. Produces a per-dataset figure plus a summary figure (mean curve with
min-max band across datasets) in the style of the Pensieve PCA plot.

Usage:
  python3 run_pca.py                       # all datasets
  python3 run_pca.py -n_jobs 12 -only tencent
  python3 run_pca.py -plot_only            # re-plot from existing CSVs
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DEFAULT_DATA_ROOT, DEFAULT_DEV_PAIR, FEATURES, discover_datasets, load_xy  # noqa: E402

THRESHOLD = 0.95
COLOR_CUM = "#2a78d6"
COLOR_BAR = "#cfcfcf"
COLOR_THRESH = "#d62728"


def run_one(dataset):
    t0 = time.time()
    X, _ = load_xy(dataset.csv_path)
    X_std = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=len(FEATURES)).fit(X_std)
    ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    rows = [
        {
            "dataset": dataset.name,
            "device": dataset.device,
            "component": k + 1,
            "explained_variance_ratio": ratio[k],
            "cumulative_explained_variance": cumulative[k],
        }
        for k in range(len(ratio))
    ]
    k95 = int(np.argmax(cumulative >= THRESHOLD - 1e-12) + 1)
    print(f"[PCA] {dataset.name}: n_rows={len(X)} k95={k95} ({time.time() - t0:.1f}s)", flush=True)
    summary = {"dataset": dataset.name, "device": dataset.device, "n_rows": len(X), "k95": k95,
               "csv_path": dataset.csv_path}
    return rows, summary


def k_for_threshold(cumulative):
    return int(np.argmax(np.asarray(cumulative) >= THRESHOLD - 1e-12) + 1)


def plot_pca(ax, components, per_comp, cum, cum_lo=None, cum_hi=None, k95=None, k95_range=None):
    ax.bar(components, per_comp * 100, color=COLOR_BAR, width=0.7, label="Per-component variance", zorder=1)
    if cum_lo is not None:
        ax.fill_between(components, cum_lo * 100, cum_hi * 100, color=COLOR_CUM, alpha=0.18, linewidth=0,
                        label="Cumulative (min–max across datasets)", zorder=2)
    ax.plot(components, cum * 100, "-o", color=COLOR_CUM, linewidth=2, markersize=6,
            label="Cumulative variance" + (" (mean)" if cum_lo is not None else ""), zorder=3)
    ax.axhline(THRESHOLD * 100, color=COLOR_THRESH, linestyle="--", linewidth=1.8, label="95% threshold", zorder=4)
    dim_label = f"Effective dim = {k95}"
    if k95_range is not None and k95_range[0] != k95_range[1]:
        dim_label += f" (per-dataset {k95_range[0]}–{k95_range[1]})"
    ax.axvline(k95, color=COLOR_THRESH, linestyle=":", linewidth=1.8, label=dim_label, zorder=4)
    ax.set_xticks(components)
    ax.set_xlim(0.4, components[-1] + 0.6)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Number of principal components", fontsize=15)
    ax.set_ylabel("Variance explained (%)", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="center right", fontsize=11, framealpha=0.95)


def make_plots(output_dir):
    long_df = pd.read_csv(os.path.join(output_dir, "pca_explained_variance.csv"))
    summary_df = pd.read_csv(os.path.join(output_dir, "pca_summary.csv"))
    n_feat = int(long_df["component"].max())
    components = np.arange(1, n_feat + 1)

    per_dir = os.path.join(output_dir, "per_dataset")
    os.makedirs(per_dir, exist_ok=True)
    for name, g in long_df.groupby("dataset"):
        g = g.sort_values("component")
        fig, ax = plt.subplots(figsize=(10, 6))
        cum = g["cumulative_explained_variance"].values
        plot_pca(ax, components, g["explained_variance_ratio"].values, cum, k95=k_for_threshold(cum))
        ax.set_title(f"PCA of Heimdall inputs ({n_feat}-d standardized features)\n{name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(per_dir, f"{name}.png"), dpi=150)
        plt.close(fig)

    piv_cum = long_df.pivot(index="component", columns="dataset", values="cumulative_explained_variance")
    piv_ratio = long_df.pivot(index="component", columns="dataset", values="explained_variance_ratio")
    mean_cum = piv_cum.mean(axis=1).values
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pca(ax, components, piv_ratio.mean(axis=1).values, mean_cum,
             cum_lo=piv_cum.min(axis=1).values, cum_hi=piv_cum.max(axis=1).values,
             k95=k_for_threshold(mean_cum),
             k95_range=(int(summary_df["k95"].min()), int(summary_df["k95"].max())))
    ax.set_title(f"PCA of Heimdall inputs ({n_feat}-d standardized features, "
                 f"{piv_cum.shape[1]} datasets)", fontsize=15)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"pca_summary.{ext}"), dpi=200)
    plt.close(fig)
    print(f"[PCA] Figures written under {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="PCA of Heimdall's input features (95% variance).")
    parser.add_argument("-data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("-dev_pair", default=DEFAULT_DEV_PAIR)
    parser.add_argument("-output_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    parser.add_argument("-n_jobs", type=int, default=12)
    parser.add_argument("-only", nargs="+", default=None, help="Keep datasets whose name contains any of these")
    parser.add_argument("-plot_only", action="store_true", help="Re-plot from existing CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.plot_only:
        datasets = discover_datasets(args.data_root, args.dev_pair, args.only)
        if not datasets:
            sys.exit(f"No datasets found under {args.data_root}")
        print(f"[PCA] {len(datasets)} datasets, n_jobs={args.n_jobs}", flush=True)
        results = Parallel(n_jobs=min(args.n_jobs, len(datasets)))(delayed(run_one)(d) for d in datasets)
        long_rows = [r for rows, _ in results for r in rows]
        pd.DataFrame(long_rows).to_csv(os.path.join(args.output_dir, "pca_explained_variance.csv"), index=False)
        summary_df = pd.DataFrame([s for _, s in results])
        summary_df.to_csv(os.path.join(args.output_dir, "pca_summary.csv"), index=False)
        print(summary_df[["dataset", "n_rows", "k95"]].to_string(index=False))

    make_plots(args.output_dir)


if __name__ == "__main__":
    main()
