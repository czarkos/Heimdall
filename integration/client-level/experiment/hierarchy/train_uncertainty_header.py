#!/usr/bin/env python3

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

EXPECTED_INPUT_FEATURES = [
    "io_type",
    "size",
    "cur_queue_len",
    "prev_queue_len_1",
    "prev_queue_len_2",
    "prev_queue_len_3",
    "prev_latency_1",
    "prev_latency_2",
    "prev_latency_3",
    "prev_throughput_1",
    "prev_throughput_2",
    "prev_throughput_3",
]


def normalize_feature_aliases(dataset: pd.DataFrame) -> pd.DataFrame:
    columns = dataset.columns.tolist()
    has_cur = "cur_queue_len" in columns
    has_queue = "queue_len" in columns
    if (not has_cur) and has_queue:
        return dataset.rename(columns={"queue_len": "cur_queue_len"})
    if has_cur and has_queue:
        return dataset.drop(columns=["queue_len"])
    return dataset


def load_feature_matrix(dataset_path: str) -> np.ndarray:
    dataset = pd.read_csv(dataset_path)
    dataset = normalize_feature_aliases(dataset)

    missing_features = [f for f in EXPECTED_INPUT_FEATURES if f not in dataset.columns]
    if missing_features:
        raise ValueError(
            "Dataset is missing required DT features: {}\nExpected feature set: {}\nAvailable columns: {}".format(
                missing_features, EXPECTED_INPUT_FEATURES, dataset.columns.tolist()
            )
        )
    x = dataset[EXPECTED_INPUT_FEATURES].to_numpy(dtype=np.float32)
    return x


def choose_reference_indices(n: int, ref_size: int, seed: int) -> np.ndarray:
    if ref_size >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=ref_size, replace=False))


def min_knn_distance(x: np.ndarray, ref: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """
    Return distance to nearest point in ref for each row in x.
    """
    x = x.astype(np.float32, copy=False)
    ref = ref.astype(np.float32, copy=False)
    ref_norm2 = np.sum(ref * ref, axis=1, keepdims=True).T  # (1, M)

    out = np.empty((x.shape[0],), dtype=np.float32)
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        xb = x[start:end]
        x_norm2 = np.sum(xb * xb, axis=1, keepdims=True)  # (B, 1)
        d2 = x_norm2 + ref_norm2 - 2.0 * (xb @ ref.T)
        d2 = np.maximum(d2, 0.0)
        out[start:end] = np.sqrt(np.min(d2, axis=1))
    return out


def fit_uncertainty_model(
    x_raw: np.ndarray, ref_size: int, tau_percentile: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    eps = 1e-6
    mean = np.mean(x_raw, axis=0).astype(np.float32)
    std = np.std(x_raw, axis=0).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)

    x = ((x_raw - mean) / std).astype(np.float32)
    ref_idx = choose_reference_indices(x.shape[0], ref_size, seed)
    ref = x[ref_idx]

    # Calibrate tau on non-reference points when possible to avoid trivial zeros.
    mask = np.ones(x.shape[0], dtype=bool)
    mask[ref_idx] = False
    cal = x[mask] if np.any(mask) else x
    dist = min_knn_distance(cal, ref)
    tau = float(np.percentile(dist, tau_percentile))
    return mean, std, ref, tau


def export_uncertainty_header(
    mean: np.ndarray, std: np.ndarray, ref: np.ndarray, tau: float, drive: str, output_dir: str
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    header_name = f"u_Trace_{drive}_uncert.h"
    header_path = os.path.join(output_dir, header_name)
    guard = f"__U_{drive.upper()}_UNCERT_H"
    ref_flat = ref.reshape(-1)

    with open(header_path, "w") as f:
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")
        f.write(f"#define UNCERT_DIM_{drive.upper()} {mean.shape[0]}\n")
        f.write(f"#define UNCERT_REF_SIZE_{drive.upper()} {ref.shape[0]}\n")
        f.write(f"#define UNCERT_TAU_{drive.upper()} {tau:.8f}f\n\n")

        f.write(f"static const float uncert_mean_{drive}[] = {{\n  ")
        f.write(", ".join(f"{float(v):.8f}f" for v in mean))
        f.write("\n};\n\n")

        f.write(f"static const float uncert_std_{drive}[] = {{\n  ")
        f.write(", ".join(f"{float(v):.8f}f" for v in std))
        f.write("\n};\n\n")

        f.write(f"static const float uncert_ref_{drive}[] = {{\n  ")
        f.write(", ".join(f"{float(v):.8f}f" for v in ref_flat))
        f.write("\n};\n\n")
        f.write(f"#endif /* {guard} */\n")

    return header_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export kNN-distance uncertainty headers for hierarchy fallback."
    )
    parser.add_argument("-dataset", type=str, required=True, help="Path to mldrive*.csv")
    parser.add_argument("-drive", type=str, required=True, help="dev_0 or dev_1")
    parser.add_argument("-output_dir", type=str, required=True, help="Output header directory")
    parser.add_argument("-ref_size", type=int, default=512, help="Reference bank size")
    parser.add_argument(
        "-tau_percentile",
        type=float,
        default=95.0,
        help="Percentile of train-distance uncertainty used as fallback threshold",
    )
    parser.add_argument("-seed", type=int, default=42, help="Random seed for reference subsampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.dataset):
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if args.ref_size <= 0:
        raise SystemExit(f"ref_size must be > 0. Got {args.ref_size}")
    if not (0.0 < args.tau_percentile < 100.0):
        raise SystemExit(f"tau_percentile must be in (0, 100). Got {args.tau_percentile}")

    x = load_feature_matrix(args.dataset)
    mean, std, ref, tau = fit_uncertainty_model(
        x, ref_size=args.ref_size, tau_percentile=args.tau_percentile, seed=args.seed
    )
    header_path = export_uncertainty_header(
        mean=mean, std=std, ref=ref, tau=tau, drive=args.drive, output_dir=args.output_dir
    )

    print("=== Hierarchy uncertainty export complete ===")
    print(f"Dataset: {args.dataset}")
    print(f"Drive: {args.drive}")
    print(f"Ref size: {ref.shape[0]}")
    print(f"Tau (p{args.tau_percentile}): {tau:.6f}")
    print(f"Header: {header_path}")


if __name__ == "__main__":
    main()

