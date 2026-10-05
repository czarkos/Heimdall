#!/usr/bin/env python3
"""
Shared helpers for the decision-boundary-complexity experiments (PCA and
smallest-p95 DT) on Heimdall's per-device training datasets.

Each dataset is one FlashNet training CSV:
  <data_root>/*/*/*/<dev_pair>/flashnet/training_results/mldrive{0,1}.csv
i.e. one dataset per (trace, device), exactly what Heimdall trains one model on.
"""

import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(EXPERIMENT_DIR, "dt"))
from train_dt import EXPECTED_INPUT_FEATURES, normalize_feature_aliases  # noqa: E402

DEFAULT_DATA_ROOT = os.path.join(
    os.environ.get("HEIMDALL", "/mnt/heimdall-exp/Heimdall"),
    "integration", "client-level", "data",
)
DEFAULT_DEV_PAIR = "nvme14n1...nvme15n1"
FEATURES = list(EXPECTED_INPUT_FEATURES)
LABEL = "reject"


@dataclass
class Dataset:
    name: str          # short unique id, e.g. alibaba.offset_p100.alibaba_9086.3__rerate_0.50-rerate_4.00__dev_1
    trace_dir: str     # <data_root>/<src>/<target>/<modification>
    device_idx: int    # 0 or 1
    csv_path: str

    @property
    def device(self) -> str:
        return f"dev_{self.device_idx}"

    @property
    def training_results_dir(self) -> str:
        return os.path.dirname(self.csv_path)


def _short_trace_name(trace_dir: str) -> str:
    target = os.path.basename(os.path.dirname(trace_dir)).replace("per_3mins.", "")
    modification = (
        os.path.basename(trace_dir)
        .replace("modified.", "")
        .replace("original", "orig")
        .replace("...", "-")
    )
    return f"{target}__{modification}"


def discover_datasets(data_root: str = DEFAULT_DATA_ROOT,
                      dev_pair: str = DEFAULT_DEV_PAIR,
                      only: Optional[List[str]] = None) -> List[Dataset]:
    """Find all mldrive{0,1}.csv datasets. `only` keeps names containing any of the substrings."""
    datasets = []
    pattern = os.path.join(data_root, "*", "*", "*", dev_pair, "flashnet", "training_results")
    for results_dir in sorted(glob.glob(pattern)):
        trace_dir = os.path.dirname(os.path.dirname(os.path.dirname(results_dir)))
        for i in (0, 1):
            csv_path = os.path.join(results_dir, f"mldrive{i}.csv")
            if not os.path.isfile(csv_path):
                print(f"[WARN] Missing dataset: {csv_path}")
                continue
            name = f"{_short_trace_name(trace_dir)}__dev_{i}"
            datasets.append(Dataset(name=name, trace_dir=trace_dir, device_idx=i, csv_path=csv_path))

    if only:
        datasets = [d for d in datasets if any(s in d.name for s in only)]

    names = [d.name for d in datasets]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Dataset names are not unique: {names}")
    return datasets


def load_xy(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the 12 DT/FlashNet input features and the `reject` label (latency is excluded)."""
    df = normalize_feature_aliases(pd.read_csv(csv_path))
    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns {missing}")
    return df[FEATURES].astype("float64"), df[LABEL].astype(int)


_VAL_ACC_RE = re.compile(r"val_accuracy:\s*([0-9.]+)")


def flashnet_val_accuracy(dataset: Dataset) -> Optional[float]:
    """Last epoch's val_accuracy from FlashNet's nnK.py log (mldrive{i}results.txt), or None."""
    path = os.path.join(dataset.training_results_dir, f"mldrive{dataset.device_idx}results.txt")
    if not os.path.isfile(path):
        return None
    with open(path, errors="replace") as f:
        matches = _VAL_ACC_RE.findall(f.read())
    return float(matches[-1]) if matches else None
