# Decision Boundary Complexity (Heimdall)

These two experiments measure how complex Heimdall's decision boundary is. Each one runs on the 12 per-device FlashNet training sets:

```
$HEIMDALL/integration/client-level/data/*/*/*/nvme14n1...nvme15n1/flashnet/training_results/mldrive{0,1}.csv
```

That is 6 traces × 2 devices, one dataset per model Heimdall trains. The inputs are the 12 features from `dt/train_dt.py` (`EXPECTED_INPUT_FEATURES`). The label is `reject`, and `latency` is excluded.

| Dir | Question | Method |
|---|---|---|
| `pca/` | How many principal components explain 95% of the variance? | StandardScaler, then PCA on all rows of each dataset |
| `smallest_p95_dt/` | What is the smallest decision tree, trained directly on the labels, that reaches 95% accuracy? | `DecisionTreeClassifier(max_depth=d)` for d = 1..40, 50/50 train/val split (`random_state=42`, same as `train_dt.py` / `nnK.py`) |

## Run

```bash
# Both experiments, unattended (logs go to logs/)
nohup ./run_overnight.sh > logs/overnight.out 2>&1 &
tail -f logs/overnight.out

# Options: --n_jobs N (default 12), --max_depth D (default 40), --resume, --only pca|dt
./run_overnight.sh --only dt --resume       # continue an interrupted DT sweep

# Or run each experiment directly
python3 pca/run_pca.py [-only <substr>...] [-plot_only]
python3 smallest_p95_dt/run_depth_sweep.py [-max_depth 40] [-only <substr>...] [-resume] [-plot_only]
```

`-plot_only` rebuilds the figures from the existing CSVs without retraining.

## Results

**`pca/results/`**
- `pca_summary.png`/`.pdf`: mean cumulative explained variance with the min–max band across datasets, mean per-component bars, the 95% line and the effective dimension.
- `pca_summary.csv`: k95 for each dataset.
- `pca_explained_variance.csv`: per-component values for each dataset.
- `per_dataset/*.png`: the same figure for each dataset.

**`smallest_p95_dt/results/`**
- `dt_depth_sweep_summary.png`/`.pdf`: mean train and validation accuracy against depth (min–max band), with the red dotted 95% line and mean FlashNet validation accuracy as a reference.
- `smallest_depth.csv`: for each dataset, the smallest depth reaching 95% on val and on train, the smallest depth matching FlashNet's validation accuracy, and the best validation accuracy with its depth.
- `dt_depth_sweep.csv`: every (dataset, depth) row.
- `per_dataset/*.png`: the same figure for each dataset.
- `partials/`: per-dataset checkpoints, saved after every depth and used by `-resume`.
