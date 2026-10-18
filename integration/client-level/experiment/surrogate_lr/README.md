# Logistic-regression surrogate of FlashNet (`surrogate_lr`)

A linear surrogate to compare against the surrogate decision trees (`small_surrogate_dt_depth5`, `surrogate_dt_depth{6,7,8,11}`). Like them it is distilled from FlashNet's own predictions on the 12 FlashNet features, replayed with the same two-client replayer, and evaluated with the same latency stats.

| Step | What runs | File |
|---|---|---|
| Training | `StandardScaler` + `LogisticRegression` (lbfgs, `C=1.0`) fitted to FlashNet's predictions on `mldrive<dev>.csv`, 50/50 split, `random_state=42`. The scaler is folded into the exported weights. | [train_surrogate_lr_fidelity.py](train_surrogate_lr_fidelity.py), driven by [train_surrogate_lr.sh](train_surrogate_lr.sh) |
| Inference | `z = bias + Σ weight[j]·x[j]` on the replayer's raw `long` feature vector; reject the read iff `z > 0`. 12 multiply-adds, no normalization, no `exp`. | [lr_algo.c](lr_algo.c), [lr_algo.h](lr_algo.h) |
| Replayer | `surrogate_dt/io_replayer.c` with the model calls renamed; routing and history updates are identical to FlashNet and the DTs | [io_replayer.c](io_replayer.c), [replay_lr.sh](replay_lr.sh) |
| Driver | train once, copy to `experiment/tmp_running/`, build, replay `-num_runs` times | [run_surrogate_lr.py](run_surrogate_lr.py), [run_overnight.sh](run_overnight.sh) |

Data prep and FlashNet teacher reconstruction are imported from `small_surrogate_dt/train_small_surrogate_dt_fidelity.py`, so the fidelity numbers are directly comparable with the trees.

The trainer verifies the export: it recomputes the decision in the exact accumulation order `lr_algo.c` uses and reports `header_match_rate` (fraction of rows where that agrees with sklearn's `predict`). It refuses to write a header if that is not 1.0 within 1e-6.

## Prerequisites (per trace, device pair `nvme0n1...nvme1n1`)

1. Baseline replay: `<trace>/<pair>/baseline/` (the replay duration is read from its stats)
2. FlashNet trained: `<trace>/<pair>/flashnet/training_results/mldrive{0,1}.csv` and the exported teacher weights `mldrive{0,1}.csv.{weight,bias}_{0..3}.csv`

`run_overnight.sh` checks both before starting.

## Run

```bash
tmux new -s lr
cd /mnt/heimdall-exp/Heimdall/integration/client-level/experiment/surrogate_lr
./run_overnight.sh --only_training   # headers + fidelity stats only (seconds per trace, no SSD access)
./run_overnight.sh                   # train (if needed) + 1 replay per trace, then latency stats (~20 min)
./run_overnight.sh --num_runs 10     # robust: 10 replays per trace (~3.5 h); a later larger N adds only the missing runs
```

Replays take about 3.5 minutes each. Everything is resumable: finished training and finished replays are skipped. Other options: `--only_replaying`, `--skip_stats`, `--devices "..."`, `--data_root DIR`.

Or call the runner directly with the same flags as the other `run_*.py` scripts:

```bash
python3 run_surrogate_lr.py -devices /dev/nvme0n1 /dev/nvme1n1 \
    -trace_dirs $HEIMDALL/integration/client-level/data/*/*/* -num_runs 1 -resume
```

`-C <float>` changes the regularization strength and `-algo_name <name>` writes to a different output directory, so variants can live side by side. `SURROGATE_LR_TARGET=label ./train_surrogate_lr.sh ...` trains directly on the ground-truth labels instead of FlashNet's predictions.

## Outputs

`<trace>/<pair>/surrogate_lr/`
- `training_results/lr_headers/w_Trace_dev_{0,1}_lr.h`: the weights used by the replayer (12 doubles + bias, scaler folded in)
- `training_results/surrogate_lr_dev_{0,1}_metrics.json`: fidelity to FlashNet, accuracy against the ground-truth labels, `header_match_rate`, standardized coefficients, scaler mean/scale, folded weights
- `training_results/surrogate_lr_dev_{0,1}_training.stats`, `surrogate_lr_training.stats`: the same, human-readable
- replays: `trace_{1,2}.trace[.stats]` directly in this directory with `-num_runs 1`, or `run_<i>/` subdirectories with more runs
- `latency_characteristic.stats`: read-latency percentiles (mean and std across runs when there are several), written by the `stats` step. `surrogate_lr` is in `ALGO_WHITELIST` of `algo_analysis/generate_latency_stats_with_dt.py`, so the regular stats pass picks it up too.

`surrogate_lr` is already in the `algos`, `label`, `font_colors` and `colors` lists of `algo_analysis/tail_improvement_with_std_dev.ipynb`, and `lr` is a model in `experiment/inference_latency/run_bench.py` (`python3 run_bench.py -models flashnet dt_depth5 lr`).

Logs from `run_overnight.sh` go to `logs/`.
