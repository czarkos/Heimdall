# Depth-11 FlashNet surrogate decision tree

A thin wrapper that trains and replays a **depth-11** surrogate DT. It reuses the depth-5 machinery unchanged:

| Step | What runs | Where it lives (tracked in git) |
|---|---|---|
| Training | `train_small_surrogate_dt_fidelity.py -target flashnet -max_depth 11`: a tree distilled from FlashNet's predictions on `mldrive<dev>.csv`, exported as C headers | `small_surrogate_dt/` |
| Replay | the `surrogate_dt` C replayer (`dt_algo.c` walks a tree of any depth) built with those headers | `surrogate_dt/` |
| Driver | `run_small_surrogate_dt.py -max_depth 11 -algo_name surrogate_dt_depth11` | `small_surrogate_dt/` |

The only depth-specific things are the two constants at the top of [run_surrogate_dt_depth11.py](run_surrogate_dt_depth11.py) (`MAX_DEPTH`, `ALGO_NAME`). `run_overnight.sh` is identical in every `surrogate_dt_depth<N>/` directory: it takes the algorithm name from its directory name.

The sibling directories `surrogate_dt_depth6/`, `7/`, `8/` and `11/` have the same layout; `../run_surrogate_depth_sweep.sh` runs all of them in sequence.

## Prerequisites (per trace, device pair `nvme0n1...nvme1n1`)

1. Baseline replay: `<trace>/<pair>/baseline/`
2. FlashNet trained: `<trace>/<pair>/flashnet/training_results/mldrive{0,1}.csv` and the exported teacher weights `mldrive{0,1}.csv.{weight,bias}_{0..3}.csv` (the trainer evaluates the teacher from these, not from the C header).

`run_overnight.sh` checks both before starting.

## Run

```bash
tmux new -s dt11
cd /mnt/heimdall-exp/Heimdall/integration/client-level/experiment/surrogate_dt_depth11
./run_overnight.sh                   # train (seconds per trace) + 1 replay per trace, then latency stats (~20 min)
./run_overnight.sh --num_runs 10     # robust: 10 replays per trace (~3.5 h); a later larger N adds only the missing runs
./run_overnight.sh --only_training   # headers + fidelity stats only (no SSD access)
```

Replays take about 3.5 minutes each. Everything is resumable: finished training and finished replays are skipped.

Or call the wrapper directly with the same flags as the other `run_*.py` scripts:

```bash
python3 run_surrogate_dt_depth11.py -devices /dev/nvme0n1 /dev/nvme1n1 \
    -trace_dirs $HEIMDALL/integration/client-level/data/*/*/* -num_runs 1 -resume
```

## Outputs

`<trace>/<pair>/surrogate_dt_depth11/`
- `training_results/surrogate_headers/w_Trace_dev_{0,1}_dt.h`: the trees used by the replayer
- `training_results/small_surrogate_dev_{0,1}_metrics.json`, `*_training.stats`: fidelity to FlashNet and accuracy against the ground-truth labels, train and test
- replays: `trace_{1,2}.trace[.stats]` directly in this directory with `-num_runs 1`, or `run_<i>/` subdirectories with more runs
- `latency_characteristic.stats`: read-latency percentiles (mean and std across runs when there are several), written by the `stats` step. `surrogate_dt_depth11` is in `ALGO_WHITELIST` of `algo_analysis/generate_latency_stats_with_dt.py`, so the regular stats pass picks it up too.

To get it into the comparison plots, add `surrogate_dt_depth11` to the `algos`, `label` and `colors` lists in `algo_analysis/tail_improvement_with_std_dev.ipynb`.

Logs from `run_overnight.sh` go to `logs/`.
