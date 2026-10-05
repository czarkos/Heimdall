# Misprediction Sensitivity Index (FlashNet / Heimdall)

**Question:** how many mispredictions, beyond what the original model already makes, can the system tolerate before FlashNet loses its advantage over the baseline? This is the Heimdall version of Pensieve's Misprediction Sensitivity Index.

## Method

- **Injection:** each read's FlashNet prediction (reject or accept) is flipped with probability **p**, in [flashnet_flip/io_replayer.c](flashnet_flip/io_replayer.c) (`should_flip()` in `flashnet_algo()`).
  - Each read consults only its home device's model, so one global p applies to both per-device models, with one independent flip per read. Writes are never touched.
  - The history update rule is unchanged, so hardware feedback from flipped decisions is part of the measured effect.
  - p is set at build time (`make FLIP_PROB=0.05 FLIP_SEED=7`). With p = 0, the binary is identical in behavior to `experiment/flashnet`.
- **Models:** the trained FlashNet weights are reused read-only from `<trace>/<pair>/flashnet/training_results/weights_header_2ssds`, the same models as the p = 0 runs.
- **Per-I/O logging:** each output line gets 3 extra columns: `orig_pred` (−1 for writes), `flipped` and `target_device`. Existing readers only use the first 7 columns.
- **p values:** {0, 1, 2, 3, 4, 5, 10}%. The baseline is a single run, as in the original paper.
- **Where p = 0 comes from** (`--p0`):
  - `robust`: the robust FlashNet runs (`flashnet/run_*`).
  - `single`: one p = 0 replay added to the overnight run, written to `flashnet_flip_p00`. It uses this directory's replayer with flips off, which makes exactly the same decisions as the original FlashNet and also logs them.
  - `auto` (default): `robust` if every trace has `flashnet/run_*`, otherwise `single`.

## Metrics

All metrics are read latency, with both clients pooled. They're computed by the same code as the robust results (`algo_analysis/generate_latency_stats_with_dt.py`): **p95 (headline)**, plus p99, p99.9, p99.99 and the average.

| Quantity | Definition |
|---|---|
| Advantage | L_baseline − L_FlashNet(p=0) |
| S | Least-squares slope of L against p (p as a fraction), fitted over p ≤ 5% (`-fit_max_p`) |
| MSI | S / Advantage (Pensieve's index) |
| **p\*** | Advantage / S = 1/MSI: the extra misprediction rate at which FlashNet falls back to the baseline |

Each curve point is the mean over traces of the per-trace mean across runs. Error bars are the mean over traces of the std across runs, as in `tail_improvement_with_std_dev.ipynb`. The 95% CIs for S, MSI and p\* come from a bootstrap over runs.

## Prerequisites (device pair `nvme0n1...nvme1n1`)

1. A baseline replay for each trace in `<trace>/<pair>/baseline/`.
2. FlashNet trained on that baseline, producing `flashnet/training_results/weights_header_2ssds/`.
3. Optional: the robust p = 0 runs (`run_flashnet.py -num_runs 10`, in `flashnet/run_*`). Without them, use `--p0 single` (or the default `auto`).

## Run (tmux)

```bash
tmux new -s msi
cd /mnt/heimdall-exp/Heimdall/integration/client-level/experiment/misprediction_sensitivity_index
./run_overnight.sh                     # auto: robust p=0 if available, else adds one p=0 replay
./run_overnight.sh --p0 single         # 1 p=0 replay + 1 run of each p>0, about 10.5 h (about 90 min per p)
./run_overnight.sh --p0 robust         # use flashnet/run_* for p=0, about 9 h
./run_overnight.sh --num_runs 3        # later: adds runs 1-2 of each p>0; finished runs are skipped
./run_overnight.sh --analyze_only      # re-analyze only
```

Detach with `Ctrl-b d` and reattach with `tmux attach -t msi`.

The p = 0 replay (`flashnet_flip_p00`) is also the reference for the model's pre-flip reject rate (decision drift), because the original replayer doesn't log decisions. In `robust` mode, add `--with_p0_reference` to get that reference too; it's then used only for drift, not for the latency curve. `--p0_runs N` sets how many p = 0 replays are added (default 1).

Run order is interleaved (run index → p → trace), so a partial night still covers every p.

## Outputs

**Replays:** `<trace>/<pair>/flashnet_flip_pXX/run_<i>/`. Each contains `trace_{1,2}.trace[.stats]` plus `flip_config.txt`, which records p, the seed and the md5 checksums of the weights.

**Analysis** (in `results/`):
- `msi_p95.png/.pdf`: the headline figure, p95 against p with the baseline line and p\*.
- `msi_all_metrics.png/.pdf`: one panel per metric.
- `msi_summary.csv`: Advantage, S, MSI and p\* with CIs, for the aggregate and for each trace.
- `msi_points.csv`: the curve points.
- `msi_runs.csv`: one row per run; also serves as a cache, so re-runs only read new replays.
- `msi_flip_stats.png/.csv`: realized flip rate and pre-flip reject rate against p, for each device.
- `per_trace/*.png`: the all-metrics figure for each trace.

Logs are written to `logs/`.
