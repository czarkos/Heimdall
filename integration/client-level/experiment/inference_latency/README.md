# Inference latency, CPU and memory: FlashNet vs. depth-5 surrogate DT

A CPU-only microbenchmark. It doesn't touch the SSDs, so it can run alongside replays.

## What is measured

[bench.c](bench.c) is compiled against the **exact inference code used in the replays**:
- `flashnet/flashnet_algo.c`, with `flashnet/training_results/weights_header_2ssds/`
- `surrogate_dt/dt_algo.c`, with `small_surrogate_dt_depth5/training_results/surrogate_headers/`

It's built once per trace with that trace's trained models, and run separately for each device's model.

- **Inputs:** real feature vectors from that trace's `flashnet/training_results/mldrive<dev>.csv`, by default 200k rows after skipping the first 1k. The replayer's history pool is filled exactly as during a replay, so the model reads the same 12 features. `feature_match_rate` checks this and should be 1.0. Only reads are scored, as in the replayer.
- **Timing:** pinned to one CPU (default 4). It warms up for at least 1 s so the `powersave` governor ramps the frequency, then runs 10 timed passes. A loop calling an empty function measures loop and call overhead (`loop_overhead_ns`).
- **Optimization levels:** **-O0 is how the replayers are built** (their makefiles pass no `-O` flag); -O2 is for reference.

### Raw CPU numbers (per inference)

| Column | Meaning |
|---|---|
| `cpu_ns_mean` | Thread CPU time (`CLOCK_THREAD_CPUTIME_ID`) per inference, mean over passes |
| `wall_ns_mean` / `wall_ns_median` / `wall_ns_std` | Wall time per inference (`CLOCK_MONOTONIC_RAW`) across passes |
| `tsc_ticks_mean` | Time-stamp-counter ticks per inference. The TSC runs at a constant 2.3 GHz (`ticks_per_ns`), so this is wall time, not core cycles. |
| `percall_ns_p50/p90/p99/p999/max` | Per-call distribution from serialized `rdtsc` around each call, minus the timer overhead (`timer_overhead_ticks`) |
| `loop_overhead_ns` | Cost of the benchmark loop plus a function call. It's included in the batch numbers above. |

### Derived CPU metric

**`core_share_pct`** = `cpu_ns_mean` × the client's read IOPS / 10⁹ × 100. This is the share of one CPU core spent on inference while replaying that trace. Read IOPS comes from the baseline replay (`baseline/trace_<dev+1>.trace.stats`), because every read is scored once by its home device's model. The summary also sums the two clients of each trace, since both replay at the same time.

### Memory

**`model_bytes`** is the size of the model's arrays for one device: FlashNet's weights and normalizer, or the DT's feature, threshold, child and leaf arrays. It's read from the compiled object's symbol sizes (`nm -S`), so it's exact. Process RSS isn't used, because it's dominated by the trace and history arrays, which are the same for every model.

## Run

```bash
python3 run_bench.py                                   # all traces, ~4 min
python3 run_bench.py -only tencent -n_rows 50000 -passes 3   # quick check
python3 run_bench.py -plot_only                        # re-plot from results/inference_raw.csv
```

## Outputs (`results/`)

- `inference_raw.csv`: one row per trace, device, model and optimization level, with every raw number, the core share and the footprint.
- `inference_summary.csv`: per model and optimization level, the mean, min and max across the 12 trace and device combinations, plus the ratio to the DT.
- `inference_overhead.png/.pdf`: (a) CPU time per inference, (b) core share per trace at -O0, (c) model size.
- `environment.txt`: CPU model, governor, turbo setting, gcc version and parameters.
