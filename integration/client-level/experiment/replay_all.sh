#!/usr/bin/env bash
set -euo pipefail

# Adjust these if needed
DEV1="/dev/nvme0n1"
DEV2="/dev/nvme2n1"
DEVICES="$DEV1 $DEV2"
TRACES="$HEIMDALL/integration/client-level/data/*/*/*"

LOG_DIR="$HEIMDALL/integration/client-level/logs"
mkdir -p "$LOG_DIR"

cd "$HEIMDALL/integration/client-level/experiment"

echo "Starting replay batch at $(date)"
echo "Devices: $DEVICES"
echo "Traces:  $TRACES"
echo

# Baseline
echo "=== Baseline replay === $(date)"
./run_baseline.py \
  -devices $DEVICES \
  -trace_dirs $TRACES \
  | tee "$LOG_DIR/baseline_replay.log"

# FlashNet (Heimdall) replay
echo "=== FlashNet replay === $(date)"
./run_flashnet.py \
  -devices $DEVICES \
  -trace_dirs $TRACES \
  -only_replaying \
  | tee "$LOG_DIR/flashnet_replay.log"

# Random
echo "=== Random replay === $(date)"
./run_random.py \
  -devices $DEVICES \
  -trace_dirs $TRACES \
  | tee "$LOG_DIR/random_replay.log"

# Hedging P95 (they used 98 percentile in the doc)
echo "=== Hedging P95 replay === $(date)"
./run_hedging.py \
  -devices $DEVICES \
  -hedging_percentile 98 \
  -trace_dirs $TRACES \
  | tee "$LOG_DIR/hedging_p95_replay.log"

# LinnOS replay
echo "=== LinnOS replay === $(date)"
./run_linnos.py \
  -devices $DEVICES \
  -trace_dirs $TRACES \
  -only_replaying \
  | tee "$LOG_DIR/linnos_replay.log"

# LinnOS + Hedging
echo "=== LinnOS + Hedging replay === $(date)"
./run_linnos_hedging.py \
  -devices $DEVICES \
  -trace_dirs $TRACES \
  -hedging_percentile 98 \
  | tee "$LOG_DIR/linnos_hedging_replay.log"

echo
echo "All replays completed at $(date)"