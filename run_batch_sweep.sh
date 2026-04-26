#!/bin/bash
# Phase 2 Batch Sweep: Runs 2-4
# Test impact of batch size at 280-step regime on GTX 1650

set -e

cd "d:/Revi/Research Projects/autoresearch-revi"

# Run 3: BATCH=2048 (baseline from Phase 1, calibration point)
echo "=== Run 3: BATCH=2048 (baseline) ==="
git reset --hard HEAD
sed -i 's/TOTAL_BATCH_SIZE = 2\*\*12/TOTAL_BATCH_SIZE = 2**11/' train.py
git add train.py && git commit -m "Run 3: BATCH=2048 (baseline calibration)" --no-verify || true
timeout 630 uv run train.py > run.log 2>&1
commit=$(git rev-parse HEAD | cut -c1-7)
metrics=$(grep "val_bpb:\|num_steps:\|peak_vram_mb:" run.log | paste -sd' ')
echo -e "${commit}\t${metrics}" >> results_temp.tsv
git reset --hard HEAD~1

# Run 4: BATCH=8192 (4× baseline, very few steps)
echo "=== Run 4: BATCH=8192 (large batch, few steps) ==="
sed -i 's/TOTAL_BATCH_SIZE = 2\*\*11/TOTAL_BATCH_SIZE = 2**13/' train.py
git add train.py && git commit -m "Run 4: BATCH=8192 (test extreme batch)" --no-verify || true
timeout 630 uv run train.py > run.log 2>&1
commit=$(git rev-parse HEAD | cut -c1-7)
metrics=$(grep "val_bpb:\|num_steps:\|peak_vram_mb:" run.log | paste -sd' ')
echo -e "${commit}\t${metrics}" >> results_temp.tsv
git reset --hard HEAD~1

echo "Batch sweep complete. Results in results_temp.tsv"
