#!/bin/bash
# Phase 2: Batch and LR Sweep
# Runs tests 2-10 in sequence, recording results

set -e

cd "d:/Revi/Research Projects/autoresearch-revi"

# Run 2: BATCH=1024
echo "=== Run 2: BATCH=1024 ==="
git show HEAD:train.py > train.py.bak
sed -i 's/TOTAL_BATCH_SIZE = 2\*\*11/TOTAL_BATCH_SIZE = 2**10/' train.py
git add train.py && git commit -m "Run 2: BATCH=1024" --no-verify || true
timeout 630 uv run train.py > run.log 2>&1
commit=$(git rev-parse HEAD | cut -c1-7)
metrics=$(grep "val_bpb:\|num_steps:\|peak_vram_mb:" run.log | paste -sd' ')
echo -e "${commit}\t${metrics}" >> results_temp.tsv
git reset --hard HEAD~1

# Run 3: BATCH=4096
echo "=== Run 3: BATCH=4096 ==="
sed -i 's/TOTAL_BATCH_SIZE = 2\*\*11/TOTAL_BATCH_SIZE = 2**12/' train.py
git add train.py && git commit -m "Run 3: BATCH=4096" --no-verify || true
timeout 630 uv run train.py > run.log 2>&1
commit=$(git rev-parse HEAD | cut -c1-7)
metrics=$(grep "val_bpb:\|num_steps:\|peak_vram_mb:" run.log | paste -sd' ')
echo -e "${commit}\t${metrics}" >> results_temp.tsv
git reset --hard HEAD~1

# Run 4: EMBEDDING_LR=1.4
echo "=== Run 4: EMBEDDING_LR=1.4 ==="
sed -i 's/EMBEDDING_LR = 1\.2/EMBEDDING_LR = 1.4/' train.py
git add train.py && git commit -m "Run 4: EMBEDDING_LR=1.4" --no-verify || true
timeout 630 uv run train.py > run.log 2>&1
commit=$(git rev-parse HEAD | cut -c1-7)
metrics=$(grep "val_bpb:\|num_steps:\|peak_vram_mb:" run.log | paste -sd' ')
echo -e "${commit}\t${metrics}" >> results_temp.tsv
git reset --hard HEAD~1

echo "Batch sweep complete. Results in results_temp.tsv"
