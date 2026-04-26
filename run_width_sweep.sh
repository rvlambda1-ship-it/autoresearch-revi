#!/bin/bash
# Phase 2 Width Sweep: Runs 10-14
# Test ASPECT_RATIO impact on model performance at fixed DEPTH=3
# Best found so far: EMBEDDING_LR=1.0, BATCH=2K, MATRIX_LR=0.015

set -e

cd "d:/Revi/Research Projects/autoresearch-revi"

# Current baseline: ASPECT_RATIO=64 → 192-dim
# Test: {56, 64, 72, 80} → {168, 192, 216, 240}-dim

echo "=== Run 10: ASPECT_RATIO=56 (168-dim, narrower) ==="
git reset --hard HEAD
sed -i 's/ASPECT_RATIO = 64/ASPECT_RATIO = 56/' train.py
git add train.py && git commit -m "Run 10: ASPECT_RATIO=56 (168-dim)" --no-verify || true
timeout 750 uv run train.py > run.log 2>&1
commit=$(git rev-parse HEAD | cut -c1-7)
val_bpb=$(strings run.log | grep "val_bpb:" | tail -1 | awk '{print $NF}')
echo -e "${commit}\t${val_bpb}\tASPECT_RATIO=56" >> results_width.tsv
git reset --hard HEAD~1

echo "=== Run 11: ASPECT_RATIO=72 (216-dim, wider) ==="
sed -i 's/ASPECT_RATIO = 56/ASPECT_RATIO = 72/' train.py
git add train.py && git commit -m "Run 11: ASPECT_RATIO=72 (216-dim)" --no-verify || true
timeout 750 uv run train.py > run.log 2>&1
commit=$(git rev-parse HEAD | cut -c1-7)
val_bpb=$(strings run.log | grep "val_bpb:" | tail -1 | awk '{print $NF}')
echo -e "${commit}\t${val_bpb}\tASPECT_RATIO=72" >> results_width.tsv
git reset --hard HEAD~1

echo "=== Run 12: ASPECT_RATIO=80 (240-dim, widest) ==="
sed -i 's/ASPECT_RATIO = 72/ASPECT_RATIO = 80/' train.py
git add train.py && git commit -m "Run 12: ASPECT_RATIO=80 (240-dim)" --no-verify || true
timeout 750 uv run train.py > run.log 2>&1
commit=$(git rev-parse HEAD | cut -c1-7)
val_bpb=$(strings run.log | grep "val_bpb:" | tail -1 | awk '{print $NF}')
echo -e "${commit}\t${val_bpb}\tASPECT_RATIO=80" >> results_width.tsv
git reset --hard HEAD~1

echo "Width sweep (partial) complete."
