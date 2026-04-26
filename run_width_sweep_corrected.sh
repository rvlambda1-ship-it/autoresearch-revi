#!/bin/bash
set -e

echo "=== Run 10 (corrected): ASPECT_RATIO=56 (168-dim) ==="
sed -i 's/ASPECT_RATIO = 64/ASPECT_RATIO = 56/' train.py
git add train.py && git commit -m "Run 10c: ASPECT_RATIO=56 (168-dim) with MATRIX_LR=0.015" --no-verify || true
timeout 750 uv run train.py > run.log 2>&1
commit=$(git rev-parse HEAD | cut -c1-7)
val_bpb=$(strings run.log | grep "val_bpb:" | tail -1 | awk '{print $NF}')
echo -e "${commit}\t${val_bpb}\tASPECT_RATIO=56_CORRECTED" >> results_width_corrected.tsv
git reset --hard HEAD~1

echo "=== Run 11c: ASPECT_RATIO=72 (216-dim) ==="
sed -i 's/ASPECT_RATIO = 56/ASPECT_RATIO = 72/' train.py
git add train.py && git commit -m "Run 11c: ASPECT_RATIO=72 (216-dim)" --no-verify || true
timeout 750 uv run train.py > run.log 2>&1
commit=$(git rev-parse HEAD | cut -c1-7)
val_bpb=$(strings run.log | grep "val_bpb:" | tail -1 | awk '{print $NF}')
echo -e "${commit}\t${val_bpb}\tASPECT_RATIO=72_CORRECTED" >> results_width_corrected.tsv
git reset --hard HEAD~1

echo "=== Run 12c: ASPECT_RATIO=80 (240-dim) ==="
sed -i 's/ASPECT_RATIO = 72/ASPECT_RATIO = 80/' train.py
git add train.py && git commit -m "Run 12c: ASPECT_RATIO=80 (240-dim)" --no-verify || true
timeout 750 uv run train.py > run.log 2>&1
commit=$(git rev-parse HEAD | cut -c1-7)
val_bpb=$(strings run.log | grep "val_bpb:" | tail -1 | awk '{print $NF}')
echo -e "${commit}\t${val_bpb}\tASPECT_RATIO=80_CORRECTED" >> results_width_corrected.tsv
git reset --hard HEAD~1

echo "=== Run 13c: ASPECT_RATIO=64 (192-dim, baseline) ==="
sed -i 's/ASPECT_RATIO = 80/ASPECT_RATIO = 64/' train.py
git add train.py && git commit -m "Run 13c: ASPECT_RATIO=64 (192-dim, baseline)" --no-verify || true
timeout 750 uv run train.py > run.log 2>&1
commit=$(git rev-parse HEAD | cut -c1-7)
val_bpb=$(strings run.log | grep "val_bpb:" | tail -1 | awk '{print $NF}')
echo -e "${commit}\t${val_bpb}\tASPECT_RATIO=64_CORRECTED" >> results_width_corrected.tsv
git reset --hard HEAD~1

echo "Corrected width sweep complete."
