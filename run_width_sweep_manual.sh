#!/bin/bash
set -e

echo "=== Run 11: ASPECT_RATIO=72 (216-dim, wider) ==="
git reset --hard HEAD
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

echo "=== Run 13: ASPECT_RATIO=64 (192-dim, baseline) ==="
sed -i 's/ASPECT_RATIO = 80/ASPECT_RATIO = 64/' train.py
git add train.py && git commit -m "Run 13: ASPECT_RATIO=64 (192-dim, baseline)" --no-verify || true
timeout 750 uv run train.py > run.log 2>&1
commit=$(git rev-parse HEAD | cut -c1-7)
val_bpb=$(strings run.log | grep "val_bpb:" | tail -1 | awk '{print $NF}')
echo -e "${commit}\t${val_bpb}\tASPECT_RATIO=64" >> results_width.tsv
git reset --hard HEAD~1

echo "Width sweep complete."
