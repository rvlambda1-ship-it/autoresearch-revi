#!/bin/bash
# Phase 2 LR Sweep: Runs 6-9
# Test EMBEDDING_LR and MATRIX_LR sensitivity at BATCH=2K

set -e

cd "d:/Revi/Research Projects/autoresearch-revi"

# Function to run and log a single experiment
run_exp() {
    local run_num=$1
    local config_change=$2
    local sed_pattern=$3
    local description=$4

    echo "=== Run ${run_num}: ${description} ==="
    git reset --hard HEAD
    sed -i "${sed_pattern}" train.py
    git add train.py && git commit -m "Run ${run_num}: ${description}" --no-verify || true
    timeout 630 uv run train.py > run.log 2>&1
    commit=$(git rev-parse HEAD | cut -c1-7)
    val_bpb=$(cat run.log | sed 's/\r/\n/g' | grep "val_bpb:" | tail -1 | awk '{print $NF}')
    echo -e "${commit}\t${val_bpb}\t${description}" >> results_sweep.tsv
    git reset --hard HEAD~1
}

# Run 6: EMBEDDING_LR=1.4 (higher)
run_exp 6 "" "s/EMBEDDING_LR = 1\.0/EMBEDDING_LR = 1.4/" "EMBEDDING_LR=1.4"

# Run 7: EMBEDDING_LR=1.6 (near Phase 1 limit)
run_exp 7 "" "s/EMBEDDING_LR = 1\.4/EMBEDDING_LR = 1.6/" "EMBEDDING_LR=1.6"

# Run 8: MATRIX_LR=0.020 (higher)
git reset --hard HEAD
sed -i 's/EMBEDDING_LR = 1\.6/EMBEDDING_LR = 1.2/' train.py  # revert to baseline
sed -i 's/MATRIX_LR = 0\.015/MATRIX_LR = 0.020/' train.py
git add train.py && git commit -m "Run 8: MATRIX_LR=0.020" --no-verify || true
timeout 630 uv run train.py > run.log 2>&1
commit=$(git rev-parse HEAD | cut -c1-7)
val_bpb=$(cat run.log | sed 's/\r/\n/g' | grep "val_bpb:" | tail -1 | awk '{print $NF}')
echo -e "${commit}\t${val_bpb}\tMATRIX_LR=0.020" >> results_sweep.tsv
git reset --hard HEAD~1

# Run 9: MATRIX_LR=0.012 (lower)
run_exp 9 "" "s/MATRIX_LR = 0\.020/MATRIX_LR = 0.012/" "MATRIX_LR=0.012"

echo "LR sweep complete. Results in results_sweep.tsv"
