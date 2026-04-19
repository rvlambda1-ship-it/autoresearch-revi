#!/bin/bash
# Wrapper script for running experiments with automatic results logging and git commits
# Usage: ./run_experiment.sh <run_number> <description>

set -e

RUN_NUM=$1
DESCRIPTION=$2
LOGFILE="run_${RUN_NUM}.log"

if [ -z "$RUN_NUM" ] || [ -z "$DESCRIPTION" ]; then
    echo "Usage: $0 <run_number> <description>"
    echo "Example: $0 317 'Test new hyperparameter'"
    exit 1
fi

echo "Starting run $RUN_NUM: $DESCRIPTION"
echo "Log file: $LOGFILE"

# Get current commit hash before running
COMMIT_HASH=$(git rev-parse --short HEAD)

# Run training with timeout
if timeout 600 uv run train.py > "$LOGFILE" 2>&1; then
    RUN_STATUS="completed"
else
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        RUN_STATUS="timeout"
    else
        RUN_STATUS="failed"
    fi
fi

# Extract results from log
VAL_BPB=$(grep "^val_bpb:" "$LOGFILE" | awk '{print $NF}' || echo "N/A")
MEMORY_GB=$(grep "^peak_vram_mb:" "$LOGFILE" | awk '{print $NF}' | awk '{printf "%.1f", $1/1024}' || echo "0.0")

if [ "$VAL_BPB" = "N/A" ]; then
    VAL_BPB="N/A"
    MEMORY_GB="0.0"
    KEEP_STATUS="crash"
elif [ "$RUN_STATUS" = "timeout" ]; then
    KEEP_STATUS="crash"
else
    KEEP_STATUS="keep"  # User will update this later if needed
fi

echo ""
echo "========================================"
echo "Run $RUN_NUM Results"
echo "========================================"
echo "Commit: $COMMIT_HASH"
echo "Val_BPB: $VAL_BPB"
echo "Memory (GB): $MEMORY_GB"
echo "Status: $KEEP_STATUS"
echo "Description: $DESCRIPTION"
echo "========================================"

# Append results to results.tsv if not a crash
if [ "$KEEP_STATUS" != "crash" ]; then
    echo -e "${COMMIT_HASH}\t${VAL_BPB}\t${MEMORY_GB}\t${KEEP_STATUS}\t${DESCRIPTION}" >> results.tsv

    # Commit results.tsv
    git add results.tsv
    git commit -m "Results: Run $RUN_NUM - val_bpb ${VAL_BPB} (${KEEP_STATUS})" || true

    echo "✓ Results logged and committed to git"
else
    echo "⚠ Run crashed/timed out. Not logging to results.tsv."
    echo "  Manually review $LOGFILE and update results.tsv if needed."
fi

exit 0
