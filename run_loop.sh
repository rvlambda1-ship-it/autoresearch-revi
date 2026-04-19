#!/bin/bash
# Autonomous research loop - runs experiments and logs results
# Usage: ./run_loop.sh <start_run> <num_runs>

START_RUN=${1:-319}
NUM_RUNS=${2:-5}

echo "======================================"
echo "AUTONOMOUS RESEARCH LOOP"
echo "Start: Run $START_RUN"
echo "Duration: $NUM_RUNS experiments"
echo "======================================"

CURRENT_RUN=$START_RUN
for i in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "[RUN $CURRENT_RUN] Starting experiment $i/$NUM_RUNS"

    # Run the experiment
    LOGFILE="run_${CURRENT_RUN}.log"
    if timeout 600 uv run train.py > "$LOGFILE" 2>&1; then
        echo "[RUN $CURRENT_RUN] Training completed, processing results..."
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "[RUN $CURRENT_RUN] TIMEOUT - exceeded 600s budget"
        else
            echo "[RUN $CURRENT_RUN] FAILED - exit code $EXIT_CODE"
        fi
    fi

    # Parse and log results
    if grep -q "^val_bpb:" "$LOGFILE"; then
        VAL_BPB=$(grep "^val_bpb:" "$LOGFILE" | awk '{print $NF}')
        MEMORY=$(grep "^peak_vram_mb:" "$LOGFILE" | awk '{print $NF}' | awk '{printf "%.1f", $1/1024}')
        STEPS=$(grep "^num_steps:" "$LOGFILE" | awk '{print $NF}')

        # Get git commit
        COMMIT=$(git rev-parse --short HEAD)

        # Get current git status to determine experiment type
        DESCRIPTION=$(git status --porcelain | grep "M train.py" > /dev/null && echo "Configuration change" || echo "Baseline run")

        echo "[RUN $CURRENT_RUN] Result: val_bpb=$VAL_BPB, steps=$STEPS, mem=${MEMORY}GB"

        # Log to results.tsv
        echo -e "${COMMIT}\t${VAL_BPB}\t${MEMORY}\t$([ \"$VAL_BPB\" != \"N/A\" ] && echo 'keep' || echo 'crash')\t${DESCRIPTION}" >> results.tsv

        # Commit results
        git add results.tsv
        git commit -m "Results: Run $CURRENT_RUN - val_bpb ${VAL_BPB}" || true

    else
        echo "[RUN $CURRENT_RUN] No val_bpb in output - run failed or timed out"
    fi

    CURRENT_RUN=$((CURRENT_RUN + 1))

    # Wait before next run to avoid thrashing
    if [ $i -lt $NUM_RUNS ]; then
        echo "[PAUSE] Waiting before next experiment..."
        sleep 2
    fi
done

echo ""
echo "======================================"
echo "LOOP COMPLETE: Ran $NUM_RUNS experiments"
echo "Latest best: $(tail -1 results.tsv | cut -f2)"
echo "======================================"
