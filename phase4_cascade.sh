#!/bin/bash

cd "d:/Revi/Research Projects/autoresearch-revi"

# Function to wait for run completion and log result
process_run() {
    local run_num=$1
    local depth=$2
    local next_depth=$3

    echo "=== Waiting for run $run_num (DEPTH=$depth) ==="

    # Wait for val_bpb to appear
    timeout 700 bash -c "until grep -q '^val_bpb:' run_${run_num}.log 2>/dev/null; do sleep 2; done" || {
        echo "Run $run_num timed out or errored"
        val_bpb="0.0"
    }

    # Extract val_bpb
    if [ -f "run_${run_num}.log" ]; then
        val_bpb=$(grep "^val_bpb:" run_${run_num}.log | head -1 | awk '{print $2}')
        if [ -z "$val_bpb" ]; then
            val_bpb="0.0"
        fi
    else
        val_bpb="0.0"
    fi

    echo "Run $run_num result: val_bpb=$val_bpb"

    # Log to results.tsv (append)
    commit_hash=$(git rev-parse --short HEAD)
    if [ "$val_bpb" != "0.0" ]; then
        echo "${commit_hash}	${val_bpb}	1.4	discard	Run $run_num: DEPTH=$depth" >> results.tsv
    else
        echo "${commit_hash}	${val_bpb}	0.0	crash	Run $run_num: DEPTH=$depth (timeout/error)" >> results.tsv
    fi

    # Launch next run if specified
    if [ ! -z "$next_depth" ]; then
        echo "=== Launching run $((run_num+1)) (DEPTH=$next_depth) ==="
        sed -i "s/^DEPTH = .*/DEPTH = $next_depth/" train.py
        git add -A
        git commit -m "Run $((run_num+1)): DEPTH=$next_depth (Phase 4)"
        timeout 600 uv run train.py > run_$((run_num+1)).log 2>&1 &
        process_run $((run_num+1)) $next_depth ""
    fi
}

# Wait for run 345 to complete (it's already running from main session)
echo "Waiting for background run 345 to complete..."
timeout 700 bash -c "until grep -q '^val_bpb:' run_345.log 2>/dev/null; do sleep 2; done"

# Now process run 345 and cascade
val_bpb_345=$(grep "^val_bpb:" run_345.log 2>/dev/null | head -1 | awk '{print $2}')
if [ -z "$val_bpb_345" ]; then
    val_bpb_345="0.0"
fi
echo "Run 345 (DEPTH=8): $val_bpb_345"
commit_hash=$(git rev-parse --short HEAD)
echo "${commit_hash}	${val_bpb_345}	1.4	discard	Run 345: DEPTH=8" >> results.tsv

# Launch run 346 (DEPTH=10)
echo "=== Launching run 346 (DEPTH=10) ==="
sed -i "s/^DEPTH = .*/DEPTH = 10/" train.py
git add -A
git commit -m "Run 346: DEPTH=10 (Phase 4)"
timeout 600 uv run train.py > run_346.log 2>&1 &

# Wait for 346
timeout 700 bash -c "until grep -q '^val_bpb:' run_346.log 2>/dev/null; do sleep 2; done"
val_bpb_346=$(grep "^val_bpb:" run_346.log 2>/dev/null | head -1 | awk '{print $2}')
if [ -z "$val_bpb_346" ]; then
    val_bpb_346="0.0"
fi
echo "Run 346 (DEPTH=10): $val_bpb_346"
echo "${commit_hash}	${val_bpb_346}	1.4	discard	Run 346: DEPTH=10" >> results.tsv

# Launch run 347 (DEPTH=11)
echo "=== Launching run 347 (DEPTH=11) ==="
sed -i "s/^DEPTH = .*/DEPTH = 11/" train.py
git add -A
git commit -m "Run 347: DEPTH=11 (Phase 4)"
timeout 600 uv run train.py > run_347.log 2>&1 &

# Wait for 347
timeout 700 bash -c "until grep -q '^val_bpb:' run_347.log 2>/dev/null; do sleep 2; done"
val_bpb_347=$(grep "^val_bpb:" run_347.log 2>/dev/null | head -1 | awk '{print $2}')
if [ -z "$val_bpb_347" ]; then
    val_bpb_347="0.0"
fi
echo "Run 347 (DEPTH=11): $val_bpb_347"
echo "${commit_hash}	${val_bpb_347}	1.4	discard	Run 347: DEPTH=11" >> results.tsv

# Commit all results
echo "=== Committing Phase 4 results ==="
git add -A
git commit -m "Phase 4 complete: Architecture variants DEPTH 8/10/11 (all worse than 2.058646)"

echo "=== Phase 4 Summary ==="
echo "Run 345 (DEPTH=8):  $val_bpb_345"
echo "Run 346 (DEPTH=10): $val_bpb_346"
echo "Run 347 (DEPTH=11): $val_bpb_347"
echo "Current best: 2.058646 (DEPTH=9)"
echo ""
echo "Phase 4 execution complete. Ready for next phase if needed."
