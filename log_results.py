#!/usr/bin/env python3
"""
Log experiment results to results.tsv and commit to git.
Usage: python log_results.py <run_number> <logfile> <description> [--discard]
"""
import sys
import subprocess
from pathlib import Path
import re

def extract_value(log_text, pattern, flags=0):
    """Extract value from log using regex pattern"""
    match = re.search(pattern, log_text, flags)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None
    return None

def main():
    if len(sys.argv) < 4:
        print("Usage: python log_results.py <run_number> <logfile> <description> [--discard]")
        sys.exit(1)

    run_num = sys.argv[1]
    logfile = sys.argv[2]
    description = sys.argv[3]
    discard_flag = "--discard" in sys.argv

    # Read log file
    log_path = Path(logfile)
    if not log_path.exists():
        print(f"Error: Log file {logfile} not found")
        sys.exit(1)

    log_text = log_path.read_text()

    # Extract metrics
    val_bpb = extract_value(log_text, r'^val_bpb:\s+([\d.]+)', re.MULTILINE)
    peak_vram = extract_value(log_text, r'^peak_vram_mb:\s+([\d.]+)', re.MULTILINE)

    # Get current git commit
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            text=True
        ).strip()
    except:
        commit = "unknown"

    # Determine status
    if val_bpb is None:
        status = "crash"
        val_bpb_str = "N/A"
        vram_gb = "0.0"
        print(f"[WARN] Run {run_num} crashed or timed out - no val_bpb result")
    else:
        status = "discard" if discard_flag else "keep"
        val_bpb_str = f"{val_bpb:.6f}"
        vram_gb = f"{peak_vram / 1024:.1f}" if peak_vram else "0.0"
        print(f"[OK] Run {run_num}: val_bpb={val_bpb_str}, memory={vram_gb}GB, status={status}")

    # Append to results.tsv
    results_file = Path(__file__).parent / "results.tsv"
    with open(results_file, "a") as f:
        f.write(f"{commit}\t{val_bpb_str}\t{vram_gb}\t{status}\t{description}\n")

    print(f"[LOG] Logged to results.tsv")

    # Commit to git
    try:
        subprocess.run(
            ["git", "add", "results.tsv"],
            cwd=Path(__file__).parent,
            check=True,
            capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", f"Results: Run {run_num} - val_bpb {val_bpb_str} ({status})"],
            cwd=Path(__file__).parent,
            check=True,
            capture_output=True
        )
        print(f"[OK] Committed results.tsv to git")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Git commit failed: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
