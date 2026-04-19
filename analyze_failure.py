#!/usr/bin/env python3
"""
Analyze run failures and identify root causes
"""
import re
import sys
from pathlib import Path

def analyze_log(logfile_path):
    """Analyze log file for failure modes"""

    log_text = Path(logfile_path).read_text()

    analysis = {
        'completed': False,
        'timeout': False,
        'oom': False,
        'cuda_error': False,
        'eval_hung': False,
        'training_completed': False,
        'eval_phase_reached': False,
        'final_steps': None,
        'errors': []
    }

    # Check for successful completion
    if 'val_bpb:' in log_text:
        analysis['completed'] = True
        val_match = re.search(r'^val_bpb:\s+([\d.]+)', log_text, re.MULTILINE)
        if val_match:
            analysis['val_bpb'] = float(val_match.group(1))
        return analysis

    # Check for training completion
    if '[DEBUG] Training complete' in log_text:
        analysis['training_completed'] = True

    # Check if eval phase was reached
    if 'entering evaluation phase' in log_text:
        analysis['eval_phase_reached'] = True

    # Check for evaluation hang
    if '[DEBUG] CUDA synchronized, calling evaluate_bpb...' in log_text and 'val_bpb:' not in log_text:
        analysis['eval_hung'] = True

    # Check for OOM
    if 'out of memory' in log_text.lower() or 'cuda:' in log_text.lower() and 'error' in log_text.lower():
        analysis['oom'] = True
        analysis['errors'].append('Out of memory during evaluation')

    # Check for CUDA errors
    if 'cuda error' in log_text.lower() or 'cudart' in log_text.lower():
        analysis['cuda_error'] = True
        analysis['errors'].append('CUDA runtime error')

    # Extract final step count
    steps = re.findall(r'step (\d+)', log_text)
    if steps:
        analysis['final_steps'] = int(steps[-1])

    # Look for exceptions/tracebacks
    if 'Traceback' in log_text or 'Error:' in log_text or 'FAILED' in log_text:
        # Extract error message
        error_match = re.search(r'(Error|Exception|Traceback).*?(?=\n\n|\Z)', log_text, re.DOTALL)
        if error_match:
            analysis['errors'].append(error_match.group(0)[:200])

    return analysis

def get_recommendations(analysis):
    """Provide recommendations based on failure analysis"""
    recommendations = []

    if analysis['eval_hung']:
        recommendations.append("EVAL HANG: evaluate_bpb is hanging despite EVAL_TOKENS monkeypatch")
        recommendations.append("  -> Reduce EVAL_TOKENS further (currently 500K, try 100K)")
        recommendations.append("  -> Or increase eval batch_size further (try 256)")
        recommendations.append("  -> Check for deadlock in evaluate_bpb loop")

    if analysis['oom']:
        recommendations.append("OUT OF MEMORY: Evaluation batch size too large")
        recommendations.append("  -> Reduce eval_batch_size from 128 to 64")
        recommendations.append("  -> Further reduce EVAL_TOKENS")

    if analysis['cuda_error']:
        recommendations.append("CUDA ERROR: GPU runtime problem")
        recommendations.append("  -> Reduce model size or batch size")
        recommendations.append("  -> Check GPU memory fragmentation")
        recommendations.append("  -> May need to restart CUDA runtime")

    if analysis['training_completed'] and analysis['eval_phase_reached'] and not analysis['completed']:
        recommendations.append("EVALUATION TIMEOUT: Training completed but evaluation exceeded time budget")
        recommendations.append("  -> Current: EVAL_TOKENS=500K, eval_batch_size=128")
        recommendations.append("  -> Next attempt: Reduce EVAL_TOKENS to 100K (5 steps per epoch)")
        recommendations.append("  -> Or use sampling-based evaluation to estimate val_bpb")

    if not recommendations:
        if analysis['final_steps'] and analysis['final_steps'] < 50:
            recommendations.append(f"EARLY TERMINATION: Only {analysis['final_steps']} training steps completed")
            recommendations.append("  -> Check for frequent CUDA OOM during training")
            recommendations.append("  -> May need to reduce DEVICE_BATCH_SIZE during training")
        else:
            recommendations.append("UNKNOWN FAILURE: Could not determine root cause from logs")
            recommendations.append("  -> Review full log output manually")

    return recommendations

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_failure.py <logfile>")
        sys.exit(1)

    logfile = sys.argv[1]
    analysis = analyze_log(logfile)

    print("=" * 70)
    print("FAILURE ANALYSIS REPORT")
    print("=" * 70)

    if analysis['completed']:
        print("✓ RUN COMPLETED SUCCESSFULLY")
        print(f"  val_bpb: {analysis.get('val_bpb', 'N/A')}")
        return 0

    print("\nFAILURE MODE DETECTION:")
    print(f"  Training completed: {analysis['training_completed']}")
    print(f"  Evaluation phase reached: {analysis['eval_phase_reached']}")
    print(f"  Evaluation hung: {analysis['eval_hung']}")
    print(f"  OOM error: {analysis['oom']}")
    print(f"  CUDA error: {analysis['cuda_error']}")
    print(f"  Final training steps: {analysis['final_steps']}")

    if analysis['errors']:
        print("\nDETECTED ERRORS:")
        for error in analysis['errors']:
            print(f"  - {error}")

    print("\nRECOMMENDATIONS:")
    recommendations = get_recommendations(analysis)
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
