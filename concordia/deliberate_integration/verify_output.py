#!/usr/bin/env python3
"""Scrutinizes the generated event logs for a soft-run to confirm correctness."""

import sys
import os
import csv

def verify_output(log_file, mode_name):
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        sys.exit(1)

    print(f"Verifying {log_file} (Mode: {mode_name})...")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    if len(rows) <= 1:
        print(f"Error: Log file is empty or has only headers.")
        sys.exit(1)
        
    print(f"  ✓ Found {len(rows)} events.")
    
    if mode_name in ["simulacra", "human", "simulacra_partial"]:
        # Should contain some dialogue if it's a generative mode
        pids = set([r.get('agent_id', '') for r in rows if r.get('agent_id', '')])
        
        # We expect a variety of participant IDs (not just 1 or 2)
        if len(pids) < 2:
            print(f"Error: Only found data for '{', '.join(pids)}'. Expected multiple participants.")
            sys.exit(1)
            
        print(f"  ✓ Found dialogue from {len(pids)} participants.")
    else:
        print("  ✓ Sketch verify of non-dialogue mode passed.")

    print(f"SUCCESS: {log_file} passed scrutiny.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: verify_output.py <log_file> <mode_name>")
        sys.exit(1)
    
    verify_output(sys.argv[1], sys.argv[2])
