import argparse
import os
import sys

# Ensure this script can import sibling modules gracefully
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stats_summary_abstract import run_abstract_stats
from analysis.analysis_side_by_side import visualize_dialogue_side_by_side

def main():
    parser = argparse.ArgumentParser(description="Unified Generic Pipeline for Human vs Concordia Simulation Data")
    parser.add_argument("--det-dir", required=True, help="Directory containing the Deterministic (Human) baseline CSVs")
    parser.add_argument("--simulacra-dir", required=True, help="Directory containing the Simulacra generative CSVs")
    parser.add_argument("--human-dir", required=True, help="Directory containing the Human generative CSVs")
    parser.add_argument("--out-prefix", default="analysis_output", help="Prefix for visual PNG exports (default: analysis_output)")
    
    args = parser.parse_args()
    
    print("\n=======================================================")
    print("1. RUNNING TEXTUAL PIPELINE (SCALAR & VERBOSITY STATS)")
    print("=======================================================\n")
    
    try:
        run_abstract_stats(args.det_dir, args.simulacra_dir, args.human_dir)
    except Exception as e:
        print(f"Error executing abstract stats: {e}")
        
    print("\n=======================================================")
    print("2. RUNNING GRAPHICAL PIPELINE (VERBOSITY & FREQUENCY)")
    print("=======================================================\n")
    
    det_csv = os.path.join(args.det_dir, "events_log.csv")
    simulacra_csv = os.path.join(args.simulacra_dir, "events_log_simulacra.csv")
    human_csv = os.path.join(args.human_dir, "events_log_human.csv")
    
    try:
         visualize_dialogue_side_by_side(det_csv, human_csv, simulacra_csv, args.out_prefix)
         print(f"Successfully generated graphic diff pipelines at {args.out_prefix}_verbosity.png and {args.out_prefix}_frequency.png")
    except Exception as e:
         print(f"Error executing visual analyzer: {e}")
         
    print("\n=======================================================")
    print("Unified Pipeline Complete.")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
