import os
import glob
import pandas as pd
import numpy as np
import re
import argparse

def compute_gini(x):
    x = np.array(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if np.amin(x) < 0:
        x -= np.amin(x)
    x += 0.0000001
    x = np.sort(x)
    index = np.arange(1, x.shape[0]+1)
    n = x.shape[0]
    return ((np.sum((2 * index - n  - 1) * x)) / (n * np.sum(x)))

def parse_dir_name(basename):
    # Example: concordia_output_f1_200_data_simulacra_partial_0.5_minimal
    
    # Split by _data_ or _data (2)_
    parts = re.split(r'_data_|_data \(2\)_', basename)
    if len(parts) != 2:
        return None
        
    experiment = parts[0].replace('concordia_output_', '')
    treatment_str = parts[1]
    
    # Deduce Mode
    if 'mask' in treatment_str.lower() or 'simulacra' in treatment_str.lower():
        mode = 'Mask'
    elif 'mirror' in treatment_str.lower() or 'human' in treatment_str.lower():
        mode = 'Mirror'
    elif 'deterministic' in treatment_str.lower():
        mode = 'Human Baseline'
    else:
        mode = 'Unknown'
        
    # Deduce Control
    if 'partial' in treatment_str.lower():
        control = 'Partial'
    elif 'nodialogue' in treatment_str.lower():
        control = 'NoDialogue'
    else:
        control = 'Full Simulation'
        
    # Deduce Agent Type
    if 'native' in treatment_str.lower():
        agent = 'Native'
    elif 'minimal' in treatment_str.lower():
        agent = 'Minimal'
    else:
        agent = 'Unknown'
        
    return {
        'Experiment': experiment,
        'Mode': mode,
        'Control': control,
        'Agent': agent
    }

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Data Analyzer for Concordia")
    parser.add_argument("--base-dir", default='.', help="Base directory containing concordia_output_* folders")
    args = parser.parse_args()
    
    results = []
    
    search_pattern = os.path.join(args.base_dir, 'concordia_output_*')
    for dir_path in glob.glob(search_pattern):
        if not os.path.isdir(dir_path):
            continue
            
        basename = os.path.basename(dir_path)
        parsed = parse_dir_name(basename)
        if not parsed:
            continue
            
        # Read Events for Dialogue Metrics
        event_files = glob.glob(os.path.join(dir_path, 'events_log*.csv'))
        avg_words = 0
        gini = 0
        
        if event_files:
            try:
                df_events = pd.read_csv(event_files[0])
                if not df_events.empty and 'content' in df_events.columns:
                    df_events = df_events.dropna(subset=['content'])
                    df_events = df_events[~df_events['agent_id'].str.contains('System', case=False, na=False)]
                    
                    df_events['word_count'] = df_events['content'].astype(str).str.split().str.len()
                    
                    if not df_events.empty:
                        verb = df_events.groupby('agent_id' if 'agent_id' in df_events.columns else 'participant_id')['word_count'].sum()
                        if not verb.empty:
                            avg_words = verb.mean()
                            gini = compute_gini(verb)
            except Exception as e:
                print(f"Error processing events in {basename}: {e}")
                
        # Read Surveys
        survey_files = glob.glob(os.path.join(dir_path, '*survey*.csv'))
        survey_mean = 0
        
        if survey_files:
            try:
                df_survey = pd.read_csv(survey_files[0])
                if not df_survey.empty and 'response' in df_survey.columns:
                    survey_mean = df_survey['response'].apply(pd.to_numeric, errors='coerce').mean()
            except Exception as e:
                print(f"Error processing surveys in {basename}: {e}")
                
        parsed.update({
            'Avg_Words': avg_words,
            'Gini': gini,
            'Survey_Mean': survey_mean,
            'Folder': basename
        })
        results.append(parsed)
        
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        print("No valid data found.")
        return
        
    print("\n=== COMPREHENSIVE ANALYSIS RESULTS ===")
    
    # Aggregate by Mode and Agent
    print("\nMetrics by Mode and Agent Type (Full Simulation Only):")
    df_full = df_results[df_results['Control'] == 'Full Simulation']
    if not df_full.empty:
        summary = df_full.groupby(['Mode', 'Agent'])[['Avg_Words', 'Gini', 'Survey_Mean']].mean().round(3)
        print(summary.to_string())
        
    # Aggregate by Control Mode
    print("\nMetrics by Control Mode:")
    summary_control = df_results.groupby(['Control'])[['Avg_Words', 'Gini', 'Survey_Mean']].mean().round(3)
    print(summary_control.to_string())
    
    # Aggregate by Experiment
    print("\nMetrics by Experiment:")
    summary_exp = df_results.groupby(['Experiment'])[['Avg_Words', 'Gini', 'Survey_Mean']].mean().round(3)
    print(summary_exp.to_string())
    
    # Save to CSV
    output_csv = 'comprehensive_analysis_metrics.csv'
    df_results.to_csv(output_csv, index=False)
    print(f"\nFull raw results saved to {output_csv}")

if __name__ == '__main__':
    main()
