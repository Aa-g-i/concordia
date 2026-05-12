import os
import glob
import pandas as pd
import argparse

def parse_logs(base_dir, output_md):
    md_content = ["# Full Sweep Behavioral Results", ""]
    
    # 1. Gather all events logs
    all_events = []
    log_files = glob.glob(os.path.join(base_dir, "concordia_output_*", "*.csv"))
    
    for log_path in log_files:
        if 'events_log' not in log_path and 'agent_states' not in log_path:
            continue
            
        parent_dir = os.path.basename(os.path.dirname(log_path))
        
        if 'mask' in parent_dir.lower() or 'simulacra' in parent_dir.lower():
            mode = 'Mask'
        elif 'mirror' in parent_dir.lower() or 'human' in parent_dir.lower():
            mode = 'Mirror'
        elif 'deterministic' in parent_dir.lower():
            mode = 'Human Baseline'
        else:
            mode = 'Unknown'
            
        # Deduce Environment
        if 'a5' in parent_dir.lower():
            env = 'A5 Charity Debate'
        else:
            env = 'Lost at Sea / Other'
            
        try:
            df = pd.read_csv(log_path)
            df['Environment'] = env
            df['Mode'] = mode
            if 'events_log' in log_path:
                df['LogType'] = 'Events'
            else:
                df['LogType'] = 'States'
            all_events.append(df)
        except Exception as e:
            pass

    if not all_events:
         with open(output_md, 'w') as f:
             f.write("# No output logs found.")
         return
         
    df_all = pd.concat(all_events, ignore_index=True)
    df_events = df_all[df_all['LogType'] == 'Events'].copy()
    
    # Analyze Dialogues (Verbosity)
    if not df_events.empty and 'content' in df_events.columns:
        df_dialogue = df_events[~df_events['agent_id'].str.contains('System', na=False, case=False)].copy()
        df_dialogue['word_count'] = df_dialogue['content'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        
        md_content.append("## Verbosity (Word Count) by Mode AND Environment")
        stats_joint = df_dialogue.groupby(['Environment', 'Mode'])['word_count'].agg(['count', 'mean']).round(2)
        md_content.append(stats_joint.to_markdown())
        md_content.append("\n")
        
        md_content.append("## Verbosity by Agent (Mode)")
        stats_mode = df_dialogue.groupby(['Mode'])['word_count'].agg(['count', 'mean']).round(2)
        md_content.append(stats_mode.to_markdown())
        md_content.append("\n")

        md_content.append("## Verbosity by Environment")
        stats_env = df_dialogue.groupby(['Environment'])['word_count'].agg(['count', 'mean']).round(2)
        md_content.append(stats_env.to_markdown())
        md_content.append("\n")

    # Analyze States
    df_states = df_all[df_all['LogType'] == 'States'].copy()
    if not df_states.empty and 'metric_name' in df_states.columns and 'value_final' in df_states.columns:
        # Extract numerical metrics
        numeric_mask = pd.to_numeric(df_states['value_final'], errors='coerce').notnull()
        numeric_states = df_states[numeric_mask].copy()
        if not numeric_states.empty:
            numeric_states['value'] = pd.to_numeric(numeric_states['value_final'])
            
            md_content.append("## Internal State Convergence (Numeric Metrics)")
            stats_states = numeric_states.groupby(['Environment', 'Mode', 'metric_name'])['value'].agg(['mean', 'std']).round(2)
            md_content.append(stats_states.to_markdown())
            md_content.append("\n")
    
    # Create the artifact dir if it doesnt exist
    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    
    with open(output_md, 'w') as f:
        f.write("\n".join(md_content))
        
    print(f"Aggregated .md report written to {output_md}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collate DeliberateLab sweep results")
    parser.add_argument("--base-dir", default='/usr/local/google/home/aarontp/INSTALLS/concordia/concordia/deliberate_integration', help="Base directory containing concordia_output_* folders")
    parser.add_argument("--output-md", default='sweep_results.md', help="Output markdown file path")
    args = parser.parse_args()
    parse_logs(args.base_dir, args.output_md)
