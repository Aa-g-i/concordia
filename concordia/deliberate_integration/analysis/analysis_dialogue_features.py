import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def calculate_gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = np.array(array, dtype=np.float64)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def compute_structural_features(det_csv: str, mirror_csv: str, mask_csv: str, output_path: str):
    print("Loading CSVs for Structural Feature Extraction...")
    
    dfs = []
    for csv_file, label in [(det_csv, 'Human'), 
                          (mirror_csv, 'Playback-Human'), 
                          (mask_csv, 'Minimal-Simulacra')]:
        if csv_file and os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['Mode'] = label
            dfs.append(df)
            
    if not dfs:
        print("No valid CSV logs found. Aborting structural features.")
        return
        
    combined_df = pd.concat(dfs, ignore_index=True)
    # Filter out System messages
    combined_df = combined_df[~combined_df['agent_id'].str.contains('System|GM|Orchestrator', na=False, case=False)]
    combined_df['word_count'] = combined_df['content'].apply(lambda x: len(str(x).split()))
    
    sns.set_theme(style="whitegrid")
    
    # -------------------------------------------------------------
    # 1. Verbosity Distribution (Participant Level)
    # -------------------------------------------------------------
    print("Computing Participant Verbosity Distributions...")
    # Calculate average words per message for each participant within each Mode
    part_verbosity = combined_df.groupby(['Mode', 'cohort_id', 'agent_id'])['word_count'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=part_verbosity, x='word_count', hue='Mode', fill=True, common_norm=False, palette='Set2')
    plt.title("Distribution of Participant Verbosity (Avg Words per Message)")
    plt.xlabel("Average Word Count")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(output_path.replace(".png", "_participant_verbosity.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. Conversational Gini Coefficient (Equality of Talk Time)
    # -------------------------------------------------------------
    print("Computing Conversational Gini Coefficients (Inequality)...")
    gini_records = []
    # Group by Mode, Cohort, and Stage to calculate inequality across participants
    for name, group in combined_df.groupby(['Mode', 'cohort_id', 'stage_id']):
        participants_word_sums = group.groupby('agent_id')['word_count'].sum().values
        if len(participants_word_sums) > 1:
            g = calculate_gini(participants_word_sums)
            gini_records.append({'Mode': name[0], 'cohort_id': name[1], 'stage_id': name[2], 'Gini': g})
            
    df_gini = pd.DataFrame(gini_records)
    if not df_gini.empty:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df_gini, x='Mode', y='Gini', hue='Mode', palette='Set2', inner='quartile', legend=False)
        plt.title("Conversational Inequality (Gini Coefficient) Across Stages")
        plt.ylabel("Gini Coefficient (0=Egalitarian, 1=Dominated)")
        plt.xlabel("Execution Mode")
        plt.tight_layout()
        plt.savefig(output_path.replace(".png", "_gini_coefficient.png"), dpi=300)
        plt.close()

    # -------------------------------------------------------------
    # 3. Inter-arrival Timing Distributions (Turn Delays)
    # -------------------------------------------------------------
    print("Computing Inter-arrival Response Timings...")
    combined_df['time_index'] = pd.to_numeric(combined_df['time_index'], errors='coerce')
    combined_df = combined_df.dropna(subset=['time_index'])
    combined_df = combined_df.sort_values(by=['Mode', 'cohort_id', 'stage_id', 'time_index'])
    # Calculate time delay between consecutive messages in the same stage
    combined_df['time_delay'] = combined_df.groupby(['Mode', 'cohort_id', 'stage_id'])['time_index'].diff()
    
    # Filter reasonable bounds to remove extreme outliers (e.g. stage transitions)
    df_timings = combined_df[(combined_df['time_delay'] > 0) & (combined_df['time_delay'] < 300)]
    
    if not df_timings.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df_timings, x='time_delay', hue='Mode', bins=50, kde=True, element="step", stat="density", common_norm=False, palette='Set2')
        plt.title("Inter-Arrive Timing Distribution (Seconds Between Messages)")
        plt.xlabel("Seconds")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(output_path.replace(".png", "_inter_arrival_timing.png"), dpi=300)
        plt.close()

    print(f"Finished Structural Features! Exported to: {output_path.replace('.png', '_*.png')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-csv", required=False)
    parser.add_argument("--mirror-csv", required=False)
    parser.add_argument("--mask-csv", required=False)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    compute_structural_features(args.det_csv, args.mirror_csv, args.mask_csv, args.out)
