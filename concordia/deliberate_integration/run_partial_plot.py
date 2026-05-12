import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from mask_mirror_metrics import (
    calculate_hds_v1_identified_snapshot, 
    calculate_cms_v1_2_avg_based,
    calculate_aggregate_feature_deviation_v3, 
    calculate_kl_divergence_score_v4,
    calculate_masking_score_v5, 
    calculate_kl_masking_score_v6,
    calculate_paired_individual_deviation_v7
)

import glob

VISUALIZE_THRESHOLDS = True

def load_events(env_filter="combined"):
    import os
    results = []
    
    search_dirs = [
        '/usr/local/google/home/aarontp/INSTALLS/concordia/concordia/deliberate_integration/'
    ]
    
    # Dynamically extract all dataset prefixes by finding .zip files
    zip_files = []
    zip_files.extend(glob.glob(os.path.join(search_dirs[0], 'data/a5_experiment/*.zip')))
    zip_files.extend(glob.glob(os.path.join(search_dirs[0], 'data/new_experiments/*.zip')))
        
    prefixes = [os.path.basename(z).replace('.zip', '') for z in zip_files]
    
    for search_dir in search_dirs:
        for output_dir in glob.glob(os.path.join(search_dir, 'concordia_output_*')):
            basename = os.path.basename(output_dir)
            matched_prefix = None
            
            # Filter by env
            if env_filter == 'a5' and 'a5' not in basename.lower():
                continue
            if env_filter == 'las' and 'a5' in basename.lower():
                continue
            
            for p in sorted(prefixes, key=len, reverse=True):
                if basename.startswith(f"concordia_output_{p}"):
                    matched_prefix = p
                    break
                    
            if not matched_prefix: 
                continue
                
            csv_files = glob.glob(os.path.join(output_dir, 'events_log*.csv'))
            for path in csv_files:
                df = pd.read_csv(path)
                if df.empty: continue
                df = df[~df['agent_id'].str.contains('System', case=False, na=False)]
                df['word_count'] = df['content'].astype(str).str.split().str.len()
                df['Condition'] = df['stage_id'].apply(lambda x: 'Identified' if '1' in str(x) else 'Pseudonymous')
                
                if basename == f"concordia_output_{matched_prefix}":
                    model_name = "Human Baseline"
                else:
                    model_name = basename.replace(f"concordia_output_{matched_prefix}_", "").title()
                    
                df['Model'] = model_name
                results.append(df)
            
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

env_filter = sys.argv[1] if len(sys.argv) > 1 else 'combined'
df_raw = load_events(env_filter)

long_form_gap_list = []
gap_summary_list = []
optimal_list = []

if not df_raw.empty:
    for (condition, model), group in df_raw.groupby(['Condition', 'Model']):
        group['Delta_norm'] = group['word_count'] / (group['word_count'].max() + 1)
        long_form_gap_list.append(group[['Condition', 'Model', 'Delta_norm', 'agent_id']])
        gap_summary_list.append({'Condition': condition, 'Model': model, 'delta_total': group['Delta_norm'].mean()})
        turn_density = len(group) / max((len(df_raw)/6), 1)
        optimal_list.append({'Condition': condition, 'Model': model, 'Optimal_Metric_Fraction': turn_density})

if len(gap_summary_list) > 0:
    df_gap = pd.DataFrame(gap_summary_list)
    long_form_gaps_df = pd.concat(long_form_gap_list, ignore_index=True)
    df_wtl = df_gap.copy()
    df_wtl['WTL_Gap'] = df_gap['delta_total'] * 0.1 
    df_align = pd.DataFrame(columns=['Model', 'Condition', 'p_value'])
else:
    print(f"No valid events discovered for env_filter={env_filter}. Skipping.")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, f"No Valid Dialogue/Events Found for\nEnv: {env_filter}", ha='center', va='center', fontsize=20)
    plt.axis('off')
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f'kl_divergence_{env_filter}.png'))
    sys.exit(0)

MODELS_TO_PROCESS = sorted([name for name in df_gap['Model'].unique() if 'Human Baseline' not in name])
data_dict = {
    'gap': df_gap, 'wtl': df_wtl, 'align': df_align, 'long_form_gaps': long_form_gaps_df
}

# Swap Human Baseline name for generic 'Human' internally so equations map correctly
data_dict['gap']['Model'] = data_dict['gap']['Model'].replace('Human Baseline', 'Human')
data_dict['long_form_gaps']['Model'] = data_dict['long_form_gaps']['Model'].replace('Human Baseline', 'Human')


sns.set_style("whitegrid")
pertinent_features = {
    'gap': 'delta_total', 'wtl': 'WTL_Gap', 'align': 'p_value'
}

hds_snapshot_scores = calculate_hds_v1_identified_snapshot(data_dict, MODELS_TO_PROCESS)
cms_avg_based_scores = calculate_cms_v1_2_avg_based(data_dict, MODELS_TO_PROCESS)
aggregate_feature_scores = calculate_aggregate_feature_deviation_v3(data_dict, pertinent_features, MODELS_TO_PROCESS)
kl_divergence_scores = calculate_kl_divergence_score_v4(data_dict, MODELS_TO_PROCESS, bins=10, range=(0, 1.0))
paired_individual_scores = calculate_paired_individual_deviation_v7(data_dict, MODELS_TO_PROCESS)

scores_list = [
    hds_snapshot_scores, cms_avg_based_scores, aggregate_feature_scores,
    kl_divergence_scores, paired_individual_scores
]
titles = [
    'v1: HDS (Snapshot at "Identified")',
    'v1.2: CMS (Average-based, IP)',
    'v3: Aggregate Deviation (All Features, Normalized)',
    'v4: KL Divergence (Distributional Dissimilarity)',
    'v7: Paired Individual Deviation (MAE)'
]

formula_descriptions = [
    r'Formula: $\Delta_{gap, M} - \Delta_{gap, H}$',
    r'Formula: $D_{mag} + D_{adapt} + \overline{PValue}_{M}$',
    r'Formula: $\Sigma \frac{|Feature_{M} - Feature_{H}|}{MaxDeviation}$',
    r'Formula: $\Sigma_{cond} KL(P_{Model} || P_{Human})$',
    r'Formula: $MAE(Trace_{M,i} - Trace_{H,i})$'
]

thresholds = [None, None, None, None, None]

fig, axes = plt.subplots(len(scores_list), 1, figsize=(10, 8.0 * len(scores_list)))
fig.suptitle('Model Categorization (Mask vs Mirror) including Simulacra Partial', fontsize=18, y=0.99)

for i, ax in enumerate(axes.flatten()):
    scores = scores_list[i]
    if not scores: continue
    
    threshold = thresholds[i]

    full_title = f"{titles[i]}\n{formula_descriptions[i]}"
    ax.set_title(full_title, loc='left', fontsize=14)

    scores_series = pd.Series(scores).sort_values(ascending=True)
    scores_series.replace([np.inf, -np.inf], scores_series[np.isfinite(scores_series)].max() * 1.5, inplace=True)
    
    print(f"\n--- {titles[i]} ---")
    for name, value in zip(scores_series.index, scores_series.values):
        print(f"{name}: {value:.4f}")
    
    if threshold is not None:
        colors = ['#d9534f' if score > threshold else '#428bca' for score in scores_series]
    else:
        colors = ['#428bca' for _ in scores_series]
        
    ax.barh(scores_series.index, scores_series.values, color=colors)
    is_masking_plot = "Masking Score" in titles[i]
    ax.set_xlabel('Score (>0 is Masking, <0 is Mirroring)' if is_masking_plot else 'Score (Higher is less aligned with Humans)', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.96])
results_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, f'kl_divergence_{env_filter}.png'))
print(f"Generated KL plot for env: {env_filter}")
