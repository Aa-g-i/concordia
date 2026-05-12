import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import fnmatch
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import entropy
import time
from sklearn.feature_extraction.text import CountVectorizer

# Suppress annoying tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

METHODS = {
    'Human': 'human_minimal',
    'Minimal-Simulacra': 'simulacra_minimal',
    'Minimal-Simulacra-Partial': 'simulacra_partial_0.5_minimal',
    'Native-Simulacra': 'simulacra_native',
    'Native-Simulacra-Partial': 'simulacra_partial_0.5_native',
    'Playback-Simulacra': 'nodialogue_mask_native',
    'Playback-Human': 'nodialogue_mirror_native'
}



def compute_gini(x):
    x = np.array(x, dtype=np.float64)
    if np.amin(x) < 0:
        x -= np.amin(x)
    x += 0.0000001
    x = np.sort(x)
    index = np.arange(1, x.shape[0]+1)
    n = x.shape[0]
    return ((np.sum((2 * index - n  - 1) * x)) / (n * np.sum(x)))

def build_dialogue_metrics():
    dialogue_records = []
    text_corpus = []

    unique_envs = set()
    unique_cohorts = set()
    unique_participants = set()

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    for base_dir in glob.glob(os.path.join(data_dir, "full_sweep_results", "concordia_output_*")) + glob.glob(os.path.join(data_dir, "concordia_output_*")):
        if not os.path.isdir(base_dir):
            continue
            
        matched_method = None
        for method_name, suffix in METHODS.items():
            if fnmatch.fnmatch(base_dir, f"*data_{suffix}"):
                matched_method = method_name
                break
        
        if not matched_method:
            continue
            
        matches = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.startswith('events_log') and f.endswith('.csv')]
        if not matches:
            continue
        csv_path = matches[0]
        
        # Extract environment name
        parts = base_dir.split("concordia_output_")
        if len(parts) > 1:
            remainder = parts[1]
            idx = remainder.find("_data_")
            if idx != -1:
                env = remainder[:idx]
            else:
                env = remainder
        else:
            env = "unknown"
            
        unique_envs.add(env)
        
        try:
            df = pd.read_csv(csv_path)
            df = df.dropna(subset=['content'])
            df = df[~df['agent_id'].astype(str).str.contains('System|GM|Orchestrator|deliberate', case=False, na=False)]
            
            if df.empty:
                continue
                
            df['word_count'] = df['content'].astype(str).str.split().str.len()
            df['time_index'] = pd.to_numeric(df['time_index'], errors='coerce')
            
            for cohort_id, group in df.groupby('cohort_id'):
                unique_cohorts.add(str(cohort_id))
                part_words = group.groupby('agent_id')['word_count'].sum()
                for agent_id in part_words.index:
                    unique_participants.add(str(agent_id))
                    
                gini = compute_gini(part_words)
                avg_verbosity = part_words.mean()
                
                # Inter-arrival
                group_sorted = group.sort_values('time_index')
                diffs = group_sorted['time_index'].diff().dropna()
                avg_timing = diffs.mean() if not diffs.empty else 0
                
                dialogue_records.append({
                    'Environment': env.upper(),
                    'Method': matched_method,
                    'Cohort': str(cohort_id),
                    'Gini': float(gini),
                    'Verbosity': float(avg_verbosity),
                    'Timing': float(avg_timing)
                })
                
            for _, row in df.iterrows():
                text_corpus.append({
                    'Environment': env.upper(),
                    'Method': matched_method,
                    'Agent': str(row['agent_id']),
                    'Cohort': str(row['cohort_id']),
                    'Text': str(row['content'])
                })
                
        except Exception as e:
            print(f"Failed to process dialogue {csv_path}: {e}")
            
    print("\n" + "="*50)
    print("DATASET SUMMARY (Dialogue)")
    print(f"Unique Experiments: {len(unique_envs)}")
    print(f"Unique Cohorts: {len(unique_cohorts)}")
    print(f"Unique Participants: {len(unique_participants)}")
    print("="*50 + "\n")
            
    return pd.DataFrame(dialogue_records), pd.DataFrame(text_corpus)

def build_survey_metrics():
    survey_records = []
    text_corpus = []

    # First, load baselines to establish "Truth" for MAE/Matches
    baselines = {}
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    for base_dir in glob.glob(os.path.join(data_dir, "concordia_output_*")):
        if not os.path.isdir(base_dir):
            continue
            
        if base_dir.endswith("data_deterministic_minimal"):
            parts = base_dir.split("concordia_output_")
            if len(parts) > 1:
                remainder = parts[1]
                idx = remainder.find("_data_")
                if idx != -1:
                    env = remainder[:idx]
                else:
                    env = remainder
            else:
                env = "unknown"
                
            matches = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.startswith('survey_responses') and f.endswith('.csv')]
            if matches:
                df = pd.read_csv(matches[0])
                df['num_val'] = pd.to_numeric(df['response'], errors='coerce')
                df['response'] = df['response'].astype(str).str.lower().str.strip()
                
                env_base = {}
                for _, row in df.iterrows():
                    env_base[(str(row.get('participant_id', '')), str(row.get('question_id', '')))] = {
                        'num_val': row['num_val'],
                        'text_val': row['response']
                    }
                baselines[env] = env_base

    for base_dir in glob.glob(os.path.join(data_dir, "concordia_output_*")):
        if not os.path.isdir(base_dir):
            continue
            
        matched_method = None
        for method_name, suffix in METHODS.items():
            if base_dir.endswith(f"data_{suffix}"):
                matched_method = method_name
                break
                
        if not matched_method:
            continue
            
        matches = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.startswith('survey_responses') and f.endswith('.csv')]
        if not matches:
            continue
        csv_path = matches[0]
        
        parts = base_dir.split("concordia_output_")
        if len(parts) > 1:
            remainder = parts[1]
            idx = remainder.find("_data_")
            if idx != -1:
                env = remainder[:idx]
            else:
                env = remainder
        else:
            env = "unknown"
            
        try:
            df = pd.read_csv(csv_path)
            df['num_val'] = pd.to_numeric(df['response'], errors='coerce')
            df['response'] = df['response'].astype(str).str.lower().str.strip()
            
            for _, row in df.iterrows():
                row_num = row['num_val']
                row_text = row['response']
                
                survey_records.append({
                    'Environment': env.upper(),
                    'Method': matched_method,
                    'Num_Value': row_num,
                    'Text_Value': row_text
                })
                
                val_rat = row.get('rationale', '')
                if pd.notna(val_rat) and str(val_rat).strip():
                    text_corpus.append({
                        'Environment': env.upper(),
                        'Method': matched_method,
                        'Text': str(val_rat)
                    })
        except Exception as e:
            print(f"Failed to process survey {csv_path}: {e}")
            
    df_records = pd.DataFrame(survey_records) if survey_records else pd.DataFrame(columns=['Environment', 'Method', 'Num_Value', 'Text_Value'])
    df_text = pd.DataFrame(text_corpus) if text_corpus else pd.DataFrame(columns=['Environment', 'Method', 'Text'])
    return df_records, df_text

def plot_massive_umap(df_texts, title, out_path):
    if len(df_texts) == 0: return
    print(f"Generating massive UMAP for {title}...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df_texts['Text'].tolist(), show_progress_bar=True)
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    df_texts['umap_x'] = embedding_2d[:, 0]
    df_texts['umap_y'] = embedding_2d[:, 1]
    
    # Kmeans clustering just for shapes/density (optional, but requested in previous visualizations)
    kmeans = KMeans(n_clusters=int(min(7, len(df_texts)/10)), random_state=42, n_init='auto').fit(embeddings)
    df_texts['cluster'] = kmeans.labels_
    
    plt.figure(figsize=(16, 12))
    # Using 'Method' as hue, and 'Environment' as style
    sns.scatterplot(data=df_texts, x='umap_x', y='umap_y', hue='Method', style='Environment', 
                    palette='tab10', s=60, alpha=0.7)
    plt.title(f"Global Unified Semantic Map: {title}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def compare_reduction_methods(df_texts, title, out_prefix):
    if df_texts.empty: return {}
    print(f"Comparing reduction methods for {title}...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df_texts['Text'].tolist(), show_progress_bar=True)
    
    methods = {
        'PCA': PCA(n_components=2, random_state=42),
        't-SNE': TSNE(n_components=2, random_state=42, n_jobs=-1),
        'UMAP': umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    }
    
    results = {}
    
    for name, reducer in methods.items():
        print(f"Running {name}...")
        start_time = time.time()
        embedding_2d = reducer.fit_transform(embeddings)
        print(f"{name} took {time.time() - start_time:.2f} seconds.")
        
        df_texts[f'{name.lower()}_x'] = embedding_2d[:, 0]
        df_texts[f'{name.lower()}_y'] = embedding_2d[:, 1]
        
        # Re-cluster in 2D space
        n_clusters = int(min(7, len(df_texts)/10))
        if n_clusters < 2: n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(embedding_2d)
        df_texts[f'{name.lower()}_cluster'] = kmeans.labels_
        
        # Compute entropy of cluster distribution
        counts = pd.Series(kmeans.labels_).value_counts(normalize=True)
        ent = entropy(counts)
        results[name] = ent
        print(f"Entropy for {name}: {ent:.4f}")
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=df_texts, x=f'{name.lower()}_x', y=f'{name.lower()}_y', hue='Method', style='Environment', 
                        palette='tab10', s=60, alpha=0.7)
        plt.title(f"{name} Projection: {title} (Entropy: {ent:.4f})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{name.lower()}.png", dpi=300)
        plt.close()
        
    return results


def compute_ttr(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def analyze_lexical_diversity(df_texts):
    print("\nAnalyzing Lexical Diversity (TTR)...")
    if 'Text' not in df_texts.columns or 'Method' not in df_texts.columns:
        print("Missing required columns for TTR analysis.")
        return {}
    
    df_texts['ttr'] = df_texts['Text'].apply(compute_ttr)
    
    results = {}
    for method in df_texts['Method'].unique():
        sub_df = df_texts[df_texts['Method'] == method]
        avg_ttr = sub_df['ttr'].mean()
        results[method] = avg_ttr
        print(f"Average TTR for {method}: {avg_ttr:.4f}")
        
    return results

def analyze_distinguishing_ngrams(df_texts):
    print("\nAnalyzing Distinguishing N-grams (Humans vs LLMs)...")
    if 'Text' not in df_texts.columns or 'Method' not in df_texts.columns:
        print("Missing required columns for N-gram analysis.")
        return
        
    df_texts['Group'] = df_texts['Method'].apply(lambda x: 'Human' if 'Human' in x or 'deterministic' in x else 'LLM')
    
    human_texts = df_texts[df_texts['Group'] == 'Human']['Text'].tolist()
    llm_texts = df_texts[df_texts['Group'] == 'LLM']['Text'].tolist()
    
    if not human_texts or not llm_texts:
        print("Insufficient data for comparison.")
        return
        
    vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english', max_features=1000)
    vectorizer.fit(df_texts['Text'])
    
    human_counts = vectorizer.transform([' '.join(human_texts)]).toarray()[0]
    llm_counts = vectorizer.transform([' '.join(llm_texts)]).toarray()[0]
    
    human_freq = human_counts / max(1, sum(human_counts))
    llm_freq = llm_counts / max(1, sum(llm_counts))
    
    diff = llm_freq - human_freq
    vocab = vectorizer.get_feature_names_out()
    
    sorted_idx = np.argsort(diff)
    
    print("\nTop 10 N-grams prominent in LLM dialogue:")
    for idx in sorted_idx[-10:][::-1]:
        print(f"  - {vocab[idx]} (diff: {diff[idx]:.4f})")
        
    print("\nTop 10 N-grams prominent in Human dialogue:")
    for idx in sorted_idx[:10]:
        print(f"  - {vocab[idx]} (diff: {-diff[idx]:.4f})")


def plot_participant_umaps(df_texts, output_base_dir):
    if df_texts.empty:
        return
    if "Agent" not in df_texts.columns:
        print("Skipping participant plots: 'Agent' column missing.")
        return

    out_dir = os.path.join(output_base_dir, "participant_plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating per-participant UMAPs in {out_dir}...")

    # Get all unique agents
    agents = df_texts["Agent"].dropna().unique()

    for agent in agents:
        agent_s = str(agent)
        # Skip system or empty agents
        if not agent_s.strip() or any(x in agent_s for x in ["None", "nan"]):
            continue

        part_df = df_texts[df_texts["Agent"] == agent]

        # Skip if very few utterances to avoid noisy plots
        if len(part_df) < 3:
            continue

        plt.figure(figsize=(10, 8))

        # Plot background in light gray
        sns.scatterplot(
            data=df_texts,
            x="umap_x",
            y="umap_y",
            color="lightgray",
            s=10,
            alpha=0.3,
            label="Other Speakers",
        )

        # Plot participant in color
        sns.scatterplot(
            data=part_df,
            x="umap_x",
            y="umap_y",
            hue="Method",
            palette="tab10",
            s=60,
            alpha=0.9,
        )

        plt.title(f"Participant Semantic Drift: {agent_s}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        # Clean filename
        safe_name = "".join([c if c.isalnum() else "_" for c in agent_s])
        plt.savefig(
            os.path.join(out_dir, f"participant_{safe_name}.png"), dpi=150
        )  # lower dpi to save space/time
        plt.close()


def plot_cohort_umaps(df_texts, output_base_dir):
    if df_texts.empty:
        return
    if "Cohort" not in df_texts.columns:
        print("Skipping cohort plots: 'Cohort' column missing.")
        return

    out_dir = os.path.join(output_base_dir, "cohort_plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating per-cohort UMAPs in {out_dir}...")

    cohorts = df_texts["Cohort"].dropna().unique()

    for cohort in cohorts:
        cohort_s = str(cohort)
        if not cohort_s.strip() or any(x in cohort_s for x in ["None", "nan"]):
            continue

        cohort_df = df_texts[df_texts["Cohort"] == cohort]

        if len(cohort_df) < 5:
            continue

        plt.figure(figsize=(10, 8))

        # Plot background in light gray
        sns.scatterplot(
            data=df_texts,
            x="umap_x",
            y="umap_y",
            color="lightgray",
            s=10,
            alpha=0.3,
            label="Full Corpus",
        )

        # Plot cohort in color
        sns.scatterplot(
            data=cohort_df,
            x="umap_x",
            y="umap_y",
            hue="Method",
            style="Environment",
            palette="tab10",
            s=60,
            alpha=0.9,
        )

        plt.title(f"Cohort Semantic Map: {cohort_s}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        safe_name = "".join([c if c.isalnum() else "_" for c in cohort_s])
        plt.savefig(os.path.join(out_dir, f"cohort_{safe_name}.png"), dpi=150)
        plt.close()


def plot_global_bar(df, metric, title, ylabel, out_path, is_boxplot=False):

    if df.empty: return
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    df = df.dropna(subset=[metric])
    
    if is_boxplot:
        sns.boxplot(data=df, x='Environment', y=metric, hue='Method', palette='Set2')
    else:
        # barplot with error bars across cohorts
        sns.barplot(data=df, x='Environment', y=metric, hue='Method', palette='Set2', errorbar='sd', capsize=0.1)
        
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def run_anova(df, metric):
    print(f"\nRunning ANOVA for {metric}...")
    methods = df['Method'].unique()
    groups = [df[df['Method'] == m][metric].dropna() for m in methods]
    groups = [g for g in groups if not g.empty]
    if len(groups) < 2:
        print("Not enough groups for ANOVA.")
        return
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_val:.4f}")

def calculate_separability(df, coord_cols=['umap_x', 'umap_y']):
    print("\nCalculating Pairwise Cluster Separability (Fisher-like Criterion)...")
    methods = df['Method'].unique()
    results = []
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            m1 = methods[i]
            m2 = methods[j]
            
            df1 = df[df['Method'] == m1]
            df2 = df[df['Method'] == m2]
            
            if df1.empty or df2.empty:
                continue
                
            pts1 = df1[coord_cols].values
            pts2 = df2[coord_cols].values
            
            c1 = pts1.mean(axis=0)
            c2 = pts2.mean(axis=0)
            
            between = np.linalg.norm(c1 - c2)**2
            
            var1 = np.mean(np.sum((pts1 - c1)**2, axis=1))
            var2 = np.mean(np.sum((pts2 - c2)**2, axis=1))
            
            fisher_score = between / (var1 + var2 + 1e-10)
            
            print(f"Pair ({m1} vs {m2}): Score = {fisher_score:.4f}")
            results.append({
                'Pair': f"{m1} vs {m2}",
                'Score': fisher_score
            })
    return pd.DataFrame(results)

if __name__ == "__main__":
    output_base_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'global_metrics')
    os.makedirs(output_base_dir, exist_ok=True)

    print("Extracting Dialogues...")
    df_events, df_events_text = build_dialogue_metrics()
    print("Extracting Surveys...")
    df_surveys, df_surveys_text = build_survey_metrics()
    
    print(f"Compacting {len(df_events)} cohort-level dialogue interactions...")
    # Plotting Structural Dialogues
    plot_global_bar(df_events, 'Gini', "Global Cohort Gini Inequality by Environment", "Gini Coefficient (0=Egalitarian, 1=Dominated)", os.path.join(output_base_dir, "global_gini_comparison.png"))
    plot_global_bar(df_events, 'Verbosity', "Global Average Participant Verbosity by Environment", "Avg Words per Participant per Cohort", os.path.join(output_base_dir, "global_verbosity_comparison.png"))
    
    # Plotting Surveys Output Value Densities
    print(f"Compacting {len(df_surveys)} numerical survey traces...")
    plot_global_bar(df_surveys, 'Num_Value', "Global Numeric Survey Responses", "Response Value", os.path.join(output_base_dir, "global_survey_numeric_boxplot.png"), is_boxplot=True)
    
    # Massive UMAPs combined globally!
    plot_massive_umap(df_surveys_text, "Survey Rationales (All Environments & Methods)", os.path.join(output_base_dir, "global_survey_rationale_umap.png"))
    plot_massive_umap(df_events_text, "Dialogue Utterances (All Environments & Methods)", os.path.join(output_base_dir, "global_dialogue_content_umap.png"))
    
    print("Comparing reduction methods for Dialogue Utterances...")
    entropy_results = compare_reduction_methods(df_events_text, "Dialogue Utterances", os.path.join(output_base_dir, "global_dialogue_content"))
    print("Entropy Results:", entropy_results)
    
    print("Analyzing Lexical Diversity...")
    analyze_lexical_diversity(df_events_text)
    
    print("Analyzing Distinguishing N-grams...")
    analyze_distinguishing_ngrams(df_events_text)
    
    # New Participant & Cohort UMAPs
    plot_participant_umaps(df_events_text, output_base_dir)
    plot_cohort_umaps(df_events_text, output_base_dir)
    
    # Run ANOVA
    run_anova(df_events, 'Gini')
    run_anova(df_events, 'Verbosity')
    
    # Run Separability on all reduction methods symmetrically
    print("\nCluster Separability for PCA:")
    calculate_separability(df_events_text, coord_cols=['pca_x', 'pca_y'])
    
    print("\nCluster Separability for t-SNE:")
    calculate_separability(df_events_text, coord_cols=['t-sne_x', 't-sne_y'])
    
    print("\nCluster Separability for UMAP:")
    calculate_separability(df_events_text, coord_cols=['umap_x', 'umap_y'])
    
    # Save data for inspection
    df_events_text.to_csv(os.path.join(output_base_dir, "dialogue_umap_data.csv"), index=False)
    
    print("\n" + "="*50)
    print("PER-EXPERIMENT DETAILED ANALYSIS")
    print("="*50)
    
    experiments = df_events_text['Environment'].dropna().unique()
    
    for exp in experiments:
        print(f"\n{'='*20} Experiment: {exp} {'='*20}")
        exp_df = df_events_text[df_events_text['Environment'] == exp].copy()
        
        if len(exp_df) < 10:
            print(f"Skipping {exp} due to insufficient samples ({len(exp_df)})")
            continue
            
        print(f"Running reduction methods for {exp}...")
        try:
            exp_entropy = compare_reduction_methods(exp_df, f"Dialogue Utterances ({exp})", f"{exp.lower()}_dialogue_content")
            print(f"Entropy Results for {exp}:", exp_entropy)
        except Exception as e:
            print(f"Failed to run reduction for {exp}: {e}")
            continue
            
        print(f"\nAnalyzing Lexical Diversity for {exp}...")
        analyze_lexical_diversity(exp_df)
        
        print(f"\nAnalyzing Distinguishing N-grams for {exp}...")
        analyze_distinguishing_ngrams(exp_df)
        
        print(f"\nCluster Separability for {exp} (PCA):")
        calculate_separability(exp_df, coord_cols=['pca_x', 'pca_y'])
        
        print(f"\nCluster Separability for {exp} (t-SNE):")
        calculate_separability(exp_df, coord_cols=['t-sne_x', 't-sne_y'])
        
        print(f"\nCluster Separability for {exp} (UMAP):")
        calculate_separability(exp_df, coord_cols=['umap_x', 'umap_y'])
        
    print("Global Analytics Pipeline Compilation Successfully Exhausted.")
