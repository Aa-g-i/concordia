import pandas as pd
import numpy as np
import os
import glob
import zipfile
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
BASE_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

METHODS = {
    'Minimal-Simulacra': 'simulacra_minimal',
    'Minimal-Simulacra-Partial': 'simulacra_partial_0.5_minimal',
    'Playback-Simulacra': 'nodialogue_mask_native',
    'Playback-Human': 'nodialogue_mirror_native',
    'Native-Simulacra': 'simulacra_native',
    'Native-Simulacra-Partial': 'simulacra_partial_0.5_native'
}
HUMAN_SUFFIX = 'human_minimal'

def load_proxy_kl_data():
    try:
        df = pd.read_csv("/usr/local/google/home/aarontp/.gemini/jetski/scratch/llm_kl_analysis/proxy_kl_experiment.csv")
        # Negate proxy_kl to make it positive (NLL)
        df['proxy_kl'] = -df['proxy_kl']
        
        mapping = {
            'LLM Simulacra': 'Minimal-Simulacra',
            'Simulacra Partial': 'Minimal-Simulacra-Partial',
            'NoDialogue Mask': 'Playback-Simulacra',
            'NoDialogue Mirror': 'Playback-Human',
            'Human Baseline': 'Human',
            'Human': 'Human'
        }
        df['Model'] = df['method'].map(mapping)
        # Drop Unknown or unmapped
        df = df.dropna(subset=['Model'])
        
        # Extract Experiment name
        def extract_exp(env):
            if '_data_' in env:
                return env.split('_data_')[0]
            return env
            
        df['Experiment'] = df['environment'].apply(extract_exp)
        
        return df
    except FileNotFoundError:
        print("Proxy KL experiment file not found.")
        return pd.DataFrame()

def load_human_id_mapping():
    """Loads mapping from Private ID to Current cohort ID from ParticipantData.csv."""
    mapping = {}
    
    exp_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'new_experiments')
    exp_zips = glob.glob(os.path.join(exp_data_dir, "*_data*.zip"))
    
    for exp_zip in exp_zips:
        if "TEMPLATE" in exp_zip or "medica" in exp_zip.lower() or "a5" in exp_zip.lower():
            continue
            
        print(f"Loading mapping from {os.path.basename(exp_zip)}...")
        try:
            with zipfile.ZipFile(exp_zip) as ez:
                pdata_files = [x for x in ez.namelist() if 'ParticipantData' in x]
                if not pdata_files:
                    print(f"  No ParticipantData file found in {os.path.basename(exp_zip)}")
                    continue
                    
                pdata_file = pdata_files[0]
                with ez.open(pdata_file) as f:
                    df_part = pd.read_csv(f, low_memory=False)
                    
                if 'Private ID' in df_part.columns and 'Current cohort ID' in df_part.columns:
                    match_count = 0
                    for _, row in df_part.iterrows():
                        pid = row['Private ID']
                        cohort = row['Current cohort ID']
                        if pd.notna(pid) and pd.notna(cohort):
                            mapping[pid] = cohort
                            match_count += 1
                    print(f"  Found {match_count} mappings in {os.path.basename(exp_zip)}")
                else:
                    print(f"  Missing required columns in {pdata_file}")
        except Exception as e:
            print(f"Error reading {exp_zip}: {e}")
            
    print(f"Total unique participants mapped: {len(mapping)}")
    return mapping

def load_survey_data():
    survey_data = []
    
    # Load Human baseline first (prefer deterministic_minimal, fallback to human_minimal)
    human_dirs = glob.glob(os.path.join(BASE_DATA_DIR, "concordia_output_*data_deterministic_minimal"))
    if not human_dirs:
        human_dirs = glob.glob(os.path.join(BASE_DATA_DIR, f"concordia_output_*data_{HUMAN_SUFFIX}"))
        
    for d in human_dirs:
        exp_name = os.path.basename(d).replace('concordia_output_', '').split('_data_')[0]
        csv_files = [os.path.join(d, f) for f in os.listdir(d) if f.startswith('survey_responses') and f.endswith('.csv')]
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            df['Model'] = 'Human'
            df['Experiment'] = exp_name
            survey_data.append(df)
            
    # Load Models
    for method_name, suffix in METHODS.items():
        dirs = glob.glob(os.path.join(BASE_DATA_DIR, f"full_sweep_results/concordia_output_*data_{suffix}")) + glob.glob(os.path.join(BASE_DATA_DIR, f"concordia_output_*data_{suffix}"))
        for d in dirs:
            exp_name = os.path.basename(d).replace('concordia_output_', '').split('_data_')[0]
            csv_files = glob.glob(os.path.join(d, 'survey_responses*.csv'))
            for csv_path in csv_files:
                df = pd.read_csv(csv_path)
                df['Model'] = method_name
                df['Experiment'] = exp_name
                survey_data.append(df)
                
    if not survey_data:
        return pd.DataFrame()
        
    return pd.concat(survey_data, ignore_index=True)

def compute_survey_kl(df):
    if df.empty or 'Model' not in df.columns or 'question_id' not in df.columns:
        print("Invalid survey data for KL computation.")
        return {}
        
    human_df = df[df['Model'] == 'Human']
    if human_df.empty:
        print("No Human survey baseline found.")
        return {}
        
    models = [m for m in df['Model'].unique() if m != 'Human']
    results = {model: {} for model in models}
    
    questions = df['question_id'].dropna().unique()
    
    for question in questions:
        h_q = human_df[human_df['question_id'] == question]
        if h_q.empty: continue
        
        # Treat responses as discrete categories
        h_counts = h_q['response'].value_counts(normalize=True)
        
        for model in models:
            m_df = df[(df['Model'] == model) & (df['question_id'] == question)]
            if m_df.empty: continue
            
            m_counts = m_df['response'].value_counts(normalize=True)
            
            # Align categories
            all_cats = list(set(h_counts.index) | set(m_counts.index))
            p = np.array([h_counts.get(c, 0.0) for c in all_cats])
            q = np.array([m_counts.get(c, 0.0) for c in all_cats])
            
            # Add small epsilon to avoid log(0)
            p = (p + 1e-10) / (p.sum() + 1e-10 * len(all_cats))
            q = (q + 1e-10) / (q.sum() + 1e-10 * len(all_cats))
            
            kl = entropy(q, p) # KL(Model || Human)
            results[model][question] = kl
            
    # Aggregate by model
    agg_results = {}
    for model, q_scores in results.items():
        if q_scores:
            agg_results[model] = np.mean(list(q_scores.values()))
        else:
            agg_results[model] = np.nan
            
    return agg_results

def compute_survey_mae(df):
    if df.empty or 'Model' not in df.columns or 'question_id' not in df.columns:
        return {}
        
    human_df = df[df['Model'] == 'Human']
    if human_df.empty:
        return {}
        
    # Filter for numerical responses
    df['num_val'] = pd.to_numeric(df['response'], errors='coerce')
    human_df = df[df['Model'] == 'Human'].dropna(subset=['num_val'])
    
    models = [m for m in df['Model'].unique() if m != 'Human']
    results = {model: {} for model in models}
    
    questions = human_df['question_id'].unique()
    
    for question in questions:
        h_vals = human_df[human_df['question_id'] == question]['num_val']
        if h_vals.empty: continue
        h_mean = h_vals.mean()
        
        for model in models:
            m_vals = df[(df['Model'] == model) & (df['question_id'] == question)]['num_val'].dropna()
            if m_vals.empty: continue
            m_mean = m_vals.mean()
            
            results[model][question] = abs(m_mean - h_mean)
            
    agg_results = {}
    for model, q_scores in results.items():
        if q_scores:
            agg_results[model] = np.mean(list(q_scores.values()))
        else:
            agg_results[model] = np.nan
            
    return agg_results

def calculate_gini(array):
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

def compute_survey_gini(df):
    if df.empty or 'Model' not in df.columns or 'rationale' not in df.columns:
        return {}
    results = {}
    for model in df['Model'].unique():
        m_df = df[df['Model'] == model]
        gini_scores = []
        for cohort, group in m_df.groupby('cohort_id'):
            if 'participant_id' in group.columns:
                part_rationale_len = group.groupby('participant_id')['rationale'].apply(lambda x: x.astype(str).str.len().sum()).values
                if len(part_rationale_len) > 1:
                    gini_scores.append(calculate_gini(part_rationale_len))
        if gini_scores:
            results[model] = np.mean(gini_scores)
        else:
            results[model] = np.nan
    return results

def compute_dialogue_gini(df):
    if df.empty or 'Model' not in df.columns or 'word_count' not in df.columns:
        return {}
    models = [m for m in df['Model'].unique() if m != 'Human']
    results = {}
    for model in ['Human'] + models:
        m_df = df[df['Model'] == model]
        if m_df.empty: continue
        gini_scores = []
        for cohort, group in m_df.groupby('cohort_id'):
            part_words = group.groupby('agent_id')['word_count'].sum().values
            if len(part_words) > 1:
                gini_scores.append(calculate_gini(part_words))
        if gini_scores:
            results[model] = np.mean(gini_scores)
        else:
            results[model] = np.nan
    return results

def compute_dialogue_timing(df):
    if df.empty or 'Model' not in df.columns or 'time_index' not in df.columns:
        return {}
    results = {}
    for model in df['Model'].unique():
        m_df = df[df['Model'] == model]
        timing_scores = []
        for cohort, group in m_df.groupby('cohort_id'):
            group['time_index'] = pd.to_numeric(group['time_index'], errors='coerce')
            group = group.dropna(subset=['time_index'])
            group = group.sort_values('time_index')
            group['time_delay'] = group['time_index'].diff()
            df_timings = group[(group['time_delay'] > 0) & (group['time_delay'] < 300)]
            if not df_timings.empty:
                timing_scores.append(df_timings['time_delay'].mean())
        if timing_scores:
            results[model] = np.mean(timing_scores)
        else:
            results[model] = np.nan
    return results

def compute_dialogue_ttr(df):
    if df.empty or 'Model' not in df.columns or 'content' not in df.columns:
        return {}
    results = {}
    for model in df['Model'].unique():
        m_df = df[df['Model'] == model]
        ttr_scores = []
        for cohort, group in m_df.groupby('cohort_id'):
            all_text = " ".join(group['content'].astype(str).tolist())
            words = all_text.split()
            total_words = len(words)
            unique_words = len(set(words))
            if total_words > 0:
                ttr_scores.append(unique_words / total_words)
        if ttr_scores:
            results[model] = np.mean(ttr_scores)
        else:
            results[model] = np.nan
    return results

def compute_dialogue_transition_entropy(df):
    if df.empty or 'Model' not in df.columns or 'agent_id' not in df.columns:
        return {}
    results = {}
    for model in df['Model'].unique():
        m_df = df[df['Model'] == model]
        entropy_scores = []
        for cohort, group in m_df.groupby('cohort_id'):
            group = group.sort_values('time_index')
            speakers = group['agent_id'].astype(str).tolist()
            if len(speakers) < 2: continue
            transitions = [(speakers[i], speakers[i+1]) for i in range(len(speakers) - 1)]
            counter = Counter(transitions)
            counts = np.array(list(counter.values()))
            probs = counts / counts.sum()
            entropy_scores.append(entropy(probs))
        if entropy_scores:
            results[model] = np.mean(entropy_scores)
        else:
            results[model] = np.nan
    return results

def get_embeddings(df, text_col='content', cache_path='comprehensive_dialogue_embeddings.npy'):
    texts = df[text_col].astype(str).tolist()
    
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        if len(embeddings) == len(texts):
            print(f"Loading cached embeddings from {cache_path}...")
            return embeddings
        else:
            print(f"Cache size mismatch ({len(embeddings)} != {len(texts)}). Recomputing...")
            
    if not HAS_SENTENCE_TRANSFORMERS:
        print("SentenceTransformer not available, cannot generate embeddings.")
        return None
        
    print("Generating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    
    np.save(cache_path, embeddings)
    print(f"Saved embeddings to {cache_path}")
    return embeddings

def compute_dialogue_semantic_convergence(df, embeddings=None):
    if df.empty or 'Model' not in df.columns or 'content' not in df.columns:
        return {}
    results = {}
    
    if embeddings is not None:
        print("Using pre-computed embeddings for semantic convergence...")
        for model_name in df['Model'].unique():
            m_df = df[df['Model'] == model_name]
            conv_scores = []
            for cohort, group in m_df.groupby('cohort_id'):
                group = group.sort_values('time_index')
                mid_point = len(group) // 2
                if mid_point < 5: continue
                
                early_idx = group.index[:mid_point]
                late_idx = group.index[mid_point:]
                
                early_emb = embeddings[early_idx]
                late_emb = embeddings[late_idx]
                
                early_var = np.var(early_emb, axis=0).sum()
                late_var = np.var(late_emb, axis=0).sum()
                conv_scores.append(early_var - late_var)
            if conv_scores:
                results[model_name] = np.mean(conv_scores)
            else:
                results[model_name] = np.nan
        return results
        
    if HAS_SENTENCE_TRANSFORMERS:
        print("Using SentenceTransformer for semantic convergence...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        for model_name in df['Model'].unique():
            m_df = df[df['Model'] == model_name]
            conv_scores = []
            for cohort, group in m_df.groupby('cohort_id'):
                group = group.sort_values('time_index')
                mid_point = len(group) // 2
                if mid_point < 5: continue
                early_text = group['content'].astype(str).iloc[:mid_point].tolist()
                late_text = group['content'].astype(str).iloc[mid_point:].tolist()
                early_emb = model.encode(early_text)
                late_emb = model.encode(late_text)
                early_var = np.var(early_emb, axis=0).sum()
                late_var = np.var(late_emb, axis=0).sum()
                conv_scores.append(early_var - late_var)
            if conv_scores:
                results[model_name] = np.mean(conv_scores)
            else:
                results[model_name] = np.nan
    else:
        print("SentenceTransformer not available. Using TF-IDF fallback for semantic convergence...")
        for model_name in df['Model'].unique():
            m_df = df[df['Model'] == model_name]
            conv_scores = []
            for cohort, group in m_df.groupby('cohort_id'):
                group = group.sort_values('time_index')
                mid_point = len(group) // 2
                if mid_point < 5: continue
                early_text = group['content'].astype(str).iloc[:mid_point].tolist()
                late_text = group['content'].astype(str).iloc[mid_point:].tolist()
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                try:
                    early_tfidf = vectorizer.fit_transform(early_text).toarray()
                    late_tfidf = vectorizer.transform(late_text).toarray()
                    early_var = np.var(early_tfidf, axis=0).sum()
                    late_var = np.var(late_tfidf, axis=0).sum()
                    conv_scores.append(early_var - late_var)
                except:
                    pass
            if conv_scores:
                results[model_name] = np.mean(conv_scores)
            else:
                results[model_name] = np.nan
    return results

def analyze_distinguishing_ngrams(df_texts):
    print("\nAnalyzing Distinguishing N-grams (Humans vs LLMs)...")
    if 'content' not in df_texts.columns or 'Model' not in df_texts.columns:
        print("Missing required columns for N-gram analysis.")
        return
        
    df_texts['Group'] = df_texts['Model'].apply(lambda x: 'Human' if 'Human' in x else 'LLM')
    
    human_texts = df_texts[df_texts['Group'] == 'Human']['content'].dropna().tolist()
    llm_texts = df_texts[df_texts['Group'] == 'LLM']['content'].dropna().tolist()
    
    if not human_texts or not llm_texts:
        print("Insufficient data for comparison.")
        return
        
    vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english', max_features=1000)
    vectorizer.fit(df_texts['content'].astype(str))
    
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

def get_umap_coordinates(df, embeddings, cache_path='comprehensive_dialogue_umap.csv'):
    if os.path.exists(cache_path):
        print(f"Loading cached UMAP coordinates from {cache_path}...")
        umap_df = pd.read_csv(cache_path)
        if len(umap_df) == len(df):
            df['umap_x'] = umap_df['umap_x'].values
            df['umap_y'] = umap_df['umap_y'].values
            return df
        else:
            print(f"Cache size mismatch ({len(umap_df)} != {len(df)}). Recomputing UMAP...")
            
    if embeddings is None:
        print("No embeddings available to calculate UMAP.")
        return df
        
    print("Calculating UMAP coordinates...")
    try:
        import umap.umap_ as umap_pkg
        reducer = umap_pkg.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        
        df['umap_x'] = embedding_2d[:, 0]
        df['umap_y'] = embedding_2d[:, 1]
        
        # Save cache
        df[['umap_x', 'umap_y']].to_csv(cache_path, index=False)
        print(f"Saved UMAP coordinates to {cache_path}")
    except ImportError:
        print("umap-learn not installed. Skipping UMAP calculation.")
        
    return df

def calculate_separability(df, coord_cols=['umap_x', 'umap_y']):
    print(f"\nCalculating Pairwise Cluster Separability for {coord_cols}...")
    if not all(col in df.columns for col in coord_cols):
        print(f"Missing coordinate columns {coord_cols} for separability.")
        return pd.DataFrame()
        
    methods = df['Model'].unique()
    results = []
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            m1 = methods[i]
            m2 = methods[j]
            
            df1 = df[df['Model'] == m1]
            df2 = df[df['Model'] == m2]
            
            if df1.empty or df2.empty:
                continue
                
            pts1 = df1[coord_cols].dropna().values
            pts2 = df2[coord_cols].dropna().values
            
            if len(pts1) < 2 or len(pts2) < 2:
                continue
                
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

def load_dialogue_data():
    dialogue_data = []
    
    # Load human ID mapping
    mapping = load_human_id_mapping()
    
    # Load Human baseline (use human_minimal for dialogue metrics as it contains actual dialogue)
    human_dirs = glob.glob(os.path.join(BASE_DATA_DIR, f"concordia_output_*data_{HUMAN_SUFFIX}"))
        
    for d in human_dirs:
        exp_name = os.path.basename(d).replace('concordia_output_', '').split('_data_')[0]
        csv_files = glob.glob(os.path.join(d, 'events_log*.csv'))
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            df = df[~df['agent_id'].str.contains('System', case=False, na=False)]
            df['word_count'] = df['content'].astype(str).str.split().str.len()
            df['Model'] = 'Human'
            df['Experiment'] = exp_name
            
            # Map agent_id to cohort_id for humans
            df['cohort_id'] = df['agent_id'].map(mapping).fillna(df['cohort_id'])
            
            dialogue_data.append(df)
            
    # Load Models
    for method_name, suffix in METHODS.items():
        dirs = glob.glob(os.path.join(BASE_DATA_DIR, f"full_sweep_results/concordia_output_*data_{suffix}")) + glob.glob(os.path.join(BASE_DATA_DIR, f"concordia_output_*data_{suffix}"))
        for d in dirs:
            exp_name = os.path.basename(d).replace('concordia_output_', '').split('_data_')[0]
            csv_files = glob.glob(os.path.join(d, 'events_log*.csv'))
            for csv_path in csv_files:
                df = pd.read_csv(csv_path)
                df = df[~df['agent_id'].str.contains('System', case=False, na=False)]
                df['word_count'] = df['content'].astype(str).str.split().str.len()
                df['Model'] = method_name
                df['Experiment'] = exp_name
                dialogue_data.append(df)
                
    if not dialogue_data:
        return pd.DataFrame()
        
    return pd.concat(dialogue_data, ignore_index=True)

def compute_dialogue_kl(df):
    if df.empty or 'Model' not in df.columns or 'word_count' not in df.columns:
        print("Invalid dialogue data for KL computation.")
        return {}
        
    human_df = df[df['Model'] == 'Human']
    if human_df.empty:
        print("No Human dialogue baseline found.")
        return {}
        
    models = [m for m in df['Model'].unique() if m != 'Human']
    scores = {}
    
    bins = 10
    h_vals = human_df['word_count'].dropna()
    p_counts, _ = np.histogram(h_vals, bins=bins, range=(0, h_vals.max()))
    p_dist = (p_counts + 1e-10) / p_counts.sum()
    
    for model in models:
        m_vals = df[df['Model'] == model]['word_count'].dropna()
        if m_vals.empty: continue
        
        q_counts, _ = np.histogram(m_vals, bins=bins, range=(0, h_vals.max()))
        q_dist = (q_counts + 1e-10) / q_counts.sum()
        
        kl = entropy(q_dist, p_dist)
        scores[model] = kl
        
    return scores

def compute_dialogue_hds(df):
    if df.empty or 'Model' not in df.columns or 'word_count' not in df.columns:
        return {}
    human_df = df[df['Model'] == 'Human']
    if human_df.empty:
        return {}
    models = [m for m in df['Model'].unique() if m != 'Human']
    scores = {}
    h_mean = human_df['word_count'].dropna().mean()
    for model in models:
        m_vals = df[df['Model'] == model]['word_count'].dropna()
        if m_vals.empty: continue
        m_mean = m_vals.mean()
        scores[model] = abs(m_mean - h_mean)
    return scores

if __name__ == "__main__":
    print("Loading Survey Data...")
    survey_df = load_survey_data()
    print("Computing Survey KL Divergence...")
    survey_kl = compute_survey_kl(survey_df)
    
    print("\nLoading Dialogue Data...")
    dialogue_df = load_dialogue_data()
    
    print("Generating/Loading Embeddings...")
    embeddings = get_embeddings(dialogue_df)
    
    print("Generating/Loading UMAP Coordinates...")
    dialogue_df = get_umap_coordinates(dialogue_df, embeddings)
    
    print("Computing Dialogue KL Divergence (Verbosity)...")
    dialogue_kl = compute_dialogue_kl(dialogue_df)
    print("Computing Dialogue HDS (Verbosity Deviation)...")
    dialogue_hds = compute_dialogue_hds(dialogue_df)
    
    print("Computing Dialogue Gini...")
    dialogue_gini = compute_dialogue_gini(dialogue_df)
    print("Computing Dialogue Timing...")
    dialogue_timing = compute_dialogue_timing(dialogue_df)
    print("Computing Dialogue TTR...")
    dialogue_ttr = compute_dialogue_ttr(dialogue_df)
    print("Computing Dialogue Transition Entropy...")
    dialogue_trans_entropy = compute_dialogue_transition_entropy(dialogue_df)
    print("Computing Dialogue Semantic Convergence...")
    dialogue_convergence = compute_dialogue_semantic_convergence(dialogue_df, embeddings=embeddings)
    
    print("Loading Proxy KL Data...")
    proxy_kl_df = load_proxy_kl_data()
    
    print("\nAnalyzing Distinguishing N-grams (Global)...")
    analyze_distinguishing_ngrams(dialogue_df)
    
    print("\nCalculating Separability (Global)...")
    separability_df = calculate_separability(dialogue_df)
    print(separability_df)
    
    print("\n" + "="*50)
    print("COMPREHENSIVE METRICS COMPARISON")
    print("="*50)
    
    all_models = set(survey_kl.keys()) | set(dialogue_kl.keys()) | set(dialogue_gini.keys())
    
    print("Computing Survey MAE...")
    survey_mae = compute_survey_mae(survey_df)
    
    print("Computing Survey Gini...")
    survey_gini = compute_survey_gini(survey_df)
    
    # For global table, group proxy_kl by Model
    proxy_kl_global = pd.DataFrame()
    if not proxy_kl_df.empty:
        proxy_kl_global = proxy_kl_df.groupby('Model')['proxy_kl'].mean().reset_index().set_index('Model')
        
    records = []
    for model in all_models:
        records.append({
            'Model': model,
            'Survey KL': survey_kl.get(model, np.nan),
            'Survey MAE': survey_mae.get(model, np.nan),
            'Survey Gini': survey_gini.get(model, np.nan),
            'Dialogue KL': dialogue_kl.get(model, np.nan),
            'Dialogue HDS': dialogue_hds.get(model, np.nan),
            'Dialogue Gini': dialogue_gini.get(model, np.nan),
            'Dialogue Timing': dialogue_timing.get(model, np.nan),
            'Dialogue TTR': dialogue_ttr.get(model, np.nan),
            'Transition Entropy': dialogue_trans_entropy.get(model, np.nan),
            'Semantic Convergence': dialogue_convergence.get(model, np.nan),
            'Proxy KL': proxy_kl_global.loc[model, 'proxy_kl'] if not proxy_kl_global.empty and model in proxy_kl_global.index else np.nan
        })
        
    df_comp = pd.DataFrame(records).set_index('Model')
    print(df_comp)
    
    # Save to CSV
    csv_out = os.path.join(BASE_RESULTS_DIR, "comprehensive_metrics_comparison.csv")
    df_comp.to_csv(csv_out)
    print(f"\nSaved comparison table to {csv_out}")
    
    # Plotting
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    axes = axes.flatten()
    
    metrics = ['Survey KL', 'Survey MAE', 'Survey Gini', 'Dialogue KL', 'Dialogue HDS', 'Dialogue Gini', 'Dialogue Timing', 'Dialogue TTR', 'Transition Entropy', 'Semantic Convergence', 'Proxy KL']
    for i, metric in enumerate(metrics):
        if metric in df_comp.columns:
            data_to_plot = df_comp[metric].dropna()
            if not data_to_plot.empty:
                data_to_plot.plot(kind='bar', ax=axes[i])
                axes[i].set_title(metric)
                axes[i].set_ylabel("Score")
                axes[i].tick_params(axis='x', rotation=45)
            else:
                axes[i].set_visible(False)
            
    # Hide unused subplots
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)
            
    plt.tight_layout()
    png_out = os.path.join(BASE_RESULTS_DIR, "comprehensive_metrics_comparison.png")
    plt.savefig(png_out, dpi=300)
    print(f"\nSaved comparison plot to {png_out}")
    
    print("\n" + "="*50)
    print("PER-EXPERIMENT METRICS BREAKDOWN")
    print("="*50)
    
    all_exps = set()
    if 'Experiment' in survey_df.columns:
        all_exps.update(survey_df['Experiment'].dropna().unique())
    if 'Experiment' in dialogue_df.columns:
        all_exps.update(dialogue_df['Experiment'].dropna().unique())
        
    for exp in sorted(list(all_exps)):
        if exp == 'Human': continue
        print(f"\nExperiment: {exp}")
        
        # Filter data
        exp_survey = survey_df[survey_df['Experiment'] == exp] if 'Experiment' in survey_df.columns else pd.DataFrame()
        exp_dialogue = dialogue_df[dialogue_df['Experiment'] == exp] if 'Experiment' in dialogue_df.columns else pd.DataFrame()
        
        if exp_survey.empty and exp_dialogue.empty:
            continue
            
        # Re-compute for this experiment
        exp_survey_kl = compute_survey_kl(exp_survey)
        exp_survey_mae = compute_survey_mae(exp_survey)
        exp_dialogue_kl = compute_dialogue_kl(exp_dialogue)
        exp_dialogue_hds = compute_dialogue_hds(exp_dialogue)
        exp_dialogue_gini = compute_dialogue_gini(exp_dialogue)
        exp_dialogue_timing = compute_dialogue_timing(exp_dialogue)
        exp_dialogue_ttr = compute_dialogue_ttr(exp_dialogue)
        exp_dialogue_trans_entropy = compute_dialogue_transition_entropy(exp_dialogue)
        exp_dialogue_convergence = compute_dialogue_semantic_convergence(exp_dialogue, embeddings=embeddings)
        
        # Load Proxy KL for this experiment
        exp_proxy_kl = {}
        if not proxy_kl_df.empty and 'Experiment' in proxy_kl_df.columns:
            exp_proxy_df = proxy_kl_df[proxy_kl_df['Experiment'] == exp]
            for _, row in exp_proxy_df.iterrows():
                exp_proxy_kl[row['Model']] = row['proxy_kl']
                
        # N-grams for this experiment
        print(f"Distinguishing N-grams for {exp}:")
        analyze_distinguishing_ngrams(exp_dialogue)
        
        # Separability for this experiment
        print(f"Cluster Separability for {exp}:")
        exp_sep = calculate_separability(exp_dialogue)
        
        exp_models = set(exp_survey_kl.keys()) | set(exp_dialogue_kl.keys()) | set(exp_dialogue_gini.keys())
        
        exp_records = []
        for model in exp_models:
            exp_records.append({
                'Model': model,
                'Survey KL': exp_survey_kl.get(model, np.nan),
                'Survey MAE': exp_survey_mae.get(model, np.nan),
                'Dialogue KL': exp_dialogue_kl.get(model, np.nan),
                'Dialogue HDS': exp_dialogue_hds.get(model, np.nan),
                'Dialogue Gini': exp_dialogue_gini.get(model, np.nan),
                'Dialogue Timing': exp_dialogue_timing.get(model, np.nan),
                'Dialogue TTR': exp_dialogue_ttr.get(model, np.nan),
                'Transition Entropy': exp_dialogue_trans_entropy.get(model, np.nan),
                'Semantic Convergence': exp_dialogue_convergence.get(model, np.nan),
                'Proxy KL': exp_proxy_kl.get(model, np.nan)
            })
            
        if exp_records:
            df_exp = pd.DataFrame(exp_records).set_index('Model')
            print(df_exp)
        else:
            print("No model data compared to human baseline for this experiment.")

