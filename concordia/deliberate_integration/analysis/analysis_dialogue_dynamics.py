import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from scipy.stats import entropy

METHODS = {
    'Human': 'human_minimal',
    'Minimal-Simulacra': 'simulacra_minimal',
    'Minimal-Simulacra-Partial': 'simulacra_partial_0.5_minimal',
    'Native-Simulacra': 'simulacra_native',
    'Native-Simulacra-Partial': 'simulacra_partial_0.5_native',
    'Playback-Simulacra': 'nodialogue_mask_native',
    'Playback-Human': 'nodialogue_mirror_native'
}

def load_all_dialogues():
    """Loads all dialogues using globbing, returning a combined DataFrame."""
    all_data = []
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
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
            
        matches = glob.glob(f"{base_dir}/events_log*.csv")
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
            df = df.dropna(subset=['content'])
            df = df[~df['agent_id'].astype(str).str.contains('System|GM|Orchestrator|deliberate', case=False, na=False)]
            
            if df.empty:
                continue
                
            df['Environment'] = env.upper()
            df['Method'] = matched_method
            all_data.append(df)
        except Exception as e:
            print(f"Failed to process dialogue {csv_path}: {e}")
            
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def analyze_information_density(df):
    """Method 3: Compute Type-Token Ratio (unique words / total words)."""
    print("Analyzing Information Density (Type-Token Ratio)...")
    results = []
    
    for (env, method, cohort), group in df.groupby(['Environment', 'Method', 'cohort_id']):
        all_text = " ".join(group['content'].astype(str).tolist())
        words = all_text.split()
        total_words = len(words)
        unique_words = len(set(words))
        
        ttr = unique_words / total_words if total_words > 0 else 0
        
        results.append({
            'Environment': env,
            'Method': method,
            'Cohort': cohort,
            'TTR': ttr,
            'Total_Words': total_words
        })
        
    return pd.DataFrame(results)

def analyze_conversational_dynamics(df):
    """Method 2: Analyze turn-taking patterns and transition entropy."""
    print("Analyzing Conversational Dynamics (Turn-Taking)...")
    results = []
    
    for (env, method, cohort), group in df.groupby(['Environment', 'Method', 'cohort_id']):
        # Ensure chronological order
        group = group.sort_values('time_index')
        
        # Get sequence of speakers
        speakers = group['agent_id'].astype(str).tolist()
        
        if len(speakers) < 2:
            continue
            
        # Count transitions
        transitions = []
        for i in range(len(speakers) - 1):
            transitions.append((speakers[i], speakers[i+1]))
            
        counter = Counter(transitions)
        
        # Compute entropy of transition distribution
        counts = np.array(list(counter.values()))
        probs = counts / counts.sum()
        ent = entropy(probs)
        
        # Also compute self-transition rate (agent speaking again)
        self_trans = sum(1 for a, b in transitions if a == b)
        self_rate = self_trans / len(transitions) if transitions else 0
        
        results.append({
            'Environment': env,
            'Method': method,
            'Cohort': cohort,
            'Transition_Entropy': ent,
            'Self_Transition_Rate': self_rate
        })
        
    return pd.DataFrame(results)

def analyze_semantic_convergence(df, model):
    """Method 1: Track variance of sentence embeddings within a cohort over time."""
    print("Analyzing Semantic Convergence...")
    results = []
    
    for (env, method, cohort), group in df.groupby(['Environment', 'Method', 'cohort_id']):
        group = group.sort_values('time_index')
        
        # Split into early and late phases (median split based on time_index or row count)
        mid_point = len(group) // 2
        if mid_point < 5: # Skip very short dialogues
            continue
            
        early_text = group['content'].astype(str).iloc[:mid_point].tolist()
        late_text = group['content'].astype(str).iloc[mid_point:].tolist()
        
        early_emb = model.encode(early_text)
        late_emb = model.encode(late_text)
        
        # Calculate variance (spread) in embedding space
        early_var = np.var(early_emb, axis=0).sum()
        late_var = np.var(late_emb, axis=0).sum()
        
        # If variance decreases, they converged
        convergence = early_var - late_var
        
        results.append({
            'Environment': env,
            'Method': method,
            'Cohort': cohort,
            'Early_Variance': early_var,
            'Late_Variance': late_var,
            'Convergence': convergence
        })
        
    return pd.DataFrame(results)

def analyze_tf_idf(df):
    """Method 4: Identify top words that distinguish LLM dialogue from Human dialogue."""
    print("Running TF-IDF Discriminator...")
    
    # We will compare 'Human' vs 'Minimal-Simulacra'
    human_text = " ".join(df[df['Method'] == 'Human']['content'].astype(str).tolist())
    llm_text = " ".join(df[df['Method'] == 'Minimal-Simulacra']['content'].astype(str).tolist())
    
    if not human_text or not llm_text:
        print("Missing either Human or LLM data for TF-IDF.")
        return
        
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X = vectorizer.fit_transform([human_text, llm_text])
    feature_names = vectorizer.get_feature_names_out()
    
    # X[0] is Human, X[1] is LLM
    human_tfidf = X[0].toarray()[0]
    llm_tfidf = X[1].toarray()[0]
    
    diff = llm_tfidf - human_tfidf
    
    # Top words for LLM
    top_llm_idx = np.argsort(diff)[-20:]
    top_llm_words = [(feature_names[i], diff[i]) for i in top_llm_idx]
    
    # Top words for Human
    top_human_idx = np.argsort(diff)[:20]
    top_human_words = [(feature_names[i], diff[i]) for i in top_human_idx]
    
    print("\nTop words over-represented in Minimal-Simulacra:")
    for word, score in reversed(top_llm_words):
        print(f"  {word}: {score:.4f}")
        
    print("\nTop words over-represented in Human:")
    for word, score in top_human_words:
        print(f"  {word}: {score:.4f}")

def main():
    df = load_all_dialogues()
    if df.empty:
        print("No dialogue data found.")
        return
        
    print(f"Loaded {len(df)} dialogue utterances.")
    
    # Ensure output directory exists
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'dynamics_plots')
    os.makedirs(results_dir, exist_ok=True)
    
    # Method 4: TF-IDF
    analyze_tf_idf(df)
    
    # Method 3: Information Density
    df_ttr = analyze_information_density(df)
    if not df_ttr.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Method', y='TTR', data=df_ttr)
        plt.title("Information Density (Type-Token Ratio) by Method")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "ttr_comparison.png"))
        plt.close()
        
    # Method 2: Conversational Dynamics
    df_turns = analyze_conversational_dynamics(df)
    if not df_turns.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Method', y='Transition_Entropy', data=df_turns)
        plt.title("Conversational Turn-Taking Entropy by Method")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "turn_entropy_comparison.png"))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Method', y='Self_Transition_Rate', data=df_turns)
        plt.title("Self-Transition Rate by Method")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "self_transition_comparison.png"))
        plt.close()
        
    # Method 1: Semantic Convergence
    print("Loading SentenceTransformer for semantic convergence...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df_conv = analyze_semantic_convergence(df, model)
    if not df_conv.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Method', y='Convergence', data=df_conv)
        plt.title("Semantic Convergence (Reduction in Variance) by Method")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "semantic_convergence_comparison.png"))
        plt.close()
        
    print(f"Advanced Dynamics Analysis Complete. Plots saved to {results_dir}")

if __name__ == "__main__":
    main()
