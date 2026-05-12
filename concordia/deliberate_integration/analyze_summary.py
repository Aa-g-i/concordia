import pandas as pd
import numpy as np

def compute_gini(x):
    x = np.array(x, dtype=np.float64)
    if np.amin(x) < 0:
        x -= np.amin(x)
    x += 0.0000001
    x = np.sort(x)
    index = np.arange(1, x.shape[0]+1)
    n = x.shape[0]
    return ((np.sum((2 * index - n  - 1) * x)) / (n * np.sum(x)))

datasets = ['f1_200', 'f1a_200', 'f2a_300']
results = []

for d in datasets:
    print(f"\n====================== {d.upper()} ======================")
    det_survey = f'concordia_output_{d}_data_deterministic_minimal/survey_responses.csv'
    mir_survey = f'concordia_output_{d}_data_human_minimal/survey_responses_human.csv'
    msk_survey = f'concordia_output_{d}_data_simulacra_minimal/survey_responses_simulacra.csv'
    
    # Surveys
    try:
        import os
        df_d = pd.read_csv(det_survey) if os.path.exists(det_survey) else pd.DataFrame()
        df_m = pd.read_csv(mir_survey) if os.path.exists(mir_survey) else pd.DataFrame()
        df_s = pd.read_csv(msk_survey) if os.path.exists(msk_survey) else pd.DataFrame()
        
        baseline_mean = df_d['response'].apply(pd.to_numeric, errors='coerce').mean() if not df_d.empty else 0
        mir_mean = df_m['response'].apply(pd.to_numeric, errors='coerce').mean() if not df_m.empty else 0
        msk_mean = df_s['response'].apply(pd.to_numeric, errors='coerce').mean() if not df_s.empty else 0
        
        print(f"SURVEY NUMERIC AVERAGES:")
        print(f"  Human Baseline: {baseline_mean:.2f}")
        if not df_m.empty: print(f"  Mirror Proxy:   {mir_mean:.2f} (Delta: {abs(baseline_mean - mir_mean):.2f})")
        if not df_s.empty: print(f"  Mask Proxy:     {msk_mean:.2f} (Delta: {abs(baseline_mean - msk_mean):.2f})")
        
        print(f"SURVEY MOST FREQUENT TEXT RESPONSE:")
        if not df_d.empty: print(f"  Human Baseline: {df_d[pd.to_numeric(df_d['response'], errors='coerce').isna()]['response'].mode()[0] if not df_d[pd.to_numeric(df_d['response'], errors='coerce').isna()].empty else 'N/A'}")
        if not df_m.empty: print(f"  Mirror Proxy:   {df_m[pd.to_numeric(df_m['response'], errors='coerce').isna()]['response'].mode()[0] if not df_m[pd.to_numeric(df_m['response'], errors='coerce').isna()].empty else 'N/A'}")
        if not df_s.empty: print(f"  Mask Proxy:     {df_s[pd.to_numeric(df_s['response'], errors='coerce').isna()]['response'].mode()[0] if not df_s[pd.to_numeric(df_s['response'], errors='coerce').isna()].empty else 'N/A'}")
        
    except Exception as e:
        print(f"Error reading surveys for {d}: {e}")

    # Events (Gini & wordcounts)
    det_ev = f'concordia_output_{d}_data_deterministic_minimal/events_log.csv'
    mir_ev = f'concordia_output_{d}_data_human_minimal/events_log_human.csv'
    msk_ev = f'concordia_output_{d}_data_simulacra_minimal/events_log_simulacra.csv'
    
    try:
        def get_structural(csv):
            import os
            if not os.path.exists(csv): return 0, 0
            df = pd.read_csv(csv)
            # drop non dialogue
            df = df.dropna(subset=['content'])
            df['word_count'] = df['content'].astype(str).str.split().str.len()
            verb = df.groupby('agent_id' if 'agent_id' in df.columns else 'participant_id')['word_count'].sum()
            if verb.empty:
                avg_words = 0
                gini = 0
            else:
                avg_words = verb.mean()
                gini = compute_gini(verb)
            return avg_words, gini
            
        d_words, d_gini = get_structural(det_ev)
        m_words, m_gini = get_structural(mir_ev)
        s_words, s_gini = get_structural(msk_ev)
        
        print(f"DIALOGUE GINI (Inequality - closer to 1 is more dominated by a few):")
        print(f"  Human Baseline: {d_gini:.3f}")
        print(f"  Mirror Proxy:   {m_gini:.3f}")
        print(f"  Mask Proxy:     {s_gini:.3f}")
        
        print(f"DIALOGUE VERBOSITY (Avg Words per Participant):")
        print(f"  Human Baseline: {d_words:.1f}")
        print(f"  Mirror Proxy:   {m_words:.1f}")
        print(f"  Mask Proxy:     {s_words:.1f}")

    except Exception as e:
        print(f"Error reading events for {d}: {e}")

print("Analysis Complete.")
