import pandas as pd
import numpy as np
import json
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score

def create_analysis_summary():
    """
    Loads all data, runs a full analysis, and saves the results to a single JSON file.
    """
    try:
        df_bge = pd.read_json('papers_bge_large.json')
        df_gte = pd.read_json('papers_gte_large.json')
        df_minilm = pd.read_json('papers_minilm.json')
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. {e}")
        return

    models_data = {
        "BGE-Large": df_bge,
        "GTE-Large": df_gte,
        "MiniLM": df_minilm
    }
    
    summary = {
        "model_metrics": {},
        "comparative_metrics": {}
    }

    # 1. Per-Model Metrics
    for name, df in models_data.items():
        coords = df[['x', 'y', 'z']].values
        labels = df['cluster_id'].values
        mask = labels != -1

        ch_score = "N/A"
        db_score = "N/A"

        if mask.sum() > 0 and len(np.unique(labels[mask])) > 1:
            coords_no_outliers = coords[mask]
            labels_no_outliers = labels[mask]
            ch_score = calinski_harabasz_score(coords_no_outliers, labels_no_outliers)
            db_score = davies_bouldin_score(coords_no_outliers, labels_no_outliers)

        # --- FIX IS HERE: Convert numpy types to standard python types ---
        summary["model_metrics"][name] = {
            "Num Clusters": int(len(df[mask]['cluster_id'].unique())),
            "Num Outliers": int((labels == -1).sum()),
            "Calinski-Harabasz Score": f"{ch_score:.2f}" if isinstance(ch_score, float) else ch_score,
            "Davies-Bouldin Score": f"{db_score:.3f}" if isinstance(db_score, float) else db_score
        }

    # 2. Comparative Metrics (Adjusted Rand Index)
    ari_bge_gte = adjusted_rand_score(df_bge['cluster_id'], df_gte['cluster_id'])
    ari_bge_minilm = adjusted_rand_score(df_bge['cluster_id'], df_minilm['cluster_id'])
    ari_gte_minilm = adjusted_rand_score(df_gte['cluster_id'], df_minilm['cluster_id'])
    
    summary["comparative_metrics"]["Adjusted Rand Index"] = {
        "BGE vs GTE": f"{ari_bge_gte:.4f}",
        "BGE vs MiniLM": f"{ari_bge_minilm:.4f}",
        "GTE vs MiniLM": f"{ari_gte_minilm:.4f}"
    }

    # 3. Save to file
    with open('analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
        
    print("Successfully generated 'analysis_summary.json'")

if __name__ == "__main__":
    create_analysis_summary()