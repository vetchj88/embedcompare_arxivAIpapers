from collections import defaultdict
import json
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score


TAXONOMY_ALIGNMENT_PATH = "taxonomy_alignment.json"


def _load_taxonomy_alignment(path: str = TAXONOMY_ALIGNMENT_PATH) -> Dict[str, dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
    except FileNotFoundError:
        print(
            "Warning: taxonomy_alignment.json not found. Run data_scripts/align_taxonomy.py first to enrich taxonomy metadata."
        )
        return {}

    return {record["id"]: record for record in records}


def _aggregate_taxonomy(cluster_df: pd.DataFrame, taxonomy_map: Dict[str, dict], taxonomy_key: str) -> dict:
    label_totals: Dict[str, float] = defaultdict(float)
    classified = 0

    for _, row in cluster_df.iterrows():
        taxonomy_record = taxonomy_map.get(row["id"])
        if not taxonomy_record:
            continue

        distribution = taxonomy_record.get(taxonomy_key, {}).get("distribution", [])
        if not distribution:
            continue

        for item in distribution:
            label_totals[item["label"]] += float(item["score"])
        classified += 1

    top_labels = []
    if label_totals and classified:
        sorted_items = sorted(label_totals.items(), key=lambda kv: kv[1], reverse=True)
        top_labels = [
            {
                "label": label,
                "score": round(total / classified, 4),
            }
            for label, total in sorted_items[:3]
        ]

    result = {
        "classified_papers": int(classified),
        "top_labels": top_labels,
    }
    if top_labels:
        result["dominant_label"] = top_labels[0]
    return result


def _summarize_clusters(df: pd.DataFrame, taxonomy_map: Dict[str, dict]) -> dict:
    cluster_summary: Dict[str, dict] = {}
    for cluster_id, cluster_df in df.groupby("cluster_id"):
        cluster_summary[str(int(cluster_id))] = {
            "paper_count": int(len(cluster_df)),
            "taxonomies": {
                "ORKG": _aggregate_taxonomy(cluster_df, taxonomy_map, "orkg"),
                "CSO": _aggregate_taxonomy(cluster_df, taxonomy_map, "cso"),
            },
        }
    return cluster_summary

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
        "comparative_metrics": {},
        "cluster_taxonomy": {}
    }

    taxonomy_map = _load_taxonomy_alignment()

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

        summary["cluster_taxonomy"][name] = _summarize_clusters(df, taxonomy_map)

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