import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import trustworthiness
from itertools import combinations, product

def run_deep_analysis():
    # --- Load Data ---
    try:
        df_bge = pd.read_json('papers_bge_large.json').rename(columns={'cluster_id': 'cluster_bge'})
        df_gte = pd.read_json('papers_gte_large.json').rename(columns={'cluster_id': 'cluster_gte'})
        df_minilm = pd.read_json('papers_minilm.json').rename(columns={'cluster_id': 'cluster_minilm'})
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure all JSON files are in the directory.")
        return

    # Merge dataframes for easier comparison
    df_merged = df_bge[['id', 'title', 'cluster_bge', 'x', 'y', 'z']].copy()
    df_merged = df_merged.rename(columns={'x':'x_bge', 'y':'y_bge', 'z':'z_bge'})
    df_merged['cluster_gte'] = df_gte['cluster_gte']
    df_merged['x_gte'] = df_gte['x']
    df_merged['y_gte'] = df_gte['y']
    df_merged['z_gte'] = df_gte['z']
    df_merged['cluster_minilm'] = df_minilm['cluster_minilm']
    df_merged['x_minilm'] = df_minilm['x']
    df_merged['y_minilm'] = df_minilm['y']
    df_merged['z_minilm'] = df_minilm['z']
    
    models = ['bge', 'gte', 'minilm']
    analysis_results = []

    print("\n" + "="*60)
    print("      Deep Spatial and Comparative Cluster Analysis")
    print("="*60 + "\n")

    # --- 1. Advanced Cluster Quality & Refinement Analysis ---
    print("--- 1. Cluster Quality and Refinement Metrics ---")
    for model in models:
        coords = df_merged[[f'x_{model}', f'y_{model}', f'z_{model}']].values
        labels = df_merged[f'cluster_{model}'].values
        
        # Exclude outliers for quality metrics
        mask = labels != -1
        if mask.sum() == 0 or len(np.unique(labels[mask])) < 2:
            print(f"\nModel: {model.upper()}")
            print("  Not enough clusters to calculate quality scores.")
            continue
            
        coords_no_outliers = coords[mask]
        labels_no_outliers = labels[mask]

        ch_score = calinski_harabasz_score(coords_no_outliers, labels_no_outliers)
        db_score = davies_bouldin_score(coords_no_outliers, labels_no_outliers)
        
        num_clusters = len(np.unique(labels_no_outliers))
        num_outliers = (labels == -1).sum()

        result = {
            "Model": model.upper(),
            "Num Clusters": num_clusters,
            "Num Outliers": num_outliers,
            "Calinski-Harabasz Score": f"{ch_score:.2f}",
            "Davies-Bouldin Score": f"{db_score:.3f}"
        }
        analysis_results.append(result)

    quality_df = pd.DataFrame(analysis_results)
    print(quality_df.to_string(index=False))
    print("\n* Calinski-Harabasz Score: Higher is better (ratio of between-cluster to within-cluster dispersion).")
    print("* Davies-Bouldin Score: Lower is better (similarity of a cluster with its most similar cluster).")
    
    # --- 2. Cluster Overlap Analysis (Jaccard Similarity) ---
    print("\n--- 2. Inter-Model Cluster Composition Comparison ---")

    fig = go.Figure()
    all_nodes = []
    all_links = []
    x_pos = [0.01, 0.5, 0.99]
    model_labels = ['BGE', 'GTE', 'MiniLM']

    # Define the flow BGE -> GTE -> MiniLM
    model_pairs = [('bge', 'gte'), ('gte', 'minilm')]
    
    # Create nodes for the Sankey diagram
    for i, model in enumerate(models):
        clusters = sorted(df_merged[f'cluster_{model}'].unique())
        for cluster_id in clusters:
            label = f"{model.upper()} C{cluster_id}"
            if cluster_id == -1:
                label = f"{model.upper()} Outliers"
            
            count = len(df_merged[df_merged[f'cluster_{model}'] == cluster_id])
            all_nodes.append(dict(label=f"{label} ({count})", x=x_pos[i], y=(clusters.index(cluster_id)+0.5)/len(clusters)))

    node_map = {node['label'].split(' ')[0] + " " + node['label'].split(' ')[1]: i for i, node in enumerate(all_nodes)}

    # Create links for the Sankey diagram
    for model1, model2 in model_pairs:
        cluster_pairs = product(
            df_merged[f'cluster_{model1}'].unique(),
            df_merged[f'cluster_{model2}'].unique()
        )
        
        for c1, c2 in cluster_pairs:
            source_label = f"{model1.upper()} C{c1}" if c1 != -1 else f"{model1.upper()} Outliers"
            target_label = f"{model2.upper()} C{c2}" if c2 != -1 else f"{model2.upper()} Outliers"
            
            source_idx = node_map[source_label.split(' ')[0] + " " + source_label.split(' ')[1]]
            target_idx = node_map[target_label.split(' ')[0] + " " + target_label.split(' ')[1]]
            
            value = len(df_merged[(df_merged[f'cluster_{model1}'] == c1) & (df_merged[f'cluster_{model2}'] == c2)])
            
            if value > 0:
                all_links.append(dict(source=source_idx, target=target_idx, value=value))

    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = [n['label'] for n in all_nodes],
          x = [n['x'] for n in all_nodes],
          y = [n['y'] for n in all_nodes]
        ),
        link = dict(
          source = [l['source'] for l in all_links],
          target = [l['target'] for l in all_links],
          value = [l['value'] for l in all_links]
      ))])

    fig.update_layout(
        title_text="Paper Flow Between Model Clusterings",
        font_size=12,
        height=800,
        annotations=[
            dict(x=0, y=1.05, text="<b>BGE-Large</b>", showarrow=False, font=dict(size=14)),
            dict(x=0.5, y=1.05, text="<b>GTE-Large</b>", showarrow=False, font=dict(size=14)),
            dict(x=1, y=1.05, text="<b>MiniLM</b>", showarrow=False, font=dict(size=14))
        ]
    )
    
    # Save to interactive HTML
    fig.write_html("cluster_overlap_analysis.html")
    print("\nGenerated interactive Sankey diagram 'cluster_overlap_analysis.html'.")
    print("This chart shows how papers are redistributed between the clusters of different models.")


if __name__ == "__main__":
    run_deep_analysis()