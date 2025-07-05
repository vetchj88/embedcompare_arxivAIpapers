import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, silhouette_score

# --- Main Analysis Function ---
def analyze_clustering_results():
    """
    Loads clustering data, performs a full statistical and comparative
    analysis, and prints the results.
    """
    # --- Load the datasets ---
    try:
        df_bge = pd.read_json('papers_bge_large.json')
        df_gte = pd.read_json('papers_gte_large.json')
        df_minilm = pd.read_json('papers_minilm.json')
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file.")
        print(f"Details: {e}")
        print("Please ensure all three JSON files (papers_bge_large.json, papers_gte_large.json, papers_minilm.json) are in the same directory as this script.")
        return # Exit the function if files are not found

    # --- 1. Basic Cluster Statistics ---
    def get_cluster_stats(df, model_name):
        """Calculates basic clustering statistics for a given model's dataframe."""
        clusters = df[df['cluster_id'] != -1]['cluster_id']
        cluster_counts = clusters.value_counts()
        stats = {
            "Model": model_name,
            "Number of Clusters": len(cluster_counts),
            "Number of Outliers": (df['cluster_id'] == -1).sum(),
            "Papers in Clusters": len(clusters),
            "Largest Cluster": cluster_counts.max() if not cluster_counts.empty else 0,
            "Smallest Cluster": cluster_counts.min() if not cluster_counts.empty else 0,
            "Avg. Cluster Size": round(cluster_counts.mean(), 2) if not cluster_counts.empty else 0,
        }
        return stats, cluster_counts

    bge_stats, bge_cluster_counts = get_cluster_stats(df_bge, 'BGE-Large')
    gte_stats, gte_cluster_counts = get_cluster_stats(df_gte, 'GTE-Large')
    minilm_stats, minilm_cluster_counts = get_cluster_stats(df_minilm, 'MiniLM')

    # --- 2. Clustering Refinement/Quality Analysis ---
    def calculate_silhouette(df):
        """Calculates the Silhouette Score for a given clustering."""
        labels = df['cluster_id']
        coords = df[['x', 'y', 'z']]
        
        # Exclude outliers from the score calculation, as they are unclustered by definition
        non_outlier_mask = labels != -1
        
        # Silhouette score is only defined if there is more than 1 cluster
        if non_outlier_mask.sum() > 0:
            labels_no_outliers = labels[non_outlier_mask]
            coords_no_outliers = coords[non_outlier_mask]
            if len(np.unique(labels_no_outliers)) > 1:
                return silhouette_score(coords_no_outliers, labels_no_outliers)
        return "N/A"

    bge_stats['Silhouette Score'] = calculate_silhouette(df_bge)
    gte_stats['Silhouette Score'] = calculate_silhouette(df_gte)
    minilm_stats['Silhouette Score'] = calculate_silhouette(df_minilm)

    # --- 3. Inter-Model Cluster Agreement ---
    ari_bge_gte = adjusted_rand_score(df_bge['cluster_id'], df_gte['cluster_id'])
    ari_bge_minilm = adjusted_rand_score(df_bge['cluster_id'], df_minilm['cluster_id'])
    ari_gte_minilm = adjusted_rand_score(df_gte['cluster_id'], df_minilm['cluster_id'])

    # --- 4. Print Combined Results ---
    print("\n" + "="*50)
    print("      Quantitative Comparison of Clustering Results")
    print("="*50 + "\n")

    # Create and format the final stats DataFrame
    stats_df = pd.DataFrame([bge_stats, gte_stats, minilm_stats])
    stats_df['Silhouette Score'] = stats_df['Silhouette Score'].apply(
        lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else x
    )
    print("--- Overall Model Performance ---\n")
    print(stats_df.to_string(index=False))
    print("\n* Silhouette Score: Measures cluster density and separation. Higher is better (range -1 to 1).")
    print("* Outliers: Papers not assigned to any cluster (ID -1).\n")

    # Create the ARI DataFrame
    ari_data = {
        'Model Comparison': ['BGE-Large vs. GTE-Large', 'BGE-Large vs. MiniLM', 'GTE-Large vs. MiniLM'],
        'Adjusted Rand Index (ARI)': [f"{ari_bge_gte:.4f}", f"{ari_bge_minilm:.4f}", f"{ari_gte_minilm:.4f}"]
    }
    ari_df = pd.DataFrame(ari_data)
    print("\n--- Inter-Model Clustering Agreement ---\n")
    print(ari_df.to_string(index=False))
    print("\n* Adjusted Rand Index: Measures similarity between two clusterings. 1.0 is identical, 0.0 is random.\n")

    # --- 5. Visualize and Save Cluster Size Distributions ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle('Distribution of Papers per Cluster (Excluding Outliers)', fontsize=16, y=1.02)

    sns.barplot(ax=axes[0], x=bge_cluster_counts.index, y=bge_cluster_counts.values, palette='viridis', order=bge_cluster_counts.index)
    axes[0].set_title(f'BGE-Large ({bge_stats["Number of Clusters"]} clusters)')
    axes[0].set_xlabel('Cluster ID')
    axes[0].set_ylabel('Number of Papers')

    sns.barplot(ax=axes[1], x=gte_cluster_counts.index, y=gte_cluster_counts.values, palette='plasma', order=gte_cluster_counts.index)
    axes[1].set_title(f'GTE-Large ({gte_stats["Number of Clusters"]} clusters)')
    axes[1].set_xlabel('Cluster ID')

    sns.barplot(ax=axes[2], x=minilm_cluster_counts.index, y=minilm_cluster_counts.values, palette='magma', order=minilm_cluster_counts.index)
    axes[2].set_title(f'MiniLM ({minilm_stats["Number of Clusters"]} clusters)')
    axes[2].set_xlabel('Cluster ID')

    plt.tight_layout()
    plt.savefig('cluster_size_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n[SUCCESS] Generated 'cluster_size_distribution.png' to visualize cluster sizes.\n")


# --- Execute the analysis ---
if __name__ == "__main__":
    analyze_clustering_results()