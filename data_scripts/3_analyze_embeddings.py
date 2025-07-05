# 3_analyze_embeddings.py
import chromadb
import numpy as np
import umap
import hdbscan
import json

# --- Configuration ---
MODELS = ["bge_large", "gte_large", "minilm"]
DB_PATH = "./chroma_db"

# --- Initialize ChromaDB Client ---
client = chromadb.PersistentClient(path=DB_PATH)

# --- Loop Through Collections and Analyze ---
for model_name in MODELS:
    collection_name = f"papers_{model_name}"
    print(f"Analyzing collection: {collection_name}")

    collection = client.get_collection(name=collection_name)

    # Pull all embeddings and metadata from the database
    data = collection.get(include=["embeddings", "metadatas"])
    embeddings = np.array(data['embeddings'])
    metadata = data['metadatas']

    # --- UMAP Dimensionality Reduction ---
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    coords_3d = reducer.fit_transform(embeddings)

    # --- HDBSCAN Clustering ---
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    cluster_labels = clusterer.fit_predict(coords_3d)

    # --- Package for Export ---
    output_data = []
    for i, paper_meta in enumerate(metadata):
        output_data.append({
            **paper_meta, # Unpack original metadata
            'x': float(coords_3d[i, 0]),
            'y': float(coords_3d[i, 1]),
            'z': float(coords_3d[i, 2]),
            'cluster_id': int(cluster_labels[i])
        })

    # --- Export for Visualization (Step 4) ---
    output_filename = f"papers_{model_name}.json"
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Analysis complete. Results saved to {output_filename}\n")

print("All analyses are complete and JSON files have been exported.")