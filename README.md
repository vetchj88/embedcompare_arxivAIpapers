# embedcompare_arxivAIpapers
A visual analysis tool comparing three sentences transformers effects on clustering with UMAP+HDBSCAN
![image](https://github.com/user-attachments/assets/55e96c76-21de-4351-ae0a-4e5ebe660653)

Comparative Analysis of Sentence Transformers

This project provides a visual and quantitative comparison of how three different sentence transformer models understand and cluster the same set of 500 AI research papers from arXiv. It's an interactive, web-based tool built with three.js that renders the semantic space of each model's embeddings as a 3D point cloud.

📜 Project Overview

The core goal of this project was to explore how different AI models create "semantic fingerprints" of complex data. We fetched 500 recent AI research papers from arXiv and processed their abstracts through three distinct sentence transformer models:

    BGE-Large (BAAI/bge-large-en-v1.5)

    GTE-Large (thenlper/gte-large)

    all-MiniLM-L6-v2 (sentence-transformers/all-MiniLM-L6-v2)

The resulting high-dimensional embeddings were then reduced to 3D coordinates using UMAP and clustered using HDBSCAN. This web application renders those 3D coordinates as interactive point clouds, allowing for a direct visual comparison of each model's clustering behavior.

✨ Features

    Interactive 3D Visualization: A smooth, responsive 3D environment built with three.js. Orbit, pan, and zoom to explore the data from any angle.

    Side-by-Side Comparison: The point clouds for all three models are rendered in their own distinct spaces, making it easy to compare their overall shape and cluster density.

    Detailed Tooltips: Hover over any point in the clouds to view the corresponding paper's title, authors, publication date, and a direct link to the arXiv page.

    Quantitative Analysis Panel: The GUI provides a detailed breakdown of each model's clustering performance, including:

        Number of clusters and outliers found.

        Calinski-Harabasz and Davies-Bouldin scores to measure cluster quality.

        Adjusted Rand Index (ARI) to measure the agreement between the clustering solutions of different models.

    Dynamic Controls: Toggle the visibility of each point cloud to focus on specific models.

🛠️ Tech Stack

    Data Fetching & Processing: Python, arxiv, pandas

    AI & ML: sentence-transformers, umap-learn, hdbscan, scikit-learn

    Frontend Visualization: JavaScript, three.js, lil-gui

    Local Server: A simple HTTP server is required to handle CORS policy for loading local files.

🚀 Getting Started

To run this project locally, follow these steps.

Prerequisites

You need a modern web browser that supports WebGL. No other dependencies are required to simply view the visualization.

Installation & Setup to run the program with provided data:

    Clone the repository:
    Bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

File Structure: Ensure your project directory is structured correctly with all the necessary data files at the root level:

/
├── index.html
├── main.js
├── style.css
├── papers_bge_large.json
├── papers_gte_large.json
├── papers_minilm.json
├── analysis_summary.json
├── circle.png
├── square.png
└── triangle.png

Run a Local Server: Due to browser security policies (CORS), you cannot open index.html directly from the file system. You must serve it from a local web server. The easiest way is using Python's built-in server.

If you have Python 3 installed:
Bash

python -m http.server

If you have Python 2 installed:
Bash

    python -m SimpleHTTPServer

    Alternatively, you can use the Live Server extension in Visual Studio Code.

    View the Application: Open your web browser and navigate to http://localhost:8000. The application should load and display the three point clouds.
📈 The Analytical Pipeline

This project's backend data processing involved several distinct steps:

    Data Ingestion (1_fetch_papers.py): Fetched metadata for 500 AI papers from the arXiv API.

    Embedding and Storage (2_embed_papers.py):

        Paper abstracts were embedded using each of the three sentence transformer models.

        The resulting vector embeddings and paper metadata were stored in a persistent ChromaDB vector database. A separate collection was created for each model.

    Analysis and Export (3_analyze_embeddings.py):

        For each model's collection, all embeddings were retrieved from ChromaDB.

        UMAP was used to reduce the dimensionality of the embeddings to 3D.

        HDBSCAN was run on the 3D coordinates to find dense clusters of semantically similar papers.

        The final data (metadata, 3D coordinates, cluster ID) was exported to the .json files used by the frontend.

    Quantitative Analysis (4_run_analysis.py, 5_deep_analysis.py):

        Cluster quality scores and the Adjusted Rand Index were calculated to create the data for analysis_summary.json.

    Visualization (main.js): The final .json data files are loaded and rendered in the browser using three.js.

    Heterogeneous Graph Construction (data_scripts/build_hetero_graph.py):

        Local metadata from papers_metadata.csv is merged with OpenAlex enrichments (concepts, venues, citations, keywords).

        Nodes are created for papers, authors, venues, topical fields, and extracted keyphrases. Each node keeps lightweight metadata.

        Typed edges include citation (paper→paper), co-author (author↔author), venue membership (paper→venue), co-citation (paper↔paper pairs sharing references), and keyphrase overlap (paper↔paper pairs sharing extracted keyphrases).

        The resulting schema and summary statistics are written to graph_data/hetero_graph.json and graph_data/graph_summary.json for downstream consumption.

    Graph Representation Learning (models/hetero_gnn.py):

        A two-layer HAN-style backbone propagates information across the heterogeneous graph using ELU activations.

        Paper nodes start from multi-hot keyphrase/field features, while other node types learn embeddings of size 128 that are updated during message passing.

        Training aligns paper representations with section-aware vectors stored in ChromaDB (collection: paper_sections) through an MSE objective, regularized by an L2 prior on all node embeddings.

        Default hyperparameters: hidden dimension = 256, output dimension = 256 (auto-matched to section vectors when available), epochs = 200, learning rate = 1e-3, weight decay = 1e-5, alignment weight = 1.0, L2 regularization weight = 1e-4, diagnostics every 10 epochs.

        Learned paper embeddings are persisted back to ChromaDB in the papers_hetero_hgnn collection and can optionally be exported as JSON for offline fusion.

To run the new graph workflow end-to-end:

```
python data_scripts/build_hetero_graph.py --metadata papers_metadata.csv --output-dir graph_data --mailto you@example.com
python models/hetero_gnn.py --graph graph_data/hetero_graph.json --chroma-path ./chroma_db --section-collection paper_sections --output-collection papers_hetero_hgnn
```
