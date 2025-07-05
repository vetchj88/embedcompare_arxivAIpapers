# embedcompare_arxivAIpapers
A visual analysis tool comparing three sentences transformers effects on clustering with UMAP+HDBSCAN

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

Installation & Setup

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

For those interested in the data generation process, the project followed these main steps:

    Data Ingestion (1_fetch_papers.py): Fetched metadata for 500 AI papers from the arXiv API.

    Embedding & Clustering (2_embed_papers.py, 3_analyze_embeddings.py): Generated vector embeddings for each paper's abstract using the three models. UMAP was used for dimensionality reduction to 3D, and HDBSCAN was used to identify clusters.

    Quantitative Analysis (4_run_analysis.py, 5_deep_analysis.py): Calculated cluster quality scores and the Adjusted Rand Index to create the data for the analysis_summary.json file.

    Visualization (main.js): The final data is loaded and rendered in the browser using three.js.
