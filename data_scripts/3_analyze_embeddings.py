"""Hybrid graph driven layout analysis for arXiv AI papers.

This module replaces the original UMAP+HDBSCAN pipeline with a
graph-first workflow that:

1. Builds a hybrid similarity graph that blends textual, author, date,
   and model-specific clustering signals.
2. Generates low-dimensional layouts using PaCMAP and a spectral
   StarMAP approximation.
3. Persists layout coordinates and metadata back into the
   ``papers_*.json`` snapshot files.
4. Records layout hyper-parameters and quality metrics for
   reproducibility in ``analysis_summary.json``.

The script assumes the ``papers_*.json`` files already contain the paper
metadata fetched in the first stage of the pipeline.  The JSON files are
updated in-place with the newly computed layouts.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pacmap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import SpectralEmbedding, trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

# --- Configuration -----------------------------------------------------------------

MODELS: Sequence[str] = ("bge_large", "gte_large", "minilm")
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "bge_large": "BGE-Large",
    "gte_large": "GTE-Large",
    "minilm": "MiniLM",
}

# Graph construction and layout parameters
@dataclass(frozen=True)
class GraphConfig:
    n_neighbors: int = 12
    text_weight: float = 0.6
    metadata_weight: float = 0.25
    cluster_weight: float = 0.15
    author_weight: float = 0.6
    time_scale_days: float = 365.0
    random_state: int = 42


@dataclass(frozen=True)
class LayoutParams:
    n_components: int = 3
    pacmap_mn_ratio: float = 0.5
    pacmap_fp_ratio: float = 2.0
    pacmap_lr: float = 1.0
    pacmap_iters: Tuple[int, int, int] = (100, 100, 250)
    metric_neighbors: int = 15


GRAPH_CONFIG = GraphConfig()
LAYOUT_PARAMS = LayoutParams()
SUMMARY_PATH = Path("analysis_summary.json")


# --- Utility functions --------------------------------------------------------------

def continuity(high_dim: np.ndarray, low_dim: np.ndarray, n_neighbors: int) -> float:
    """Compute the continuity metric between high- and low-dimensional spaces.

    Continuity rewards preservation of high-dimensional neighbours in the
    low-dimensional embedding.  The implementation follows equation (4)
    from Venna & Kaski (2001).
    """

    n_samples = high_dim.shape[0]
    if n_samples <= n_neighbors:
        raise ValueError("n_neighbors must be smaller than the number of samples")

    # Rank distances in both spaces.
    high_dist = pairwise_distances(high_dim, metric="euclidean")
    low_dist = pairwise_distances(low_dim, metric="euclidean")

    # argsort twice to obtain ranks, excluding self (rank 0)
    high_rank = np.argsort(np.argsort(high_dist, axis=1), axis=1)
    low_rank = np.argsort(np.argsort(low_dist, axis=1), axis=1)

    high_neighbors = np.argsort(high_dist, axis=1)[:, 1 : n_neighbors + 1]
    low_neighbors = np.argsort(low_dist, axis=1)[:, 1 : n_neighbors + 1]
    low_neighbor_sets = [set(row) for row in low_neighbors]

    accum = 0.0
    for i in range(n_samples):
        missing = [j for j in high_neighbors[i] if j not in low_neighbor_sets[i]]
        if not missing:
            continue
        ranks_low = low_rank[i, missing] + 1  # convert to 1-indexed ranks
        accum += np.sum(ranks_low - n_neighbors)

    denom = n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1)
    if denom <= 0:
        raise ValueError("Invalid denominator encountered while computing continuity")
    score = 1.0 - (2.0 / denom) * accum
    return float(score)


def prepare_text_features(documents: Sequence[str]) -> np.ndarray:
    """Tokenise abstracts into TF-IDF vectors for layout construction."""

    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix.astype(np.float32).toarray()


def parse_authors(authors_field: str) -> List[str]:
    if not authors_field:
        return []
    return [author.strip() for author in authors_field.split(",") if author.strip()]


def authors_similarity(authors_a: Sequence[str], authors_b: Sequence[str]) -> float:
    set_a = set(authors_a)
    set_b = set(authors_b)
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def time_similarity(date_a: str, date_b: str, scale_days: float) -> float:
    # Work with ordinal dates for stability.
    ordinal_a = np.datetime64(date_a).astype("int64")
    ordinal_b = np.datetime64(date_b).astype("int64")
    diff_days = abs(int(ordinal_a - ordinal_b))
    return float(math.exp(-diff_days / scale_days))


def build_hybrid_graph(
    features: np.ndarray,
    metadata: Sequence[Dict[str, object]],
    cluster_labels: Sequence[int],
    config: GraphConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a hybrid similarity graph for the provided snapshot.

    Returns
    -------
    pair_indices : np.ndarray
        ``(n_samples * n_neighbors, 2)`` array of neighbour pairs.
    adjacency_matrix : np.ndarray
        Symmetric dense affinity matrix for spectral layouts.
    """

    n_samples = features.shape[0]
    nn = NearestNeighbors(
        n_neighbors=config.n_neighbors + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(features)
    distances, neighbor_indices = nn.kneighbors(features)

    # Discard self-neighbours (first column)
    neighbor_indices = neighbor_indices[:, 1 : config.n_neighbors + 1]
    distances = distances[:, 1 : config.n_neighbors + 1]
    text_similarity = 1.0 - distances

    metadata_similarity = np.zeros_like(text_similarity, dtype=np.float32)
    cluster_bonus = np.zeros_like(text_similarity, dtype=np.float32)

    for i in range(n_samples):
        authors_i = parse_authors(metadata[i].get("authors", ""))
        date_i = metadata[i].get("date", "1970-01-01")
        cluster_i = cluster_labels[i]

        for pos, neighbor_idx in enumerate(neighbor_indices[i]):
            authors_j = parse_authors(metadata[int(neighbor_idx)].get("authors", ""))
            date_j = metadata[int(neighbor_idx)].get("date", "1970-01-01")
            metadata_similarity[i, pos] = (
                config.author_weight * authors_similarity(authors_i, authors_j)
                + (1.0 - config.author_weight)
                * time_similarity(date_i, date_j, config.time_scale_days)
            )
            if cluster_i != -1 and cluster_i == cluster_labels[int(neighbor_idx)]:
                cluster_bonus[i, pos] = 1.0

    combined = (
        config.text_weight * text_similarity
        + config.metadata_weight * metadata_similarity
        + config.cluster_weight * cluster_bonus
    )

    # Normalise per node to [0, 1]
    min_vals = combined.min(axis=1, keepdims=True)
    max_vals = combined.max(axis=1, keepdims=True)
    span = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
    normalised = (combined - min_vals) / span

    sources = np.repeat(np.arange(n_samples), config.n_neighbors)
    targets = neighbor_indices.reshape(-1)
    pair_indices = np.column_stack([sources, targets]).astype(np.int32)
    pair_weights = normalised.reshape(-1).astype(np.float32)

    adjacency = np.zeros((n_samples, n_samples), dtype=np.float32)
    adjacency[sources, targets] = pair_weights
    adjacency[targets, sources] = np.maximum(adjacency[targets, sources], pair_weights)

    return pair_indices, adjacency


def run_pacmap_layout(
    features: np.ndarray,
    pair_neighbors: np.ndarray,
    params: LayoutParams,
    seed: int,
) -> np.ndarray:
    model = pacmap.PaCMAP(
        n_components=params.n_components,
        n_neighbors=GRAPH_CONFIG.n_neighbors,
        MN_ratio=params.pacmap_mn_ratio,
        FP_ratio=params.pacmap_fp_ratio,
        lr=params.pacmap_lr,
        num_iters=params.pacmap_iters,
        pair_neighbors=pair_neighbors,
        random_state=seed,
        verbose=False,
    )
    embedding = model.fit_transform(features, init="pca")
    return embedding.astype(np.float32)


def run_starmap_layout(adjacency: np.ndarray, params: LayoutParams, seed: int) -> np.ndarray:
    # Add a small epsilon to ensure positivity and avoid disconnected graphs causing failures.
    affinity = adjacency.copy()
    max_val = affinity.max()
    if max_val > 0:
        affinity /= max_val
    np.fill_diagonal(affinity, 1.0)

    spectral = SpectralEmbedding(
        n_components=params.n_components,
        affinity="precomputed",
        random_state=seed,
    )
    coords = spectral.fit_transform(affinity)
    return coords.astype(np.float32)


def update_snapshot_with_layout(
    papers: List[Dict[str, object]],
    pacmap_coords: np.ndarray,
    starmap_coords: np.ndarray,
) -> None:
    for idx, record in enumerate(papers):
        pac = pacmap_coords[idx]
        sta = starmap_coords[idx]
        record["x"] = float(pac[0])
        record["y"] = float(pac[1])
        record["z"] = float(pac[2])
        record["layouts"] = {
            "pacmap": {
                "x": float(pac[0]),
                "y": float(pac[1]),
                "z": float(pac[2]),
            },
            "starmap": {
                "x": float(sta[0]),
                "y": float(sta[1]),
                "z": float(sta[2]),
            },
        }
        record["layout_algorithm"] = "pacmap"
        record["layout_version"] = "hybrid-graph-v1"


# --- Main pipeline ------------------------------------------------------------------

def analyse_snapshots() -> None:
    np.random.seed(GRAPH_CONFIG.random_state)

    summary: Dict[str, object] = {}
    if SUMMARY_PATH.exists():
        with SUMMARY_PATH.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)

    layout_summary = {
        "hyperparameters": {
            "graph": {
                "n_neighbors": GRAPH_CONFIG.n_neighbors,
                "text_weight": GRAPH_CONFIG.text_weight,
                "metadata_weight": GRAPH_CONFIG.metadata_weight,
                "cluster_weight": GRAPH_CONFIG.cluster_weight,
                "author_weight": GRAPH_CONFIG.author_weight,
                "time_scale_days": GRAPH_CONFIG.time_scale_days,
            },
            "pacmap": {
                "n_components": LAYOUT_PARAMS.n_components,
                "MN_ratio": LAYOUT_PARAMS.pacmap_mn_ratio,
                "FP_ratio": LAYOUT_PARAMS.pacmap_fp_ratio,
                "lr": LAYOUT_PARAMS.pacmap_lr,
                "num_iters": LAYOUT_PARAMS.pacmap_iters,
                "random_state": GRAPH_CONFIG.random_state,
            },
            "starmap": {
                "n_components": LAYOUT_PARAMS.n_components,
                "solver": "SpectralEmbedding",
                "random_state": GRAPH_CONFIG.random_state,
            },
        },
        "quality": {},
    }

    for model_name in MODELS:
        snapshot_path = Path(f"papers_{model_name}.json")
        if not snapshot_path.exists():
            print(f"[WARN] Snapshot {snapshot_path} is missing; skipping.")
            continue

        with snapshot_path.open("r", encoding="utf-8") as fh:
            papers: List[Dict[str, object]] = json.load(fh)

        documents = [paper.get("abstract", "") for paper in papers]
        features = prepare_text_features(documents)
        cluster_labels = [int(paper.get("cluster_id", -1)) for paper in papers]

        pair_indices, adjacency = build_hybrid_graph(
            features, papers, cluster_labels, GRAPH_CONFIG
        )

        pacmap_coords = run_pacmap_layout(features, pair_indices, LAYOUT_PARAMS, GRAPH_CONFIG.random_state)
        starmap_coords = run_starmap_layout(adjacency, LAYOUT_PARAMS, GRAPH_CONFIG.random_state)

        tw_pacmap = trustworthiness(features, pacmap_coords, n_neighbors=LAYOUT_PARAMS.metric_neighbors)
        cont_pacmap = continuity(features, pacmap_coords, n_neighbors=LAYOUT_PARAMS.metric_neighbors)
        tw_starmap = trustworthiness(features, starmap_coords, n_neighbors=LAYOUT_PARAMS.metric_neighbors)
        cont_starmap = continuity(features, starmap_coords, n_neighbors=LAYOUT_PARAMS.metric_neighbors)

        update_snapshot_with_layout(papers, pacmap_coords, starmap_coords)
        with snapshot_path.open("w", encoding="utf-8") as fh:
            json.dump(papers, fh, indent=2)

        layout_summary["quality"][MODEL_DISPLAY_NAMES.get(model_name, model_name)] = {
            "graph": {
                "nodes": len(papers),
                "edges": int(pair_indices.shape[0]),
            },
            "PaCMAP": {
                "trustworthiness": round(float(tw_pacmap), 6),
                "continuity": round(float(cont_pacmap), 6),
            },
            "StarMAP": {
                "trustworthiness": round(float(tw_starmap), 6),
                "continuity": round(float(cont_starmap), 6),
            },
        }

        print(f"[INFO] Updated {snapshot_path} with hybrid graph layouts.")

    summary["layout"] = layout_summary
    with SUMMARY_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=4)
    print(f"[INFO] Layout metadata saved to {SUMMARY_PATH}.")


if __name__ == "__main__":
    analyse_snapshots()
