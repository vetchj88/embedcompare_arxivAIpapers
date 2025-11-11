import json
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

DATASET_KEYWORDS = {
    "imagenet": ["imagenet", "image-net"],
    "imagenet-21k": ["imagenet-21k", "imagenet21k"],
    "cifar-10": ["cifar-10", "cifar10"],
    "cifar-100": ["cifar-100", "cifar100"],
    "ms coco": ["ms coco", "mscoco", "coco dataset", "coco"],
    "squad": ["squad"],
    "humaneval": ["humaneval", "human-eval"],
    "mmlu": ["mmlu", "massive multitask"],
    "commonsenseqa": ["commonsenseqa"],
    "kitti": ["kitti"],
    "nuscenes": ["nuscene", "nuscenes"],
    "waymo": ["waymo"],
    "librispeech": ["librispeech"],
    "pile": ["the pile", "pile dataset", "pile"],
    "laion": ["laion", "laion-5b", "laion5b"],
    "c4": ["colossal clean crawled corpus", "c4 dataset"],
    "math": ["gsm8k", "math dataset", "math benchmark"],
    "scannet": ["scannet"],
}

TAXONOMY_RULES = {
    "Multimodal & Vision": ["vision", "image", "video", "multimodal", "visual"],
    "Language Models": ["language model", "llm", "gpt", "transformer", "generation"],
    "Robotics & Control": ["robot", "autonomous", "control", "manipulation", "navigation"],
    "Explainability": ["explain", "interpret", "explanation", "interpretability"],
    "Evaluation & Benchmarks": ["benchmark", "evaluation", "metric", "assessment"],
    "Healthcare & Bio": ["medical", "health", "biomedical", "diagnosis", "clinical"],
    "Finance & Economics": ["financial", "trading", "econom", "market", "risk"],
}


def extract_dataset_tags(text: str) -> set[str]:
    text_lower = text.lower()
    tags: set[str] = set()
    for canonical, variants in DATASET_KEYWORDS.items():
        if any(variant in text_lower for variant in variants):
            tags.add(canonical)
    return tags


def classify_taxonomy(text: str) -> set[str]:
    text_lower = text.lower()
    tags: set[str] = set()
    for label, keywords in TAXONOMY_RULES.items():
        if any(keyword in text_lower for keyword in keywords):
            tags.add(label)
    return tags


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    if not union:
        return 0.0
    return len(intersection) / len(union)


def normalized_distance_matrix(coords: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    max_distance = distances.max()
    if max_distance > 0:
        distances = distances / max_distance
    return distances


def temporal_alignment_matrix(dates: np.ndarray) -> np.ndarray:
    deltas = np.abs(dates[:, None] - dates[None, :]).astype("timedelta64[D]").astype(float)
    deltas = np.nan_to_num(deltas, nan=0.0)
    return np.exp(-deltas / 365.0)


def round_float(value: float, decimals: int = 3) -> float:
    return float(np.round(float(value), decimals))


def create_analysis_summary():
    """Load all model exports, compute metrics, and persist a consolidated summary JSON."""
    try:
        df_bge = pd.read_json("papers_bge_large.json")
        df_gte = pd.read_json("papers_gte_large.json")
        df_minilm = pd.read_json("papers_minilm.json")
    except FileNotFoundError as exc:
        print(f"Error: Could not find a required data file. {exc}")
        return

    models_data = {
        "BGE-Large": df_bge,
        "GTE-Large": df_gte,
        "MiniLM": df_minilm,
    }

    summary: dict[str, dict] = {
        "model_metrics": {},
        "comparative_metrics": {},
    }

    # 1. Per-model metrics
    for name, df in models_data.items():
        coords = df[["x", "y", "z"]].values
        labels = df["cluster_id"].values
        mask = labels != -1

        ch_score: float | str = "N/A"
        db_score: float | str = "N/A"

        if mask.sum() > 0 and len(np.unique(labels[mask])) > 1:
            coords_no_outliers = coords[mask]
            labels_no_outliers = labels[mask]
            ch_score = calinski_harabasz_score(coords_no_outliers, labels_no_outliers)
            db_score = davies_bouldin_score(coords_no_outliers, labels_no_outliers)

        summary["model_metrics"][name] = {
            "Num Clusters": int(len(df[mask]["cluster_id"].unique())),
            "Num Outliers": int((labels == -1).sum()),
            "Calinski-Harabasz Score": f"{ch_score:.2f}" if isinstance(ch_score, float) else ch_score,
            "Davies-Bouldin Score": f"{db_score:.3f}" if isinstance(db_score, float) else db_score,
        }

    # 2. Comparative metrics
    ari_bge_gte = adjusted_rand_score(df_bge["cluster_id"], df_gte["cluster_id"])
    ari_bge_minilm = adjusted_rand_score(df_bge["cluster_id"], df_minilm["cluster_id"])
    ari_gte_minilm = adjusted_rand_score(df_gte["cluster_id"], df_minilm["cluster_id"])

    summary["comparative_metrics"]["Adjusted Rand Index"] = {
        "BGE vs GTE": f"{ari_bge_gte:.4f}",
        "BGE vs MiniLM": f"{ari_bge_minilm:.4f}",
        "GTE vs MiniLM": f"{ari_gte_minilm:.4f}",
    }

    # 3. Explanation graph assembly
    merged = df_bge[
        ["id", "title", "authors", "date", "abstract", "cluster_id", "x", "y", "z"]
    ].copy()
    merged = merged.rename(
        columns={
            "cluster_id": "cluster_bge",
            "x": "x_bge",
            "y": "y_bge",
            "z": "z_bge",
        }
    )
    merged["date"] = merged["date"].astype(str)
    merged["cluster_gte"] = df_gte["cluster_id"]
    merged["cluster_minilm"] = df_minilm["cluster_id"]
    merged["x_gte"], merged["y_gte"], merged["z_gte"] = df_gte["x"], df_gte["y"], df_gte["z"]
    merged["x_minilm"], merged["y_minilm"], merged["z_minilm"] = (
        df_minilm["x"],
        df_minilm["y"],
        df_minilm["z"],
    )

    merged["parsed_date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["dataset_tags"] = merged["abstract"].apply(extract_dataset_tags)
    merged["taxonomy_tags"] = (merged["title"] + " " + merged["abstract"]).apply(
        classify_taxonomy
    )
    merged["author_set"] = merged["authors"].apply(
        lambda names: {author.strip() for author in names.split(",") if author.strip()}
    )
    merged["author_last_names"] = merged["author_set"].apply(
        lambda names: {name.split(" ")[-1].lower() for name in names if name}
    )
    merged["abstract_lower"] = merged["abstract"].str.lower()

    coords_bge = merged[["x_bge", "y_bge", "z_bge"]].values
    coords_gte = merged[["x_gte", "y_gte", "z_gte"]].values
    coords_minilm = merged[["x_minilm", "y_minilm", "z_minilm"]].values

    hybrid_matrix = 1.0 - np.mean(
        [
            normalized_distance_matrix(coords_bge),
            normalized_distance_matrix(coords_gte),
            normalized_distance_matrix(coords_minilm),
        ],
        axis=0,
    )

    date_array = merged["parsed_date"].values.astype("datetime64[D]")
    temporal_matrix = temporal_alignment_matrix(date_array)

    upper_i, upper_j = np.triu_indices(len(merged), k=1)
    hybrid_values = hybrid_matrix[upper_i, upper_j]
    candidate_indices = np.argsort(hybrid_values)[::-1][:400]

    edges: list[dict] = []
    seen_pairs: set[tuple[int, int]] = set()

    def build_edge(i: int, j: int) -> dict:
        hybrid_score = hybrid_matrix[i, j]
        temporal_score = temporal_matrix[i, j]

        authors_i = merged.iloc[i]["author_set"]
        authors_j = merged.iloc[j]["author_set"]
        datasets_i = merged.iloc[i]["dataset_tags"]
        datasets_j = merged.iloc[j]["dataset_tags"]
        taxonomy_i = merged.iloc[i]["taxonomy_tags"]
        taxonomy_j = merged.iloc[j]["taxonomy_tags"]

        author_overlap = jaccard_similarity(authors_i, authors_j)
        dataset_overlap = jaccard_similarity(datasets_i, datasets_j)
        taxonomy_overlap = jaccard_similarity(taxonomy_i, taxonomy_j)
        graph_score = 0.5 * author_overlap + 0.3 * dataset_overlap + 0.2 * taxonomy_overlap

        last_names_i = merged.iloc[i]["author_last_names"]
        last_names_j = merged.iloc[j]["author_last_names"]
        abstract_i = merged.iloc[i]["abstract_lower"]
        abstract_j = merged.iloc[j]["abstract_lower"]
        cites_i = any(last_name in abstract_j for last_name in last_names_i if len(last_name) > 2)
        cites_j = any(last_name in abstract_i for last_name in last_names_j if len(last_name) > 2)
        mutual_citations = cites_i and cites_j

        return {
            "source_id": merged.iloc[i]["id"],
            "target_id": merged.iloc[j]["id"],
            "scores": {
                "hybrid_similarity": round_float(hybrid_score),
                "temporal_alignment": round_float(temporal_score),
                "graph_affinity": round_float(graph_score),
            },
            "signals": {
                "shared_datasets": sorted(datasets_i.intersection(datasets_j)),
                "taxonomy_matches": sorted(taxonomy_i.intersection(taxonomy_j)),
                "mutual_citations": bool(mutual_citations),
                "shared_authors": sorted(authors_i.intersection(authors_j)),
            },
            "metadata": {
                "source": {
                    "title": merged.iloc[i]["title"],
                    "authors": merged.iloc[i]["authors"],
                    "date": str(merged.iloc[i]["date"]),
                    "clusters": {
                        "BGE-Large": int(merged.iloc[i]["cluster_bge"]),
                        "GTE-Large": int(merged.iloc[i]["cluster_gte"]),
                        "MiniLM": int(merged.iloc[i]["cluster_minilm"]),
                    },
                },
                "target": {
                    "title": merged.iloc[j]["title"],
                    "authors": merged.iloc[j]["authors"],
                    "date": str(merged.iloc[j]["date"]),
                    "clusters": {
                        "BGE-Large": int(merged.iloc[j]["cluster_bge"]),
                        "GTE-Large": int(merged.iloc[j]["cluster_gte"]),
                        "MiniLM": int(merged.iloc[j]["cluster_minilm"]),
                    },
                },
            },
        }

    for idx in candidate_indices:
        i = int(upper_i[idx])
        j = int(upper_j[idx])
        pair_key = (min(i, j), max(i, j))
        if pair_key in seen_pairs:
            continue
        edges.append(build_edge(i, j))
        seen_pairs.add(pair_key)

    dataset_map: dict[str, list[int]] = {}
    for idx, tags in enumerate(merged["dataset_tags"]):
        for tag in tags:
            dataset_map.setdefault(tag, []).append(idx)

    for indices in dataset_map.values():
        if len(indices) < 2:
            continue
        for i, j in combinations(indices, 2):
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen_pairs:
                continue
            edges.append(build_edge(i, j))
            seen_pairs.add(pair_key)

    def edge_priority(edge: dict) -> float:
        base = edge["scores"].get("hybrid_similarity", 0.0)
        bonus = 0.0
        if edge["signals"].get("shared_datasets"):
            bonus += 0.05
        if edge["signals"].get("taxonomy_matches"):
            bonus += 0.02
        if edge["signals"].get("mutual_citations"):
            bonus += 0.04
        if edge["signals"].get("shared_authors"):
            bonus += 0.01
        return base + bonus

    edges.sort(key=edge_priority, reverse=True)
    edges = edges[:150]

    summary["explanation_graph"] = {
        "schema_version": "1.0",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "edge_count": len(edges),
        "topology": {
            "nodes": int(len(merged)),
            "max_edges": 150,
        },
        "attribution_models": {
            "hybrid_similarity": "1 - mean normalized Euclidean distance across BGE-Large, GTE-Large, MiniLM embeddings.",
            "temporal_alignment": "Exponential decay of publication gap in days (exp(-|Δdays| / 365)).",
            "graph_affinity": "Weighted Jaccard overlap of authors (50%), dataset tags (30%), taxonomy tags (20%).",
        },
        "edges": edges,
    }

    with open("analysis_summary.json", "w") as handle:
        json.dump(summary, handle, indent=4)

    print("Successfully generated 'analysis_summary.json'")


if __name__ == "__main__":
    create_analysis_summary()
