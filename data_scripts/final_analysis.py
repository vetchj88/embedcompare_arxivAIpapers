from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score


TAXONOMY_ALIGNMENT_PATH = "taxonomy_alignment.json"
KNOWLEDGE_GRAPH_PATH = "knowledge_graph.json"
JOINT_TAXONOMY_EMBEDDINGS_PATH = "joint_taxonomy_embeddings.json"
PAPER_EXPLANATIONS_PATH = Path("paper_explanations.json")


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


def _load_knowledge_graph(path: str = KNOWLEDGE_GRAPH_PATH) -> Dict[str, dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            graph_payload = json.load(handle)
    except FileNotFoundError:
        print(
            "Warning: knowledge_graph.json not found. Run data_scripts/build_hetero_graph.py to enrich knowledge graph metadata."
        )
        return {}

    label_lookup = {node["id"]: node.get("label", node["id"]) for node in graph_payload.get("nodes", [])}

    relation_to_key = {
        "MENTIONS_DATASET": "datasets",
        "MENTIONS_METHOD": "methods",
        "MENTIONS_TASK": "tasks",
    }

    contexts: Dict[str, dict] = defaultdict(lambda: {key: set() for key in relation_to_key.values()})

    for edge in graph_payload.get("edges", []):
        relation = edge.get("relation")
        key = relation_to_key.get(relation)
        if not key:
            continue

        source = edge.get("source", "")
        target = edge.get("target", "")

        if source.startswith("paper:"):
            paper_id = source.split(":", 1)[1]
            label = label_lookup.get(target, target.split(":", 1)[-1])
            contexts[paper_id][key].add(label)

        if target.startswith("paper:"):
            paper_id = target.split(":", 1)[1]
            label = label_lookup.get(source, source.split(":", 1)[-1])
            contexts[paper_id][key].add(label)

    # Convert sets to sorted lists for JSON serialisation
    finalized_contexts: Dict[str, dict] = {}
    for paper_id, buckets in contexts.items():
        finalized_contexts[paper_id] = {
            key: sorted(values) for key, values in buckets.items() if values
        }

    return finalized_contexts


def _load_joint_taxonomy_embeddings(path: str = JOINT_TAXONOMY_EMBEDDINGS_PATH) -> Dict[str, List[dict]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
    except FileNotFoundError:
        print(
            "Warning: joint_taxonomy_embeddings.json not found. Run data_scripts/align_taxonomy.py to generate interpretable axes."
        )
        return {}

    axes: Dict[str, List[dict]] = {}
    for record in records:
        embedding = record.get("embedding", {})
        if not embedding:
            continue
        sorted_axes = sorted(
            ((label, float(score)) for label, score in embedding.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        top_axes = [
            {"label": label, "score": round(score, 3)}
            for label, score in sorted_axes[:3]
        ]
        if top_axes:
            axes[record["id"]] = top_axes
    return axes


def _format_taxonomy_items(distribution: Iterable[dict]) -> List[dict]:
    result: List[dict] = []
    for item in distribution or []:
        label = item.get("label")
        score = item.get("score")
        if label is None or score is None:
            continue
        result.append({"label": label, "score": round(float(score), 4)})
    return result


def _build_taxonomy_section(taxonomy_record: dict) -> dict:
    if not taxonomy_record:
        return {}
    section = {}
    top_label = taxonomy_record.get("top_label") or taxonomy_record.get("dominant_label")
    if top_label:
        section["dominant_label"] = {
            "label": top_label.get("label"),
            "score": round(float(top_label.get("score", 0.0)), 4),
        }
    section["top_labels"] = _format_taxonomy_items(
        taxonomy_record.get("top_k") or taxonomy_record.get("top_labels")
    )
    section["distribution"] = _format_taxonomy_items(taxonomy_record.get("distribution"))
    # Remove empty keys for cleaner JSON output
    return {key: value for key, value in section.items() if value}


def _build_paper_explanations(
    paper_catalog: Dict[str, dict],
    taxonomy_map: Dict[str, dict],
    knowledge_context: Dict[str, dict],
    taxonomy_axes: Dict[str, List[dict]],
    cluster_memberships: Dict[str, Dict[str, int]],
    cluster_taxonomy_summary: Dict[str, dict],
) -> List[dict]:
    explanations: List[dict] = []

    for paper_id, metadata in sorted(paper_catalog.items(), key=lambda item: item[1]["title"].lower()):
        taxonomy_record = taxonomy_map.get(paper_id, {})
        orkg_section = _build_taxonomy_section(taxonomy_record.get("orkg", {}))
        cso_section = _build_taxonomy_section(taxonomy_record.get("cso", {}))
        axes_section = taxonomy_axes.get(paper_id, [])
        knowledge_section = knowledge_context.get(paper_id, {})

        cluster_items: List[str] = []
        for model_name, cluster_id in sorted(cluster_memberships.get(paper_id, {}).items()):
            cluster_descriptor = cluster_taxonomy_summary.get(model_name, {}).get(str(cluster_id), {})
            dominant_label = (
                cluster_descriptor.get("taxonomies", {})
                .get("ORKG", {})
                .get("dominant_label", {})
                .get("label")
            )
            if cluster_id == -1:
                label_fragment = "Outlier"
            else:
                label_fragment = f"Cluster {cluster_id}"
            if dominant_label:
                label_fragment = f"{label_fragment} · {dominant_label}"
            cluster_items.append(f"{model_name}: {label_fragment}")

        sections: List[dict] = []
        if cluster_items:
            sections.append({"title": "Cluster Context", "items": cluster_items})

        taxonomy_items: List[str] = []
        if orkg_section.get("dominant_label"):
            dom = orkg_section["dominant_label"]
            taxonomy_items.append(f"ORKG anchor: {dom['label']} ({dom['score']:.2f})")
        if cso_section.get("dominant_label"):
            dom = cso_section["dominant_label"]
            taxonomy_items.append(f"CSO anchor: {dom['label']} ({dom['score']:.2f})")
        if axes_section:
            axis_summary = ", ".join(f"{axis['label']}" for axis in axes_section)
            taxonomy_items.append(f"Top latent axes: {axis_summary}")
        if taxonomy_items:
            sections.append({"title": "Taxonomy Anchors", "items": taxonomy_items})

        knowledge_items: List[str] = []
        datasets = knowledge_section.get("datasets", [])
        methods = knowledge_section.get("methods", [])
        tasks = knowledge_section.get("tasks", [])
        if datasets:
            knowledge_items.append("Datasets: " + ", ".join(datasets))
        if methods:
            knowledge_items.append("Methods: " + ", ".join(methods))
        if tasks:
            knowledge_items.append("Tasks: " + ", ".join(tasks))
        if knowledge_items:
            sections.append({"title": "Knowledge Graph Cues", "items": knowledge_items})

        highlights: List[str] = []
        for block in sections:
            if block.get("items"):
                highlights.append(block["items"][0])
        highlights = highlights[:3]

        explanations.append(
            {
                "id": paper_id,
                "title": metadata["title"],
                "orkg": orkg_section,
                "cso": cso_section,
                "taxonomy_axes": axes_section,
                "knowledge_graph": knowledge_section,
                "cluster_memberships": cluster_memberships.get(paper_id, {}),
                "sections": sections,
                "highlights": highlights,
            }
        )

    return explanations

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
    knowledge_context = _load_knowledge_graph()
    taxonomy_axes = _load_joint_taxonomy_embeddings()

    paper_catalog: Dict[str, dict] = {}
    cluster_memberships: Dict[str, Dict[str, int]] = defaultdict(dict)

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

        for _, row in df[['id', 'title', 'cluster_id']].iterrows():
            paper_id = str(row['id']) if pd.notna(row['id']) else None
            title = row.get('title') if isinstance(row.get('title'), str) else None
            if not paper_id or not title:
                continue
            if paper_id not in paper_catalog:
                paper_catalog[paper_id] = {"title": title}
            try:
                cluster_value = int(row['cluster_id'])
            except (TypeError, ValueError):
                continue
            cluster_memberships[paper_id][name] = cluster_value

    # 2. Comparative Metrics (Adjusted Rand Index)
    ari_bge_gte = adjusted_rand_score(df_bge['cluster_id'], df_gte['cluster_id'])
    ari_bge_minilm = adjusted_rand_score(df_bge['cluster_id'], df_minilm['cluster_id'])
    ari_gte_minilm = adjusted_rand_score(df_gte['cluster_id'], df_minilm['cluster_id'])
    
    summary["comparative_metrics"]["Adjusted Rand Index"] = {
        "BGE vs GTE": f"{ari_bge_gte:.4f}",
        "BGE vs MiniLM": f"{ari_bge_minilm:.4f}",
        "GTE vs MiniLM": f"{ari_gte_minilm:.4f}"
    }

    # 3. Persist to disk
    with open('analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    paper_explanations = _build_paper_explanations(
        paper_catalog,
        taxonomy_map,
        knowledge_context,
        taxonomy_axes,
        cluster_memberships,
        summary["cluster_taxonomy"],
    )
    PAPER_EXPLANATIONS_PATH.write_text(json.dumps(paper_explanations, indent=2), encoding='utf-8')

    print("Successfully generated 'analysis_summary.json' and 'paper_explanations.json'")

if __name__ == "__main__":
    create_analysis_summary()
