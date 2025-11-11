"""Utilities for aligning papers with ORKG/CSO taxonomies and generating
knowledge-aware embeddings.

This module performs four high-level tasks:

1. Encode each paper with an enriched BERT-style encoder (SentenceTransformer
   when available, TF-IDF fallback otherwise) and classify papers into ORKG and
   CSO taxonomy labels.  The script stores the full label distributions together
   with top-1 confidence scores so downstream tooling can reason about
   uncertainty.
2. Extract lightweight task/dataset/method entities and build a knowledge graph
   that links papers to those entities.  The graph is serialized to JSON using a
   NetworkX-inspired structure (nodes + edges).
3. Train a joint text+KG projection that makes taxonomy axes explicit latent
   dimensions.  We fit a Ridge regressor that maps concatenated text and KG
   features to the taxonomy distributions, yielding a representation in which
   each dimension corresponds to a taxonomy label.  The resulting embeddings are
   exported for reuse.
4. Persist the taxonomy alignment, knowledge graph, and joint embeddings to
   disk.

Typical usage:

>>> from data_scripts.align_taxonomy import main
>>> main()

The script is intentionally self-contained and can be invoked as a CLI:
`python data_scripts/align_taxonomy.py --input papers_bge_large.json`
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # Prefer a real BERT-style encoder when available.
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - handled gracefully below.
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import normalize
except ImportError as exc:  # pragma: no cover - the project already depends on sklearn.
    raise ImportError(
        "align_taxonomy.py requires scikit-learn. Install it via `pip install scikit-learn`."
    ) from exc


LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT = "papers_bge_large.json"
TAXONOMY_ALIGNMENT_PATH = "taxonomy_alignment.json"
KNOWLEDGE_GRAPH_PATH = "knowledge_graph.json"
JOINT_EMBEDDINGS_PATH = "joint_taxonomy_embeddings.json"


@dataclass(frozen=True)
class TaxonomyEntry:
    label: str
    description: str


ORKG_CATEGORIES: Tuple[TaxonomyEntry, ...] = (
    TaxonomyEntry(
        "Machine Learning",
        "Learning algorithms, optimization techniques, and model development for AI systems.",
    ),
    TaxonomyEntry(
        "Natural Language Processing",
        "Linguistic modeling, language understanding, and text generation tasks.",
    ),
    TaxonomyEntry(
        "Computer Vision",
        "Visual recognition, perception, and image/video understanding techniques.",
    ),
    TaxonomyEntry(
        "Robotics",
        "Autonomous systems, planning, control, and embodied intelligence.",
    ),
    TaxonomyEntry(
        "Knowledge Representation",
        "Ontologies, semantic web, knowledge graphs, and reasoning frameworks.",
    ),
)

CSO_CATEGORIES: Tuple[TaxonomyEntry, ...] = (
    TaxonomyEntry(
        "Deep Learning",
        "Neural architectures, representation learning, and large-scale optimization.",
    ),
    TaxonomyEntry(
        "Reinforcement Learning",
        "Sequential decision making, policy optimization, and reward-driven learning.",
    ),
    TaxonomyEntry(
        "Information Retrieval",
        "Search, recommendation, and knowledge access methodologies.",
    ),
    TaxonomyEntry(
        "Computer Vision",
        "Object detection, segmentation, and multimodal perception techniques.",
    ),
    TaxonomyEntry(
        "Natural Language Processing",
        "Machine translation, dialogue systems, and language understanding.",
    ),
)


TASK_KEYWORDS = {
    "Machine Translation": {"translation", "translate", "multilingual"},
    "Question Answering": {"question answering", "qa", "questions answering"},
    "Summarization": {"summarization", "summarize", "abstract"},
    "Recommendation": {"recommender", "recommendation", "recommend"},
    "Planning": {"planning", "planner", "plan generation"},
    "Segmentation": {"segmentation", "segmentation task", "segment"},
}

DATASET_KEYWORDS = {
    "ImageNet": {"imagenet"},
    "COCO": {"mscoco", "coco dataset", "coco"},
    "GLUE": {"glue", "general language understanding evaluation"},
    "SQuAD": {"squad", "stanford question answering dataset"},
    "UCF-101": {"ucf-101", "ucf101"},
    "MNIST": {"mnist"},
}

METHOD_KEYWORDS = {
    "Transformers": {"transformer", "self-attention", "attention"},
    "Graph Neural Networks": {"graph neural network", "gnn", "graph convolution"},
    "Autoencoders": {"autoencoder", "variational autoencoder", "vae"},
    "Contrastive Learning": {"contrastive", "contrastive learning"},
    "Diffusion Models": {"diffusion model", "denoising diffusion", "ddpm"},
    "Hypernetworks": {"hyper-network", "hypernetwork"},
}


class EnrichedBertEncoder:
    """A light wrapper around SentenceTransformer with TF-IDF fallback.

    The class exposes a uniform ``encode`` API that returns L2-normalized
    embeddings.  When SentenceTransformer is not available (for instance in
    resource constrained CI environments) the class gracefully falls back to a
    TF-IDF encoder seeded with the category descriptions and paper abstracts.
    """

    def __init__(self, corpus: Sequence[str], model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._uses_transformer = SentenceTransformer is not None

        if self._uses_transformer:
            try:
                self._encoder = SentenceTransformer(model_name)
                LOGGER.info("Loaded SentenceTransformer model '%s'", model_name)
            except Exception as exc:  # pragma: no cover - log and fallback.
                LOGGER.warning(
                    "Falling back to TF-IDF encoder because SentenceTransformer failed: %s",
                    exc,
                )
                self._uses_transformer = False

        if not self._uses_transformer:
            LOGGER.info("Using TF-IDF fallback encoder. Install sentence-transformers for BERT embeddings.")
            self._vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
            self._vectorizer.fit(corpus)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._uses_transformer:
            embeddings = self._encoder.encode(  # type: ignore[attr-defined]
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(embeddings)

        matrix = self._vectorizer.transform(texts)
        embeddings = normalize(matrix, norm="l2").toarray()
        return embeddings


def _softmax(scores: np.ndarray, temperature: float = 0.07) -> np.ndarray:
    scaled = scores / max(temperature, 1e-6)
    scaled -= scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


def _prepare_corpus(papers: Sequence[Dict[str, str]]) -> List[str]:
    return [f"{paper['title']}\n{paper['abstract']}" for paper in papers]


def _build_taxonomy_matrix(categories: Sequence[TaxonomyEntry], encoder: EnrichedBertEncoder) -> np.ndarray:
    category_texts = [f"{entry.label}: {entry.description}" for entry in categories]
    return encoder.encode(category_texts)


def _score_taxonomy(text_embedding: np.ndarray, taxonomy_matrix: np.ndarray) -> np.ndarray:
    return taxonomy_matrix @ text_embedding


def _format_distribution(categories: Sequence[TaxonomyEntry], scores: np.ndarray) -> Dict[str, object]:
    probs = _softmax(scores)
    pairs = [
        {"label": entry.label, "score": float(prob)}
        for entry, prob in zip(categories, probs)
    ]
    ranked = sorted(pairs, key=lambda item: item["score"], reverse=True)
    top = ranked[0]
    return {
        "top_label": top,
        "top_k": ranked[:5],
        "distribution": pairs,
    }


class KnowledgeGraphBuilder:
    """Collects paper-to-entity relationships and serializes them as a graph."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, object]] = {}
        self.edges: List[Dict[str, str]] = []
        self._feature_indices: Dict[str, Dict[str, int]] = {
            "task": {},
            "dataset": {},
            "method": {},
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        lowered = text.lower()
        entities = {"tasks": [], "datasets": [], "methods": []}

        for label, keywords in TASK_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                entities["tasks"].append(label)

        for label, keywords in DATASET_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                entities["datasets"].append(label)

        for label, keywords in METHOD_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                entities["methods"].append(label)

        # Deduplicate while preserving order.
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))

        return entities

    def add_paper(self, paper: Dict[str, str], entity_map: Dict[str, List[str]]) -> None:
        paper_id = f"paper:{paper['id']}"
        if paper_id not in self.nodes:
            self.nodes[paper_id] = {"id": paper_id, "type": "paper", "label": paper["title"]}

        for entity_type, labels in (
            ("task", entity_map.get("tasks", [])),
            ("dataset", entity_map.get("datasets", [])),
            ("method", entity_map.get("methods", [])),
        ):
            for label in labels:
                node_id = f"{entity_type}:{label}"
                if node_id not in self.nodes:
                    self.nodes[node_id] = {"id": node_id, "type": entity_type, "label": label}

                self.edges.append({
                    "source": paper_id,
                    "target": node_id,
                    "relation": f"MENTIONS_{entity_type.upper()}",
                })

                index_map = self._feature_indices[entity_type]
                if label not in index_map:
                    index_map[label] = len(index_map)

    def feature_vector(self, entity_map: Dict[str, List[str]]) -> np.ndarray:
        total_dims = sum(len(mapping) for mapping in self._feature_indices.values())
        vector = np.zeros(total_dims, dtype=float)
        offset = 0
        for entity_type in ("task", "dataset", "method"):
            indices = self._feature_indices[entity_type]
            labels = entity_map.get(f"{entity_type}s", [])
            for label in labels:
                if label in indices:
                    vector[offset + indices[label]] = 1.0
            offset += len(indices)
        return vector

    def to_dict(self) -> Dict[str, object]:
        return {
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
        }


class JointEmbeddingTrainer:
    """Fit a joint text+KG projection that aligns with taxonomy distributions."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.model = Ridge(alpha=alpha)
        self.taxonomy_labels: List[str] = []

    def fit_transform(
        self,
        text_embeddings: np.ndarray,
        kg_features: np.ndarray,
        taxonomy_scores: np.ndarray,
        labels: Sequence[str],
        paper_ids: Sequence[str],
    ) -> Dict[str, Dict[str, float]]:
        features = np.concatenate([text_embeddings, kg_features], axis=1)
        self.taxonomy_labels = list(labels)
        self.model.fit(features, taxonomy_scores)
        projected = self.model.predict(features)
        results: Dict[str, Dict[str, float]] = {}
        for paper_id, row in zip(paper_ids, projected):
            results[paper_id] = {label: float(value) for label, value in zip(self.taxonomy_labels, row)}
        return results


def align_taxonomy(
    papers: Sequence[Dict[str, str]],
    encoder: EnrichedBertEncoder,
    kg_builder: KnowledgeGraphBuilder,
) -> Tuple[List[Dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    taxonomy_matrix_orkg = _build_taxonomy_matrix(ORKG_CATEGORIES, encoder)
    taxonomy_matrix_cso = _build_taxonomy_matrix(CSO_CATEGORIES, encoder)

    paper_corpus = _prepare_corpus(papers)
    text_embeddings = encoder.encode(paper_corpus)

    alignment_records: List[Dict[str, object]] = []
    taxonomy_targets: List[np.ndarray] = []

    for emb, paper, text in zip(text_embeddings, papers, paper_corpus):
        scores_orkg = _score_taxonomy(emb, taxonomy_matrix_orkg)
        scores_cso = _score_taxonomy(emb, taxonomy_matrix_cso)

        record = {
            "id": paper["id"],
            "title": paper["title"],
            "orkg": _format_distribution(ORKG_CATEGORIES, scores_orkg),
            "cso": _format_distribution(CSO_CATEGORIES, scores_cso),
        }

        entity_map = kg_builder.extract_entities(text)
        kg_builder.add_paper(paper, entity_map)

        # Store metadata for downstream aggregation.
        record["entities"] = entity_map
        alignment_records.append(record)

        taxonomy_targets.append(
            np.array([item["score"] for item in record["orkg"]["distribution"]] + [
                item["score"] for item in record["cso"]["distribution"]
            ])
        )

    if alignment_records:
        kg_matrix = np.vstack([
            kg_builder.feature_vector(record.get("entities", {}))
            for record in alignment_records
        ])
    else:
        kg_matrix = np.zeros((0, 0))

    taxonomy_matrix = np.vstack(taxonomy_targets)
    return alignment_records, text_embeddings, np.asarray(kg_matrix), taxonomy_matrix


def serialize_alignment(records: Sequence[Dict[str, object]], path: str = TAXONOMY_ALIGNMENT_PATH) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    LOGGER.info("Saved taxonomy alignment to %s", path)


def serialize_knowledge_graph(builder: KnowledgeGraphBuilder, path: str = KNOWLEDGE_GRAPH_PATH) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(builder.to_dict(), handle, indent=2)
    LOGGER.info("Saved knowledge graph to %s", path)


def serialize_joint_embeddings(
    embeddings: Dict[int, Dict[str, float]],
    papers: Sequence[Dict[str, str]],
    taxonomy_labels: Sequence[str],
    path: str = JOINT_EMBEDDINGS_PATH,
) -> None:
    payload = []
    for paper in papers:
        payload.append(
            {
                "id": paper["id"],
                "title": paper["title"],
                "embedding": embeddings.get(paper["id"], {}),
                "taxonomy_labels": list(taxonomy_labels),
            }
        )

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    LOGGER.info("Saved joint embeddings to %s", path)


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Align papers with ORKG/CSO taxonomies.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to the paper JSON dataset.")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name. Ignored when TF-IDF fallback is used.",
    )
    parsed = parser.parse_args(args=args)

    if not os.path.exists(parsed.input):
        raise FileNotFoundError(f"Could not locate paper dataset: {parsed.input}")

    with open(parsed.input, "r", encoding="utf-8") as handle:
        papers = json.load(handle)

    if not isinstance(papers, list):
        raise ValueError("Input dataset must be a list of paper records.")

    corpus = _prepare_corpus(papers)
    encoder = EnrichedBertEncoder(corpus + [entry.description for entry in ORKG_CATEGORIES], parsed.model)
    kg_builder = KnowledgeGraphBuilder()

    records, text_embeddings, kg_matrix, taxonomy_matrix = align_taxonomy(papers, encoder, kg_builder)
    serialize_alignment(records)
    serialize_knowledge_graph(kg_builder)

    if taxonomy_matrix.size == 0:
        LOGGER.warning("No taxonomy scores were generated; skipping joint embedding training.")
        return

    joint_labels = [entry.label for entry in ORKG_CATEGORIES] + [entry.label for entry in CSO_CATEGORIES]
    trainer = JointEmbeddingTrainer(alpha=0.25)
    paper_ids = [paper["id"] for paper in papers]
    joint_embeddings = trainer.fit_transform(
        text_embeddings,
        kg_matrix,
        taxonomy_matrix,
        joint_labels,
        paper_ids,
    )
    serialize_joint_embeddings(joint_embeddings, papers, joint_labels)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()
