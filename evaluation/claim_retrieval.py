"""Evaluate claim-level retrieval improvements for enhanced embeddings."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

from models.claim_contrastive import ClaimContrastiveModel
from transformers import AutoTokenizer

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("sentence-transformers is required for evaluation") from exc


LOGGER = logging.getLogger("claim_retrieval")


@dataclass
class ClaimQuery:
    paper_id: str
    claim: str
    datasets: List[str]
    relevant_ids: Set[str]


def load_claims(path: Path) -> List[Dict]:
    with path.open() as fh:
        return json.load(fh)


def build_query_set(
    claims_data: Iterable[Dict],
    candidate_ids: Set[str],
) -> List[ClaimQuery]:
    dataset_index: Dict[str, Set[str]] = {}
    claim_index: Dict[str, Set[str]] = {}
    for entry in claims_data:
        paper_id = str(entry["id"])
        extracted = entry.get("extracted", {})
        if paper_id not in candidate_ids:
            continue
        for claim in extracted.get("claims", []):
            claim_index.setdefault(claim, set()).add(paper_id)
        for dataset in extracted.get("datasets", []):
            dataset_index.setdefault(dataset, set()).add(paper_id)

    queries: List[ClaimQuery] = []
    for entry in claims_data:
        paper_id = str(entry["id"])
        extracted = entry.get("extracted", {})
        if paper_id not in candidate_ids:
            continue
        for claim in extracted.get("claims", []):
            datasets = extracted.get("datasets", [])
            relevant = set()
            relevant |= claim_index.get(claim, set())
            for dataset in datasets:
                relevant |= dataset_index.get(dataset, set())
            relevant.discard(paper_id)
            relevant &= candidate_ids
            if not relevant:
                continue
            queries.append(ClaimQuery(paper_id=paper_id, claim=claim, datasets=list(datasets), relevant_ids=relevant))
    return queries


def compute_metrics(
    ranked_ids: Sequence[str],
    relevant: Set[str],
    ks: Sequence[int],
) -> Tuple[Dict[int, float], float, Dict[int, float]]:
    recalls: Dict[int, float] = {}
    ndcg: Dict[int, float] = {}
    for k in ks:
        topk = ranked_ids[:k]
        hit = 1.0 if any(doc_id in relevant for doc_id in topk) else 0.0
        recalls[k] = hit
        gains = [1.0 if doc_id in relevant else 0.0 for doc_id in topk]
        dcg = sum(g / np.log2(idx + 2) for idx, g in enumerate(gains))
        ideal_len = min(len(relevant), k)
        ideal_dcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_len))
        ndcg[k] = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    mrr = 0.0
    for idx, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            mrr = 1.0 / idx
            break
    return recalls, mrr, ndcg


def aggregate_metrics(records: List[Tuple[Dict[int, float], float, Dict[int, float]]], ks: Sequence[int]) -> Dict[str, object]:
    if not records:
        return {"recall": {k: 0.0 for k in ks}, "mrr": 0.0, "ndcg": {k: 0.0 for k in ks}, "queries": 0}
    total = len(records)
    recall_avg = {k: float(np.mean([rec[0][k] for rec in records])) for k in ks}
    ndcg_avg = {k: float(np.mean([rec[2][k] for rec in records])) for k in ks}
    mrr_avg = float(np.mean([rec[1] for rec in records]))
    return {"recall": recall_avg, "mrr": mrr_avg, "ndcg": ndcg_avg, "queries": total}


def load_enhanced_embeddings(path: Path) -> Dict[str, np.ndarray]:
    with path.open() as fh:
        data = json.load(fh)
    return {str(entry["id"]): np.asarray(entry["embedding"], dtype=np.float32) for entry in data}


def evaluate_system(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_ids: Sequence[str],
    queries: Sequence[ClaimQuery],
    ks: Sequence[int],
) -> Dict[str, object]:
    records: List[Tuple[Dict[int, float], float, Dict[int, float]]] = []
    for query, q_emb in zip(queries, query_embeddings):
        sims = candidate_embeddings @ q_emb
        order = np.argsort(-sims)
        ranked_ids = [candidate_ids[idx] for idx in order]
        records.append(compute_metrics(ranked_ids, query.relevant_ids, ks))
    return aggregate_metrics(records, ks)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("papers_metadata.csv"))
    parser.add_argument("--claims", type=Path, default=Path("extracted_claims.json"))
    parser.add_argument("--baseline-model", type=str, default="thenlper/gte-large")
    parser.add_argument("--enhanced-model-dir", type=Path, default=Path("checkpoints/claim_contrastive"))
    parser.add_argument("--enhanced-embeddings", type=Path, default=Path("enhanced_embeddings.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/claim_retrieval_metrics.json"))
    parser.add_argument("--ks", nargs="*", type=int, default=[1, 5, 10])
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    metadata = pd.read_csv(args.metadata)
    if "id" not in metadata.columns:
        if "paper_id" in metadata.columns:
            metadata = metadata.rename(columns={"paper_id": "id"})
        else:
            raise ValueError("Metadata must include an 'id' column")
    metadata["id"] = metadata["id"].astype(str)
    metadata = metadata.set_index("id")

    enhanced_map = load_enhanced_embeddings(args.enhanced_embeddings)
    candidate_ids = [pid for pid in metadata.index if pid in enhanced_map]
    if not candidate_ids:
        raise RuntimeError("No overlapping papers between metadata and enhanced embeddings")
    candidate_set = set(candidate_ids)

    LOGGER.info("Evaluating over %d candidate papers", len(candidate_ids))

    corpus_texts = [f"{metadata.loc[pid].get('title', '')}\n{metadata.loc[pid].get('abstract', '')}" for pid in candidate_ids]
    baseline_model = SentenceTransformer(args.baseline_model)
    baseline_matrix = baseline_model.encode(
        corpus_texts, convert_to_numpy=True, normalize_embeddings=True
    )
    enhanced_matrix = np.stack([enhanced_map[pid] for pid in candidate_ids])

    claims_data = load_claims(args.claims)
    queries = build_query_set(claims_data, candidate_set)
    if args.max_queries:
        queries = queries[: args.max_queries]
    if not queries:
        raise RuntimeError("No claim queries with cross-paper relevance were found")
    LOGGER.info("Prepared %d evaluation queries", len(queries))

    baseline_query_texts = [format_query_text(query) for query in queries]
    baseline_query_embeddings = baseline_model.encode(
        baseline_query_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.enhanced_model_dir)
    enhanced_model = ClaimContrastiveModel.load(args.enhanced_model_dir, device=device)
    enhanced_query_embeddings = enhanced_model.encode(
        tokenizer,
        baseline_query_texts,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    metrics_baseline = evaluate_system(
        query_embeddings=baseline_query_embeddings,
        candidate_embeddings=baseline_matrix,
        candidate_ids=candidate_ids,
        queries=queries,
        ks=args.ks,
    )
    metrics_enhanced = evaluate_system(
        query_embeddings=enhanced_query_embeddings,
        candidate_embeddings=enhanced_matrix,
        candidate_ids=candidate_ids,
        queries=queries,
        ks=args.ks,
    )

    deltas = {
        "recall": {k: metrics_enhanced["recall"][k] - metrics_baseline["recall"][k] for k in args.ks},
        "mrr": metrics_enhanced["mrr"] - metrics_baseline["mrr"],
        "ndcg": {k: metrics_enhanced["ndcg"][k] - metrics_baseline["ndcg"][k] for k in args.ks},
    }

    results = {
        "baseline": metrics_baseline,
        "enhanced": metrics_enhanced,
        "delta": deltas,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(results, fh, indent=2)
    LOGGER.info("Saved evaluation metrics to %s", args.output)


def format_query_text(query: ClaimQuery) -> str:
    if query.datasets:
        return f"claim: {query.claim}\nrelated datasets: {', '.join(query.datasets)}"
    return f"claim: {query.claim}"


if __name__ == "__main__":
    main()
