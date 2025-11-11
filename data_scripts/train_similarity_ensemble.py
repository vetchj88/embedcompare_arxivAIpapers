"""train_similarity_ensemble.py

Train a hybrid similarity scorer that combines dense, sparse, and graph-based
signals using SciRepEval benchmark relevance judgments.

The script expects the following inputs:

* ``--dense`` – Dense similarity matrix (NumPy ``.npy`` or ``.npz`` with array
  name ``arr_0``) aligned with ``paper_ids.json``
* ``--bm25`` and ``--splade`` – Sparse lexical matrices produced by
  ``build_sparse_index.py``
* ``--citation``, ``--coauthor``, ``--venue`` – Graph proximity matrices from
  ``compute_graph_scores.py``
* ``--benchmark`` – Directory containing SciRepEval data (``queries.jsonl``,
  ``corpus.jsonl``, and ``qrels/test.tsv`` or ``qrels/dev.tsv``)

The model is trained via ridge regression by default.  Optionally, a shallow
neural scorer can be enabled with ``--neural``.

Outputs stored in ``--output-dir``:

* ``weights.json`` – learned coefficients or neural state dict
* ``hybrid_similarity.npz`` – CSR matrix capturing the fused similarities
* ``knn_edges.parquet`` – optional k-nearest neighbour edges for fast lookup
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - neural scorer is optional
    torch = None
    nn = None


@dataclass
class EnsembleConfig:
    dense_path: Path
    bm25_path: Path
    splade_path: Path
    citation_path: Path
    coauthor_path: Path
    venue_path: Path
    benchmark_dir: Path
    output_dir: Path
    neural: bool = False
    knn: int | None = 50


def parse_args() -> EnsembleConfig:
    parser = argparse.ArgumentParser(description="Train a hybrid similarity ensemble")
    parser.add_argument("--dense", type=Path, required=True, help="Dense similarity matrix (.npy or .npz)")
    parser.add_argument("--bm25", type=Path, required=True, help="BM25 sparse matrix (.npz)")
    parser.add_argument("--splade", type=Path, required=True, help="SPLADE sparse matrix (.npz)")
    parser.add_argument("--citation", type=Path, required=True, help="Citation proximity matrix (.npz)")
    parser.add_argument("--coauthor", type=Path, required=True, help="Co-author proximity matrix (.npz)")
    parser.add_argument("--venue", type=Path, required=True, help="Venue proximity matrix (.npz)")
    parser.add_argument("--benchmark", type=Path, required=True, help="SciRepEval benchmark directory")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/ensemble"), help="Directory for trained ensemble outputs"
    )
    parser.add_argument(
        "--neural", action="store_true", help="Use a shallow neural scorer instead of linear ridge regression"
    )
    parser.add_argument("--knn", type=int, default=50, help="Number of neighbours to store in the output k-NN graph")
    args = parser.parse_args()
    return EnsembleConfig(
        dense_path=args.dense,
        bm25_path=args.bm25,
        splade_path=args.splade,
        citation_path=args.citation,
        coauthor_path=args.coauthor,
        venue_path=args.venue,
        benchmark_dir=args.benchmark,
        output_dir=args.output_dir,
        neural=args.neural,
        knn=args.knn if args.knn > 0 else None,
    )


def load_similarity_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        loaded = np.load(path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            return loaded[loaded.files[0]]
        return loaded[()]
    raise ValueError(f"Unsupported similarity format: {path}")


def load_sparse_matrix(path: Path) -> sparse.csr_matrix:
    return sparse.load_npz(path).tocsr()


def ensure_alignment(*matrices: sparse.csr_matrix | np.ndarray) -> None:
    shapes = {matrix.shape for matrix in matrices}
    if len(shapes) != 1:
        raise ValueError(f"Matrices must share the same shape, received: {shapes}")


def load_paper_ids(paths: Iterable[Path]) -> List[str]:
    for path in paths:
        ids_path = path.parent / "paper_ids.json"
        if ids_path.exists():
            return json.loads(ids_path.read_text())
    raise FileNotFoundError("Could not locate paper_ids.json next to provided matrices")


# --- SciRepEval utilities -------------------------------------------------


def load_scirepeval(benchmark_dir: Path) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    queries_path = benchmark_dir / "queries.jsonl"
    corpus_path = benchmark_dir / "corpus.jsonl"
    qrels_path = None
    for candidate in ["qrels/test.tsv", "qrels/dev.tsv", "qrels/train.tsv"]:
        maybe = benchmark_dir / candidate
        if maybe.exists():
            qrels_path = maybe
            break
    if not queries_path.exists() or not corpus_path.exists() or qrels_path is None:
        raise FileNotFoundError(
            "Benchmark directory must contain queries.jsonl, corpus.jsonl, and a qrels split (test/dev/train)."
        )

    queries = pd.read_json(queries_path, lines=True)
    corpus = pd.read_json(corpus_path, lines=True)
    qrels = pd.read_csv(qrels_path, sep="\t", names=["query_id", "corpus_id", "score"], header=None)
    relevant = qrels.groupby("query_id")["corpus_id"].apply(list).to_dict()
    return queries, relevant


def build_training_pairs(
    queries: pd.DataFrame,
    relevant: Dict[str, List[str]],
    paper_ids: List[str],
) -> List[Tuple[int, int, int]]:
    paper_index = {pid: idx for idx, pid in enumerate(paper_ids)}
    pairs: List[Tuple[int, int, int]] = []
    for query_id, positives in relevant.items():
        if query_id not in paper_index:
            continue
        q_idx = paper_index[query_id]
        for doc_id in positives:
            d_idx = paper_index.get(doc_id)
            if d_idx is None:
                continue
            pairs.append((q_idx, d_idx, 1))
    return pairs


def sample_negatives(
    pairs: List[Tuple[int, int, int]],
    num_papers: int,
    negatives_per_positive: int = 4,
) -> List[Tuple[int, int, int]]:
    rng = np.random.default_rng(42)
    positives_by_query: Dict[int, set[int]] = {}
    for q, d, _ in pairs:
        positives_by_query.setdefault(q, set()).add(d)
    augmented = pairs.copy()
    for q_idx, positives in positives_by_query.items():
        candidates = np.setdiff1d(np.arange(num_papers), np.array(list(positives)))
        if len(candidates) == 0:
            continue
        sample_size = min(len(candidates), len(positives) * negatives_per_positive)
        negatives = rng.choice(candidates, size=sample_size, replace=False)
        for neg in negatives:
            augmented.append((q_idx, neg, 0))
    return augmented


def extract_features(
    pairs: List[Tuple[int, int, int]],
    dense: np.ndarray,
    bm25: sparse.csr_matrix,
    splade: sparse.csr_matrix,
    citation: sparse.csr_matrix,
    coauthor: sparse.csr_matrix,
    venue: sparse.csr_matrix,
) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for q_idx, d_idx, label in pairs:
        dense_score = dense[q_idx, d_idx]
        lexical_bm25 = bm25[q_idx, d_idx]
        lexical_splade = splade[q_idx, d_idx]
        citation_score = citation[q_idx, d_idx]
        coauthor_score = coauthor[q_idx, d_idx]
        venue_score = venue[q_idx, d_idx]
        features.append(
            [
                float(dense_score),
                float(lexical_bm25),
                float(lexical_splade),
                float(citation_score),
                float(coauthor_score),
                float(venue_score),
            ]
        )
        labels.append(label)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


# --- Models ---------------------------------------------------------------


def train_linear_model(features: np.ndarray, labels: np.ndarray) -> Ridge:
    model = Ridge(alpha=1.0)
    model.fit(features, labels)
    return model


class NeuralScorer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


def train_neural_model(features: np.ndarray, labels: np.ndarray) -> NeuralScorer:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for the neural scorer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralScorer(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(labels).to(device)

    for _ in range(200):
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
    return model.cpu()


def apply_model(model, features: np.ndarray) -> np.ndarray:
    if isinstance(model, Ridge):
        return model.predict(features)
    if torch is not None and isinstance(model, NeuralScorer):
        with torch.no_grad():
            tensor = torch.from_numpy(features).float()
            return model(tensor).numpy()
    raise TypeError("Unsupported model type")


# --- Evaluation & storage ------------------------------------------------


def evaluate_map(
    model,
    pairs: List[Tuple[int, int, int]],
    dense: np.ndarray,
    bm25: sparse.csr_matrix,
    splade: sparse.csr_matrix,
    citation: sparse.csr_matrix,
    coauthor: sparse.csr_matrix,
    venue: sparse.csr_matrix,
    num_papers: int,
) -> float:
    by_query: Dict[int, List[Tuple[int, float, int]]] = {}
    features_cache: Dict[Tuple[int, int], np.ndarray] = {}
    for q_idx, d_idx, label in pairs:
        key = (q_idx, d_idx)
        if key not in features_cache:
            feats = np.array(
                [
                    dense[q_idx, d_idx],
                    bm25[q_idx, d_idx],
                    splade[q_idx, d_idx],
                    citation[q_idx, d_idx],
                    coauthor[q_idx, d_idx],
                    venue[q_idx, d_idx],
                ],
                dtype=np.float32,
            )
            features_cache[key] = feats
        score = apply_model(model, features_cache[key][None, :])[0]
        by_query.setdefault(q_idx, []).append((d_idx, float(score), label))

    ap_scores = []
    for results in by_query.values():
        results.sort(key=lambda x: x[1], reverse=True)
        y_true = [label for _, _, label in results]
        y_scores = [score for _, score, _ in results]
        ap = average_precision_score(y_true, y_scores)
        if not math.isnan(ap):
            ap_scores.append(ap)
    return float(np.mean(ap_scores)) if ap_scores else 0.0


def build_hybrid_matrix(
    model,
    dense: np.ndarray,
    bm25: sparse.csr_matrix,
    splade: sparse.csr_matrix,
    citation: sparse.csr_matrix,
    coauthor: sparse.csr_matrix,
    venue: sparse.csr_matrix,
) -> sparse.csr_matrix:
    num_papers = dense.shape[0]
    data: List[float] = []
    rows: List[int] = []
    cols: List[int] = []

    for i in range(num_papers):
        dense_row = dense[i]
        bm25_row = bm25.getrow(i)
        splade_row = splade.getrow(i)
        citation_row = citation.getrow(i)
        coauthor_row = coauthor.getrow(i)
        venue_row = venue.getrow(i)

        candidates = set(np.nonzero(dense_row)[0])
        candidates.update(bm25_row.indices)
        candidates.update(splade_row.indices)
        candidates.update(citation_row.indices)
        candidates.update(coauthor_row.indices)
        candidates.update(venue_row.indices)
        if i in candidates:
            candidates.remove(i)

        for j in candidates:
            feature_vec = np.array(
                [
                    dense_row[j],
                    bm25_row[0, j],
                    splade_row[0, j],
                    citation_row[0, j],
                    coauthor_row[0, j],
                    venue_row[0, j],
                ],
                dtype=np.float32,
            )
            score = float(apply_model(model, feature_vec[None, :])[0])
            if score <= 0:
                continue
            rows.append(i)
            cols.append(j)
            data.append(score)
    return sparse.csr_matrix((data, (rows, cols)), shape=dense.shape)


def build_knn_graph(matrix: sparse.csr_matrix, k: int) -> pd.DataFrame:
    matrix = matrix.tocsr()
    rows, cols = [], []
    scores = []
    for idx in range(matrix.shape[0]):
        row = matrix.getrow(idx)
        if row.nnz == 0:
            continue
        top_idx = np.argsort(row.data)[::-1][:k]
        rows.extend([idx] * len(top_idx))
        cols.extend(row.indices[top_idx])
        scores.extend(row.data[top_idx])
    return pd.DataFrame({"source": rows, "target": cols, "score": scores})


def serialize_model(model, output_path: Path) -> None:
    if isinstance(model, Ridge):
        payload = {
            "type": "ridge",
            "coef": model.coef_.tolist(),
            "intercept": float(model.intercept_),
        }
        output_path.write_text(json.dumps(payload, indent=2))
        return
    if torch is not None and isinstance(model, NeuralScorer):
        payload = {
            "type": "neural",
            "state_dict": {k: v.numpy().tolist() for k, v in model.state_dict().items()},
        }
        output_path.write_text(json.dumps(payload))
        return
    raise TypeError("Unsupported model type")


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    dense = load_similarity_matrix(config.dense_path)
    bm25 = load_sparse_matrix(config.bm25_path)
    splade = load_sparse_matrix(config.splade_path)
    citation = load_sparse_matrix(config.citation_path)
    coauthor = load_sparse_matrix(config.coauthor_path)
    venue = load_sparse_matrix(config.venue_path)

    ensure_alignment(dense, bm25, splade, citation, coauthor, venue)

    paper_ids = load_paper_ids(
        [
            config.bm25_path,
            config.splade_path,
            config.citation_path,
            config.coauthor_path,
            config.venue_path,
        ]
    )

    queries, relevant = load_scirepeval(config.benchmark_dir)
    pairs = build_training_pairs(queries, relevant, paper_ids)
    if not pairs:
        raise RuntimeError("No overlapping query/document ids found between SciRepEval and paper ids")
    pairs = sample_negatives(pairs, num_papers=len(paper_ids))

    features, labels = extract_features(pairs, dense, bm25, splade, citation, coauthor, venue)

    model = train_neural_model(features, labels) if config.neural else train_linear_model(features, labels)
    map_score = evaluate_map(model, pairs, dense, bm25, splade, citation, coauthor, venue, len(paper_ids))
    print(f"[✓] Trained ensemble model – MAP@ benchmark pairs: {map_score:.4f}")

    hybrid = build_hybrid_matrix(model, dense, bm25, splade, citation, coauthor, venue)
    sparse.save_npz(config.output_dir / "hybrid_similarity.npz", hybrid)
    serialize_model(model, config.output_dir / "weights.json")

    if config.knn:
        knn_df = build_knn_graph(hybrid, config.knn)
        knn_df.to_parquet(config.output_dir / "knn_edges.parquet", index=False)

    (config.output_dir / "paper_ids.json").write_text(json.dumps(paper_ids))
    print(f"[✓] Hybrid similarity artifacts saved to {config.output_dir}")


if __name__ == "__main__":
    main()
