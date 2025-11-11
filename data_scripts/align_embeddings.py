"""Align per-model embedding spaces and build ensemble representations.

Steps performed by this utility:

1. Load the dense embedding matrices written by ``2_embed_papers.py``.
2. Project all vectors into the smallest shared dimensionality and apply a
   universal translation followed by Procrustes alignment.
3. Compute diagnostic metrics (hubness, trustworthiness) for transparency.
4. Learn ensemble pooling weights from SciRepEval retrieval metrics and export
   fused representations for downstream analysis.
5. Update ``analysis_summary.json`` with alignment metadata and diagnostics.

The script is intentionally conservative: when optional dependencies (e.g.
``scikit-learn``) are unavailable the diagnostic metrics gracefully degrade to
``None`` so the pipeline remains usable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import numpy as np

try:  # Optional diagnostics dependencies
    from sklearn.manifold import trustworthiness as sklearn_trustworthiness
    from sklearn.neighbors import NearestNeighbors
except ImportError:  # pragma: no cover - optional path
    sklearn_trustworthiness = None  # type: ignore[assignment]
    NearestNeighbors = None  # type: ignore[assignment]


VECTOR_STORE_ROOT = Path("./vector_stores")
ENSEMBLE_DIR = VECTOR_STORE_ROOT / "ensembles"
SCIREPEVAL_RESULTS_PATH = Path("./data/sci_rep_eval_results.json")
ANALYSIS_PATH = Path("analysis_summary.json")


@dataclass
class AlignmentResult:
    aligned_embeddings: Dict[str, np.ndarray]
    projected_embeddings: Dict[str, np.ndarray]
    rotation_maps: Dict[str, np.ndarray]
    global_mean: np.ndarray
    target_dim: int
    reference_model: str


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_embeddings(root: Path) -> Dict[str, np.ndarray]:
    embeddings: Dict[str, np.ndarray] = {}
    if not root.exists():
        raise FileNotFoundError(
            f"Vector store directory {root} does not exist. Run 2_embed_papers.py first."
        )

    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        path = model_dir / "embeddings.npy"
        if path.exists():
            embeddings[model_dir.name] = np.load(path)
    if not embeddings:
        raise FileNotFoundError(
            f"No embeddings found in {root}. Ensure 2_embed_papers.py completed successfully."
        )
    return embeddings


def project_to_dim(matrix: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Project *matrix* to the provided dimension using SVD components.

    Returns the projected matrix and the projection components used.
    """

    if matrix.shape[1] == dim:
        return matrix.copy(), np.eye(matrix.shape[1], dtype=matrix.dtype)

    matrix_centered = matrix - matrix.mean(axis=0, keepdims=True)
    u, _, vh = np.linalg.svd(matrix_centered, full_matrices=False)
    components = vh[:dim].T  # shape: original_dim x dim
    projected = matrix_centered @ components
    return projected, components


def centre_embeddings(embeddings: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    stacked = np.vstack(list(embeddings.values()))
    global_mean = stacked.mean(axis=0, keepdims=True)
    centred = {name: matrix - global_mean for name, matrix in embeddings.items()}
    return centred, global_mean.squeeze(0)


def procrustes_align(embeddings: Dict[str, np.ndarray]) -> AlignmentResult:
    dims = {name: matrix.shape[1] for name, matrix in embeddings.items()}
    target_dim = min(dims.values())

    projected: Dict[str, np.ndarray] = {}
    for name, matrix in embeddings.items():
        projected_matrix, _ = project_to_dim(matrix, target_dim)
        projected[name] = projected_matrix.astype(np.float32)

    centred, global_mean = centre_embeddings(projected)
    reference_model = sorted(centred.keys())[0]
    reference_matrix = centred[reference_model]

    aligned: Dict[str, np.ndarray] = {reference_model: reference_matrix}
    rotation_maps: Dict[str, np.ndarray] = {reference_model: np.eye(target_dim, dtype=np.float32)}

    for name, matrix in centred.items():
        if name == reference_model:
            continue
        m = matrix.T @ reference_matrix
        u, _, vh = np.linalg.svd(m, full_matrices=False)
        rotation = (u @ vh).astype(np.float32)
        aligned[name] = matrix @ rotation
        rotation_maps[name] = rotation

    return AlignmentResult(aligned, projected, rotation_maps, global_mean, target_dim, reference_model)


def compute_hubness(matrix: np.ndarray, k: int = 10) -> Optional[float]:
    if NearestNeighbors is None:
        return None
    n_samples = matrix.shape[0]
    if n_samples <= 1:
        return None
    k = min(k, n_samples - 1)
    if k < 1:
        return None
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(matrix)
    indices = nbrs.kneighbors(return_distance=False)[:, 1:]
    counts = np.bincount(indices.ravel(), minlength=n_samples)
    mean = counts.mean()
    if mean == 0:
        return 0.0
    return float(counts.std() / mean)


def compute_trustworthiness(
    original: np.ndarray, embedded: np.ndarray, k: int = 10
) -> Optional[float]:
    if sklearn_trustworthiness is None:
        return None
    n_samples = embedded.shape[0]
    if n_samples <= 1:
        return None
    k = min(k, n_samples - 1)
    if k < 2:
        return None
    return float(sklearn_trustworthiness(original, embedded, n_neighbors=k))


def softmax(matrix: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    scaled = matrix / max(temperature, 1e-6)
    scaled -= np.max(scaled, axis=axis, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=axis, keepdims=True)


def load_scirepeval_scores(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        print(
            f"SciRepEval results not found at {path}. Using uniform weights for ensemble pooling."
        )
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def aggregate_model_weights(scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not scores:
        return {}
    aggregated: Dict[str, float] = {}
    for model, task_scores in scores.items():
        if task_scores:
            aggregated[model] = float(np.mean(list(task_scores.values())))
    return aggregated


def normalise_weights(models: Iterable[str], weights: Mapping[str, float]) -> Dict[str, float]:
    model_list = list(models)
    values = np.array([weights.get(model, 1.0) for model in model_list], dtype=np.float32)
    if not np.any(values):
        values = np.ones_like(values)
    norm_values = values / values.sum()
    return {model: float(weight) for model, weight in zip(model_list, norm_values)}


def weighted_average_ensemble(
    aligned: Mapping[str, np.ndarray], weights: Mapping[str, float]
) -> np.ndarray:
    model_names = sorted(aligned.keys())
    norms = normalise_weights(model_names, weights)
    weighted_sum = sum(aligned[name] * norms[name] for name in model_names)
    return weighted_sum.astype(np.float32)


def gated_fusion_ensemble(
    aligned: Mapping[str, np.ndarray], weights: Mapping[str, float], temperature: float = 0.5
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    model_names = sorted(aligned.keys())
    stacked = np.stack([aligned[name] for name in model_names], axis=-1)
    norms = normalise_weights(model_names, weights)
    weight_vector = np.array([norms[name] for name in model_names], dtype=np.float32)
    norms_matrix = np.linalg.norm(stacked, axis=1)
    gating_logits = norms_matrix[:, np.newaxis] * weight_vector[np.newaxis, :]
    gating_weights = softmax(gating_logits, axis=-1, temperature=temperature)
    fused = np.sum(stacked * gating_weights[:, np.newaxis, :], axis=-1)
    stats = {
        name: {
            "mean_weight": float(gating_weights[:, idx].mean()),
            "std_weight": float(gating_weights[:, idx].std()),
        }
        for idx, name in enumerate(model_names)
    }
    return fused.astype(np.float32), stats


def save_aligned_embeddings(root: Path, aligned: Mapping[str, np.ndarray]) -> None:
    for name, matrix in aligned.items():
        store_dir = ensure_directory(root / name / "aligned")
        np.save(store_dir / "embeddings.npy", matrix)


def save_ensemble_embeddings(weighted: np.ndarray, gated: np.ndarray) -> None:
    ensure_directory(ENSEMBLE_DIR)
    np.save(ENSEMBLE_DIR / "weighted_average.npy", weighted)
    np.save(ENSEMBLE_DIR / "gated_fusion.npy", gated)


def update_analysis_summary(alignment_info: Mapping[str, object]) -> None:
    summary: MutableMapping[str, object]
    if ANALYSIS_PATH.exists():
        with ANALYSIS_PATH.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
    else:
        summary = {}
    summary.setdefault("alignment_diagnostics", {}).update(alignment_info)
    with ANALYSIS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=4)


def main() -> None:
    embeddings = load_embeddings(VECTOR_STORE_ROOT)
    alignment = procrustes_align(embeddings)
    save_aligned_embeddings(VECTOR_STORE_ROOT, alignment.aligned_embeddings)

    diagnostics = {}
    for name, aligned_matrix in alignment.aligned_embeddings.items():
        original = alignment.projected_embeddings[name]
        diagnostics[name] = {
            "hubness_ratio": compute_hubness(aligned_matrix),
            "trustworthiness": compute_trustworthiness(original, aligned_matrix),
            "embedding_norm_mean": float(np.linalg.norm(aligned_matrix, axis=1).mean()),
        }

    scores = load_scirepeval_scores(SCIREPEVAL_RESULTS_PATH)
    aggregated_scores = aggregate_model_weights(scores)
    weights = normalise_weights(alignment.aligned_embeddings.keys(), aggregated_scores)

    weighted_average = weighted_average_ensemble(alignment.aligned_embeddings, weights)
    gated_fusion, gating_stats = gated_fusion_ensemble(alignment.aligned_embeddings, weights)
    save_ensemble_embeddings(weighted_average, gated_fusion)

    alignment_summary = {
        "reference_model": alignment.reference_model,
        "target_dimension": alignment.target_dim,
        "global_mean_norm": float(np.linalg.norm(alignment.global_mean)),
        "weights": weights,
        "diagnostics_per_model": diagnostics,
        "gated_fusion_stats": gating_stats,
    }

    update_analysis_summary(alignment_summary)
    print("Alignment complete. Diagnostics written to analysis_summary.json.")


if __name__ == "__main__":
    main()
