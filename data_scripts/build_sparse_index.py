"""build_sparse_index.py

Generate sparse lexical representations (BM25 and SPLADE) for each paper
using segmented text passages.

The script expects an input JSONL/JSON file where each record contains
```
{
    "paper_id": "2506.12345",
    "segments": ["text chunk 1", "text chunk 2", ...]
}
```
Additional metadata keys are ignored.  Segments are concatenated when
building document level vectors.

Two sparse matrices are produced:

* ``bm25_index.npz`` – CSR matrix of BM25 weights
* ``splade_index.npz`` – CSR matrix of SPLADE activations

The script also stores ``paper_ids.json`` to preserve the row ordering and
``splade_vocabulary.json`` containing the tokenizer vocabulary used by the
SPLADE encoder.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

try:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "transformers and torch are required for SPLADE encoding. "
        "Install them with `pip install torch transformers`."
    ) from exc


@dataclass
class SparseIndexConfig:
    segmented_path: Path
    output_dir: Path
    splade_model_name: str = "naver/splade-cocondenser-ensembledistil"
    max_splade_tokens: int = 512
    splade_top_k: int = 256


def parse_args() -> SparseIndexConfig:
    parser = argparse.ArgumentParser(description="Build sparse representations for segmented papers")
    parser.add_argument("segmented", type=Path, help="Path to JSON/JSONL file with segmented papers")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sparse"),
        help="Directory where sparse indices will be stored",
    )
    parser.add_argument(
        "--splade-model",
        default="naver/splade-cocondenser-ensembledistil",
        help="Hugging Face model used for SPLADE encoding",
    )
    parser.add_argument(
        "--splade-max-length",
        type=int,
        default=512,
        help="Maximum number of wordpieces fed to the SPLADE encoder",
    )
    parser.add_argument(
        "--splade-top-k",
        type=int,
        default=256,
        help="Number of most activated vocabulary items to retain per document",
    )
    args = parser.parse_args()
    return SparseIndexConfig(
        segmented_path=args.segmented,
        output_dir=args.output_dir,
        splade_model_name=args.splade_model,
        max_splade_tokens=args.splade_max_length,
        splade_top_k=args.splade_top_k,
    )


def load_segmented_records(path: Path) -> pd.DataFrame:
    """Load segmented passages from JSON or JSONL into a DataFrame."""

    if not path.exists():
        raise FileNotFoundError(f"Segmented file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        records = json.loads(path.read_text())
    df = pd.DataFrame(records)
    if "paper_id" not in df.columns:
        raise ValueError("Expected `paper_id` column in segmented file")
    if "segments" not in df.columns:
        raise ValueError("Expected `segments` column in segmented file")
    return df


def flatten_segments(segments: Sequence[str]) -> str:
    return "\n".join(segment.strip() for segment in segments if segment)


def compute_bm25_matrix(documents: Sequence[str]) -> sparse.csr_matrix:
    vectorizer = CountVectorizer(token_pattern=r"(?u)\\b\\w+\\b")
    term_freq = vectorizer.fit_transform(documents)

    # BM25 weighting parameters
    k1 = 1.5
    b = 0.75

    doc_lengths = term_freq.sum(axis=1).A1
    avg_doc_length = doc_lengths.mean() if len(doc_lengths) else 0.0
    df = (term_freq > 0).sum(axis=0).A1
    n_docs = term_freq.shape[0]
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    rows, cols = term_freq.nonzero()
    data = term_freq.data
    bm25_data = []
    for idx, tf in enumerate(data):
        row = rows[idx]
        col = cols[idx]
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_lengths[row] / (avg_doc_length + 1e-9))
        weight = idf[col] * numerator / (denominator + 1e-9)
        bm25_data.append(weight)

    bm25_matrix = sparse.csr_matrix((bm25_data, (rows, cols)), shape=term_freq.shape)
    return bm25_matrix, vectorizer


def compute_splade_matrix(
    documents: Sequence[str],
    model_name: str,
    max_length: int,
    top_k: int,
) -> tuple[sparse.csr_matrix, dict[int, str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    model.to(device)

    vocab_size = model.config.vocab_size
    data: List[float] = []
    indices: List[int] = []
    indptr = [0]

    for text in tqdm(documents, desc="Encoding SPLADE", unit="doc"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits  # [1, seq_len, vocab]
        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # [seq_len, vocab]
        pooled = torch.log1p(torch.relu(log_probs)).sum(dim=0)
        if top_k is not None and top_k < vocab_size:
            values, idxs = torch.topk(pooled, k=top_k)
        else:
            values, idxs = pooled, torch.arange(vocab_size, device=pooled.device)
        values = values.cpu()
        idxs = idxs.cpu()
        mask = values > 0
        data.extend(values[mask].tolist())
        indices.extend(idxs[mask].tolist())
        indptr.append(len(indices))

    splade_matrix = sparse.csr_matrix((data, indices, indptr), shape=(len(documents), vocab_size))
    id_to_token = {idx: token for token, idx in tokenizer.get_vocab().items()}
    return splade_matrix, id_to_token


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_segmented_records(config.segmented_path)
    df["segments"] = df["segments"].apply(lambda segs: segs if isinstance(segs, list) else [])
    documents = [flatten_segments(segs) for segs in df["segments"].tolist()]

    print("[+] Building BM25 matrix...")
    bm25_matrix, vectorizer = compute_bm25_matrix(documents)
    sparse.save_npz(config.output_dir / "bm25_index.npz", bm25_matrix)
    (config.output_dir / "bm25_vocabulary.json").write_text(json.dumps(vectorizer.vocabulary_))

    print("[+] Building SPLADE matrix...")
    splade_matrix, id_to_token = compute_splade_matrix(
        documents,
        model_name=config.splade_model_name,
        max_length=config.max_splade_tokens,
        top_k=config.splade_top_k,
    )
    sparse.save_npz(config.output_dir / "splade_index.npz", splade_matrix)
    (config.output_dir / "splade_vocabulary.json").write_text(json.dumps(id_to_token))

    (config.output_dir / "paper_ids.json").write_text(json.dumps(df["paper_id"].tolist()))
    print(f"[✓] Sparse indices written to {config.output_dir}")


if __name__ == "__main__":
    main()
