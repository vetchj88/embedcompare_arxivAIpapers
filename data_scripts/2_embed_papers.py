"""Embed paper abstracts with multiple models and persist results per model.

This script stores embeddings in two forms:

1. ChromaDB collections for quick retrieval experiments.
2. Model specific directories under ``vector_stores`` containing the dense
   embedding matrix, metadata, and (when available) late-interaction
   token-level representations.

The additional persistence enables downstream alignment and ensemble
experiments that rely on the on-disk artefacts created here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- Configuration ---
MODELS: Dict[str, str] = {
    "bge_large": "BAAI/bge-large-en-v1.5",
    "gte_large": "thenlper/gte-large",
    "minilm": "all-MiniLM-L6-v2",
}
CSV_FILE = "papers_metadata.csv"
DB_PATH = "./chroma_db"
VECTOR_STORE_ROOT = Path("./vector_stores")

# --- Initialize ChromaDB Client ---
client = chromadb.PersistentClient(path=DB_PATH)

# --- Load Paper Metadata ---
df = pd.read_csv(CSV_FILE)
df["authors"] = df["authors"].apply(eval)
df["authors"] = df["authors"].apply(lambda authors: ", ".join(authors))

documents: List[str] = df["abstract"].tolist()
metadatas: List[Dict] = df.to_dict("records")
ids = [str(i) for i in range(len(df))]


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def persist_metadata(store_dir: Path, metadatas: List[Dict]) -> None:
    metadata_path = store_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadatas, fh, ensure_ascii=False, indent=2)


def persist_dense_embeddings(store_dir: Path, embeddings: np.ndarray) -> None:
    embeddings_path = store_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)


def persist_late_interaction(
    store_dir: Path, model: SentenceTransformer, documents: Iterable[str], ids: Iterable[str]
) -> bool:
    """Persist late-interaction/token embeddings when the model exposes them.

    Returns True when at least one representation is written, False otherwise.
    """

    late_dir = ensure_directory(store_dir / "late_interaction")
    try:
        token_embeddings = model.encode(
            documents,
            show_progress_bar=True,
            output_value="token_embeddings",
            convert_to_numpy=True,
        )
    except (TypeError, ValueError, AttributeError) as exc:
        print(f"  Skipping late-interaction export (not supported by this model): {exc}")
        return False

    if not isinstance(token_embeddings, (list, tuple)):
        token_embeddings = list(token_embeddings)

    written = False
    for doc_id, token_matrix in zip(ids, token_embeddings):
        if token_matrix is None:
            continue
        np.save(late_dir / f"{doc_id}.npy", np.asarray(token_matrix, dtype=np.float32))
        written = True

    if written:
        print(f"  Persisted late-interaction representations to {late_dir}")
    else:
        print("  Model returned no token-level representations; nothing persisted.")

    return written


def process_model(model_name: str, model_id: str) -> None:
    print(f"Processing model: {model_name} ({model_id})")

    model = SentenceTransformer(model_id)
    collection_name = f"papers_{model_name}"
    collection = client.get_or_create_collection(name=collection_name)

    embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)

    collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
    print(f"  Stored embeddings in ChromaDB collection: {collection_name}")

    store_dir = ensure_directory(VECTOR_STORE_ROOT / model_name)
    persist_dense_embeddings(store_dir, embeddings)
    persist_metadata(store_dir, metadatas)
    persist_late_interaction(store_dir, model, documents, ids)


for model_key, model_path in MODELS.items():
    process_model(model_key, model_path)

print("All models have been processed and data is stored in ChromaDB and vector stores.")
