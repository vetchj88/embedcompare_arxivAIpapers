"""
Unified embedding script: abstracts OR segmented sections, multiple models/backends.

Features
--------
- Abstract mode: CSV with title/authors/abstract → 1 vector per paper per model.
- Section mode: JSONL with sections per paper → 1 vector per section per model.
- Backends:
    * sentence-transformers  (SentenceTransformer)  [optional late-interaction export]
    * transformers (AutoTokenizer + AutoModel mean pooled)
- Persistence:
    * ChromaDB collections
    * Optional on-disk vector stores under ./vector_stores/<model>/<mode>/
      - embeddings.npy
      - metadata.json
      - (optional) late_interaction/*.npy (if backend supports token embeddings)

Environment variables
---------------------
EMBED_MODE = "abstracts" | "sections"             (default: abstracts)
CSV_FILE = "papers_metadata.csv"                   (for abstracts mode)
SEGMENTS_JSONL = "data/papers_with_sections.jsonl" (for sections mode)
DB_PATH = "./chroma_db"
VECTOR_STORE_ROOT = "./vector_stores"
EMBED_BATCH_SIZE = int (default 16 for ST; 8 for HF)
EMBED_MAX_LENGTH = int (default 512)

Model configs can be overridden via env (see MODEL_CONFIGS below).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Literal

import numpy as np
import pandas as pd
import chromadb

# Backends are optional; import lazily where possible
import torch

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    from transformers import AutoModel, AutoTokenizer
    _HAS_HF = True
except Exception:
    _HAS_HF = False


# -----------------------
# Configuration & inputs
# -----------------------
EMBED_MODE: Literal["abstracts", "sections"] = os.environ.get("EMBED_MODE", "abstracts").lower()  # noqa: E713
CSV_FILE = Path(os.environ.get("CSV_FILE", "papers_metadata.csv"))
SEGMENTS_JSONL = Path(os.environ.get("SEGMENTS_JSONL", "data/papers_with_sections.jsonl"))
DB_PATH = Path(os.environ.get("DB_PATH", "./chroma_db"))
VECTOR_STORE_ROOT = Path(os.environ.get("VECTOR_STORE_ROOT", "./vector_stores"))

# sensible batch defaults for each backend; can override via env
BATCH_SIZE_ST = int(os.environ.get("EMBED_BATCH_SIZE", 16))
BATCH_SIZE_HF = int(os.environ.get("EMBED_BATCH_SIZE", 8))
MAX_LENGTH = int(os.environ.get("EMBED_MAX_LENGTH", 512))

# Model registry: name → {backend, model_name, persist_late}
# You can safely enable/disable any subset here or via env overrides.
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # legacy models you already used
    "bge_large": {
        "backend": "st",
        "model_name": os.environ.get("BGE_MODEL", "BAAI/bge-large-en-v1.5"),
        "persist_late": True,  # try token embeddings if supported
    },
    "gte_large": {
        "backend": "st",
        "model_name": os.environ.get("GTE_MODEL", "thenlper/gte-large"),
        "persist_late": True,
    },
    "minilm": {
        "backend": "st",
        "model_name": os.environ.get("MINILM_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "persist_late": False,
    },
    # newer science-leaning models
    "specter2": {
        "backend": "hf",
        "model_name": os.environ.get("SPECTER2_MODEL", "allenai/specter2_base"),
        "persist_late": False,
    },
    "qwen3_embedding": {
        "backend": "hf",
        "model_name": os.environ.get("QWEN3_EMBED_MODEL", "Qwen/Qwen1.5-embedding-1024"),
        "persist_late": False,
    },
    "embedding_gemma": {
        "backend": "hf",
        "model_name": os.environ.get("EMBEDDING_GEMMA_MODEL", "google/embedding-gecko"),
        "persist_late": False,
    },
}


# -----------------------
# Utilities
# -----------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_metadata(store_dir: Path, metadatas: List[Dict[str, Any]]) -> None:
    with (store_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)


def save_dense(store_dir: Path, embeddings: np.ndarray) -> None:
    np.save(store_dir / "embeddings.npy", embeddings.astype("float32", copy=False))


def chunk_iter(it: Iterable[str], size: int) -> Iterable[Tuple[str, ...]]:
    batch: List[str] = []
    for x in it:
        batch.append(x)
        if len(batch) >= size:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


# -----------------------
# Data loading
# -----------------------
def load_abstracts(csv_path: Path) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    df = pd.read_csv(csv_path)
    # normalize authors column to a string list → "A, B"
    if "authors" in df.columns:
        try:
            df["authors"] = df["authors"].apply(eval)
        except Exception:
            pass
        df["authors"] = df["authors"].apply(lambda a: ", ".join(a) if isinstance(a, (list, tuple)) else str(a))
    documents = df["abstract"].fillna("").astype(str).tolist()
    metadatas = df.to_dict("records")
    # prefer arXiv ID if present; else index
    if "id" in df.columns:
        ids = [str(x) for x in df["id"].tolist()]
    elif "arxiv_id" in df.columns:
        ids = [str(x) for x in df["arxiv_id"].tolist()]
    else:
        ids = [str(i) for i in range(len(df))]
    return documents, metadatas, ids


def load_sections(jsonl_path: Path) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    segments: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            base = {
                "paper_id": row["paper_id"],
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "authors": ", ".join(row.get("authors", [])),
                "date": row.get("date"),
                "pdf_path": row.get("pdf_path"),
            }
            sections = row.get("sections", {}) or {}
            for sec_name, text in sections.items():
                if not text:
                    continue
                segments.append(
                    {
                        "paper_id": row["paper_id"],
                        "section": sec_name,
                        "text": text,
                        "metadata": {**base, "section": sec_name},
                    }
                )
    if not segments:
        raise FileNotFoundError(f"No segmented content in {jsonl_path}")
    texts = [s["text"] for s in segments]
    metadatas = [s["metadata"] for s in segments]
    # stable unique IDs per segment
    ids = [f"{s['paper_id']}::{s['section']}::{i}" for i, s in enumerate(segments)]
    return texts, metadatas, ids


# -----------------------
# Encoders
# -----------------------
class STEncoder:
    def __init__(self, model_name: str):
        if not _HAS_ST:
            raise ImportError("sentence-transformers not installed")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Iterable[str], batch_size: int = BATCH_SIZE_ST) -> np.ndarray:
        return self.model.encode(
            list(texts),
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=batch_size,
        ).astype("float32")

    # optional, not all ST models support token outputs
    def try_token_embeddings(self, texts: Iterable[str], batch_size: int = BATCH_SIZE_ST):
        try:
            toks = self.model.encode(
                list(texts),
                show_progress_bar=True,
                output_value="token_embeddings",
                convert_to_numpy=True,
                batch_size=batch_size,
            )
            # normalize shape: list of (seq_len, dim) ndarrays
            if not isinstance(toks, (list, tuple)):
                toks = list(toks)
            return toks
        except Exception as e:
            print(f"  Skipping late-interaction export (not supported): {e}")
            return None


class HFEncoder:
    def __init__(self, model_name: str, max_length: int = MAX_LENGTH):
        if not _HAS_HF:
            raise ImportError("transformers not installed")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.max_length = max_length

    @torch.inference_mode()
    def encode(self, texts: Iterable[str], batch_size: int = BATCH_SIZE_HF) -> np.ndarray:
        outs: List[np.ndarray] = []
        for batch in chunk_iter(texts, batch_size):
            toks = self.tok(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            hidden = self.model(**toks).last_hidden_state
            pooled = mean_pool(hidden, toks["attention_mask"])
            normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outs.append(normed.cpu().numpy().astype("float32"))
        return np.vstack(outs)

    def try_token_embeddings(self, texts: Iterable[str], batch_size: int = BATCH_SIZE_HF):
        # Not supported in this generic HF path (no standardized token embedding output)
        return None


# -----------------------
# Main processing routine
# -----------------------
def run_for_model(
    model_key: str,
    cfg: Dict[str, Any],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    mode_tag: str,
    persist_vector_store: bool = True,
) -> None:
    backend = cfg.get("backend", "st")
    model_name = cfg["model_name"]
    persist_late = bool(cfg.get("persist_late", False))

    print(f"[+] Processing model: {model_key} ({model_name}) | backend={backend} | mode={mode_tag}")

    # encoder
    if backend == "st":
        encoder = STEncoder(model_name)
    elif backend == "hf":
        encoder = HFEncoder(model_name)
    else:
        raise ValueError(f"Unknown backend '{backend}' for model {model_key}")

    # embeddings
    embeddings = encoder.encode(documents)
    assert embeddings.shape[0] == len(documents), "Embedding count mismatch"

    # ChromaDB upsert
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection_name = f"{mode_tag}_{model_key}"
    col = client.get_or_create_collection(name=collection_name)
    # clear existing to keep id spaces clean (you may remove this if you prefer upserts)
    try:
        col.delete(where={})
    except Exception:
        pass
    col.add(embeddings=embeddings.tolist(), documents=documents, metadatas=metadatas, ids=ids)
    print(f"    Stored {len(ids)} embeddings in ChromaDB collection: {collection_name}")

    # vector store persistence
    if persist_vector_store:
        store_dir = ensure_dir(VECTOR_STORE_ROOT / model_key / mode_tag)
        save_dense(store_dir, embeddings)
        save_metadata(store_dir, metadatas)
        # late-interaction (token-level) only for ST when requested
        if persist_late and backend == "st":
            toks = encoder.try_token_embeddings(documents)
            if toks is not None:
                late_dir = ensure_dir(store_dir / "late_interaction")
                written = 0
                for doc_id, tmat in zip(ids, toks):
                    if tmat is None:
                        continue
                    np.save(late_dir / f"{doc_id}.npy", np.asarray(tmat, dtype=np.float32))
                    written += 1
                if written:
                    print(f"    Persisted {written} token-level representations to {late_dir}")
                else:
                    print("    Model returned no token-level representations; nothing persisted.")

    print(f"[✓] Done model: {model_key}\n")


def main() -> None:
    if EMBED_MODE == "abstracts":
        documents, metadatas, ids = load_abstracts(CSV_FILE)
        mode_tag = "abstracts"
    elif EMBED_MODE == "sections":
        documents, metadatas, ids = load_sections(SEGMENTS_JSONL)
        mode_tag = "sections"
    else:
        raise ValueError("EMBED_MODE must be 'abstracts' or 'sections'")

    if not documents:
        raise RuntimeError("No documents to embed.")

    print(f"Mode: {EMBED_MODE} | Documents: {len(documents)} | Models: {list(MODEL_CONFIGS.keys())}")

    for model_key, cfg in MODEL_CONFIGS.items():
        run_for_model(model_key, cfg, documents, metadatas, ids, mode_tag, persist_vector_store=True)

    print("All models processed and data persisted to ChromaDB (and vector_stores where enabled).")


if __name__ == "__main__":
    main()
