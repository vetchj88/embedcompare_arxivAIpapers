"""Embed segmented paper sections with multiple transformer encoders."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import chromadb
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DATA_DIR = Path("data")
SEGMENTS_JSONL = DATA_DIR / "papers_with_sections.jsonl"
DB_PATH = Path("./chroma_db")
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", 8))
MAX_LENGTH = int(os.environ.get("EMBED_MAX_LENGTH", 512))

MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "specter2": {
        "model_name": os.environ.get("SPECTER2_MODEL", "allenai/specter2_base"),
    },
    "qwen3_embedding": {
        "model_name": os.environ.get("QWEN3_EMBED_MODEL", "Qwen/Qwen1.5-embedding-1024"),
    },
    "embedding_gemma": {
        "model_name": os.environ.get("EMBEDDING_GEMMA_MODEL", "google/embedding-gecko"),
    },
}


def load_segmented_texts(jsonl_path: Path) -> List[Dict[str, str]]:
    segments: List[Dict[str, str]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for row in handle:
            if not row.strip():
                continue
            payload = json.loads(row)
            base_metadata = {
                "paper_id": payload["paper_id"],
                "title": payload.get("title", ""),
                "abstract": payload.get("abstract", ""),
                "authors": ", ".join(payload.get("authors", [])),
                "date": payload.get("date"),
                "pdf_path": payload.get("pdf_path"),
            }
            for section_name, text in payload.get("sections", {}).items():
                if not text:
                    continue
                segments.append(
                    {
                        "paper_id": payload["paper_id"],
                        "section": section_name,
                        "text": text,
                        "metadata": {**base_metadata, "section": section_name},
                    }
                )
    return segments


class HuggingFaceEncoder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: Iterable[str]) -> np.ndarray:
        batched_embeddings: List[np.ndarray] = []
        for batch in _batch(texts, BATCH_SIZE):
            tokens = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            outputs = self.model(**tokens)
            embeddings = _mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
            normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            batched_embeddings.append(normalized.cpu().numpy().astype("float32"))
        return np.vstack(batched_embeddings)


def _batch(iterable: Iterable[str], size: int) -> Iterable[Tuple[str, ...]]:
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def upsert_segments(collection: chromadb.Collection, segments: List[Dict[str, str]], embeddings: np.ndarray) -> None:
    ids = []
    documents = []
    metadatas = []
    for index, segment in enumerate(segments):
        ids.append(f"{segment['paper_id']}::{segment['section']}::{index}")
        documents.append(segment["text"])
        metadatas.append(segment["metadata"])

    collection.add(embeddings=embeddings.tolist(), ids=ids, metadatas=metadatas, documents=documents)


def embed_segments() -> None:
    segments = load_segmented_texts(SEGMENTS_JSONL)
    if not segments:
        raise FileNotFoundError(
            f"No segmented content found at {SEGMENTS_JSONL}. Run 1_fetch_papers.py first."
        )

    client = chromadb.PersistentClient(path=str(DB_PATH))

    for model_key, config in MODEL_CONFIGS.items():
        model_name = config["model_name"]
        print(f"[+] Embedding with {model_key} ({model_name})")
        encoder = HuggingFaceEncoder(model_name)

        collection_name = f"sections_{model_key}"
        collection = client.get_or_create_collection(name=collection_name)
        collection.delete(where={})

        texts = [segment["text"] for segment in segments]
        embeddings = encoder.encode(texts)
        upsert_segments(collection, segments, embeddings)
        print(f"[✅] Stored {len(segments)} segment embeddings in {collection_name}")

    print("All models processed. Segment embeddings are stored in ChromaDB.")


if __name__ == "__main__":
    embed_segments()
