"""Train a learned pooling layer over section embeddings stored in ChromaDB."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import chromadb
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path("data")
DEFAULT_SEGMENTS_JSONL = DATA_DIR / "papers_with_sections.jsonl"
DB_PATH = Path("./chroma_db")


@dataclass
class TrainingExample:
    anchor: str
    positive: str
    negative: Optional[str] = None
    weight: float = 1.0


def load_training_examples(path: Path) -> List[TrainingExample]:
    examples: List[TrainingExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            examples.append(
                TrainingExample(
                    anchor=payload["anchor"],
                    positive=payload["positive"],
                    negative=payload.get("negative"),
                    weight=float(payload.get("weight", 1.0)),
                )
            )
    if not examples:
        raise ValueError(f"No training examples found in {path}")
    return examples


class SectionEmbeddingStore:
    def __init__(self, collection: chromadb.Collection):
        self.collection = collection
        self._cache: Dict[str, torch.Tensor] = {}

    def __call__(self, paper_id: str) -> torch.Tensor:
        if paper_id not in self._cache:
            record = self.collection.get(where={"paper_id": paper_id}, include=["ids", "embeddings", "metadatas"])
            ids = record.get("ids", [])
            embeddings = record.get("embeddings", [])
            if not ids:
                raise KeyError(f"No embeddings found for paper_id={paper_id}")
            ordered = sorted(zip(ids, embeddings), key=lambda item: item[0])
            tensor = torch.tensor(np.array([item[1] for item in ordered], dtype=np.float32))
            self._cache[paper_id] = tensor
        return self._cache[paper_id]


class TripletDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], float]]):
    def __init__(self, examples: Sequence[TrainingExample], store: SectionEmbeddingStore):
        self.examples = list(examples)
        self.store = store

    def __len__(self) -> int:  # pragma: no cover - simple forwarder
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], float]:
        example = self.examples[index]
        anchor = self.store(example.anchor)
        positive = self.store(example.positive)
        negative = self.store(example.negative) if example.negative else None
        return anchor, positive, negative, example.weight


def pad_segments(segments: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = [segment.size(0) for segment in segments]
    max_len = max(lengths)
    embed_dim = segments[0].size(1)
    batch_size = len(segments)
    padded = torch.zeros(batch_size, max_len, embed_dim, dtype=segments[0].dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for idx, segment in enumerate(segments):
        length = segment.size(0)
        padded[idx, :length] = segment
        mask[idx, :length] = True
    return padded, mask


def collate_triplets(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], float]]):
    anchors, positives, negatives, weights = zip(*batch)
    anchor_padded, anchor_mask = pad_segments(anchors)
    positive_padded, positive_mask = pad_segments(positives)
    negative_padded = negative_mask = None
    if any(negative is not None for negative in negatives):
        placeholder = torch.zeros((1, anchors[0].size(1)), dtype=anchors[0].dtype)
        filtered_negatives = [neg if neg is not None else placeholder for neg in negatives]
        negative_padded, negative_mask = pad_segments(filtered_negatives)
        for idx, negative in enumerate(negatives):
            if negative is None:
                negative_mask[idx] = False
    return {
        "anchor": anchor_padded,
        "anchor_mask": anchor_mask,
        "positive": positive_padded,
        "positive_mask": positive_mask,
        "negative": negative_padded,
        "negative_mask": negative_mask,
        "weight": torch.tensor(weights, dtype=torch.float32),
    }


class SectionPooler(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Linear(embed_dim, 1)

    def forward(self, segments: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attention_scores = self.attention(segments).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(attention_scores, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * segments, dim=1)
        return pooled


def train_pooler(
    pooler: SectionPooler,
    dataloader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    margin: float,
    lr: float,
) -> None:
    optimizer = torch.optim.Adam(pooler.parameters(), lr=lr)
    pooler.to(device)

    for epoch in range(epochs):
        pooler.train()
        running_loss = 0.0
        for batch in dataloader:
            anchor = batch["anchor"].to(device)
            anchor_mask = batch["anchor_mask"].to(device)
            positive = batch["positive"].to(device)
            positive_mask = batch["positive_mask"].to(device)
            weights = batch["weight"].to(device)

            anchor_repr = pooler(anchor, anchor_mask)
            positive_repr = pooler(positive, positive_mask)
            positive_sim = torch.cosine_similarity(anchor_repr, positive_repr)

            if batch["negative"] is not None and batch["negative_mask"] is not None:
                negative = batch["negative"].to(device)
                negative_mask = batch["negative_mask"].to(device)
                negative_repr = pooler(negative, negative_mask)
                negative_sim = torch.cosine_similarity(anchor_repr, negative_repr)
                loss = torch.relu(margin - positive_sim + negative_sim)
            else:
                loss = 1.0 - positive_sim

            loss = (loss * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(dataloader), 1)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")


def pool_embeddings_for_papers(
    pooler: SectionPooler,
    store: SectionEmbeddingStore,
    paper_ids: Iterable[str],
    *,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    pooler.eval()
    pooled: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for paper_id in paper_ids:
            segments = store(paper_id)
            padded, mask = pad_segments([segments])
            pooled_vec = pooler(padded.to(device), mask.to(device))
            pooled[paper_id] = pooled_vec.cpu().numpy()[0].astype("float32")
    return pooled


def save_pooled_embeddings(
    client: chromadb.PersistentClient,
    collection_name: str,
    pooled_embeddings: Dict[str, np.ndarray],
    *,
    metadata: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    collection = client.get_or_create_collection(name=collection_name)
    collection.delete(where={})

    ids = []
    embeddings = []
    metadatas = []
    for paper_id, vector in pooled_embeddings.items():
        ids.append(paper_id)
        embeddings.append(vector.tolist())
        meta = {"paper_id": paper_id}
        if metadata and paper_id in metadata:
            meta.update(metadata[paper_id])
        metadatas.append(meta)

    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)


def load_metadata_lookup(jsonl_path: Path) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            lookup[payload["paper_id"]] = {
                "title": payload.get("title"),
                "date": payload.get("date"),
                "authors": ", ".join(payload.get("authors", [])),
            }
    return lookup


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a pooling model over segmented paper embeddings")
    parser.add_argument("training_file", type=Path, help="JSONL file with anchor/positive[/negative] triples")
    parser.add_argument("--collection", default="sections_specter2", help="ChromaDB collection containing segment embeddings")
    parser.add_argument("--output-collection", default="pooled_specter2", help="Collection to store pooled embeddings")
    parser.add_argument("--segments-jsonl", type=Path, default=DEFAULT_SEGMENTS_JSONL, help="Segmented papers JSONL for metadata")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    client = chromadb.PersistentClient(path=str(DB_PATH))
    segment_collection = client.get_or_create_collection(args.collection)

    examples = load_training_examples(args.training_file)
    store = SectionEmbeddingStore(segment_collection)
    dataset = TripletDataset(examples, store)

    # Peek at embedding dimensionality
    sample_tensor = dataset[0][0]
    embed_dim = sample_tensor.size(1)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_triplets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pooler = SectionPooler(embed_dim)

    train_pooler(pooler, dataloader, device=device, epochs=args.epochs, margin=args.margin, lr=args.lr)

    metadata_lookup = load_metadata_lookup(args.segments_jsonl)
    paper_ids = sorted(metadata_lookup.keys())
    pooled_embeddings = pool_embeddings_for_papers(pooler, store, paper_ids, device=device)

    save_pooled_embeddings(
        client,
        args.output_collection,
        pooled_embeddings,
        metadata=metadata_lookup,
    )
    print(f"[✅] Pooled embeddings stored in collection '{args.output_collection}'")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
