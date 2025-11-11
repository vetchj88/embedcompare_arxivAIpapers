"""Contrastive learner that aligns paper views and extracted claims."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase


LOGGER = logging.getLogger("claim_contrastive")
DEFAULT_VIEWS = ["title", "abstract", "claims", "methods", "conclusion", "datasets"]


@dataclass
class ViewPair:
    paper_id: str
    view_a: str
    view_b: str
    text_a: str
    text_b: str


class PaperViewsDataset(Dataset[ViewPair]):
    """Dataset yielding positive pairs across paper views."""

    def __init__(
        self,
        metadata_path: Path,
        claims_path: Path,
        include_views: Optional[Sequence[str]] = None,
        max_pairs_per_paper: Optional[int] = None,
        min_tokens: int = 5,
    ) -> None:
        if include_views is None:
            include_views = DEFAULT_VIEWS
        self.include_views = list(include_views)

        self.metadata = self._load_metadata(metadata_path)
        self.claims = self._load_claims(claims_path)
        self.paper_views: Dict[str, Dict[str, str]] = {}
        self.pairs: List[ViewPair] = []
        self._build_pairs(max_pairs_per_paper=max_pairs_per_paper, min_tokens=min_tokens)

    @staticmethod
    def _load_metadata(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "id" not in df.columns:
            if "paper_id" in df.columns:
                df = df.rename(columns={"paper_id": "id"})
            else:
                raise ValueError("Metadata must include an 'id' column")
        df["id"] = df["id"].astype(str)
        return df.set_index("id")

    @staticmethod
    def _load_claims(path: Path) -> Dict[str, Dict[str, List[str]]]:
        with path.open() as fh:
            data = json.load(fh)
        claims: Dict[str, Dict[str, List[str]]] = {}
        for entry in data:
            extracted = entry.get("extracted", {})
            claims[str(entry["id"])] = {
                "claims": entry.get("extracted", {}).get("claims", []),
                "tasks": extracted.get("tasks", []),
                "datasets": extracted.get("datasets", []),
                "methods": extracted.get("methods", []),
            }
        return claims

    def _build_pairs(self, max_pairs_per_paper: Optional[int], min_tokens: int) -> None:
        for paper_id, row in self.metadata.iterrows():
            extracted = self.claims.get(paper_id, {"claims": [], "datasets": [], "methods": [], "tasks": []})
            views: Dict[str, str] = {}
            title = str(row.get("title", "")).strip()
            abstract = str(row.get("abstract", "")).strip()
            if title:
                views["title"] = title
            if abstract:
                views["abstract"] = abstract
            if extracted.get("claims"):
                views["claims"] = "; ".join(extracted["claims"])
                # Treat claims summary as a pseudo-conclusion to enrich the view set.
                views.setdefault("conclusion", views["claims"])
            if extracted.get("methods"):
                views["methods"] = "; ".join(extracted["methods"])
            if extracted.get("datasets"):
                views["datasets"] = "; ".join(extracted["datasets"])
            if extracted.get("tasks") and "tasks" in self.include_views:
                views["tasks"] = "; ".join(extracted["tasks"])

            filtered_views = {
                key: value
                for key, value in views.items()
                if key in self.include_views and len(value.split()) >= min_tokens
            }
            if len(filtered_views) < 2:
                continue
            self.paper_views[paper_id] = filtered_views

            combos = list(combinations(filtered_views.items(), 2))
            if max_pairs_per_paper is not None and len(combos) > max_pairs_per_paper:
                combos = combos[:max_pairs_per_paper]
            for (view_a, text_a), (view_b, text_b) in combos:
                self.pairs.append(
                    ViewPair(
                        paper_id=paper_id,
                        view_a=view_a,
                        view_b=view_b,
                        text_a=text_a,
                        text_b=text_b,
                    )
                )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> ViewPair:
        return self.pairs[index]

    def iter_documents(self, strategy: str = "all") -> Iterator[Tuple[str, str]]:
        for paper_id, views in self.paper_views.items():
            if strategy == "all":
                text = " \n".join(dict.fromkeys(views.values()))
            elif strategy == "claims":
                text = views.get("claims") or " \n".join(dict.fromkeys(views.values()))
            else:
                raise ValueError(f"Unknown export strategy: {strategy}")
            yield paper_id, text


class ClaimContrastiveModel(nn.Module):
    def __init__(self, model_name: str, pooling: str = "mean") -> None:
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0]
        elif self.pooling == "mean":
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).type_as(hidden)
            summed = torch.sum(hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-6)
            embeddings = summed / counts
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling}")
        return F.normalize(embeddings, p=2, dim=-1)

    @torch.no_grad()
    def encode(
        self,
        tokenizer: PreTrainedTokenizerBase,
        texts: Sequence[str],
        device: Optional[torch.device] = None,
        batch_size: int = 16,
        max_length: int = 256,
    ) -> np.ndarray:
        self.eval()
        device = device or next(self.parameters()).device
        results: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            embeddings = self.forward(**encoded).cpu().numpy()
            results.append(embeddings)
        if not results:
            return np.zeros((0, self.encoder.config.hidden_size))
        return np.vstack(results)

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "model_name": self.model_name,
                "pooling": self.pooling,
            },
            output_dir / "contrastive.pt",
        )

    @classmethod
    def load(cls, checkpoint_dir: Path, device: Optional[torch.device] = None) -> "ClaimContrastiveModel":
        payload = torch.load(checkpoint_dir / "contrastive.pt", map_location=device or "cpu")
        model = cls(model_name=payload["model_name"], pooling=payload.get("pooling", "mean"))
        model.load_state_dict(payload["state_dict"])
        if device:
            model.to(device)
        return model


def build_dataloader(
    dataset: PaperViewsDataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_length: int,
) -> DataLoader:
    def collate(batch: Sequence[ViewPair]) -> Dict[str, torch.Tensor]:
        texts_a = [item.text_a for item in batch]
        texts_b = [item.text_b for item in batch]
        encoded_a = tokenizer(
            texts_a,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded_b = tokenizer(
            texts_b,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "input_ids_a": encoded_a["input_ids"],
            "attention_mask_a": encoded_a["attention_mask"],
            "input_ids_b": encoded_b["input_ids"],
            "attention_mask_b": encoded_b["attention_mask"],
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)


def contrastive_loss(emb_a: torch.Tensor, emb_b: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = emb_a @ emb_b.t() / temperature
    labels = torch.arange(len(emb_a), device=emb_a.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return (loss_a + loss_b) / 2


def train_contrastive(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PaperViewsDataset(
        metadata_path=Path(args.metadata),
        claims_path=Path(args.claims),
        include_views=args.views,
        max_pairs_per_paper=args.max_pairs_per_paper,
        min_tokens=args.min_tokens,
    )
    if not len(dataset):
        raise RuntimeError("No training pairs were generated; check inputs.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = ClaimContrastiveModel(model_name=args.model_name, pooling=args.pooling)
    model.to(device)

    dataloader = build_dataloader(dataset, tokenizer, args.batch_size, args.max_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(dataloader)
    LOGGER.info("Starting training for %d epochs (%d steps)", args.epochs, total_steps)

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids_a = batch["input_ids_a"].to(device)
            attn_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attn_b = batch["attention_mask_b"].to(device)

            embeddings_a = model(input_ids_a, attn_a)
            embeddings_b = model(input_ids_b, attn_b)
            loss = contrastive_loss(embeddings_a, embeddings_b, temperature=args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
        LOGGER.info("Epoch %d average loss: %.4f", epoch + 1, epoch_loss / len(dataloader))

    output_dir = Path(args.output_dir)
    model.save(output_dir)
    tokenizer.save_pretrained(output_dir)
    LOGGER.info("Saved contrastive model to %s", output_dir)

    if args.export_embeddings:
        export_path = Path(args.export_embeddings)
        export_embeddings(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            output_path=export_path,
            device=device,
            max_length=args.max_length,
            batch_size=args.export_batch_size,
            strategy=args.export_strategy,
        )


def export_embeddings(
    model: ClaimContrastiveModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: PaperViewsDataset,
    output_path: Path,
    device: torch.device,
    max_length: int,
    batch_size: int,
    strategy: str = "all",
) -> None:
    paper_ids: List[str] = []
    texts: List[str] = []
    for paper_id, text in dataset.iter_documents(strategy=strategy):
        paper_ids.append(paper_id)
        texts.append(text)
    if not paper_ids:
        raise RuntimeError("No documents available for export")
    embeddings = model.encode(tokenizer, texts, device=device, max_length=max_length, batch_size=batch_size)
    records = [
        {"id": paper_id, "embedding": embedding.tolist()}
        for paper_id, embedding in zip(paper_ids, embeddings)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(records, fh, indent=2)
    LOGGER.info("Exported %d enhanced embeddings to %s", len(records), output_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=str, default="papers_metadata.csv")
    parser.add_argument("--claims", type=str, default="extracted_claims.json")
    parser.add_argument("--model-name", type=str, default="thenlper/gte-large")
    parser.add_argument("--pooling", type=str, choices=["mean", "cls"], default="mean")
    parser.add_argument("--views", nargs="*", default=DEFAULT_VIEWS, help="Views to include in training pairs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--export-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-pairs-per-paper", type=int, default=None)
    parser.add_argument("--min-tokens", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="checkpoints/claim_contrastive")
    parser.add_argument("--export-embeddings", type=str, default="enhanced_embeddings.json")
    parser.add_argument("--export-strategy", type=str, default="all", choices=["all", "claims"])
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train_contrastive(args)
