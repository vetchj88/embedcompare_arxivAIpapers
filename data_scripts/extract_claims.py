"""Run an LLM over paper segments to extract normalized claims metadata.

This script reads paper metadata and optional pre-segmented content, calls a
large language model for each segment, and stores normalized claims, tasks,
datasets, and method descriptions.  The output JSON can be consumed by the
contrastive learner and evaluation pipeline.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency for offline use
    OpenAI = None  # type: ignore


LOGGER = logging.getLogger("extract_claims")
DEFAULT_SEGMENT_KEYS = ("title", "abstract")


@dataclass
class Segment:
    """Represents a textual segment of a paper to send to the LLM."""

    name: str
    text: str


@dataclass
class ExtractedMetadata:
    """Normalized structured metadata produced by the LLM."""

    claims: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)

    @classmethod
    def from_response(cls, response: Dict[str, Iterable[str]]) -> "ExtractedMetadata":
        return cls(
            claims=_normalize_list(response.get("claims", [])),
            tasks=_normalize_list(response.get("tasks", [])),
            datasets=_normalize_list(response.get("datasets", [])),
            methods=_normalize_list(
                response.get("method_descriptions", response.get("methods", []))
            ),
        )

    def merge(self, other: "ExtractedMetadata") -> None:
        """Merge another metadata object into this one with deduplication."""

        self.claims = _merge_unique(self.claims, other.claims)
        self.tasks = _merge_unique(self.tasks, other.tasks)
        self.datasets = _merge_unique(self.datasets, other.datasets)
        self.methods = _merge_unique(self.methods, other.methods)

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "claims": self.claims,
            "tasks": self.tasks,
            "datasets": self.datasets,
            "methods": self.methods,
        }


def _normalize_entry(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _normalize_list(values: Iterable[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for value in values:
        if not value:
            continue
        item = _normalize_entry(str(value))
        if item and item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


def _merge_unique(left: List[str], right: Iterable[str]) -> List[str]:
    seen = set(left)
    merged = list(left)
    for item in right:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return merged


def chunk_text(text: str, max_words: int = 220) -> List[str]:
    """Split large segments into roughly max_words sized chunks."""

    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks: List[str] = []
    for start in range(0, len(words), max_words):
        chunk = " ".join(words[start : start + max_words])
        chunks.append(chunk)
    return chunks


class ClaimExtractor:
    """Encapsulates calls to the LLM or a deterministic mock extractor."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        dry_run: bool = False,
        api_base: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.dry_run = dry_run or OpenAI is None
        self.client = None
        if not self.dry_run:
            self.client = OpenAI(base_url=api_base) if api_base else OpenAI()
            LOGGER.info("Using OpenAI model %s", model)
        else:
            LOGGER.warning(
                "Running in dry-run mode; outputs will be heuristic placeholders."
            )

    def extract(self, segment: Segment) -> ExtractedMetadata:
        if self.dry_run:
            return self._mock_response(segment)
        assert self.client is not None  # mypy hint
        prompt = self._build_prompt(segment)
        response = self.client.responses.create(
            model=self.model,
            response_format={"type": "json_object"},
            temperature=self.temperature,
            input=[{"role": "user", "content": prompt}],
        )
        content = response.output[0].content[0].text  # type: ignore[attr-defined]
        return ExtractedMetadata.from_response(json.loads(content))

    def _build_prompt(self, segment: Segment) -> str:
        return (
            "You are an expert research curator. For the following paper segment, "
            "list distinct scientific claims, associated tasks, datasets, and "
            "method descriptions. Respond as JSON with keys 'claims', 'tasks', "
            "'datasets', and 'method_descriptions'. Use concise noun phrases and "
            "normalize dataset/task names. If information is unavailable, return "
            "an empty list for that field.\n\n"
            f"Segment ({segment.name}):\n" + segment.text.strip()
        )

    def _mock_response(self, segment: Segment) -> ExtractedMetadata:
        words = segment.text.split()
        seed = sum(ord(c) for c in segment.name) + len(words)
        random.seed(seed)
        sample_claim = " ".join(words[: min(12, len(words))]) or "no claim"
        mock_data = {
            "claims": [sample_claim],
            "tasks": [random.choice(["classification", "generation", "analysis"])],
            "datasets": [random.choice(["synthetic", "imagenet", "custom dataset"])],
            "method_descriptions": [
                random.choice([
                    "transformer-based approach",
                    "bayesian optimization pipeline",
                    "novel contrastive learner",
                ])
            ],
        }
        return ExtractedMetadata.from_response(mock_data)


def build_segments(row: pd.Series, extra_segments: Optional[Dict[str, List[str]]] = None) -> List[Segment]:
    segments: List[Segment] = []
    for key in DEFAULT_SEGMENT_KEYS:
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        for idx, chunk in enumerate(chunk_text(value)):
            suffix = f"_{idx}" if idx else ""
            segments.append(Segment(name=f"{key}{suffix}", text=chunk))
    if extra_segments:
        for key, values in extra_segments.items():
            for idx, value in enumerate(values):
                clean_value = str(value).strip()
                if not clean_value:
                    continue
                suffix = f"_{idx}" if idx else ""
                segments.append(Segment(name=f"{key}{suffix}", text=clean_value))
    return segments


def load_extra_segments(path: Optional[Path]) -> Dict[str, Dict[str, List[str]]]:
    if not path:
        return {}
    with path.open() as fh:
        data = json.load(fh)
    return {entry["id"]: entry.get("segments", {}) for entry in data}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("papers_metadata.csv"),
        help="Path to CSV file containing paper metadata.",
    )
    parser.add_argument(
        "--segments-json",
        type=Path,
        default=None,
        help="Optional JSON file with additional segments keyed by paper id.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("extracted_claims.json"),
        help="Output JSON path for normalized claims.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="LLM model identifier to use for extraction.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, skip remote LLM calls and generate heuristic outputs.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Optional custom API base URL for the OpenAI-compatible endpoint.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    if not args.metadata.exists():
        LOGGER.error("Metadata file %s not found", args.metadata)
        sys.exit(1)

    df = pd.read_csv(args.metadata)
    if "id" not in df.columns:
        if "paper_id" in df.columns:
            df = df.rename(columns={"paper_id": "id"})
        else:
            raise ValueError("Metadata must contain an 'id' or 'paper_id' column identifying each paper")

    if df['id'].isnull().any():
        raise ValueError("Paper id column contains missing values")

    extra_segments = load_extra_segments(args.segments_json)
    extractor = ClaimExtractor(
        model=args.model,
        temperature=args.temperature,
        dry_run=args.dry_run,
        api_base=args.api_base,
    )

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting claims"):
        paper_id = str(row["id"])
        segments = build_segments(row, extra_segments.get(paper_id))
        metadata = ExtractedMetadata()
        segment_outputs = {}
        for segment in segments:
            try:
                segment_metadata = extractor.extract(segment)
            except Exception as exc:  # pragma: no cover - best effort logging
                LOGGER.exception("Failed to extract claims for %s segment %s", paper_id, segment.name)
                continue
            metadata.merge(segment_metadata)
            segment_outputs[segment.name] = segment_metadata.to_dict()
        results.append(
            {
                "id": paper_id,
                "title": row.get("title", ""),
                "segments": [segment.__dict__ for segment in segments],
                "extracted": metadata.to_dict(),
                "segment_outputs": segment_outputs,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(results, fh, indent=2)
    LOGGER.info("Wrote normalized claim annotations for %d papers to %s", len(results), args.output)


if __name__ == "__main__":
    main()
