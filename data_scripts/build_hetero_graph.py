"""Build a heterogeneous scholarly graph from local metadata and OpenAlex."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - pandas is expected in the environment
    raise SystemExit("pandas is required to run this script") from exc


EDGE_TYPES = {"citation", "co_author", "venue_membership", "co_citation", "keyphrase_overlap"}
DEFAULT_OUTPUT_DIR = Path("graph_data")
STOPWORDS = {
    "and",
    "for",
    "from",
    "have",
    "https",
    "https",
    "into",
    "over",
    "such",
    "that",
    "the",
    "their",
    "this",
    "with",
}


@dataclass
class PaperRecord:
    """Normalized metadata for a paper."""

    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    published: Optional[str]


class OpenAlexClient:
    """Thin wrapper around the OpenAlex API with basic caching."""

    def __init__(self, cache_path: Path, mailto: Optional[str] = None) -> None:
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            self._cache: Dict[str, dict] = json.loads(cache_path.read_text())
        else:
            self._cache = {}
        self.session = requests.Session()
        self.mailto = mailto

    def _write_cache(self) -> None:
        tmp_path = self.cache_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self._cache, indent=2))
        tmp_path.replace(self.cache_path)

    def fetch_work(self, arxiv_id: str) -> Optional[dict]:
        """Fetch a work by its arXiv identifier (without the version suffix)."""

        if arxiv_id in self._cache:
            return self._cache[arxiv_id]

        url = f"https://api.openalex.org/works/arXiv:{arxiv_id}"
        params = {}
        if self.mailto:
            params["mailto"] = self.mailto

        response = self.session.get(url, params=params, timeout=20)
        if response.status_code == 404:
            self._cache[arxiv_id] = None
            self._write_cache()
            return None
        response.raise_for_status()
        payload = response.json()
        self._cache[arxiv_id] = payload
        self._write_cache()
        return payload


def load_local_metadata(csv_path: Path, limit: Optional[int] = None) -> List[PaperRecord]:
    """Load core paper metadata produced by earlier data scripts."""

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Metadata file '{csv_path}' does not exist. Run data_scripts/1_fetch_papers.py first."
        )

    df = pd.read_csv(csv_path)
    if "authors" in df.columns:
        df["authors"] = df["authors"].apply(_normalize_authors)

    records = []
    for _, row in df.head(limit).iterrows():
        records.append(
            PaperRecord(
                paper_id=str(row.get("id")),
                title=str(row.get("title", "")),
                abstract=str(row.get("abstract", "")),
                authors=list(row.get("authors", [])),
                published=row.get("date"),
            )
        )
    return records


def _normalize_authors(value: str) -> List[str]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(author) for author in parsed]
    except json.JSONDecodeError:
        pass
    return [name.strip() for name in value.split(",") if name.strip()]


def extract_keyphrases(text: str, top_k: int = 8) -> List[str]:
    tokens = [tok for tok in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower()) if tok not in STOPWORDS]
    if not tokens:
        return []
    freq = Counter(tokens)
    return [word for word, _ in freq.most_common(top_k)]


def normalize_arxiv_id(paper_id: str) -> str:
    base = paper_id.split("v")[0]
    return base


def normalize_openalex_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    return value.split("/")[-1]


def build_graph(
    papers: Sequence[PaperRecord],
    client: OpenAlexClient,
    output_dir: Path,
    mailto: Optional[str] = None,
) -> dict:
    """Construct the heterogeneous scholarly graph."""

    nodes: Dict[str, Dict[str, dict]] = defaultdict(dict)
    edges: List[dict] = []

    feature_vocab: List[str] = []
    vocab_index: Dict[str, int] = {}

    paper_references: Dict[str, set] = {}
    keyphrase_index: Dict[str, List[str]] = defaultdict(list)
    reference_to_papers: Dict[str, List[str]] = defaultdict(list)

    openalex_to_paper: Dict[str, str] = {}

    for paper in papers:
        arxiv_id = normalize_arxiv_id(paper.paper_id)
        openalex_payload = client.fetch_work(arxiv_id) if client else None

        keyphrases = extract_keyphrases(paper.abstract)
        venue_name = None
        fields: List[str] = []
        references: List[str] = []
        openalex_id: Optional[str] = None

        if openalex_payload:
            openalex_id = openalex_payload.get("id")
            venue = openalex_payload.get("host_venue") or {}
            venue_name = venue.get("display_name") or venue.get("publisher")
            concepts = openalex_payload.get("concepts") or []
            fields = sorted({concept.get("display_name") for concept in concepts if concept.get("display_name")})
            references = [normalize_openalex_id(ref) for ref in (openalex_payload.get("referenced_works") or [])]
            keywords = openalex_payload.get("keywords") or []
            if keywords:
                keyphrases = list({*keyphrases, *(kw.get("display_name") for kw in keywords if kw.get("display_name"))})

        feature_tokens = [
            *(keyphrases or []),
            *(fields or []),
        ]
        feature_indices: List[int] = []
        for token in feature_tokens:
            if not token:
                continue
            if token not in vocab_index:
                vocab_index[token] = len(feature_vocab)
                feature_vocab.append(token)
            feature_indices.append(vocab_index[token])

        node_payload = {
            "id": paper.paper_id,
            "arxiv_id": arxiv_id,
            "title": paper.title,
            "published": paper.published,
            "authors": paper.authors,
            "venue": venue_name,
            "fields": fields,
            "keyphrases": keyphrases,
            "feature_indices": feature_indices,
        }
        if openalex_id:
            short_id = normalize_openalex_id(openalex_id)
            node_payload["openalex_id"] = openalex_id
            node_payload["openalex_short_id"] = short_id
            openalex_to_paper[openalex_id] = paper.paper_id
            if short_id:
                openalex_to_paper[short_id] = paper.paper_id
        nodes["paper"][paper.paper_id] = node_payload
        paper_references[paper.paper_id] = set(references)
        for keyphrase in keyphrases:
            keyphrase_index[keyphrase].append(paper.paper_id)
        for ref in references:
            if ref:
                reference_to_papers[ref].append(paper.paper_id)

        for author in paper.authors:
            author_id = f"author::{author.lower()}"
            if author_id not in nodes["author"]:
                nodes["author"][author_id] = {"id": author_id, "name": author}

        if venue_name:
            venue_id = f"venue::{venue_name.lower()}"
            if venue_id not in nodes["venue"]:
                nodes["venue"][venue_id] = {"id": venue_id, "name": venue_name}

        for field in fields:
            field_id = f"field::{field.lower()}"
            if field_id not in nodes["field"]:
                nodes["field"][field_id] = {"id": field_id, "name": field}

        for keyphrase in keyphrases:
            keyphrase_id = f"keyphrase::{keyphrase.lower()}"
            if keyphrase_id not in nodes["keyphrase"]:
                nodes["keyphrase"][keyphrase_id] = {"id": keyphrase_id, "name": keyphrase}

    # Citation edges (only when the cited paper is part of the dataset)
    paper_ids = set(nodes["paper"].keys())
    for src_id, references in paper_references.items():
        for ref in references:
            target_id = openalex_to_paper.get(ref)
            if target_id and target_id in paper_ids:
                edges.append(
                    {
                        "source_type": "paper",
                        "source": src_id,
                        "target_type": "paper",
                        "target": target_id,
                        "type": "citation",
                    }
                )

    # Co-author edges
    for paper in nodes["paper"].values():
        authors = paper.get("authors", [])
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                a_id = f"author::{authors[i].lower()}"
                b_id = f"author::{authors[j].lower()}"
                edges.append(
                    {
                        "source_type": "author",
                        "source": a_id,
                        "target_type": "author",
                        "target": b_id,
                        "type": "co_author",
                    }
                )
                edges.append(
                    {
                        "source_type": "author",
                        "source": b_id,
                        "target_type": "author",
                        "target": a_id,
                        "type": "co_author",
                    }
                )

    # Venue membership edges
    for paper in nodes["paper"].values():
        venue = paper.get("venue")
        if venue:
            venue_id = f"venue::{venue.lower()}"
            edges.append(
                {
                    "source_type": "paper",
                    "source": paper["id"],
                    "target_type": "venue",
                    "target": venue_id,
                    "type": "venue_membership",
                }
            )

    # Co-citation edges
    for ref, citing_papers in reference_to_papers.items():
        if len(citing_papers) < 2:
            continue
        for i in range(len(citing_papers)):
            for j in range(i + 1, len(citing_papers)):
                edges.append(
                    {
                        "source_type": "paper",
                        "source": citing_papers[i],
                        "target_type": "paper",
                        "target": citing_papers[j],
                        "type": "co_citation",
                        "weight": 1.0,
                    }
                )
                edges.append(
                    {
                        "source_type": "paper",
                        "source": citing_papers[j],
                        "target_type": "paper",
                        "target": citing_papers[i],
                        "type": "co_citation",
                        "weight": 1.0,
                    }
                )

    # Keyphrase overlap edges
    for keyphrase, papers_with_term in keyphrase_index.items():
        if len(papers_with_term) < 2:
            continue
        for i in range(len(papers_with_term)):
            for j in range(i + 1, len(papers_with_term)):
                edges.append(
                    {
                        "source_type": "paper",
                        "source": papers_with_term[i],
                        "target_type": "paper",
                        "target": papers_with_term[j],
                        "type": "keyphrase_overlap",
                        "weight": 1.0,
                        "keyphrase": keyphrase,
                    }
                )
                edges.append(
                    {
                        "source_type": "paper",
                        "source": papers_with_term[j],
                        "target_type": "paper",
                        "target": papers_with_term[i],
                        "type": "keyphrase_overlap",
                        "weight": 1.0,
                        "keyphrase": keyphrase,
                    }
                )

    # Sanity filter: keep only supported edge types
    edges = [edge for edge in edges if edge["type"] in EDGE_TYPES]

    graph_payload = {
        "nodes": {node_type: list(payloads.values()) for node_type, payloads in nodes.items()},
        "edges": edges,
        "metadata": {
            "feature_vocab": feature_vocab,
            "edge_types": sorted(EDGE_TYPES),
            "mailto": mailto,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hetero_graph.json"
    output_path.write_text(json.dumps(graph_payload, indent=2))
    print(f"[✅] Heterogeneous graph saved to {output_path}")
    return graph_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a heterogeneous knowledge graph for the paper corpus.")
    parser.add_argument("--metadata", default="papers_metadata.csv", help="Path to the paper metadata CSV file")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the hetero graph JSON will be stored",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit the number of papers processed")
    parser.add_argument(
        "--mailto",
        default=os.environ.get("OPENALEX_MAILTO"),
        help="Contact email for polite OpenAlex API usage",
    )
    parser.add_argument(
        "--cache",
        default=str(DEFAULT_OUTPUT_DIR / "openalex_cache.json"),
        help="Path to the JSON cache for OpenAlex responses",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_path = Path(args.cache)

    papers = load_local_metadata(Path(args.metadata), args.limit)
    print(f"[+] Loaded {len(papers)} local paper records")

    client = OpenAlexClient(cache_path=cache_path, mailto=args.mailto)
    print("[+] Fetching OpenAlex enrichments (cached when available)...")
    graph_payload = build_graph(papers, client=client, output_dir=output_dir, mailto=args.mailto)

    summary_path = output_dir / "graph_summary.json"
    summary = {
        "num_nodes": {node_type: len(entries) for node_type, entries in graph_payload["nodes"].items()},
        "num_edges": len(graph_payload["edges"]),
        "edge_type_counts": dict(Counter(edge["type"] for edge in graph_payload["edges"])),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[✅] Graph summary stored at {summary_path}")


if __name__ == "__main__":
    main()
