"""compute_graph_scores.py

Compute graph-based proximity matrices (citation, co-author, venue) for the
papers in the metadata table.

The script expects a CSV file with at least the following columns:

* ``id`` – unique paper identifier
* ``authors`` – list-like column (JSON / Python literal / semi-colon string)
* ``venue`` – optional textual venue / publication source
* ``references`` – optional list of referenced paper ids
* ``cited_by`` – optional list of citing paper ids

Only the identifiers that appear in the metadata will be considered when
constructing the proximity matrices.

Outputs are stored in ``--output-dir`` as ``*.npz`` files (CSR matrices) plus
``paper_ids.json`` which captures the row/column ordering.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute citation / co-author / venue proximity matrices")
    parser.add_argument("metadata", type=Path, help="CSV file with paper metadata")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/graph"),
        help="Directory to save graph proximity matrices",
    )
    parser.add_argument(
        "--min-shared-authors",
        type=int,
        default=1,
        help="Minimum number of shared authors to create a co-author link",
    )
    return parser.parse_args()


def _safe_parse_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, (set, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        # Try JSON first
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, (list, tuple, set)):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except (ValueError, SyntaxError):
            pass
        return [part.strip() for part in stripped.split(";") if part.strip()]
    return []


def compute_citation_matrix(df: pd.DataFrame, paper_index: dict[str, int]) -> sparse.csr_matrix:
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for paper_id, refs in zip(df["id"], df.get("references", [[]])):
        references = _safe_parse_list(refs)
        src_idx = paper_index.get(paper_id)
        if src_idx is None:
            continue
        for ref in references:
            dst_idx = paper_index.get(ref)
            if dst_idx is None:
                continue
            rows.append(src_idx)
            cols.append(dst_idx)
            data.append(1.0)

    # Symmetrize by adding transpose to capture undirected proximity
    citation_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(df), len(df)))
    citation_matrix = citation_matrix.maximum(citation_matrix.transpose())
    return citation_matrix


def compute_coauthor_matrix(
    df: pd.DataFrame,
    paper_index: dict[str, int],
    min_shared_authors: int,
) -> sparse.csr_matrix:
    author_to_papers: defaultdict[str, List[int]] = defaultdict(list)
    for paper_id, authors in zip(df["id"], df.get("authors", [[]])):
        parsed_authors = _safe_parse_list(authors)
        idx = paper_index.get(paper_id)
        if idx is None:
            continue
        for author in parsed_authors:
            author_to_papers[author].append(idx)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for papers in author_to_papers.values():
        if len(papers) < 2:
            continue
        for i, j in combinations(sorted(set(papers)), 2):
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([1.0, 1.0])

    coauthor_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(df), len(df)))
    if min_shared_authors > 1:
        # Re-compute weights to count duplicates when authors overlap more than once
        coauthor_matrix = coauthor_matrix.power(min_shared_authors)
    return coauthor_matrix


def compute_venue_matrix(df: pd.DataFrame, paper_index: dict[str, int]) -> sparse.csr_matrix:
    venue_groups: defaultdict[str, List[int]] = defaultdict(list)
    for paper_id, venue in zip(df["id"], df.get("venue", "")):
        if venue is None:
            continue
        venue_str = str(venue).strip()
        if not venue_str:
            continue
        idx = paper_index.get(paper_id)
        if idx is None:
            continue
        venue_groups[venue_str].append(idx)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for members in venue_groups.values():
        if len(members) < 2:
            continue
        for i, j in combinations(sorted(set(members)), 2):
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([1.0, 1.0])

    return sparse.csr_matrix((data, (rows, cols)), shape=(len(df), len(df)))


def normalize_matrix(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    if matrix.nnz == 0:
        return matrix
    matrix = matrix.tocsr()
    row_sums = np.sqrt(matrix.power(2).sum(axis=1)).A1
    row_sums[row_sums == 0] = 1.0
    normalized = matrix.multiply(1.0 / row_sums[:, None])
    normalized = normalized.multiply(1.0 / row_sums[None, :])
    return normalized.tocsr()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata)
    if "id" not in df.columns:
        raise ValueError("Metadata CSV must contain an `id` column")

    paper_ids = df["id"].astype(str).tolist()
    paper_index = {pid: idx for idx, pid in enumerate(paper_ids)}

    citation_matrix = compute_citation_matrix(df, paper_index)
    coauthor_matrix = compute_coauthor_matrix(df, paper_index, args.min_shared_authors)
    venue_matrix = compute_venue_matrix(df, paper_index)

    citation_norm = normalize_matrix(citation_matrix)
    coauthor_norm = normalize_matrix(coauthor_matrix)
    venue_norm = normalize_matrix(venue_matrix)

    sparse.save_npz(args.output_dir / "citation_scores.npz", citation_norm)
    sparse.save_npz(args.output_dir / "coauthor_scores.npz", coauthor_norm)
    sparse.save_npz(args.output_dir / "venue_scores.npz", venue_norm)
    (args.output_dir / "paper_ids.json").write_text(json.dumps(paper_ids))
    print(f"[✓] Graph proximity matrices saved to {args.output_dir}")


if __name__ == "__main__":
    main()
