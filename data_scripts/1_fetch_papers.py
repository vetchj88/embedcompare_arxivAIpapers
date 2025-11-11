"""Fetch arXiv metadata, download PDFs, and persist segmented content."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import arxiv
import pandas as pd

try:  # pragma: no cover - import resolution depends on execution context
    from . import segment_papers
except ImportError:  # pragma: no cover
    import segment_papers


# --- Configuration ---
DEFAULT_CATEGORIES: Sequence[str] = ("cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE")
QUERY = " OR ".join(f"cat:{cat}" for cat in DEFAULT_CATEGORIES)
MAX_RESULTS = 500
DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
METADATA_CSV = DATA_DIR / "papers_metadata.csv"
SEGMENTS_JSONL = DATA_DIR / "papers_with_sections.jsonl"


@dataclass
class PaperRecord:
    """Container for paper metadata and segmented sections."""

    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    date: str
    pdf_path: Path
    sections: Dict[str, str] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, str]:
        data = asdict(self)
        data["authors"] = ", ".join(self.authors)
        data["pdf_path"] = str(self.pdf_path)
        data["sections"] = list(self.sections.keys())
        return data

    def to_json_line(self) -> str:
        payload = asdict(self)
        payload["authors"] = self.authors
        payload["pdf_path"] = str(self.pdf_path)
        payload["sections"] = self.sections
        return json.dumps(payload, ensure_ascii=False)


def prepare_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)


def download_pdf(result: arxiv.Result, pdf_path: Path) -> Optional[Path]:
    """Download the PDF for the given result, returning the path or ``None``."""

    if pdf_path.exists():
        return pdf_path

    try:
        result.download_pdf(filename=str(pdf_path))
        return pdf_path
    except Exception as exc:  # pragma: no cover - network failures should not crash script
        print(f"[!] Failed to download PDF for {result.entry_id}: {exc}")
        return None


def segment_paper(pdf_path: Path, title: str, abstract: str) -> Dict[str, str]:
    """Segment the PDF into logical sections using :mod:`segment_papers`."""

    try:
        segments = segment_papers.segment_pdf(
            pdf_path,
            title=title,
            abstract=abstract,
        )
        return segments
    except Exception as exc:  # pragma: no cover - segmentation errors shouldn't halt pipeline
        print(f"[!] Failed to segment {pdf_path.name}: {exc}")
        return {}


def normalise_date(value: Optional[str], default: str) -> str:
    """Convert ``YYYY-MM-DD`` strings to the format required by the API."""

    if not value:
        return default
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:  # pragma: no cover - validated by CLI/UI
        raise ValueError(f"Invalid date '{value}'. Expected YYYY-MM-DD.") from exc
    return parsed.strftime("%Y%m%d0000")


def build_query(
    *,
    query: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
    search_terms: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Return an arXiv query string composed from the provided parameters."""

    if query:
        return query

    filters = []
    cats = [c.strip() for c in (categories or DEFAULT_CATEGORIES) if c and c.strip()]
    if cats:
        if len(cats) == 1:
            filters.append(f"cat:{cats[0]}")
        else:
            filters.append("(" + " OR ".join(f"cat:{cat}" for cat in cats) + ")")

    if search_terms:
        filters.append(f"({search_terms})")

    if start_date or end_date:
        start = normalise_date(start_date, "*")
        end = normalise_date(end_date, "*")
        # arXiv expects a closing time as HHMM; assume end of day when supplied
        if end != "*":
            end = end[:-4] + "2359"
        filters.append(f"submittedDate:[{start} TO {end}]")

    if not filters:
        return QUERY

    return " AND ".join(filters)


def fetch_papers(
    *,
    query: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
    search_terms: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_results: int = MAX_RESULTS,
) -> Iterable[PaperRecord]:
    """Fetch arXiv papers according to the provided configuration."""

    resolved_query = build_query(
        query=query,
        categories=categories,
        search_terms=search_terms,
        start_date=start_date,
        end_date=end_date,
    )
    print(
        "[*] Starting search for "
        f"{max_results} papers with query: '{resolved_query}'..."
    )

    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    search = arxiv.Search(
        query=resolved_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    prepare_directories()

    papers: List[PaperRecord] = []
    try:
        for index, result in enumerate(client.results(search)):
            if (index + 1) % 25 == 0:
                print(f"[*] Fetched {index + 1} / {max_results} papers...")

            short_id = result.get_short_id()
            pdf_path = PDF_DIR / f"{short_id}.pdf"
            downloaded = download_pdf(result, pdf_path)
            if downloaded is None:
                continue

            sections = segment_paper(pdf_path, title=result.title, abstract=result.summary)

            papers.append(
                PaperRecord(
                    paper_id=short_id,
                    title=result.title,
                    abstract=result.summary.replace("\n", " "),
                    authors=[author.name for author in result.authors],
                    date=result.published.strftime("%Y-%m-%d"),
                    pdf_path=downloaded,
                    sections=sections,
                )
            )

            time.sleep(0.25)

    except arxiv.UnexpectedEmptyPageError as error:
        print("\n[!] Error: Encountered an empty page from arXiv API. Saving partial results.")
        print(f"[!] Details: {error}")

    return papers


def save_metadata(papers: Iterable[PaperRecord]) -> None:
    records = list(papers)
    if not records:
        print("[❌] No papers were downloaded. Exiting.")
        return

    metadata_rows = [record.to_metadata() for record in records]
    df = pd.DataFrame(metadata_rows)
    df.to_csv(METADATA_CSV, index=False)

    with SEGMENTS_JSONL.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.to_json_line() + "\n")

    print(f"[✅] Saved metadata to {METADATA_CSV.resolve()}")
    print(f"[✅] Saved segmented content to {SEGMENTS_JSONL.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and segment arXiv papers.")
    parser.add_argument(
        "--query",
        help="Full arXiv query string (overrides categories/search parameters).",
    )
    parser.add_argument(
        "--categories",
        help="Comma separated list of arXiv categories (e.g. cs.AI,cs.CL).",
    )
    parser.add_argument(
        "--search",
        dest="search_terms",
        help="Additional keyword query appended with AND semantics.",
    )
    parser.add_argument(
        "--start-date",
        help="Earliest submission date in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        help="Latest submission date in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=MAX_RESULTS,
        help=f"Maximum number of results to request (default: {MAX_RESULTS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = None
    if args.categories:
        categories = [part.strip() for part in args.categories.split(",") if part.strip()]
    papers = list(
        fetch_papers(
            query=args.query,
            categories=categories,
            search_terms=args.search_terms,
            start_date=args.start_date,
            end_date=args.end_date,
            max_results=args.max_results,
        )
    )
    save_metadata(papers)


if __name__ == "__main__":
    main()
