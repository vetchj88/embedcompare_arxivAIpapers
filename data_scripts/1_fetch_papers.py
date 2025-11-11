"""Fetch arXiv metadata, download PDFs, and persist segmented content."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import arxiv
import pandas as pd

try:  # pragma: no cover - import resolution depends on execution context
    from . import segment_papers
except ImportError:  # pragma: no cover
    import segment_papers


# --- Configuration ---
QUERY = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:cs.NE"
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


def fetch_papers() -> Iterable[PaperRecord]:
    print(f"[*] Starting search for {MAX_RESULTS} papers with query: '{QUERY}'...")

    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    search = arxiv.Search(
        query=QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    prepare_directories()

    papers: List[PaperRecord] = []
    try:
        for index, result in enumerate(client.results(search)):
            if (index + 1) % 25 == 0:
                print(f"[*] Fetched {index + 1} / {MAX_RESULTS} papers...")

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


if __name__ == "__main__":
    papers = list(fetch_papers())
    save_metadata(papers)
