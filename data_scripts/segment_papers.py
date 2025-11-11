"""Utilities for segmenting scientific PDFs into canonical sections."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency at runtime
    from pdfminer.high_level import extract_text
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pdfminer.six is required for PDF segmentation. Install with 'pip install pdfminer.six'."
    ) from exc


SECTION_PATTERNS: Dict[str, List[re.Pattern[str]]] = {
    "abstract": [re.compile(r"^abstract\b", re.IGNORECASE)],
    "introduction": [
        re.compile(r"^(?:\d+\.?\s*)?introduction\b", re.IGNORECASE),
        re.compile(r"^background\b", re.IGNORECASE),
    ],
    "methods": [
        re.compile(r"^(?:\d+\.?\s*)?(materials\s+and\s+methods|methodology|methods?)\b", re.IGNORECASE),
    ],
    "results": [
        re.compile(r"^(?:\d+\.?\s*)?results?\b", re.IGNORECASE),
        re.compile(r"^(?:\d+\.?\s*)?experiments?\b", re.IGNORECASE),
    ],
    "conclusion": [
        re.compile(r"^(?:\d+\.?\s*)?conclusions?\b", re.IGNORECASE),
        re.compile(r"^(?:\d+\.?\s*)?discussion\b", re.IGNORECASE),
        re.compile(r"^(?:\d+\.?\s*)?summary\b", re.IGNORECASE),
    ],
}

CAPTION_PATTERN = re.compile(r"^(figure|table)\s*\d+[:\.]", re.IGNORECASE)


@dataclass
class SegmentedPaper:
    paper_id: str
    sections: Dict[str, str]

    def to_json(self) -> str:
        payload = {"paper_id": self.paper_id, "sections": self.sections}
        return json.dumps(payload, ensure_ascii=False)


def _extract_text(pdf_path: Path) -> str:
    text = extract_text(str(pdf_path))
    return text or ""


def _detect_section(line: str) -> Optional[str]:
    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if pattern.match(line):
                return section
    return None


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_into_sections(text: str) -> Dict[str, str]:
    """Heuristically split the provided text into canonical sections."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sections: Dict[str, List[str]] = {name: [] for name in SECTION_PATTERNS}
    sections["captions"] = []

    current_section: Optional[str] = None

    for line in lines:
        normalized = line.lower()
        caption_match = CAPTION_PATTERN.match(normalized)
        if caption_match:
            sections["captions"].append(line)
            continue

        detected = _detect_section(normalized)
        if detected is not None:
            current_section = detected
            continue

        if current_section is not None:
            sections[current_section].append(line)

    return {name: _normalize_whitespace(" ".join(content)) for name, content in sections.items() if content}


def segment_pdf(pdf_path: Path, *, title: Optional[str] = None, abstract: Optional[str] = None) -> Dict[str, str]:
    """Segment a paper's PDF into key sections.

    Parameters
    ----------
    pdf_path:
        Path to the PDF to parse.
    title:
        Optional title supplied from metadata. If provided, it will override the
        parsed title when persisting sections.
    abstract:
        Optional abstract supplied from metadata.
    """

    raw_text = _extract_text(Path(pdf_path))
    sections = split_into_sections(raw_text)

    if title:
        sections["title"] = _normalize_whitespace(title)
    if abstract:
        sections.setdefault("abstract", _normalize_whitespace(abstract))

    return sections


def segment_directory(pdf_dir: Path, *, output_jsonl: Path) -> None:
    records: List[SegmentedPaper] = []

    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        paper_id = pdf_path.stem
        sections = segment_pdf(pdf_path)
        if not sections:
            continue
        records.append(SegmentedPaper(paper_id=paper_id, sections=sections))

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment PDFs into canonical sections")
    parser.add_argument("pdf_dir", type=Path, help="Directory containing PDF files")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/paper_segments.jsonl"),
        help="Destination JSONL path",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    segment_directory(args.pdf_dir, output_jsonl=args.output)
    print(f"[✅] Segmented papers written to {args.output.resolve()}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
