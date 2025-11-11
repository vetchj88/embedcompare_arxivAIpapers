"""Utilities for splitting paper content into semantically meaningful segments.

This module previously handled high level section and paragraph segmentation.
It now explicitly captures figure and table captions as dedicated segments so
that downstream models can reason about them separately.  The implementation is
kept lightweight and does not require PDF parsing – it assumes the caller has
already extracted structured content from a paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional
import itertools


@dataclass
class Segment:
    """Representation of a contiguous chunk of a document.

    Attributes
    ----------
    segment_id:
        Unique identifier for the segment.  Generated deterministically from
        section and position information so downstream consumers can join on it.
    section:
        The logical section heading the segment belongs to.  May be ``None`` for
        front-matter content.
    segment_type:
        Semantic type of the segment.  Common values include ``"paragraph"``,
        ``"figure_caption"`` and ``"table_caption"``.
    text:
        Normalised textual content.
    metadata:
        Additional context such as figure identifiers or table numbers.
    """

    segment_id: str
    section: Optional[str]
    segment_type: str
    text: str
    metadata: MutableMapping[str, object] = field(default_factory=dict)


class PaperSegmenter:
    """Splits structured paper content into model-ready segments.

    The expected input is a mapping with an optional ``"sections"`` key.  Each
    section is a mapping containing:

    ``heading``
        Section heading text.
    ``paragraphs``
        Iterable of paragraph strings.
    ``figures`` / ``tables`` (optional)
        Iterable of dictionaries with at least a ``caption`` key.  The dict can
        optionally include ``id`` / ``label`` / ``thumbnail`` values which are
        propagated into the segment metadata.
    """

    def __init__(self, include_captions: bool = True) -> None:
        self.include_captions = include_captions

    def segment(self, paper: Mapping[str, object]) -> List[Segment]:
        """Return a list of :class:`Segment` objects for ``paper``.

        Figure and table captions are emitted as dedicated segments tagged with
        ``segment_type`` values ``"figure_caption"`` and ``"table_caption"``.  The
        resulting list preserves document order: paragraphs first, followed by
        any figures/tables encountered within a section.
        """

        sections: Iterable[Mapping[str, object]] = paper.get("sections", [])  # type: ignore[assignment]
        all_segments: List[Segment] = []

        # Running counter used when sections do not specify explicit identifiers.
        section_counter = itertools.count(1)

        for section in sections:
            heading = section.get("heading")  # type: ignore[assignment]
            section_id = section.get("id") or f"sec-{next(section_counter)}"
            paragraphs: Iterable[str] = section.get("paragraphs", [])  # type: ignore[assignment]

            for paragraph_idx, paragraph in enumerate(paragraphs, start=1):
                text = (paragraph or "").strip()
                if not text:
                    continue
                segment_id = f"{section_id}-p{paragraph_idx}"
                all_segments.append(
                    Segment(
                        segment_id=segment_id,
                        section=str(heading) if heading is not None else None,
                        segment_type="paragraph",
                        text=text,
                        metadata={"section_id": section_id, "paragraph_index": paragraph_idx},
                    )
                )

            if not self.include_captions:
                continue

            self._extend_with_caption_segments(
                segments=all_segments,
                section=section,
                section_id=str(section_id),
                heading=str(heading) if heading is not None else None,
            )

        return all_segments

    def _extend_with_caption_segments(
        self,
        segments: List[Segment],
        section: Mapping[str, object],
        section_id: str,
        heading: Optional[str],
    ) -> None:
        """Create figure/table caption segments for ``section`` if available."""

        for caption_type in ("figures", "tables"):
            entries: Iterable[Mapping[str, object]] = section.get(caption_type, [])  # type: ignore[assignment]
            for entry_idx, entry in enumerate(entries, start=1):
                caption_text = str(entry.get("caption", "")).strip()
                if not caption_text:
                    continue

                label = entry.get("label") or entry.get("id") or entry.get("name")
                segment_id = f"{section_id}-{caption_type[0]}{entry_idx}"
                metadata: Dict[str, object] = {
                    "section_id": section_id,
                    "section_heading": heading,
                    "caption_index": entry_idx,
                    "caption_type": caption_type,
                }
                if label:
                    metadata["label"] = label
                if entry.get("thumbnail"):
                    metadata["thumbnail"] = entry["thumbnail"]
                if entry.get("uri"):
                    metadata["uri"] = entry["uri"]

                segment_type = "figure_caption" if caption_type == "figures" else "table_caption"
                segments.append(
                    Segment(
                        segment_id=segment_id,
                        section=heading,
                        segment_type=segment_type,
                        text=caption_text,
                        metadata=metadata,
                    )
                )


__all__ = ["PaperSegmenter", "Segment"]
