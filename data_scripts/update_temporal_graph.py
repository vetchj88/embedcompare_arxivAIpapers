"""Utilities for maintaining a temporally ordered stream of paper events.

This script transforms the static metadata that powers the embedding
visualisation into a temporal event stream.  Two event types are handled:
submission events (the first time a paper appears) and citation events
(the moment one paper cites another).  The resulting stream is written as a
JSON Lines file sorted by event time so that downstream components—such as
our temporal graph neural network—can consume the events incrementally.

Usage
-----
python data_scripts/update_temporal_graph.py \
    --metadata papers_metadata.csv \
    --citations citations.csv \
    --output data/temporal_edge_stream.jsonl

The citations file is optional.  If it is omitted, the output will contain
only submission events sorted by submission date.  The citations file is
expected to contain the columns ``citing_id``, ``cited_id`` and
``citation_date`` in ISO 8601 format.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pandas as pd


@dataclass(frozen=True)
class TemporalEvent:
    """Representation of a single temporal graph update."""

    timestamp: datetime
    event_type: str
    source: str
    target: str
    metadata: Dict[str, object]

    def serialise(self) -> Dict[str, object]:
        """Convert the event to a JSON-serialisable dictionary."""

        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


def _parse_timestamp(value: str) -> datetime:
    """Parse timestamps and normalise them to UTC."""

    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    return timestamp


def load_paper_metadata(path: Path) -> pd.DataFrame:
    """Load paper metadata and ensure submission timestamps are valid."""

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column containing submission dates")

    df["timestamp"] = df["date"].apply(_parse_timestamp)
    if "authors" in df.columns and df["authors"].dtype == object:
        try:
            df["authors"] = df["authors"].apply(lambda x: x if isinstance(x, list) else eval(str(x)))
        except Exception:  # pragma: no cover - fallback for malformed records
            df["authors"] = df["authors"].apply(lambda x: [x] if isinstance(x, str) else [])
    return df


def load_citation_events(path: Path) -> pd.DataFrame:
    """Load citation events if available."""

    df = pd.read_csv(path)
    required_columns = {"citing_id", "cited_id", "citation_date"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Citation file {path} is missing required columns: {sorted(missing)}"
        )

    df["timestamp"] = df["citation_date"].apply(_parse_timestamp)
    return df


def iter_submission_events(df: pd.DataFrame) -> Iterator[TemporalEvent]:
    """Yield submission events for every paper."""

    for row in df.itertuples(index=False):
        metadata = {
            "title": getattr(row, "title", ""),
            "authors": getattr(row, "authors", []),
            "abstract": getattr(row, "abstract", ""),
            "date": getattr(row, "date", ""),
        }
        paper_id = str(getattr(row, "id"))
        yield TemporalEvent(
            timestamp=getattr(row, "timestamp"),
            event_type="submission",
            source="__source__",  # virtual source node
            target=paper_id,
            metadata=metadata,
        )


def iter_citation_events(df: pd.DataFrame) -> Iterator[TemporalEvent]:
    """Yield citation events sorted by citation timestamp."""

    for row in df.sort_values("timestamp").itertuples(index=False):
        metadata = {
            "context": getattr(row, "context", ""),
        }
        yield TemporalEvent(
            timestamp=getattr(row, "timestamp"),
            event_type="citation",
            source=str(getattr(row, "citing_id")),
            target=str(getattr(row, "cited_id")),
            metadata=metadata,
        )


def build_event_stream(
    metadata_path: Path,
    citation_path: Optional[Path],
    output_path: Path,
) -> List[TemporalEvent]:
    """Build and persist the combined event stream."""

    metadata_df = load_paper_metadata(metadata_path)
    events: List[TemporalEvent] = list(iter_submission_events(metadata_df))

    if citation_path and citation_path.exists():
        citation_df = load_citation_events(citation_path)
        events.extend(list(iter_citation_events(citation_df)))

    events.sort(key=lambda event: event.timestamp)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for event in events:
            json.dump(event.serialise(), fh)
            fh.write("\n")

    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a temporally ordered graph event stream.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("papers_metadata.csv"),
        help="Path to the CSV file containing paper metadata (with submission dates).",
    )
    parser.add_argument(
        "--citations",
        type=Path,
        default=None,
        help="Optional CSV containing citation edges and timestamps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/temporal_edge_stream.jsonl"),
        help="Destination for the JSONL event stream.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    citation_path = args.citations if args.citations else None
    events = build_event_stream(args.metadata, citation_path, args.output)
    print(f"Wrote {len(events)} events to {args.output}")


if __name__ == "__main__":
    main()
