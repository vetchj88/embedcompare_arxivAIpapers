"""Periodic jobs for maintaining temporal embedding snapshots."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from data_scripts.update_temporal_graph import TemporalEvent
from models.temporal_gnn import TemporalGraphNetwork


def _parse_timestamp(value: str) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def load_event_stream(path: Path) -> List[TemporalEvent]:
    events: List[TemporalEvent] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            payload = json.loads(line)
            events.append(
                TemporalEvent(
                    timestamp=_parse_timestamp(payload["timestamp"]),
                    event_type=payload["event_type"],
                    source=payload["source"],
                    target=payload["target"],
                    metadata=payload.get("metadata", {}),
                )
            )
    return events


@dataclass
class SnapshotDescriptor:
    path: Path
    timestamp: datetime
    label: str


class TemporalUpdateScheduler:
    """Replay historical events and maintain fresh temporal snapshots."""

    def __init__(
        self,
        event_stream: Path,
        snapshot_dir: Path = Path("temporal_snapshots"),
        snapshot_interval: timedelta = timedelta(days=7),
        tail_interval: timedelta = timedelta(minutes=10),
    ) -> None:
        self.event_stream = event_stream
        self.snapshot_dir = snapshot_dir
        self.snapshot_interval = snapshot_interval
        self.tail_interval = tail_interval
        self.model = TemporalGraphNetwork(memory_dim=128, embedding_dim=3, hidden_dim=128)
        self.base_metadata: Dict[str, Dict[str, object]] = {}
        self.last_processed_index = 0
        self.next_snapshot_after: Optional[datetime] = None
        self.manifest_path = snapshot_dir / "manifest.json"
        self.manifest: Dict[str, object] = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "snapshots": [],
        }
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def load_base_metadata(self, metadata_files: Iterable[Path]) -> None:
        for file_path in metadata_files:
            if not file_path.exists():
                continue
            with file_path.open("r", encoding="utf-8") as fh:
                records = json.load(fh)
            for record in records:
                paper_id = str(record.get("id"))
                self.base_metadata.setdefault(paper_id, record)

    def _manifest_entries(self) -> List[Dict[str, object]]:
        return list(self.manifest.get("snapshots", []))

    def _update_manifest(self, descriptor: SnapshotDescriptor) -> None:
        entries = self._manifest_entries()
        entry = {
            "id": descriptor.timestamp.strftime("%Y-%m-%d"),
            "timestamp": descriptor.timestamp.isoformat(),
            "label": descriptor.label,
            "path": descriptor.path.as_posix(),
        }
        entries = [e for e in entries if e["path"] != descriptor.path.as_posix()]
        entries.append(entry)
        entries.sort(key=lambda item: item["timestamp"])
        self.manifest["snapshots"] = entries
        with self.manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(self.manifest, fh, indent=2)

    def _build_snapshot_payload(self) -> Dict[str, Dict[str, object]]:
        embeddings = self.model.snapshot_embeddings()
        payload: Dict[str, Dict[str, object]] = {}
        for paper_id, node in embeddings.items():
            base = self.base_metadata.get(paper_id, {})
            authors = base.get("authors") or node["metadata"].get("authors", [])
            if isinstance(authors, list):
                authors_str = ", ".join(authors)
            else:
                authors_str = str(authors)
            payload[paper_id] = {
                "id": paper_id,
                "title": base.get("title") or node["metadata"].get("title", ""),
                "authors": authors_str,
                "date": base.get("date") or node["metadata"].get("date", ""),
                "cluster_id": base.get("cluster_id", -1),
                "x": node["embedding"][0] if node["embedding"] else 0.0,
                "y": node["embedding"][1] if len(node["embedding"]) > 1 else 0.0,
                "z": node["embedding"][2] if len(node["embedding"]) > 2 else 0.0,
                "temporal_embedding": node["embedding"],
                "last_updated": node["last_updated"],
            }
        return payload

    def _write_snapshot(self, timestamp: datetime) -> None:
        payload = self._build_snapshot_payload()
        if not payload:
            return
        records = sorted(payload.values(), key=lambda row: row.get("date", ""))
        filename = f"snapshot_{timestamp.strftime('%Y-%m-%dT%H-%M-%S')}.json"
        path = self.snapshot_dir / filename
        snapshot_data = {
            "timestamp": timestamp.isoformat(),
            "models": {
                "BGE-Large": [dict(record) for record in records],
                "GTE-Large": [dict(record) for record in records],
                "MiniLM": [dict(record) for record in records],
            },
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(snapshot_data, fh, indent=2)
        descriptor = SnapshotDescriptor(
            path=path,
            timestamp=timestamp,
            label=timestamp.strftime("%d %b %Y"),
        )
        self._update_manifest(descriptor)

    def _maybe_emit_snapshot(self, timestamp: datetime) -> None:
        if self.next_snapshot_after is None or timestamp >= self.next_snapshot_after:
            self._write_snapshot(timestamp)
            self.next_snapshot_after = timestamp + self.snapshot_interval

    def _process_events(self, events: List[TemporalEvent]) -> None:
        for idx, event in enumerate(events[self.last_processed_index :], start=self.last_processed_index):
            self.model.process_event(event)
            self._maybe_emit_snapshot(event.timestamp)
            self.last_processed_index = idx + 1
        if events:
            self._write_snapshot(events[-1].timestamp)

    async def replay_historical(self) -> None:
        events = load_event_stream(self.event_stream)
        if not events:
            return
        self._process_events(events)

    async def tail_new_events(self) -> None:
        await asyncio.sleep(self.tail_interval.total_seconds())
        events = load_event_stream(self.event_stream)
        if len(events) > self.last_processed_index:
            self._process_events(events)

    async def start(self) -> None:
        await self.replay_historical()
        while True:
            await self.tail_new_events()


async def main() -> None:
    scheduler = TemporalUpdateScheduler(Path("data/temporal_edge_stream.jsonl"))
    scheduler.load_base_metadata(
        [
            Path("papers_bge_large.json"),
            Path("papers_gte_large.json"),
            Path("papers_minilm.json"),
        ]
    )
    await scheduler.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        pass
