"""Temporal Graph Network implementation for streaming paper embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import torch
from torch import nn

from data_scripts.update_temporal_graph import TemporalEvent


def _to_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


@dataclass
class MemoryState:
    vector: torch.Tensor
    last_update: datetime


class TemporalMemory(nn.Module):
    """Lightweight memory module that stores hidden states per node."""

    def __init__(self, memory_dim: int) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.gru = nn.GRUCell(memory_dim, memory_dim)
        self.register_buffer("_default_state", torch.zeros(memory_dim))
        self.reset_state()

    def reset_state(self) -> None:
        self._state: Dict[str, MemoryState] = {}

    def _ensure_node(self, node_id: str) -> MemoryState:
        if node_id not in self._state:
            self._state[node_id] = MemoryState(
                vector=self._default_state.clone(),
                last_update=datetime.fromtimestamp(0, tz=timezone.utc),
            )
        return self._state[node_id]

    def get_state(self, node_id: str) -> MemoryState:
        return self._ensure_node(node_id)

    def update(self, node_id: str, message: torch.Tensor, timestamp: datetime) -> None:
        state = self._ensure_node(node_id)
        hidden = self.gru(message.unsqueeze(0), state.vector.unsqueeze(0))
        self._state[node_id] = MemoryState(vector=hidden.squeeze(0), last_update=_to_utc(timestamp))

    def items(self) -> Iterator[Tuple[str, MemoryState]]:
        return iter(self._state.items())


class TemporalGraphNetwork(nn.Module):
    """Temporal Graph Network with message passing and node memory."""

    def __init__(
        self,
        memory_dim: int = 128,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.memory = TemporalMemory(memory_dim)
        feature_dim = 6
        self.message_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(memory_dim),
            nn.Linear(memory_dim, embedding_dim),
        )
        self.time_projection = nn.Linear(1, 1)
        self.adjacency: Dict[str, set[str]] = {}
        self.node_metadata: Dict[str, Dict[str, object]] = {}

    def reset_state(self) -> None:
        self.memory.reset_state()
        self.adjacency.clear()
        self.node_metadata.clear()

    def _time_feature(self, node_id: str, timestamp: datetime) -> float:
        state = self.memory.get_state(node_id)
        delta = (_to_utc(timestamp) - state.last_update).total_seconds()
        projected = self.time_projection(torch.tensor([[delta]], dtype=torch.float32))
        return projected.item()

    def _encode_features(
        self,
        event: TemporalEvent,
        node_id: str,
        direction: str,
        timestamp: datetime,
    ) -> torch.Tensor:
        time_component = self._time_feature(node_id, timestamp)
        event_flag = 1.0 if event.event_type == "citation" else 0.0
        direction_flag = 1.0 if direction == "outgoing" else -1.0
        author_count = float(len(event.metadata.get("authors", [])))
        abstract_len = float(len(event.metadata.get("abstract", "")))
        context_len = float(len(event.metadata.get("context", "")))
        features = torch.tensor(
            [time_component, event_flag, direction_flag, author_count, abstract_len, context_len],
            dtype=torch.float32,
        )
        return self.message_encoder(features)

    def _update_adjacency(self, source: str, target: str) -> None:
        self.adjacency.setdefault(source, set()).add(target)
        self.adjacency.setdefault(target, set()).add(source)

    def _update_metadata(self, node_id: str, metadata: Dict[str, object]) -> None:
        existing = self.node_metadata.get(node_id, {})
        merged = {**existing, **metadata}
        self.node_metadata[node_id] = merged

    def process_event(self, event: TemporalEvent) -> None:
        timestamp = event.timestamp
        if event.event_type == "submission":
            self._update_metadata(event.target, event.metadata)
            message = self._encode_features(event, event.target, "incoming", timestamp)
            self.memory.update(event.target, message, timestamp)
        elif event.event_type == "citation":
            self._update_adjacency(event.source, event.target)
            out_message = self._encode_features(event, event.source, "outgoing", timestamp)
            in_message = self._encode_features(event, event.target, "incoming", timestamp)
            self.memory.update(event.source, out_message, timestamp)
            self.memory.update(event.target, in_message, timestamp)
        else:  # pragma: no cover - defensive branch for future events
            raise ValueError(f"Unknown event type: {event.event_type}")

    def process_stream(self, events: Iterable[TemporalEvent]) -> None:
        for event in sorted(events, key=lambda evt: evt.timestamp):
            self.process_event(event)

    def get_embedding(self, node_id: str) -> torch.Tensor:
        state = self.memory.get_state(node_id)
        return self.embedding_head(state.vector)

    def snapshot_embeddings(self) -> Dict[str, Dict[str, object]]:
        snapshot: Dict[str, Dict[str, object]] = {}
        for node_id, state in self.memory.items():
            embedding = self.embedding_head(state.vector).detach().numpy().tolist()
            metadata = self.node_metadata.get(node_id, {})
            snapshot[node_id] = {
                "embedding": embedding,
                "metadata": metadata,
                "last_updated": state.last_update.isoformat(),
            }
        return snapshot

    def export_snapshot(self, output_path: Path, timestamp: datetime) -> None:
        data = {
            "timestamp": _to_utc(timestamp).isoformat(),
            "nodes": self.snapshot_embeddings(),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
