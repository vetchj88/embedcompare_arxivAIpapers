"""Section-level pooling with modality-aware attention fusion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SectionPoolerOutput:
    """Output container including fused embeddings and modality diagnostics."""

    embedding: torch.Tensor
    modality_weights: Dict[str, torch.Tensor]
    attention_maps: Dict[str, torch.Tensor]


class SectionAttentionPooler(nn.Module):
    """Aggregate section-level representations while learning modality weights.

    Parameters
    ----------
    hidden_dim:
        Dimensionality of the textual embeddings used as the reference modality.
    visual_dim:
        Dimensionality of incoming visual embeddings.  Set to ``hidden_dim`` when
        the modalities already share the same space.
    num_heads:
        Number of attention heads used for intra-modality pooling.
    dropout:
        Dropout applied to modality logits before softmax fusion.
    """

    def __init__(
        self,
        hidden_dim: int,
        visual_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.visual_dim = visual_dim or hidden_dim

        self.text_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.caption_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.visual_project = nn.Linear(self.visual_dim, hidden_dim)
        self.visual_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.modality_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        caption_embeddings: Optional[torch.Tensor] = None,
        caption_mask: Optional[torch.Tensor] = None,
        visual_embeddings: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> SectionPoolerOutput:
        """Return fused section embedding and modality attribution signals."""

        modality_vectors = {}
        modality_weights = {}
        attention_maps = {}

        text_vector, text_attn = self._pool_sequence(
            embeddings=text_embeddings,
            attention_layer=self.text_attention,
            mask=text_mask,
        )
        modality_vectors["text"] = text_vector
        attention_maps["text"] = text_attn

        if caption_embeddings is not None:
            caption_vector, caption_attn = self._pool_sequence(
                embeddings=caption_embeddings,
                attention_layer=self.caption_attention,
                mask=caption_mask,
            )
            modality_vectors["captions"] = caption_vector
            attention_maps["captions"] = caption_attn

        if visual_embeddings is not None:
            projected = self.visual_project(visual_embeddings)
            visual_vector, visual_attn = self._pool_sequence(
                embeddings=projected,
                attention_layer=self.visual_attention,
                mask=visual_mask,
            )
            modality_vectors["visual"] = visual_vector
            attention_maps["visual"] = visual_attn

        fused_embedding, modality_distribution = self._fuse_modalities(modality_vectors)
        for key, weight in modality_distribution.items():
            modality_weights[key] = weight

        output = SectionPoolerOutput(
            embedding=fused_embedding,
            modality_weights=modality_weights,
            attention_maps=attention_maps,
        )
        self.last_output = output  # type: ignore[attr-defined]
        return output

    def _pool_sequence(
        self,
        embeddings: torch.Tensor,
        attention_layer: nn.MultiheadAttention,
        mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool a variable-length sequence with self-attention."""

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()

        attn_output, attn_weights = attention_layer(
            embeddings,
            embeddings,
            embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        pooled = attn_output.mean(dim=1)
        # attention weights returned as (batch, heads, query, key)
        attention_summary = attn_weights.mean(dim=1)
        return pooled, attention_summary

    def _fuse_modalities(self, modality_vectors: Mapping[str, torch.Tensor]):
        """Combine modality vectors using learned scalar weights."""

        if not modality_vectors:
            raise ValueError("At least one modality must be provided to the pooler.")

        weights: Dict[str, torch.Tensor] = {}

        modality_names = list(modality_vectors.keys())
        logits = []
        for name in modality_names:
            vector = modality_vectors[name]
            logit = self.modality_score(vector)
            logits.append(logit)
        stacked_logits = torch.cat(logits, dim=1)
        alpha = F.softmax(stacked_logits, dim=1)

        fused_vector = torch.zeros_like(next(iter(modality_vectors.values())))
        for idx, name in enumerate(modality_names):
            weight = alpha[:, idx : idx + 1]
            weights[name] = weight.squeeze(1)
            fused_vector = fused_vector + modality_vectors[name] * weight

        return fused_vector, weights


__all__ = ["SectionAttentionPooler", "SectionPoolerOutput"]
