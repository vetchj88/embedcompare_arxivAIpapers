"""Train a HAN-style heterogeneous GNN and persist graph embeddings."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:  # chromadb is used elsewhere in the project, so we expect it to be available.
    import chromadb
    from chromadb import errors as chromadb_errors
except Exception as exc:  # pragma: no cover - defensive guard for optional dependency
    raise SystemExit("chromadb is required for hetero_gnn training") from exc


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


@dataclass
class TrainerConfig:
    graph_path: Path
    chroma_path: Optional[Path]
    section_collection: Optional[str]
    output_collection: Optional[str]
    output_json: Optional[Path]
    hidden_dim: int = 256
    embedding_dim: int = 256
    num_layers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    alignment_weight: float = 1.0
    l2_weight: float = 1e-4
    epochs: int = 200
    log_every: int = 10
    init_dim: int = 128


class HeteroGraphDataset:
    """Utility wrapper that converts the JSON graph into tensors."""

    def __init__(self, graph_path: Path) -> None:
        if not graph_path.exists():
            raise FileNotFoundError(
                f"Graph file '{graph_path}' not found. Run data_scripts/build_hetero_graph.py first."
            )
        payload = json.loads(graph_path.read_text())
        self.nodes: Dict[str, List[dict]] = payload.get("nodes", {})
        self.edges: List[dict] = payload.get("edges", [])
        self.metadata: dict = payload.get("metadata", {})
        self.feature_vocab: List[str] = self.metadata.get("feature_vocab", [])

        self.node_types: List[str] = list(self.nodes.keys())
        self.id_to_index: Dict[str, Dict[str, int]] = {
            node_type: {node["id"]: idx for idx, node in enumerate(entries)}
            for node_type, entries in self.nodes.items()
        }
        self.num_nodes: Dict[str, int] = {node_type: len(entries) for node_type, entries in self.nodes.items()}
        self.paper_ids: List[str] = [node["id"] for node in self.nodes.get("paper", [])]
        self.paper_metadata: List[dict] = self.nodes.get("paper", [])

        self.edge_index_dict = self._build_edge_index()
        self.features = self._build_initial_features()

    def _build_edge_index(self) -> Dict[Tuple[str, str, str], Dict[str, torch.Tensor]]:
        edge_dict: Dict[Tuple[str, str, str], Dict[str, List[int]]] = {}
        weights_dict: Dict[Tuple[str, str, str], List[float]] = {}
        for edge in self.edges:
            src_type = edge.get("source_type")
            dst_type = edge.get("target_type")
            rel_type = edge.get("type")
            src_id = edge.get("source")
            dst_id = edge.get("target")
            if (
                src_type not in self.id_to_index
                or dst_type not in self.id_to_index
                or src_id not in self.id_to_index[src_type]
                or dst_id not in self.id_to_index[dst_type]
            ):
                continue
            key = (src_type, rel_type, dst_type)
            if key not in edge_dict:
                edge_dict[key] = {"src": [], "dst": []}
                weights_dict[key] = []
            edge_dict[key]["src"].append(self.id_to_index[src_type][src_id])
            edge_dict[key]["dst"].append(self.id_to_index[dst_type][dst_id])
            weights_dict[key].append(float(edge.get("weight", 1.0)))

        tensor_dict: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]] = {}
        for key, idx_lists in edge_dict.items():
            src_tensor = torch.tensor(idx_lists["src"], dtype=torch.long)
            dst_tensor = torch.tensor(idx_lists["dst"], dtype=torch.long)
            weight_values = weights_dict.get(key, [])
            weight_tensor = (
                torch.tensor(weight_values, dtype=torch.float32) if any(weight_values) else torch.ones(len(src_tensor))
            )
            tensor_dict[key] = {"src": src_tensor, "dst": dst_tensor, "weight": weight_tensor}
        return tensor_dict

    def _build_initial_features(self) -> Dict[str, Optional[torch.Tensor]]:
        features: Dict[str, Optional[torch.Tensor]] = {}
        vocab_size = max(len(self.feature_vocab), 1)
        paper_nodes = self.nodes.get("paper", [])
        if paper_nodes:
            matrix = torch.zeros(len(paper_nodes), vocab_size, dtype=torch.float32)
            for row_idx, node in enumerate(paper_nodes):
                for feat_idx in node.get("feature_indices", []):
                    if 0 <= feat_idx < vocab_size:
                        matrix[row_idx, feat_idx] = 1.0
            if vocab_size == 1:
                matrix[:, 0] = 1.0
            features["paper"] = matrix
        else:
            features["paper"] = None

        for node_type in self.node_types:
            if node_type == "paper":
                continue
            features[node_type] = None
        return features


class HeteroLayer(nn.Module):
    def __init__(self, in_dims: Dict[str, int], out_dim: int, relations: Iterable[Tuple[str, str, str]], activation: bool = True):
        super().__init__()
        self.node_types = list(in_dims.keys())
        self.self_linears = nn.ModuleDict({node_type: nn.Linear(in_dims[node_type], out_dim) for node_type in self.node_types})
        self.rel_linears = nn.ModuleDict()
        for src_type, rel_type, dst_type in relations:
            key = self._format_key(src_type, rel_type, dst_type)
            if src_type not in in_dims:
                continue
            self.rel_linears[key] = nn.Linear(in_dims[src_type], out_dim)
        self.activation = activation

    @staticmethod
    def _format_key(src: str, rel: str, dst: str) -> str:
        return f"{src}__{rel}__{dst}"

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        outputs = {node_type: self.self_linears[node_type](inputs[node_type]) for node_type in inputs}
        for (src_type, rel_type, dst_type), tensors in edge_index_dict.items():
            key = self._format_key(src_type, rel_type, dst_type)
            if key not in self.rel_linears or src_type not in inputs or dst_type not in outputs:
                continue
            src_idx = tensors["src"]
            dst_idx = tensors["dst"]
            weight = tensors.get("weight")
            src_messages = self.rel_linears[key](inputs[src_type])
            gathered = src_messages.index_select(0, src_idx)
            if weight is not None:
                gathered = gathered * weight.unsqueeze(-1).to(gathered.device)
            aggregated = scatter_add(gathered, dst_idx.to(gathered.device), outputs[dst_type].size(0))
            outputs[dst_type] = outputs[dst_type] + aggregated
        if self.activation:
            outputs = {node_type: F.elu(tensor) for node_type, tensor in outputs.items()}
        return outputs


class HeteroGraphEncoder(nn.Module):
    def __init__(self, dataset: HeteroGraphDataset, config: TrainerConfig) -> None:
        super().__init__()
        self.node_types = dataset.node_types
        self.relations = list(dataset.edge_index_dict.keys())
        self.embeddings = nn.ModuleDict()

        # Determine input dimensionality per node type.
        self.input_dims: Dict[str, int] = {}
        for node_type in self.node_types:
            feature = dataset.features.get(node_type)
            if feature is None:
                self.input_dims[node_type] = config.init_dim
                self.embeddings[node_type] = nn.Embedding(dataset.num_nodes.get(node_type, 1), config.init_dim)
            else:
                dim = feature.size(1)
                self.input_dims[node_type] = dim

        layers: List[HeteroLayer] = []
        current_dims = dict(self.input_dims)
        for layer_idx in range(config.num_layers):
            is_last = layer_idx == config.num_layers - 1
            next_dim = config.embedding_dim if is_last else config.hidden_dim
            layers.append(HeteroLayer(current_dims, next_dim, self.relations, activation=not is_last))
            current_dims = {node_type: next_dim for node_type in current_dims}
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        features: Dict[str, Optional[torch.Tensor]],
        edge_index_dict: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        inputs: Dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            if node_type in self.embeddings:
                inputs[node_type] = self.embeddings[node_type].weight
            else:
                assert features[node_type] is not None
                inputs[node_type] = features[node_type]
        for layer in self.layers:
            inputs = layer(inputs, edge_index_dict)
        return inputs


class HeteroGNNTrainer:
    def __init__(self, dataset: HeteroGraphDataset, config: TrainerConfig) -> None:
        self.dataset = dataset
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        targets, mask, target_dim = self._load_alignment_targets()
        if target_dim and target_dim != self.config.embedding_dim:
            print(
                f"[!] Alignment vector dim {target_dim} differs from requested embedding dim {self.config.embedding_dim}. Using {target_dim}."
            )
            self.config.embedding_dim = target_dim
        elif not target_dim and self.config.embedding_dim <= 0:
            self.config.embedding_dim = self.config.hidden_dim

        self.encoder = HeteroGraphEncoder(dataset, self.config).to(self.device)
        self.features = {
            node_type: (tensor.to(self.device) if tensor is not None else None)
            for node_type, tensor in dataset.features.items()
        }
        self.edge_index_dict = {
            key: {
                "src": tensors["src"].to(self.device),
                "dst": tensors["dst"].to(self.device),
                "weight": tensors["weight"].to(self.device),
            }
            for key, tensors in dataset.edge_index_dict.items()
        }
        self.targets = targets.to(self.device)
        self.target_mask = mask.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )

    def _load_alignment_targets(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        num_papers = len(self.dataset.paper_ids)
        placeholder_dim = max(self.config.embedding_dim, 1)
        targets = torch.zeros(num_papers, placeholder_dim, dtype=torch.float32)
        mask = torch.zeros(num_papers, dtype=torch.bool)
        if not self.config.section_collection or not self.config.chroma_path:
            return targets, mask, 0

        client = chromadb.PersistentClient(path=str(self.config.chroma_path))
        try:
            collection = client.get_collection(self.config.section_collection)
        except chromadb_errors.NotFoundError:
            print(
                f"[!] Section collection '{self.config.section_collection}' not found in ChromaDB at {self.config.chroma_path}."
            )
            return targets, mask, 0

        results = collection.get(ids=self.dataset.paper_ids, include=["embeddings", "metadatas"])
        embeddings = results.get("embeddings", [])
        ids = results.get("ids", [])
        dim = 0
        embedding_map = {pid: emb for pid, emb in zip(ids, embeddings)}
        for idx, paper_id in enumerate(self.dataset.paper_ids):
            vector = embedding_map.get(paper_id)
            if vector is None:
                continue
            array = np.asarray(vector, dtype=np.float32)
            if array.ndim != 1:
                array = array.reshape(-1)
            if not dim:
                dim = array.shape[0]
                targets = torch.zeros(num_papers, dim, dtype=torch.float32)
            elif array.shape[0] != dim:
                continue
            targets[idx] = torch.from_numpy(array)
            mask[idx] = True
        return targets, mask, dim

    def train(self) -> torch.Tensor:
        self.encoder.train()
        for epoch in range(1, self.config.epochs + 1):
            self.optimizer.zero_grad()
            outputs = self.encoder(self.features, self.edge_index_dict)
            paper_embeddings = outputs.get("paper")
            if paper_embeddings is None:
                raise RuntimeError("Graph is missing 'paper' node type.")

            loss_align = torch.tensor(0.0, device=self.device)
            if self.target_mask.any():
                predicted = paper_embeddings[self.target_mask]
                target = self.targets[self.target_mask]
                loss_align = F.mse_loss(predicted, target)

            loss_reg = sum(tensor.pow(2).mean() for tensor in outputs.values())
            loss = self.config.alignment_weight * loss_align + self.config.l2_weight * loss_reg
            loss.backward()
            self.optimizer.step()

            if epoch % self.config.log_every == 0 or epoch == 1:
                align_val = float(loss_align.detach().cpu())
                reg_val = float(loss_reg.detach().cpu())
                print(f"[Epoch {epoch:04d}] loss={loss.item():.4f} align={align_val:.4f} reg={reg_val:.4f}")

        self.encoder.eval()
        with torch.no_grad():
            final_outputs = self.encoder(self.features, self.edge_index_dict)
        return final_outputs["paper"].detach().cpu()

    def persist_embeddings(self, embeddings: torch.Tensor) -> None:
        embeddings = embeddings.detach().cpu()
        if self.config.output_collection and self.config.chroma_path:
            client = chromadb.PersistentClient(path=str(self.config.chroma_path))
            collection = client.get_or_create_collection(self.config.output_collection)
            metadatas = [
                {
                    "paper_id": node.get("id"),
                    "title": node.get("title"),
                    "openalex_id": node.get("openalex_id"),
                    "source": "hetero_gnn",
                }
                for node in self.dataset.paper_metadata
            ]
            collection.upsert(
                ids=self.dataset.paper_ids,
                embeddings=embeddings.numpy().tolist(),
                metadatas=metadatas,
            )
            print(
                f"[✅] Persisted {len(self.dataset.paper_ids)} graph embeddings to collection '{self.config.output_collection}'."
            )

        if self.config.output_json:
            records = []
            for idx, node in enumerate(self.dataset.paper_metadata):
                records.append(
                    {
                        "paper_id": node.get("id"),
                        "embedding": embeddings[idx].tolist(),
                        "metadata": {
                            "title": node.get("title"),
                            "authors": node.get("authors"),
                            "venue": node.get("venue"),
                            "fields": node.get("fields"),
                            "keyphrases": node.get("keyphrases"),
                        },
                    }
                )
            self.config.output_json.parent.mkdir(parents=True, exist_ok=True)
            self.config.output_json.write_text(json.dumps(records, indent=2))
            print(f"[✅] Graph embeddings exported to {self.config.output_json}")


def parse_args() -> TrainerConfig:
    parser = argparse.ArgumentParser(description="Train a heterogeneous GNN over the scholarly graph.")
    parser.add_argument("--graph", default="graph_data/hetero_graph.json", help="Path to the heterogeneous graph JSON")
    parser.add_argument("--chroma-path", default="./chroma_db", help="Path to the persistent ChromaDB directory")
    parser.add_argument("--section-collection", default="paper_sections", help="Chroma collection with section-aware vectors")
    parser.add_argument(
        "--output-collection", default="papers_hetero_hgnn", help="Chroma collection for the learned graph embeddings"
    )
    parser.add_argument(
        "--output-json", default=None, help="Optional path to export embeddings as JSON in addition to ChromaDB"
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimensionality for intermediate GNN layers")
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Target embedding dimensionality. Set to 0 to infer from section vectors.",
    )
    parser.add_argument("--init-dim", type=int, default=128, help="Dimension of learned node embeddings for nodes without features")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of relational message passing layers")
    parser.add_argument("--epochs", type=int, default=200, help="Number of full-batch training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for Adam optimizer")
    parser.add_argument(
        "--alignment-weight", type=float, default=1.0, help="Weight for the alignment loss against section vectors"
    )
    parser.add_argument("--l2-weight", type=float, default=1e-4, help="Weight for embedding norm regularization")
    parser.add_argument("--log-every", type=int, default=10, help="How frequently to print training diagnostics")

    args = parser.parse_args()
    output_json = Path(args.output_json) if args.output_json else None
    return TrainerConfig(
        graph_path=Path(args.graph),
        chroma_path=Path(args.chroma_path) if args.chroma_path else None,
        section_collection=args.section_collection,
        output_collection=args.output_collection,
        output_json=output_json,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alignment_weight=args.alignment_weight,
        l2_weight=args.l2_weight,
        epochs=args.epochs,
        log_every=args.log_every,
        init_dim=args.init_dim,
    )


def main() -> None:
    config = parse_args()
    dataset = HeteroGraphDataset(config.graph_path)
    trainer = HeteroGNNTrainer(dataset, config)
    embeddings = trainer.train()
    trainer.persist_embeddings(embeddings)


if __name__ == "__main__":
    main()
