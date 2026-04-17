"""
Graph Neural Network fraud detector using PyTorch Geometric.

Architecture: GraphSAGE → GAT → MLP head
Input: transaction graph where nodes are (customer, device, merchant, ip)
       and edges are relationships (MADE_TRANSACTION, USES_DEVICE, …)

torch and torch-geometric are optional; if unavailable the GNN is skipped
by the ensemble and the model save/load methods raise ImportError.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.common.logging import get_logger
from src.models.base import FraudClassifier

logger = get_logger(__name__)

# Optional heavy deps – imported lazily so the rest of the codebase
# (config, schemas, feature engineering, API) can load without GPU deps.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
    _TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    _TORCH_AVAILABLE = False
    DEVICE = None  # type: ignore[assignment]

NODE_FEATURE_DIM = 16


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch and torch-geometric are required for GNN features. "
            "Install them with: pip install torch torch-geometric"
        )


class GraphSAGE_GAT:
    """Defined only when torch is available."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "GraphSAGE_GAT":  # type: ignore[misc]
        _require_torch()
        return object.__new__(cls)

    def __init__(
        self,
        in_channels: int = NODE_FEATURE_DIM,
        hidden_channels: int = 64,
        out_channels: int = 32,
        gat_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        import torch.nn as nn
        from torch_geometric.nn import GATConv, SAGEConv
        # Store as plain attributes; actual nn.Module behaviour wired below
        self._sage1 = SAGEConv(in_channels, hidden_channels)
        self._sage2 = SAGEConv(hidden_channels, hidden_channels)
        self._gat = GATConv(hidden_channels, out_channels, heads=gat_heads, dropout=dropout, concat=False)
        self._mlp = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )
        self._dropout = dropout

    def forward(self, x: Any, edge_index: Any, batch: Optional[Any] = None) -> Any:
        import torch.nn.functional as F
        from torch_geometric.nn import global_mean_pool
        x = F.relu(self._sage1(x, edge_index))
        x = F.dropout(x, p=self._dropout, training=False)
        x = F.relu(self._sage2(x, edge_index))
        x = self._gat(x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return self._mlp(x)


class _TorchModule:
    """Mixin that makes GraphSAGE_GAT a proper nn.Module when torch is available."""


def _build_module(hidden_channels: int = 64, gat_heads: int = 4, dropout: float = 0.3) -> Any:
    _require_torch()
    import torch.nn as nn
    from torch_geometric.nn import GATConv, SAGEConv

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sage1 = SAGEConv(NODE_FEATURE_DIM, hidden_channels)
            self.sage2 = SAGEConv(hidden_channels, hidden_channels)
            self.gat = GATConv(hidden_channels, 32, heads=gat_heads, dropout=dropout, concat=False)
            self.mlp = nn.Sequential(
                nn.Linear(32, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 2)
            )
            self._dropout = dropout

        def forward(self, x: Any, edge_index: Any, batch: Optional[Any] = None) -> Any:
            import torch.nn.functional as F
            from torch_geometric.nn import global_mean_pool
            x = F.relu(self.sage1(x, edge_index))
            x = F.dropout(x, p=self._dropout, training=self.training)
            x = F.relu(self.sage2(x, edge_index))
            x = self.gat(x, edge_index)
            if batch is not None:
                x = global_mean_pool(x, batch)
            return self.mlp(x)

    return _Net().to(DEVICE)


class GNNFraudClassifier(FraudClassifier):
    """PyG-based GNN fraud detector operating on transaction subgraphs.

    Falls back gracefully when torch is not installed.
    """

    name = "gnn"

    def __init__(
        self,
        hidden_channels: int = 64,
        gat_heads: int = 4,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
    ) -> None:
        self.hidden_channels = hidden_channels
        self.gat_heads = gat_heads
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self._model: Any = None

    @property
    def available(self) -> bool:
        return _TORCH_AVAILABLE

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        graph_data_list: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> "GNNFraudClassifier":
        _require_torch()
        import torch
        import torch.nn as nn
        from torch_geometric.loader import DataLoader

        if graph_data_list is None:
            graph_data_list = self._tabular_to_graphs(X, y)

        self._model = _build_module(self.hidden_channels, self.gat_heads, self.dropout)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-4)
        loader = DataLoader(graph_data_list, batch_size=32, shuffle=True)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 50.0], device=DEVICE))

        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                out = self._model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info("GNN epoch", epoch=epoch + 1, loss=round(total_loss / len(loader), 4))
        return self

    def _tabular_to_graphs(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> List[Any]:
        _require_torch()
        import torch
        from torch_geometric.data import Data

        graphs = []
        X_proc = self.preprocess(X).values.astype(np.float32)
        labels = y.values.astype(np.int64) if y is not None else np.zeros(len(X), dtype=np.int64)
        for row, label in zip(X_proc, labels):
            x = torch.zeros((3, NODE_FEATURE_DIM), dtype=torch.float)
            x[0, : min(len(row), NODE_FEATURE_DIM)] = torch.tensor(row[:NODE_FEATURE_DIM])
            edge_index = torch.tensor([[0, 1, 2, 0], [2, 2, 0, 1]], dtype=torch.long)
            graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long)))
        return graphs

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        _require_torch()
        import torch
        import torch.nn.functional as F
        from torch_geometric.loader import DataLoader

        if self._model is None:
            raise RuntimeError("Model not trained – call fit() first")
        self._model.eval()
        graphs = self._tabular_to_graphs(X)
        loader = DataLoader(graphs, batch_size=256, shuffle=False)
        probs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                logits = self._model(batch.x, batch.edge_index, batch.batch)
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def save(self, path: Path) -> None:
        _require_torch()
        import torch
        if self._model is None:
            raise RuntimeError("No trained model to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "hidden_channels": self.hidden_channels,
                "gat_heads": self.gat_heads,
                "dropout": self.dropout,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "GNNFraudClassifier":
        _require_torch()
        import torch
        ckpt = torch.load(path, map_location=DEVICE)
        instance = cls(
            hidden_channels=ckpt.get("hidden_channels", 64),
            gat_heads=ckpt.get("gat_heads", 4),
            dropout=ckpt.get("dropout", 0.3),
        )
        instance._model = _build_module(instance.hidden_channels, instance.gat_heads, instance.dropout)
        instance._model.load_state_dict(ckpt["state_dict"])
        instance._model.eval()
        return instance
