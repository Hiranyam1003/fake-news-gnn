import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import scatter, softmax

from config import LABELS


class ContentMLP(nn.Module):
    """Content-only baseline: source-tweet text vector -> class logits."""

    def __init__(self, text_dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, len(LABELS))
        )

    def forward(self, data, return_attention=False):
        logits = self.net(data.text)
        return (logits, None) if return_attention else logits


class SAGEClassifier(nn.Module):
    """GraphSAGE over the propagation tree with attention pooling.

    use_text=False -> structure-only. use_text=True -> joint: the projected source-text
    vector is concatenated to every node before every layer (Bi-GCN's root enhancement)
    and to the pooled graph vector. The pooling attention weights are the explanation.
    """

    def __init__(self, node_dim, text_dim, hidden, layers, dropout, use_text, root_enhance=True):
        super().__init__()
        self.use_text = use_text
        self.root_enhance = use_text and root_enhance
        self.dropout = dropout
        extra = hidden if use_text else 0
        if use_text:
            self.text_proj = nn.Linear(text_dim, hidden)
        dims = [node_dim] + [hidden] * layers
        conv_extra = hidden if self.root_enhance else 0
        self.convs = nn.ModuleList(SAGEConv(dims[i] + conv_extra, dims[i + 1]) for i in range(layers))
        self.gate = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.head = nn.Sequential(
            nn.Linear(hidden + extra, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, len(LABELS))
        )

    def forward(self, data, return_attention=False):
        h, edge_index, batch = data.x, data.edge_index, data.batch
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        t = F.relu(self.text_proj(data.text)) if self.use_text else None
        for conv in self.convs:
            if self.root_enhance:
                h = torch.cat([h, t[batch]], dim=1)
            h = F.dropout(F.relu(conv(h, edge_index)), self.dropout, self.training)
        attention = softmax(self.gate(h).squeeze(-1), batch)
        g = scatter(attention.unsqueeze(-1) * h, batch, dim=0, dim_size=int(batch.max()) + 1, reduce="sum")
        if self.use_text:
            g = torch.cat([g, t], dim=1)
        logits = self.head(g)
        return (logits, attention) if return_attention else logits


def build_model(mode, node_dim, text_dim, cfg):
    if mode == "content":
        return ContentMLP(text_dim, cfg["hidden"], cfg["dropout"])
    return SAGEClassifier(node_dim, text_dim, cfg["hidden"], cfg["layers"], cfg["dropout"],
                          use_text=mode == "joint", root_enhance=cfg.get("root_enhance", True))
