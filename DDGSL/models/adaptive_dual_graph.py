"""Equations (8)-(12): current-window cosine, scalar gate, sparse-view fusion."""
import torch
from torch import nn
from .direction_aware_graph import DirectionGraph
from .graph_operations import cosine_graph, positive_topk


class DualGraph(nn.Module):
    def __init__(self, config, embedding_f, embedding_b):
        super().__init__()
        self.config = config
        self.direction = DirectionGraph(config, embedding_f, embedding_b)
        self.gate = nn.Linear(config.window_size, 1)

    def forward(self, x):
        cfg = self.config
        result = self.direction(x)
        c, a_cos = cosine_graph(x, cfg.topk, cfg.eps)
        pooled = x.mean(dim=1) if cfg.pooling == 'mean' else x.max(dim=1).values  # (10)
        lam = torch.sigmoid(self.gate(pooled)).unsqueeze(-1)  # (11): [batch,1,1]
        b = lam * result['A_dir'] + (1 - lam) * a_cos  # (12): both views ALREADY sparse
        a = positive_topk(b, cfg.topk)
        result.update({'C': c, 'A_cos': a_cos, 'lambda': lam, 'B': b, 'A': a})
        return result
