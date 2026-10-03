"""Single-head weighted graph attention, equations (13)-(16), pure PyTorch.
A[...,i,j] denotes source j -> target i. Only A > 0 defines neighbors.
No self-connections are added; empty rows have zero aggregate before sigma.
"""
import torch
from torch import nn
import torch.nn.functional as F


def activate(x, name):
    if name == 'sigmoid':
        return torch.sigmoid(x)
    if name == 'relu':
        return torch.relu(x)
    if name == 'tanh':
        return torch.tanh(x)
    if name == 'identity':
        return x
    raise ValueError('sigma must be selected explicitly')


class GraphLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_map = nn.Linear(config.window_size, config.gat_hidden_dim, bias=False)
        self.node_embedding = nn.Embedding(config.node_num, config.embedding_dim)
        h_dim = config.gat_hidden_dim + config.embedding_dim
        self.att_target = nn.Parameter(torch.empty(h_dim))
        self.att_source = nn.Parameter(torch.empty(h_dim))
        # W in (16) differs in shape from W in (13).
        self.message_map = nn.Linear(h_dim, config.gat_output_dim, bias=False)
        nn.init.normal_(self.att_target, std=h_dim ** -0.5)
        nn.init.normal_(self.att_source, std=h_dim ** -0.5)

    def forward(self, x, adjacency, return_attention_weights=False):
        cfg = self.config
        if x.ndim != 3 or x.shape[1:] != (cfg.node_num, cfg.window_size):
            raise ValueError('invalid node feature shape')
        if adjacency.shape != (x.shape[0], cfg.node_num, cfg.node_num):
            raise ValueError('invalid adjacency shape')
        if not torch.isfinite(adjacency).all() or (adjacency < 0).any():
            raise ValueError('adjacency must contain finite, nonnegative weights')
        embedding = self.node_embedding.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        h = activate(torch.cat((self.feature_map(x), embedding), dim=-1), cfg.gat_activation)  # (13)
        target = (h * self.att_target).sum(-1).unsqueeze(-1)
        source = (h * self.att_source).sum(-1).unsqueeze(-2)
        score = F.leaky_relu(target + source + cfg.gat_eta * torch.log(adjacency + cfg.eps),
                             negative_slope=cfg.gat_negative_slope)  # (14)
        valid = adjacency > 0
        logits = score.masked_fill(~valid, float('-inf'))
        has_neighbors = valid.any(-1, keepdim=True)
        safe_logits = torch.where(has_neighbors, logits, torch.zeros_like(logits))
        attention = torch.softmax(safe_logits, dim=-1).masked_fill(~valid, 0)  # (15)
        z = activate(attention @ self.message_map(h), cfg.gat_activation)  # (16)
        if return_attention_weights:
            return z, attention, h
        return z
