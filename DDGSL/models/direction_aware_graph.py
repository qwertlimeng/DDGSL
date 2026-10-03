"""Window-dependent directional graph, equations (2)-(7).
The Embedding operator in (2) is injected explicitly by the caller.
"""
import math
import torch
from torch import nn
from .graph_operations import positive_topk


class DirectionGraph(nn.Module):
    def __init__(self, config, embedding_f, embedding_b):
        super().__init__()
        if embedding_f is embedding_b:
            raise ValueError('Eq. (2) requires independent forward/backward mappings')
        if {id(p) for p in embedding_f.parameters()} & {id(p) for p in embedding_b.parameters()}:
            raise ValueError('forward/backward mappings may not share parameters')
        if not any(p.requires_grad for p in embedding_f.parameters()) or not any(
                p.requires_grad for p in embedding_b.parameters()):
            raise ValueError('Eq. (2) mappings must each contain trainable parameters')
        self.config = config
        self.embedding_f = embedding_f
        self.embedding_b = embedding_b
        d = config.embedding_dim
        # Linear stores the transpose of the d x d matrix in Eq. (3).
        self.forward_map = nn.Linear(d, d, bias=False)
        self.backward_map = nn.Linear(d, d, bias=False)
        self.beta_logit = nn.Parameter(torch.tensor(math.log(config.beta_init / (1-config.beta_init))))
        # Paper imposes no sign constraint on learnable gamma.
        self.gamma = nn.Parameter(torch.tensor(float(config.gamma_init)))

    @property
    def beta(self):
        return torch.sigmoid(self.beta_logit)

    def forward(self, x):
        cfg = self.config
        if x.ndim != 3 or x.shape[1:] != (cfg.node_num, cfg.window_size):
            raise ValueError('expected [batch, node_num, window_size]')
        ef, eb = self.embedding_f(x), self.embedding_b(x)  # (2)
        expected = (x.shape[0], cfg.node_num, cfg.embedding_dim)
        if ef.shape != expected or eb.shape != expected:
            raise ValueError('Embedding must produce [batch, node_num, embedding_dim]')
        ef_tilde = torch.tanh(cfg.xi * self.forward_map(ef))  # (3)
        eb_tilde = torch.tanh(cfg.xi * self.backward_map(eb))
        sf = ef_tilde @ eb_tilde.transpose(-1, -2)  # (4)
        sb = eb_tilde @ ef_tilde.transpose(-1, -2)
        s_dir = self.beta * sf + (1 - self.beta) * sb  # (5)
        s_tilde = torch.relu(torch.tanh(self.gamma * s_dir))  # (7)
        a_dir = positive_topk(s_tilde, cfg.topk, exclude_self=not cfg.direction_self_loops)
        return {'E_f': ef, 'E_b': eb, 'E_f_tilde': ef_tilde, 'E_b_tilde': eb_tilde,
                'S_f': sf, 'S_b': sb, 'S_dir': s_dir, 'S_tilde': s_tilde,
                'A_dir': a_dir, 'beta': self.beta, 'gamma': self.gamma}

