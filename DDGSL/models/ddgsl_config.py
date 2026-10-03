"""Core formula configuration and author-confirmed STIB conventions."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class CoreConfig:
    window_size: int
    node_num: int
    embedding_dim: int
    topk: int
    stib_depth: int
    # Deliberately required: the paper does not uniquely specify these choices.
    direction_self_loops: bool
    pooling: str
    gat_activation: str
    gat_hidden_dim: int
    gat_output_dim: int
    beta_init: float = 0.5
    gamma_init: float = 1.0
    xi: float = 1.0
    mu1: float = 1e-3
    mu2: float = 1e-3
    eps: float = 1e-8
    gat_eta: float = 1.0
    gat_negative_slope: float = 0.2
    gat_heads: int = 1
    beta_parameterization: str = 'sigmoid'
    gamma_parameterization: str = 'direct'
    temporal_hidden_dim: int = 16
    temporal_kernel_size: int = 3
    temporal_negative_slope: float = 0.2
    architecture_version: str = 'ddgsl-core-v1-author-confirmed-20261002'
    length_alignment: str = 'linear_align_corners_false'

    def __post_init__(self):
        for name in ('window_size', 'node_num', 'embedding_dim', 'topk',
                     'stib_depth', 'gat_hidden_dim', 'gat_output_dim',
                     'temporal_hidden_dim', 'temporal_kernel_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be a positive integer')
        if type(self.direction_self_loops) is not bool:
            raise ValueError('direction_self_loops must be explicitly True or False')
        if not 0 < self.beta_init < 1:
            raise ValueError('finite sigmoid initialization requires 0 < beta_init < 1')
        for name in ('gamma_init', 'xi', 'mu1', 'mu2', 'eps', 'gat_eta',
                     'gat_negative_slope', 'temporal_negative_slope'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(name + ' must be finite')
        if self.eps <= 0 or min(self.mu1, self.mu2) < 0:
            raise ValueError('eps must be positive; mu1/mu2 must be nonnegative')
        if self.gat_negative_slope < 0 or self.temporal_negative_slope < 0:
            raise ValueError('LeakyReLU slopes must be nonnegative')
        if self.pooling not in ('mean', 'max'):
            raise ValueError('pooling must be explicitly mean or max')
        if self.gat_activation not in ('sigmoid', 'relu', 'tanh', 'identity'):
            raise ValueError('unsupported explicit sigma activation')
        if self.gat_heads != 1:
            raise ValueError('Equations (13)-(16) define one head; multi-head extension unconfirmed')
        if self.beta_parameterization != 'sigmoid' or self.gamma_parameterization != 'direct':
            raise ValueError('supported parameterizations: beta=sigmoid(logit), gamma=Parameter')
        if self.temporal_kernel_size % 2 != 1:
            raise ValueError('use an odd temporal kernel to preserve length')
        if self.length_alignment != 'linear_align_corners_false':
            raise ValueError('only the author-confirmed interpolation is implemented')

    @classmethod
    def confirmed(cls, window_size, node_num, embedding_dim=64, topk=10,
                  stib_depth=1, **kwargs):
        """Local choices explicitly confirmed by the author in this conversation."""
        return cls(window_size=window_size, node_num=node_num,
                   embedding_dim=embedding_dim, topk=topk, stib_depth=stib_depth,
                   direction_self_loops=False, pooling='mean', gat_activation='sigmoid',
                   gat_hidden_dim=embedding_dim, gat_output_dim=window_size, **kwargs)

    def to_dict(self):
        return asdict(self)
