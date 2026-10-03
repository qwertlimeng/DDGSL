"""
depth=1/2 -> 3/7 block applications. ONE parameter set across all branches.
All levels are concatenated per node, then a Linear restores original width.
"""
from functools import lru_cache

import torch
from torch import nn
from .spatiotemporal_graph_block import GDN
from .gated_temporal_convolution import GatedTemporalEncoder


@lru_cache(maxsize=64)
def _linear_weights(input_length, output_length):
    """Fixed half-pixel interpolation coefficients, computed on CPU."""
    positions = ((torch.arange(output_length, dtype=torch.float64) + .5)
                 * input_length / output_length - .5).clamp(0, input_length - 1)
    left = positions.floor().long()
    right = (left + 1).clamp(max=input_length - 1)
    fraction = positions - left
    columns = torch.arange(output_length)
    weights = torch.zeros(input_length, output_length, dtype=torch.float64)
    weights[left, columns] = 1 - fraction
    weights[right, columns] += fraction
    return weights


def align_time(x, length):
    """Linear align_corners=False interpolation with deterministic gradients.

    The same fixed linear operator is applied by matmul, avoiding CUDA's
    nondeterministic upsample_linear1d backward. No learned weights are added.
    """
    if x.shape[-1] == length:
        return x
    weights = _linear_weights(x.shape[-1], length).to(device=x.device, dtype=x.dtype)
    return x.matmul(weights)


class SharedSTIB(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.gat_output_dim != config.window_size:
            raise ValueError('shared STIB needs gat_output_dim == canonical window_size')
        self.config = config
        self.temporal = GatedTemporalEncoder(config.temporal_hidden_dim,
                                             config.temporal_kernel_size,
                                             config.temporal_negative_slope)
        self.gdn = GDN(config=config,
                       embedding_f=nn.Linear(config.window_size, config.embedding_dim),
                       embedding_b=nn.Linear(config.window_size, config.embedding_dim))

    def forward(self, x):
        length = x.shape[-1]
        canonical = align_time(x, self.config.window_size)
        temporal = self.temporal(canonical)
        result = self.gdn(temporal, return_details=True)
        result['temporal'] = temporal
        result['Z'] = align_time(result['Z'], length)
        return result


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.window_size < 2 ** config.stib_depth:
            raise ValueError('window too short for nonempty even/odd branches at requested depth')
        self.config = config
        self.shared_stib = SharedSTIB(config)
        # Each complete tree level has total time length w, including odd w.
        self.concat_dim = (config.stib_depth + 1) * config.window_size
        self.align = nn.Linear(self.concat_dim, config.window_size)

    @staticmethod
    def _split_even_odd_torch(x):
        return x[..., ::2], x[..., 1::2]

    def forward(self, x, return_details=False):
        cfg = self.config
        if x.ndim != 3 or x.shape[1:] != (cfg.node_num, cfg.window_size):
            raise ValueError('expected [batch,node_num,window_size]')
        branches = []
        paths = []

        def visit(branch_input, level, path):
            block = self.shared_stib(branch_input)
            branches.append(block)
            paths.append(path)
            if level < cfg.stib_depth:
                even, odd = self._split_even_odd_torch(block['Z'])
                visit(even, level + 1, path + '/even')
                visit(odd, level + 1, path + '/odd')

        visit(x, 0, 'root')
        concatenated = torch.cat([branch['Z'] for branch in branches], dim=-1)
        aligned = self.align(concatenated)  # Eq. (17), author-confirmed concat + FC
        if return_details:
            return {'aligned': aligned, 'Z_cat': concatenated,
                    'branches': branches, 'branch_paths': paths}
        return aligned
