"""Full DDGSL recursive STIB model and next-step prediction."""
import torch
from torch import nn
from .recursive_stib import MultiScaleTemporalEncoder
from .ddgsl_losses import temporal_smoothness


class DDGSL(nn.Module):
    """Confirmed tree -> concat/FC -> next-step regression, without extra ensemble.

    Debug graph tensors: [batch, branch, target, source]. Branch order is preorder:
    root, even subtree, odd subtree. lambda is [batch,branch,1,1].
    L_smooth=None denotes unavailable temporal metadata at inference; it is NOT
    silently zeroed when mu1 > 0 during loss computation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.multiscale = MultiScaleTemporalEncoder(config)
        self.regression = nn.Linear(config.window_size, 1)

    def forward(self, x, edge_index=None, *, return_details=False,
                metadata=None, previous_x=None, previous_metadata=None):
        # Legacy loaders still supply a static graph; it is intentionally unused.
        current = self.multiscale(x, return_details=True)
        prediction = self.regression(current['aligned']).squeeze(-1)
        if not return_details:
            if metadata is not None or previous_x is not None or previous_metadata is not None:
                raise ValueError('paired regularization requires return_details=True')
            return prediction
        branches = current['branches']
        previous = None
        if previous_x is not None:
            if previous_x.shape != x.shape:
                raise ValueError('predecessor batch shape must match current batch')
            previous = self.multiscale(previous_x, return_details=True)
        if metadata is None and (previous_x is not None or previous_metadata is not None):
            raise ValueError('current metadata required for paired windows')
        smooth, per_smooth = None, None
        smooth_terms = []
        if metadata is not None:
            for index, branch in enumerate(branches):
                previous_a = None if previous is None else previous['branches'][index]['A']
                term, per_window = temporal_smoothness(branch['A'], metadata, previous_a, previous_metadata)
                smooth_terms.append(per_window)
                branch['L_smooth'] = term
                branch['L_smooth_per_window'] = per_window
            per_smooth = torch.stack(smooth_terms, dim=1)
            smooth = per_smooth.mean()  # equal weighting across all STIBs and windows
        result = {'prediction': prediction, 'L_smooth': smooth,
                  'L_smooth_per_window_branch': per_smooth,
                  'L_direction': torch.stack([b['L_direction'] for b in branches]).mean(),
                  'branch_paths': current['branch_paths'], 'branches': branches,
                  'Z_cat': current['Z_cat'], 'aligned': current['aligned'],
                  'A_previous': None if previous is None else
                  torch.stack([b['A'] for b in previous['branches']], dim=1)}
        for key in ('A_dir', 'A_cos', 'B', 'A', 'lambda', 'attention', 'E_asym',
                    'L_direction_per_window'):
            result[key] = torch.stack([b[key] for b in branches], dim=1)
        return result
