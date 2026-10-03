"""Graph/prediction building block used inside the author-confirmed shared STIB.
No detach, static graph branch, or extra prediction ensemble.
"""
from torch import nn
from .ddgsl_losses import direction_regularization, temporal_smoothness
from .adaptive_dual_graph import DualGraph
from .directed_graph_attention import GraphLayer


class GDN(nn.Module):
    def __init__(self, *legacy_args, config=None, embedding_f=None, embedding_b=None,
                 prediction_head=None, **legacy_kwargs):
        super().__init__()
        if legacy_args or legacy_kwargs or config is None:
            raise ValueError(
                'Legacy GDN/ensemble configuration is incompatible. Construct the full model '
                'as DDGSL(CoreConfig.confirmed(...)), or supply explicit CoreConfig and '
                'embedding operators for this graph block. See README.md.')
        if embedding_f is None or embedding_b is None:
            raise ValueError('Explicit Embedding operators required')
        self.config = config
        self.graphs = DualGraph(config, embedding_f, embedding_b)
        self.attention = GraphLayer(config)
        self.prediction_head = prediction_head

    def forward(self, data, org_edge_index=None, *, return_details=False,
                metadata=None, previous_data=None, previous_metadata=None):
        if org_edge_index is not None:
            raise ValueError('DDGSL learns A per window; external graph cannot override it')
        details = self.graphs(data)  # keeps the temporal encoder attached
        z, attention, h = self.attention(data, details['A'], return_attention_weights=True)
        prediction = None
        if self.prediction_head is not None:
            prediction = self.prediction_head(z)
            if prediction.shape != (data.shape[0], self.config.node_num, 1):
                raise ValueError('single-block test head must produce [batch,node_num,1]')
            prediction = prediction.squeeze(-1)
        direction, per_direction, strength = direction_regularization(details['A_dir'], self.config.eps)
        smooth, per_smooth, previous_a = None, None, None
        if metadata is not None:
            if previous_data is not None:
                previous_a = self.graphs(previous_data)['A']
            smooth, per_smooth = temporal_smoothness(details['A'], metadata, previous_a, previous_metadata)
        elif previous_data is not None or previous_metadata is not None:
            raise ValueError('current metadata required with predecessor data')
        details.update({'prediction': prediction, 'Z': z, 'H': h, 'attention': attention,
                        'A_previous': previous_a, 'L_smooth': smooth,
                        'L_smooth_per_window': per_smooth, 'L_direction': direction,
                        'L_direction_per_window': per_direction, 'E_asym': strength})
        return details if return_details else (z if prediction is None else prediction)
