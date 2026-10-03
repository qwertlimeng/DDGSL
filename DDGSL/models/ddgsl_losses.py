"""Equations (23)-(27), independent of DataLoader order and hidden state."""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WindowMetadata:
    # segment_id identifies a contiguous partition WITHIN a sequence.
    sequence_id: torch.Tensor
    segment_id: torch.Tensor
    window_index: torch.Tensor
    time_index: torch.Tensor
    stride: torch.Tensor
    is_first: torch.Tensor

    def to(self, device):
        return WindowMetadata(**{name: getattr(self, name).to(device) for name in
                                 ('sequence_id', 'segment_id', 'window_index',
                                  'time_index', 'stride', 'is_first')})

    def validate(self, batch, device):
        for name in ('sequence_id', 'segment_id', 'window_index', 'time_index', 'stride', 'is_first'):
            value = getattr(self, name)
            if value.shape != (batch,) or value.device != device:
                raise ValueError(name + ' must be [batch] on the graph device')
            if name == 'is_first':
                if value.dtype != torch.bool:
                    raise ValueError('is_first must be bool')
            elif value.dtype != torch.long:
                raise ValueError(name + ' must be torch.long')
        if (self.stride <= 0).any() or (self.window_index < 0).any():
            raise ValueError('positive stride and nonnegative window_index required')
        if not torch.equal(self.is_first, self.window_index == 0):
            raise ValueError('is_first must identify index zero within each continuous segment')


def temporal_smoothness(current, metadata, previous=None, previous_metadata=None):
    """Eq. (24). Explicit predecessor pairs; never compare neighboring batch rows.

    Each non-first window MUST have its true predecessor, even across batches.
    Boundaries contribute zero; batch mean includes these zero terms. Both graphs
    remain in autograd. No cache, detach, reordering, or silent missing-pair skip.
    """
    batch = current.shape[0]
    metadata.validate(batch, current.device)
    valid = ~metadata.is_first
    if previous is None:
        if valid.any():
            raise ValueError('missing predecessor for a non-first window')
        if previous_metadata is not None:
            raise ValueError('previous metadata without previous graph')
        per_window = current.sum(dim=(-1, -2)) * 0
        return per_window.mean(), per_window
    if previous.shape != current.shape or previous_metadata is None:
        raise ValueError('matching previous graphs and metadata are required')
    previous_metadata.validate(batch, current.device)
    same_sequence = metadata.sequence_id == previous_metadata.sequence_id
    same_segment = metadata.segment_id == previous_metadata.segment_id
    adjacent = metadata.window_index == previous_metadata.window_index + 1
    same_stride = metadata.stride == previous_metadata.stride
    adjacent_time = metadata.time_index - previous_metadata.time_index == metadata.stride
    if (valid & ~(same_sequence & same_segment & adjacent & same_stride & adjacent_time)).any():
        raise ValueError('predecessor must be adjacent in the same sequence and continuous segment')
    differences = (current - previous).abs().mean(dim=(-1, -2))
    per_window = torch.where(valid, differences, differences * 0)
    return per_window.mean(), per_window


def direction_regularization(a_dir, eps):
    """Equations (25)-(26), ratio computed separately for each window."""
    numerator = (a_dir - a_dir.transpose(-1, -2)).abs().sum(dim=(-1, -2))
    denominator = a_dir.abs().sum(dim=(-1, -2)) + eps
    strength = numerator / denominator
    per_window = 1 / (1 + strength)
    return per_window.mean(), per_window, strength


def core_objective(prediction, target, smooth, direction, mu1, mu2):
    """Eq. (27). Return independent scalar terms without detaching gradients."""
    if not all(math.isfinite(v) and v >= 0 for v in (mu1, mu2)):
        raise ValueError('mu1/mu2 must be finite and nonnegative')
    if smooth is None:
        if mu1 != 0:
            raise ValueError('mu1 > 0 requires explicit temporal predecessor metadata')
        smooth = prediction.sum() * 0
    if prediction.shape != target.shape:
        raise ValueError('prediction/target shapes must match; broadcasting is forbidden')
    mse = F.mse_loss(prediction, target, reduction='mean')
    total = mse + mu1 * smooth + mu2 * direction
    return {'total': total, 'mse': mse, 'smooth': smooth, 'direction': direction}
