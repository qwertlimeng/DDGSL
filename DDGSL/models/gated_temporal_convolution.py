# Shared gated temporal convolution for each recursive STIB call.
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedTemporalEncoder(nn.Module):
    """Eq. (1), [B,N,L] -> [B,N,L]; gates at each STIB, not after tree fusion.

    Conv kernels act only along time. Zero padding preserves L. No residual,
    extra dropout, or BSF module is added to the two-convolution formula.
    """
    def __init__(self, hidden_dim=16, kernel_size=3, negative_slope=0.2):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError('kernel_size must be a positive odd integer')
        self.negative_slope = negative_slope
        padding = (0, kernel_size // 2)
        self.conv1 = nn.Conv2d(1, hidden_dim, (1, kernel_size), padding=padding)
        self.conv2 = nn.Conv2d(hidden_dim, 2, (1, kernel_size), padding=padding)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] < 1:
            raise ValueError('expected nonempty [batch,node,time] input')
        u = F.leaky_relu(self.conv1(x.unsqueeze(1)), negative_slope=self.negative_slope)
        return F.glu(torch.tanh(self.conv2(u)), dim=1).squeeze(1)
