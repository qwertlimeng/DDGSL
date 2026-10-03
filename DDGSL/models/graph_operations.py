"""A[..., i, j] is the weight of source j -> target i throughout."""
import torch


def positive_topk(matrix, k, exclude_self=False):
    """Keep at most K positive incoming weights. No jitter or synthetic edges.

    Stable sorting resolves exact ties by column index. Selected values keep
    their autograd graph; discrete selection indices are not differentiable.
    """
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError('expected square adjacency matrices')
    if type(k) is not int or k < 0:
        raise ValueError('k must be a nonnegative integer')
    n = matrix.shape[-1]
    values = torch.relu(matrix)
    if exclude_self:
        values = values.masked_fill(torch.eye(n, dtype=torch.bool, device=values.device), 0)
    k = min(k, max(n - int(exclude_self), 0))
    indices = torch.argsort(values, dim=-1, descending=True, stable=True)[..., :k]
    return torch.zeros_like(values).scatter(-1, indices, values.gather(-1, indices))


def cosine_graph(x, k, eps):
    """Equations (8)-(9): denominator is the norm product PLUS epsilon."""
    norms = x.norm(p=2, dim=-1, keepdim=True)
    cosine = torch.relu((x @ x.transpose(-1, -2)) /
                        (norms @ norms.transpose(-1, -2) + eps))
    diagonal = torch.eye(x.shape[-2], dtype=torch.bool, device=x.device)
    cosine = cosine.masked_fill(diagonal, 0)
    return cosine, positive_topk(cosine, k, exclude_self=True)
