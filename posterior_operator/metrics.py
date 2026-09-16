"""Evaluation metrics for conditional distribution estimates.

Density- and CDF-based discrepancies are computed on a shared 1-D grid by the
trapezoid rule; the calibration metrics take the intervals or quantiles a
method produced and the realised responses.
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import Tensor

__all__ = [
    "trapezoid",
    "hellinger",
    "total_variation",
    "kl_divergence",
    "js_divergence",
    "kolmogorov_smirnov",
    "wasserstein1",
    "coverage",
    "interval_width",
    "pinball_loss",
]


def _check_grid(values: Tensor, grid: Tensor) -> Tensor:
    grid = grid.reshape(-1)
    if values.shape[-1] != grid.numel():
        raise ValueError(f"last axis of values has {values.shape[-1]} entries but the grid has {grid.numel()}")
    if grid.numel() < 2:
        raise ValueError("need at least 2 grid points")
    return grid


def trapezoid(values: Tensor, grid: Tensor) -> Tensor:
    """Integrate ``values`` over the last axis against ``grid``."""
    grid = _check_grid(values, grid)
    widths = grid[1:] - grid[:-1]
    return (0.5 * (values[..., 1:] + values[..., :-1]) * widths).sum(dim=-1)


def _normalise(density: Tensor, grid: Tensor) -> Tensor:
    mass = trapezoid(density, grid).unsqueeze(-1)
    return density / mass.clamp_min(torch.finfo(density.dtype).eps)


def hellinger(p: Tensor, q: Tensor, grid: Tensor, normalise: bool = True) -> Tensor:
    r""":math:`H(p, q) = \sqrt{\tfrac12 \int (\sqrt{p} - \sqrt{q})^2}`, in ``[0, 1]``."""
    if normalise:
        p, q = _normalise(p, grid), _normalise(q, grid)
    integrand = (p.clamp_min(0).sqrt() - q.clamp_min(0).sqrt()) ** 2
    return (0.5 * trapezoid(integrand, grid)).clamp_min(0).sqrt()


def total_variation(p: Tensor, q: Tensor, grid: Tensor, normalise: bool = True) -> Tensor:
    r""":math:`\mathrm{TV}(p, q) = \tfrac12 \int |p - q|`, in ``[0, 1]``."""
    if normalise:
        p, q = _normalise(p, grid), _normalise(q, grid)
    return 0.5 * trapezoid((p - q).abs(), grid)


def kl_divergence(p: Tensor, q: Tensor, grid: Tensor, eps: float = 1e-12, normalise: bool = True) -> Tensor:
    r""":math:`\mathrm{KL}(p \Vert q) = \int p \log(p/q)`, with ``q`` floored at ``eps``."""
    if normalise:
        p, q = _normalise(p, grid), _normalise(q, grid)
    p_safe = p.clamp_min(eps)
    q_safe = q.clamp_min(eps)
    return trapezoid(p.clamp_min(0) * (torch.log(p_safe) - torch.log(q_safe)), grid)


def js_divergence(p: Tensor, q: Tensor, grid: Tensor, **kw) -> Tensor:
    """Jensen-Shannon divergence, the symmetrised and bounded relative entropy."""
    m = 0.5 * (_normalise(p, grid) + _normalise(q, grid))
    return 0.5 * kl_divergence(p, m, grid, normalise=True, **kw) + 0.5 * kl_divergence(q, m, grid, normalise=True, **kw)


def kolmogorov_smirnov(cdf_p: Tensor, cdf_q: Tensor) -> Tensor:
    """Sup-norm distance between two CDFs evaluated on a common grid."""
    if cdf_p.shape != cdf_q.shape:
        raise ValueError(f"shape mismatch: {tuple(cdf_p.shape)} vs {tuple(cdf_q.shape)}")
    return (cdf_p - cdf_q).abs().amax(dim=-1)


def wasserstein1(cdf_p: Tensor, cdf_q: Tensor, grid: Tensor) -> Tensor:
    r""":math:`W_1 = \int |F_p - F_q|`, the 1-Wasserstein distance in 1-D."""
    return trapezoid((cdf_p - cdf_q).abs(), grid)


def coverage(intervals: Tensor, y_true: Tensor) -> Tensor:
    """Fraction of responses falling inside their own interval.

    Args:
        intervals: ``(n, 2)`` lower/upper endpoints.
        y_true: ``(n,)`` or ``(n, 1)`` realised responses.
    """
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError(f"intervals must have shape (n, 2), got {tuple(intervals.shape)}")
    y = y_true.reshape(-1)
    if y.numel() != intervals.shape[0]:
        raise ValueError(f"{intervals.shape[0]} intervals but {y.numel()} responses")
    inside = (y >= intervals[:, 0]) & (y <= intervals[:, 1])
    return inside.to(intervals.dtype).mean()


def interval_width(intervals: Tensor) -> Tuple[Tensor, Tensor]:
    """``(mean, std)`` of the interval widths -- the efficiency side of coverage."""
    widths = intervals[:, 1] - intervals[:, 0]
    return widths.mean(), widths.std(unbiased=widths.numel() > 1)


def pinball_loss(predicted: Tensor, y_true: Tensor, levels: Union[Tensor, float]) -> Tensor:
    r"""Average quantile (pinball) loss :math:`\max(p\,e, (p-1)\,e)`, ``e = y - \hat{q}_p``.

    Args:
        predicted: ``(n, k)`` predicted quantiles.
        y_true: ``(n,)`` realised responses.
        levels: ``(k,)`` quantile levels matching the columns of ``predicted``.
    """
    pred = predicted if predicted.ndim == 2 else predicted.reshape(-1, 1)
    lv = torch.as_tensor(levels, dtype=pred.dtype, device=pred.device).reshape(1, -1)
    if lv.shape[1] != pred.shape[1]:
        raise ValueError(f"{pred.shape[1]} predicted quantiles but {lv.shape[1]} levels")
    err = y_true.reshape(-1, 1) - pred
    return torch.maximum(lv * err, (lv - 1.0) * err).mean()
