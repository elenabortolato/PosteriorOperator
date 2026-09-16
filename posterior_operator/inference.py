r"""Inference on the conditional law produced by a fitted :class:`NCPOperator`.

Everything here is a functional of the same object: the discrete conditional
law that NCP produces by reweighting a reference sample
:math:`\{y_j\}_{j=1}^m \sim \pi_Y`,

.. math:: \widehat{p}(\cdot \mid x) = \sum_{j=1}^{m} w_j(x)\, \delta_{y_j},
          \qquad w_j(x) = \tfrac{1}{m}\big(1 + \hat{r}(x, y_j)\big).

Conditional expectations, CDFs, quantiles, moments and confidence regions are
then plain weighted statistics of that law -- no sampling, no per-:math:`x`
optimisation, no retraining.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

__all__ = ["ConditionalDistribution", "GaussianKDE", "isotonic_regression"]

Observable = Union[None, int, Callable[[Tensor], Tensor]]


def isotonic_regression(values: Tensor) -> Tensor:
    """Least-squares projection onto non-decreasing sequences (pool adjacent violators).

    Applied row-wise to ``(..., n)`` inputs. Used to repair a CDF estimate when
    negative density-ratio values have made it non-monotone.

    The pool-adjacent-violators stack is carried for every row at once, so the
    Python loop runs once over the atoms rather than once per row. That matters
    because the natural call here is one row per observation with one atom per
    stored prior draw, and a per-row implementation costs about a second per
    observation at twenty thousand atoms.
    """
    import numpy as np

    original_shape = values.shape
    flat = values.reshape(-1, original_shape[-1])
    array = flat.detach().cpu().numpy().astype(np.float64, copy=True)
    n_rows, n = array.shape
    if n == 0:
        return values.clone()

    rows = np.arange(n_rows)
    block_val = np.empty((n_rows, n), dtype=np.float64)
    block_len = np.zeros((n_rows, n), dtype=np.int64)
    top = np.zeros(n_rows, dtype=np.int64)

    for i in range(n):
        block_val[rows, top] = array[:, i]
        block_len[rows, top] = 1
        top += 1
        # Pool while the last block undercuts the one before it. Different rows
        # need different numbers of merges, so iterate on the set that still
        # violates instead of on a per-row while loop.
        while True:
            active = top > 1
            if not active.any():
                break
            last = np.where(active, top - 1, 0)
            prev = np.where(active, top - 2, 0)
            violating = active & (block_val[rows, prev] > block_val[rows, last])
            if not violating.any():
                break
            sel = rows[violating]
            a, b = prev[violating], last[violating]
            w1 = block_len[sel, a]
            w2 = block_len[sel, b]
            block_val[sel, a] = (block_val[sel, a] * w1 + block_val[sel, b] * w2) / (w1 + w2)
            block_len[sel, a] = w1 + w2
            top[violating] -= 1

    # Expand the blocks back to length n. Each row has `top[r]` blocks whose
    # lengths sum to n, so a cumulative-count lookup assigns every position.
    out = np.empty_like(array)
    valid = np.arange(n)[None, :] < top[:, None]
    lengths = np.where(valid, block_len, 0)
    ends = np.cumsum(lengths, axis=1)
    for r in range(n_rows):
        index = np.searchsorted(ends[r, : top[r]], np.arange(n), side="right")
        out[r] = block_val[r, index]

    return torch.as_tensor(out, dtype=values.dtype, device=values.device).reshape(original_shape)


class GaussianKDE:
    r"""Gaussian kernel density estimate of the marginal :math:`\pi_Y`.

    NCP models the *ratio* :math:`p(y \mid x)/\pi_Y(y)`, so turning it into an
    actual conditional density needs an estimate of :math:`\pi_Y` -- which is a
    plain unconditional density estimation problem on data you already have.
    Bandwidth defaults to Scott's rule, per coordinate.
    """

    def __init__(self, samples: Tensor, bandwidth: Optional[float] = None):
        samples = samples if samples.ndim == 2 else samples.reshape(samples.shape[0], -1)
        if samples.shape[0] < 2:
            raise ValueError("need at least 2 samples for a KDE")
        self.samples = samples
        n, d = samples.shape
        scale = samples.std(dim=0, unbiased=True).clamp_min(torch.finfo(samples.dtype).eps)
        factor = n ** (-1.0 / (d + 4)) if bandwidth is None else float(bandwidth)
        self.bandwidth = scale * factor
        self._log_norm = -0.5 * d * torch.log(torch.tensor(2 * torch.pi, dtype=samples.dtype)) - torch.log(
            self.bandwidth
        ).sum()

    def log_density(self, points: Tensor) -> Tensor:
        points = points if points.ndim == 2 else points.reshape(points.shape[0], -1)
        z = (points.unsqueeze(1) - self.samples.unsqueeze(0)) / self.bandwidth  # (q, n, d)
        log_kernels = self._log_norm - 0.5 * (z**2).sum(dim=-1)  # (q, n)
        return torch.logsumexp(log_kernels, dim=-1) - torch.log(
            torch.tensor(float(self.samples.shape[0]), dtype=points.dtype)
        )

    def __call__(self, points: Tensor) -> Tensor:
        return self.log_density(points).exp()


class ConditionalDistribution:
    r"""The conditional law :math:`Y \mid X = x_i` for a batch of conditioning values.

    Obtained from :meth:`NCPOperator.condition`. Indexing follows the batch of
    conditioning values: every method returns one row per :math:`x_i`.

    Attributes:
        weights: ``(n_x, m)`` masses on the reference atoms.
        atoms: ``(m, y_dim)`` reference sample standing in for :math:`\pi_Y`.
    """

    def __init__(
        self,
        weights: Tensor,
        atoms: Tensor,
        operator: Optional[Any] = None,
        x: Optional[Tensor] = None,
        rank: Optional[int] = None,
    ):
        if weights.ndim != 2:
            raise ValueError(f"weights must be 2D (n_x, m), got {tuple(weights.shape)}")
        if atoms.ndim != 2:
            raise ValueError(f"atoms must be 2D (m, y_dim), got {tuple(atoms.shape)}")
        if weights.shape[1] != atoms.shape[0]:
            raise ValueError(f"weights has {weights.shape[1]} columns but there are {atoms.shape[0]} atoms")
        self.weights = weights
        self.atoms = atoms
        self._operator = operator
        self._x = x
        self._rank = rank

    # ------------------------------------------------------------------ basics

    def __len__(self) -> int:
        return self.weights.shape[0]

    @property
    def n_atoms(self) -> int:
        return self.atoms.shape[0]

    @property
    def y_dim(self) -> int:
        return self.atoms.shape[1]

    def _values(self, observable: Observable) -> Tensor:
        """Reduce the atoms to the scalar quantity an observable asks for."""
        if observable is None:
            if self.y_dim != 1:
                raise ValueError(
                    f"Y is {self.y_dim}-dimensional, so a scalar observable is required: pass a "
                    "coordinate index or a callable mapping (m, y_dim) -> (m,)"
                )
            return self.atoms[:, 0]
        if isinstance(observable, int):
            if not -self.y_dim <= observable < self.y_dim:
                raise ValueError(f"coordinate {observable} out of range for y_dim={self.y_dim}")
            return self.atoms[:, observable]
        out = observable(self.atoms)
        out = out.reshape(out.shape[0], -1)
        if out.shape[1] != 1:
            raise ValueError(f"observable must be scalar-valued, got {out.shape[1]} outputs")
        return out[:, 0]

    # ------------------------------------------------------------------ moments

    def expectation(self, fn: Callable[[Tensor], Tensor]) -> Tensor:
        r""":math:`\mathbb{E}[f(Y) \mid X = x]` for any vector-valued ``fn``.

        ``fn`` maps the atoms ``(m, y_dim)`` to ``(m,)`` or ``(m, k)``; the
        result is ``(n_x,)`` or ``(n_x, k)``.
        """
        vals = fn(self.atoms)
        if vals.shape[0] != self.n_atoms:
            raise ValueError(f"observable returned {vals.shape[0]} rows for {self.n_atoms} atoms")
        if vals.ndim == 1:
            return self.weights @ vals
        return self.weights @ vals.reshape(self.n_atoms, -1)

    def mean(self) -> Tensor:
        r""":math:`\mathbb{E}[Y \mid X = x]`, shape ``(n_x, y_dim)``."""
        return self.weights @ self.atoms

    def covariance(self) -> Tensor:
        r""":math:`\operatorname{Cov}[Y \mid X = x]`, shape ``(n_x, y_dim, y_dim)``."""
        mu = self.mean()  # (n_x, y_dim)
        second = torch.einsum("nm,mi,mj->nij", self.weights, self.atoms, self.atoms)
        return second - torch.einsum("ni,nj->nij", mu, mu)

    def variance(self) -> Tensor:
        r"""Per-coordinate conditional variance, shape ``(n_x, y_dim)``.

        Clamped at zero: the plug-in second moment of a signed weight vector
        can dip below the squared mean.
        """
        mu = self.mean()
        second = self.weights @ (self.atoms**2)
        return (second - mu**2).clamp_min(0.0)

    def std(self) -> Tensor:
        return self.variance().sqrt()

    def moment(self, order: int, observable: Observable = None, central: bool = False) -> Tensor:
        """Conditional moment of a scalar observable."""
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        vals = self._values(observable)
        if not central:
            return self.weights @ vals**order
        mu = self.weights @ vals  # (n_x,)
        return (self.weights * (vals.unsqueeze(0) - mu.unsqueeze(1)) ** order).sum(dim=-1)

    # --------------------------------------------------------------- cdf/quantile

    def _sorted_cdf(self, observable: Observable) -> Tuple[Tensor, Tensor]:
        """Atom values in increasing order plus the inclusive cumulative masses."""
        vals = self._values(observable)
        order = torch.argsort(vals)
        return vals[order], self.weights[:, order].cumsum(dim=-1)

    def cdf(
        self,
        observable: Observable = None,
        grid: Optional[Tensor] = None,
        monotone: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        r"""Conditional CDF :math:`\widehat{F}(t \mid x) = \mathbb{P}(f(Y) \le t \mid X = x)`.

        Args:
            observable: scalar reduction of ``Y`` (default: the single coordinate).
            grid: evaluation points. Defaults to the sorted atom values, on
                which the estimate is a step function.
            monotone: project onto monotone functions and clip to ``[0, 1]``.
                Only needed when the operator was conditioned with
                ``clip=False``, which allows negative masses.

        Returns:
            ``(points, cdf)`` with shapes ``(q,)`` and ``(n_x, q)``.
        """
        sorted_vals, cum = self._sorted_cdf(observable)
        if grid is None:
            points, out = sorted_vals, cum
        else:
            grid = grid.reshape(-1).to(dtype=sorted_vals.dtype, device=sorted_vals.device)
            points = grid
            # Number of atoms with value <= t, hence which cumulative entry to read.
            idx = torch.searchsorted(sorted_vals.contiguous(), grid.contiguous(), right=True) - 1
            padded = torch.cat([torch.zeros_like(cum[:, :1]), cum], dim=-1)
            out = padded.gather(1, (idx + 1).clamp_min(0).expand(cum.shape[0], -1))
        if monotone:
            out = isotonic_regression(out).clamp(0.0, 1.0)
        return points, out

    def quantile(self, levels: Union[float, Sequence[float], Tensor], observable: Observable = None) -> Tensor:
        r"""Conditional quantiles: the smallest atom value with :math:`\widehat{F} \ge p`.

        Returns shape ``(n_x, n_levels)``, or ``(n_x,)`` for a scalar level.
        """
        scalar = isinstance(levels, float) or (isinstance(levels, Tensor) and levels.ndim == 0)
        lv = torch.as_tensor(levels, dtype=self.weights.dtype, device=self.weights.device).reshape(-1)
        if bool(((lv < 0) | (lv > 1)).any()):
            raise ValueError("quantile levels must lie in [0, 1]")
        sorted_vals, cum = self._sorted_cdf(observable)
        # cum is non-decreasing (weights are non-negative after clipping), so a
        # batched binary search gives the quantile index directly.
        cum = cum.contiguous()
        target = lv.expand(cum.shape[0], -1).contiguous()
        idx = torch.searchsorted(cum, target, right=False).clamp_max(sorted_vals.shape[0] - 1)
        out = sorted_vals[idx]
        return out[:, 0] if scalar else out

    def median(self, observable: Observable = None) -> Tensor:
        return self.quantile(0.5, observable=observable)

    # ---------------------------------------------------------------- regions

    def interval(self, alpha: float = 0.05, observable: Observable = None) -> Tensor:
        r"""Shortest conditional confidence interval at level :math:`1 - \alpha`.

        Sweeps every left endpoint and binary-searches the matching right
        endpoint, returning the shortest :math:`[a, b]` with
        :math:`\widehat{F}(b \mid x) - \widehat{F}(a^- \mid x) \ge 1 - \alpha`.
        Both endpoints are atom values, so the interval is always inside the
        observed support.

        Returns shape ``(n_x, 2)``.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        sorted_vals, cum = self._sorted_cdf(observable)
        n_x, m = cum.shape
        # Mass strictly below each atom, i.e. F(a^-).
        below = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], dim=-1)
        target = (below + (1.0 - alpha)).contiguous()
        right = torch.searchsorted(cum.contiguous(), target, right=False)  # (n_x, m)

        feasible = right < m
        right_clamped = right.clamp_max(m - 1)
        lengths = sorted_vals[right_clamped] - sorted_vals.unsqueeze(0)
        lengths = torch.where(feasible, lengths, torch.full_like(lengths, float("inf")))

        best_left = lengths.argmin(dim=-1)
        rows = torch.arange(n_x, device=cum.device)
        best_right = right_clamped[rows, best_left]
        # No left endpoint reaches the target (possible only with clipped mass):
        # fall back to the full observed support.
        none_feasible = ~feasible.any(dim=-1)
        lo = torch.where(none_feasible, sorted_vals[0].expand(n_x), sorted_vals[best_left])
        hi = torch.where(none_feasible, sorted_vals[-1].expand(n_x), sorted_vals[best_right])
        return torch.stack([lo, hi], dim=-1)

    def density(self, grid: Tensor, marginal: Callable[[Tensor], Tensor]) -> Tensor:
        r"""Conditional density :math:`\widehat{p}(y \mid x) = \hat\pi_Y(y)\,(1 + \hat{r}(x, y))`.

        Args:
            grid: ``(q, y_dim)`` evaluation points (``(q,)`` accepted when ``y_dim == 1``).
            marginal: density of :math:`\pi_Y` at those points, e.g. a
                :class:`GaussianKDE` fitted on the reference sample, or the
                exact marginal when it is known.

        Returns shape ``(n_x, q)``. Requires the distribution to come from
        :meth:`NCPOperator.condition`, which retains the operator and ``x``.
        """
        if self._operator is None or self._x is None:
            raise RuntimeError(
                "density() needs the operator and conditioning values; build this object "
                "via NCPOperator.condition(x) rather than constructing it directly"
            )
        pts = grid if grid.ndim == 2 else grid.reshape(-1, 1)
        ratio = 1.0 + self._operator.deflated_ratio(self._x, pts, rank=self._rank)
        return ratio.clamp_min(0.0) * marginal(pts).reshape(1, -1)

    def highest_density_region(
        self,
        grid: Tensor,
        marginal: Callable[[Tensor], Tensor],
        alpha: float = 0.05,
    ) -> Tuple[Tensor, Tensor]:
        r"""Highest-density region at level :math:`1 - \alpha` on a 1-D grid.

        Adds grid cells in decreasing density order until they hold
        :math:`1 - \alpha` of the mass, giving the smallest-volume region --
        which, unlike :meth:`interval`, can be a union of disjoint pieces and
        so stays tight on multimodal conditionals.

        Returns ``(mask, level)``: a ``(n_x, q)`` boolean membership mask and
        the ``(n_x,)`` density thresholds defining the region.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        flat = grid.reshape(-1)
        if flat.numel() < 2:
            raise ValueError("need at least 2 grid points")
        order = torch.argsort(flat)
        pts = flat[order]
        dens = self.density(pts, marginal)  # (n_x, q)

        # Midpoint cell widths, so the mass integrates with the trapezoid rule.
        edges = torch.cat([pts[:1], 0.5 * (pts[1:] + pts[:-1]), pts[-1:]])
        cell = (edges[1:] - edges[:-1]).clamp_min(0.0)
        mass = dens * cell
        mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(mass.dtype).eps)

        desc = torch.argsort(dens, dim=-1, descending=True)
        cum = mass.gather(1, desc).cumsum(dim=-1)
        # First position where the accumulated mass reaches the target.
        reached = (cum >= 1.0 - alpha).float().argmax(dim=-1)
        never = ~(cum >= 1.0 - alpha).any(dim=-1)
        reached = torch.where(never, torch.full_like(reached, cum.shape[-1] - 1), reached)
        level = dens.gather(1, desc).gather(1, reached.unsqueeze(-1)).squeeze(-1)

        mask_sorted = dens >= level.unsqueeze(-1)
        mask = torch.empty_like(mask_sorted)
        mask[:, order] = mask_sorted
        return mask, level

    # ---------------------------------------------------------------- sampling

    def sample(self, n_samples: int, generator: Optional[torch.Generator] = None) -> Tensor:
        """Draw from the conditional law by resampling the reference atoms.

        Returns shape ``(n_x, n_samples, y_dim)``.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        idx = torch.multinomial(self.weights, n_samples, replacement=True, generator=generator)
        return self.atoms[idx]

    def __repr__(self) -> str:
        return f"ConditionalDistribution(n_x={len(self)}, n_atoms={self.n_atoms}, y_dim={self.y_dim})"
