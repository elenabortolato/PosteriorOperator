r"""Training objectives for Neural Conditional Probability.

Setting
-------
Let :math:`(X, Y) \sim \pi_{XY}` with marginals :math:`\pi_X, \pi_Y`, and let

.. math:: r(x, y) = \frac{p(x, y)}{\pi_X(x)\,\pi_Y(y)} - 1
          = \frac{p(y \mid x)}{\pi_Y(y)} - 1

be the *deflated* density ratio -- the kernel of the conditional expectation
operator :math:`E: L^2(\pi_Y) \to L^2(\pi_X)`, :math:`(Eg)(x) =
\mathbb{E}[g(Y) \mid X = x]`, after subtracting its trivial component
:math:`\mathbb{1} \otimes \mathbb{1}`. NCP models it with a bilinear form in two
learned embeddings :math:`u_\theta: \mathcal{X} \to \mathbb{R}^d` and
:math:`v_\phi: \mathcal{Y} \to \mathbb{R}^d`, weighted by learned singular
values :math:`s \in (0, 1]^d`:

.. math:: h(x, y) = \sum_{k=1}^{d} s_k\, u_k(x)\, v_k(y).

Objective
---------
All estimators here target the *same* population objective, the squared
:math:`L^2(\pi_X \otimes \pi_Y)` distance to :math:`r` up to a constant:

.. math::
    \mathcal{L}(\theta, \phi, s)
      = \mathbb{E}_{\pi_X \otimes \pi_Y}\!\big[h^2\big]
        - 2\Big(\mathbb{E}_{\pi_{XY}}[h] - \mathbb{E}_{\pi_X \otimes \pi_Y}[h]\Big)
      = \lVert h - r \rVert^2_{L^2(\pi_X \otimes \pi_Y)} - \lVert r \rVert^2 .

The identity uses :math:`\mathbb{E}_{\pi_X \otimes \pi_Y}[h\,r] =
\mathbb{E}_{\pi_{XY}}[h] - \mathbb{E}_{\pi_X \otimes \pi_Y}[h]`. Crucially the
objective needs *no* conditioning value and *no* normalising constant: training
is a single unconditional pass over the joint sample, which is what lets
inference re-condition on new :math:`x` without retraining.

Regularisation
--------------
:math:`\mathcal{L}` is invariant to any reparametrisation :math:`u \mapsto
A^{-\top} u`, :math:`v \mapsto A v`, so the embeddings are only identified up
to a linear map. The penalty pushes the second-moment matrices
:math:`\mathbb{E}[uu^\top]` and :math:`\mathbb{E}[vv^\top]` towards the
identity, i.e. towards orthonormal features, which both conditions the
optimisation and makes ``s`` interpretable as singular values.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

__all__ = [
    "NCPLoss",
    "ustat_objective",
    "split_objective",
    "orthonormality_penalty",
    "log_fro_penalty",
    "centering_penalty",
]


def _check_shapes(u: Tensor, v: Tensor, s: Tensor) -> None:
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError(f"embeddings must be 2D (batch, latent), got {tuple(u.shape)} and {tuple(v.shape)}")
    if u.shape != v.shape:
        raise ValueError(f"embeddings must have the same shape, got {tuple(u.shape)} and {tuple(v.shape)}")
    if s.shape != (u.shape[1],):
        raise ValueError(f"expected {u.shape[1]} singular values, got {tuple(s.shape)}")
    if u.shape[0] < 2:
        raise ValueError("at least 2 samples are required to form a product-measure pair")


def ustat_objective(u: Tensor, v: Tensor, s: Tensor) -> Tensor:
    r"""Unbiased U-statistic estimate of :math:`\mathcal{L}`.

    All :math:`n(n-1)` off-diagonal pairs :math:`(x_i, y_j)`, :math:`i \neq j`,
    serve as independent draws from :math:`\pi_X \otimes \pi_Y`, while the
    diagonal carries the joint term. Evaluated through Gram-free identities
    (:math:`\sum_{ij} H_{ij}^2 = \langle U_s^\top U_s, V^\top V\rangle` for
    :math:`H = U_s V^\top`), so the cost is :math:`O(nd^2)` and no
    :math:`n \times n` matrix is ever materialised.
    """
    _check_shapes(u, v, s)
    n = u.shape[0]
    us = u * s

    # Diagonal of H: the joint-measure evaluations h(x_i, y_i).
    h_diag = (us * v).sum(dim=-1)
    # Sums over all pairs, then peel off the diagonal to get the i != j sums.
    sum_sq_all = ((us.T @ us) * (v.T @ v)).sum()
    sum_all = us.sum(dim=0) @ v.sum(dim=0)
    n_pairs = n * (n - 1)
    prod_h2 = (sum_sq_all - (h_diag**2).sum()) / n_pairs
    prod_h = (sum_all - h_diag.sum()) / n_pairs

    return prod_h2 - 2.0 * (h_diag.mean() - prod_h)


def split_objective(u: Tensor, v: Tensor, s: Tensor, generator: torch.Generator | None = None) -> Tensor:
    r"""Unbiased two-split estimate of :math:`\mathcal{L}`, at :math:`O(nd)` cost.

    The batch is shuffled and halved into independent copies
    :math:`(X_1, Y_1)` and :math:`(X_2, Y_2)`. Then

    .. math::
        \tfrac12\big(h(X_1, Y_2)^2 + h(X_2, Y_1)^2\big)
        - \big\langle s \odot (u(X_1) - u(X_2)),\; v(Y_1) - v(Y_2) \big\rangle

    is unbiased for :math:`\mathcal{L}`: the first bracket estimates
    :math:`\mathbb{E}_{\pi_X \otimes \pi_Y}[h^2]` because :math:`X_1 \perp Y_2`,
    and the difference term expands to exactly
    :math:`2(\mathbb{E}_{\pi_{XY}}[h] - \mathbb{E}_{\pi_X \otimes \pi_Y}[h])`.
    Cheaper than :func:`ustat_objective` but higher variance, since it uses
    :math:`n/2` pairs instead of :math:`n(n-1)`.
    """
    _check_shapes(u, v, s)
    n = u.shape[0]
    half = n // 2
    if half < 1:
        raise ValueError(f"need at least 2 samples to split, got {n}")
    perm = torch.randperm(n, generator=generator, device=u.device)
    i1, i2 = perm[:half], perm[half : 2 * half]
    u1, u2, v1, v2 = u[i1], u[i2], v[i1], v[i2]

    cross_1 = ((u1 * s) * v2).sum(dim=-1)  # h(X1, Y2)
    cross_2 = ((u2 * s) * v1).sum(dim=-1)  # h(X2, Y1)
    deflated = (((u1 - u2) * s) * (v1 - v2)).sum(dim=-1)

    return (0.5 * cross_1**2 + 0.5 * cross_2**2 - deflated).mean()


def orthonormality_penalty(z: Tensor) -> Tensor:
    r"""Unbiased estimate of :math:`\lVert \mathbb{E}[zz^\top] - I_d \rVert_F^2`.

    Expanding the Frobenius norm gives
    :math:`\lVert A \rVert_F^2 - 2\operatorname{tr}A + d` with
    :math:`A = \mathbb{E}[zz^\top]`. The quadratic term is estimated by the
    off-diagonal pairs, :math:`\mathbb{E}[(z_i \cdot z_j)^2] = \lVert A
    \rVert_F^2` for :math:`i \neq j`, which keeps the estimate unbiased (the
    plug-in :math:`\lVert \hat{A} \rVert_F^2` is biased upwards by
    :math:`O(d/n)` and would silently shrink the embeddings on small batches).
    """
    if z.ndim != 2:
        raise ValueError(f"expected a 2D (batch, latent) tensor, got {tuple(z.shape)}")
    n, d = z.shape
    if n < 2:
        raise ValueError("at least 2 samples are required")
    gram = z.T @ z  # (d, d), equals sum_i z_i z_i^T
    sq_norms = (z * z).sum(dim=-1)  # (n,)
    sum_sq_off = (gram * gram).sum() - (sq_norms**2).sum()  # sum_{i != j} (z_i . z_j)^2
    return sum_sq_off / (n * (n - 1)) - 2.0 * sq_norms.mean() + d


def centering_penalty(z: Tensor) -> Tensor:
    r"""Squared norm of the empirical mean, :math:`\lVert \frac1n \sum_i z_i \rVert^2`.

    The singular functions of the *deflated* operator are mean zero, so this
    pushes the embeddings towards that constraint during training. It is
    optional: the whitening step in
    :meth:`~posterior_operator.operator.NCPOperator.fit_statistics` subtracts
    the empirical means exactly afterwards. Adding it with weight 2 reproduces
    the centering terms of the regulariser written in Bortolato (2026).
    """
    if z.ndim != 2:
        raise ValueError(f"expected a 2D (batch, latent) tensor, got {tuple(z.shape)}")
    return (z.mean(dim=0) ** 2).sum()


def log_fro_penalty(z: Tensor) -> Tensor:
    r"""Metric-deformation penalty :math:`\operatorname{mean}(\lambda^2 - \lambda - \log\lambda)`.

    Taken over the eigenvalues of the second-moment matrix. It is minimised at
    :math:`\lambda = 1` like :func:`orthonormality_penalty`, but diverges as
    :math:`\lambda \to 0^+`, so it actively prevents the embedding from
    collapsing onto a lower-dimensional subspace.
    """
    if z.ndim != 2:
        raise ValueError(f"expected a 2D (batch, latent) tensor, got {tuple(z.shape)}")
    n = z.shape[0]
    cov = (z.T @ z) / n
    eps = torch.finfo(cov.dtype).eps * cov.shape[0]
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(eps)
    return (eigvals * (eigvals - 1.0) - torch.log(eigvals)).mean()


class NCPLoss:
    """The NCP training loss: a fit term plus an identifiability penalty.

    Args:
        mode: ``"ustat"`` (default) for the unbiased all-pairs U-statistic, or
            ``"split"`` for the cheaper two-half estimator.
        gamma: weight of the penalty. ``0`` disables it, leaving the embeddings
            identified only up to a linear map -- usually still trainable, but
            the whitening step then has to undo a badly scaled basis.
        penalty: ``"orthonormality"`` (default) or ``"log_fro"``.
        center_weight: weight on :func:`centering_penalty`, relative to
            ``gamma``. Zero by default, since the whitening step centers
            exactly after training; set it to ``2.0`` to match the regulariser
            written in Bortolato (2026).
        generator: optional RNG for the ``"split"`` shuffle, for reproducibility.
    """

    MODES = ("ustat", "split")
    PENALTIES = ("orthonormality", "log_fro")

    def __init__(
        self,
        mode: str = "ustat",
        gamma: float = 1e-3,
        penalty: str = "orthonormality",
        center_weight: float = 0.0,
        generator: torch.Generator | None = None,
    ):
        if mode not in self.MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {self.MODES}")
        if penalty not in self.PENALTIES:
            raise ValueError(f"unknown penalty {penalty!r}, expected one of {self.PENALTIES}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if center_weight < 0:
            raise ValueError(f"center_weight must be non-negative, got {center_weight}")
        self.mode = mode
        self.gamma = gamma
        self.penalty = penalty
        self.center_weight = center_weight
        self.generator = generator

    def fit_term(self, u: Tensor, v: Tensor, s: Tensor) -> Tensor:
        if self.mode == "ustat":
            return ustat_objective(u, v, s)
        return split_objective(u, v, s, generator=self.generator)

    def penalty_term(self, u: Tensor, v: Tensor) -> Tensor:
        fn = orthonormality_penalty if self.penalty == "orthonormality" else log_fro_penalty
        total = fn(u) + fn(v)
        if self.center_weight > 0:
            total = total + self.center_weight * (centering_penalty(u) + centering_penalty(v))
        return total

    def parts(self, u: Tensor, v: Tensor, s: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(fit_term, penalty_term)`` separately, for diagnostics."""
        fit = self.fit_term(u, v, s)
        pen = self.penalty_term(u, v) if self.gamma > 0 else torch.zeros((), dtype=fit.dtype, device=fit.device)
        return fit, pen

    def __call__(self, u: Tensor, v: Tensor, s: Tensor) -> Tensor:
        fit, pen = self.parts(u, v, s)
        return fit + self.gamma * pen
