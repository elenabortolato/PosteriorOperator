r"""Simulators for likelihood-free posterior-functional inference.

Each simulator draws from the prior-predictive law
:math:`\rho(d\theta, dy) = \pi(d\theta)\, p(dy \mid \theta)`, which is the only
access the estimator needs. Where a tractable likelihood happens to exist it is
exposed as well -- not because the method uses it, but so that estimated
posterior functionals can be checked against the truth.

The three models cover the regimes the theory singles out:

* :class:`GaussianLinear` -- jointly Gaussian, so the conditional expectation
  operator *is* classical CCA and the low-rank formula is exact at
  :math:`d = \operatorname{rank}(\Sigma_{\Theta Y})`.
* :class:`MA2` -- the standard likelihood-free benchmark, with a non-Gaussian
  posterior on the invertibility triangle and an exactly computable Gaussian
  likelihood for reference.
* :class:`SumIdentified` -- only :math:`\theta_1 + \theta_2` is identified, so
  the operator's leading singular function should recover the identified
  direction and the orthogonal direction should keep its prior.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

__all__ = ["Simulator", "GaussianLinear", "MA2", "SumIdentified"]


class Simulator:
    """Base class: a prior and a forward model that can only be sampled."""

    theta_dim: int
    data_dim: int

    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype

    # --- the only interface the estimator needs --------------------------
    def sample_prior(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        raise NotImplementedError

    def simulate(self, theta: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        raise NotImplementedError

    def sample_joint(self, n: int, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor]:
        r"""Draw :math:`(\theta_i, y_i) \sim \rho`, the prior-predictive law."""
        theta = self.sample_prior(n, generator)
        return theta, self.simulate(theta, generator)

    # --- reference quantities, for validation only ------------------------
    def log_likelihood(self, theta: Tensor, y_obs: Tensor) -> Tensor:
        """``(n,)`` log-likelihood of one observation under each row of ``theta``."""
        raise NotImplementedError

    def log_prior(self, theta: Tensor) -> Tensor:
        """``(n,)`` log prior density, up to a constant."""
        raise NotImplementedError


class GaussianLinear(Simulator):
    r"""``Theta ~ N(0, Sigma_theta)``, ``Y = A Theta + eps``, ``eps ~ N(0, sigma^2 I)``.

    Everything is available in closed form: the posterior is Gaussian with mean
    :math:`\Sigma_{\Theta Y}\Sigma_{YY}^{-1} y_0`, and the singular system of the
    deflated operator is the canonical correlation analysis of
    :math:`(\Theta, Y)` -- canonical correlations as singular values, canonical
    variates as singular functions.
    """

    def __init__(
        self,
        theta_dim: int = 3,
        data_dim: int = 5,
        noise: float = 0.5,
        design: Optional[Tensor] = None,
        prior_cov: Optional[Tensor] = None,
        seed: int = 0,
        **kw,
    ):
        super().__init__(**kw)
        self.theta_dim, self.data_dim = int(theta_dim), int(data_dim)
        self.noise = float(noise)
        g = torch.Generator().manual_seed(seed)
        if design is None:
            design = torch.randn(self.data_dim, self.theta_dim, generator=g, dtype=self.dtype)
            design = design / math.sqrt(self.theta_dim)
        self.design = torch.as_tensor(design, dtype=self.dtype)
        if self.design.shape != (self.data_dim, self.theta_dim):
            raise ValueError(f"design must be ({self.data_dim}, {self.theta_dim}), got {tuple(self.design.shape)}")
        self.prior_cov = (
            torch.eye(self.theta_dim, dtype=self.dtype)
            if prior_cov is None
            else torch.as_tensor(prior_cov, dtype=self.dtype)
        )
        self._prior_chol = torch.linalg.cholesky(self.prior_cov)

    # --- simulation --------------------------------------------------------
    def sample_prior(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        z = torch.randn(n, self.theta_dim, generator=generator, dtype=self.dtype)
        return z @ self._prior_chol.T

    def simulate(self, theta: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        n = theta.shape[0]
        noise = torch.randn(n, self.data_dim, generator=generator, dtype=self.dtype)
        return theta @ self.design.T + self.noise * noise

    # --- closed-form reference --------------------------------------------
    @property
    def cov_yy(self) -> Tensor:
        return self.design @ self.prior_cov @ self.design.T + self.noise**2 * torch.eye(
            self.data_dim, dtype=self.dtype
        )

    @property
    def cov_theta_y(self) -> Tensor:
        return self.prior_cov @ self.design.T

    def posterior_mean(self, y_obs: Tensor) -> Tensor:
        r""":math:`\Sigma_{\Theta Y}\Sigma_{YY}^{-1} y_0`, shape ``(n, theta_dim)``."""
        y_obs = torch.as_tensor(y_obs, dtype=self.dtype).reshape(-1, self.data_dim)
        # Sigma_yy^{-1} Sigma_{Y Theta} is (q, p), so right-multiplying the batch
        # of observations gives one posterior mean per row.
        return y_obs @ torch.linalg.solve(self.cov_yy, self.cov_theta_y.T)

    def posterior_cov(self) -> Tensor:
        """Posterior covariance, the same for every observation in a linear model."""
        return self.prior_cov - self.cov_theta_y @ torch.linalg.solve(self.cov_yy, self.cov_theta_y.T)

    def canonical_correlations(self) -> Tensor:
        r"""Canonical correlations :math:`\rho_1 \ge \dots \ge \rho_{r^*}` of :math:`(\Theta, Y)`.

        These are the singular values of the operator **restricted to linear
        functionals** of :math:`\theta`. They are *not* the whole spectrum of
        the deflated operator on :math:`L^2_\pi(\Theta)` -- see
        :meth:`exact_spectrum`.
        """
        return torch.linalg.svdvals(self._whitened_cross())

    def exact_spectrum(self, max_order: int = 30) -> Tuple[Tensor, list]:
        r"""The exact :math:`L^2` spectrum of the deflated operator.

        In canonical coordinates a jointly Gaussian pair factorises into
        independent scalar pairs :math:`(v_i, u_i)` with correlation
        :math:`\rho_i`, so :math:`L^2_\pi(\Theta)` is a tensor product of
        one-dimensional Hermite spaces and the operator is diagonal in the
        Hermite tensor basis:

        .. math:: \mathbb{E}\Big[\prod_i H_{a_i}(v_i) \,\Big|\, u\Big]
                  = \Big(\prod_i \rho_i^{a_i}\Big) \prod_i H_{a_i}(u_i),

        giving one singular value :math:`\sigma_a = \prod_i \rho_i^{a_i}` for
        every multi-index :math:`a \neq 0`. The operator therefore has
        *infinitely many* non-zero singular values whenever any
        :math:`\rho_i > 0`, however small :math:`\operatorname{rank}
        (\Sigma_{\Theta Y})` is, and the canonical correlations are only the
        subset with :math:`|a| = 1`.

        Args:
            max_order: largest total order :math:`|a|` enumerated.

        Returns:
            ``(values, multi_indices)`` sorted by decreasing singular value.
        """
        import itertools

        rho = [r for r in self.canonical_correlations().tolist() if r > 0]
        items = []
        for a in itertools.product(range(max_order + 1), repeat=len(rho)):
            order = sum(a)
            if order == 0 or order > max_order:
                continue
            items.append((math.prod(r**ai for r, ai in zip(rho, a)), a))
        items.sort(key=lambda t: -t[0])
        return torch.tensor([v for v, _ in items], dtype=self.dtype), [a for _, a in items]

    def linear_direction_ranks(self, max_order: int = 30) -> list:
        r"""1-based spectral position of each linear canonical direction.

        The rank-:math:`d` truncation of the low-rank formula recovers
        :math:`\mathbb{E}[\Theta \mid Y = y]` exactly only once :math:`d`
        reaches the largest of these positions: below that, a *nonlinear*
        Hermite direction of a strongly correlated canonical pair outranks a
        weakly correlated *linear* one, and the linear one is dropped.
        """
        _, indices = self.exact_spectrum(max_order)
        n_rho = len(indices[0])
        ranks = []
        for i in range(n_rho):
            target = tuple(1 if j == i else 0 for j in range(n_rho))
            ranks.append(indices.index(target) + 1)
        return ranks

    def truncation_error_posterior_mean(self, rank: int, max_order: int = 30) -> float:
        r"""Exact RMS truncation error of the rank-``d`` posterior mean.

        Only the linear canonical directions contribute to
        :math:`\mathbb{E}[\Theta \mid Y]`; each one that the top-``rank``
        truncation drops contributes its own :math:`\rho_i^2` (the canonical
        variates are orthonormal), so the error is available in closed form and
        is exactly zero once every linear direction is retained.
        """
        rho = self.canonical_correlations()
        ranks = self.linear_direction_ranks(max_order)
        dropped = torch.tensor(
            [float(rho[i]) ** 2 for i, position in enumerate(ranks) if position > rank], dtype=self.dtype
        )
        if dropped.numel() == 0:
            return 0.0
        return float(dropped.sum().sqrt())

    def _whitened_cross(self) -> Tensor:
        inv_sqrt_theta = _inv_sqrt(self.prior_cov.double())
        inv_sqrt_y = _inv_sqrt(self.cov_yy.double())
        return inv_sqrt_theta @ self.cov_theta_y.double() @ inv_sqrt_y

    def log_likelihood(self, theta: Tensor, y_obs: Tensor) -> Tensor:
        theta = torch.as_tensor(theta, dtype=self.dtype).reshape(-1, self.theta_dim)
        y_obs = torch.as_tensor(y_obs, dtype=self.dtype).reshape(self.data_dim)
        resid = y_obs.unsqueeze(0) - theta @ self.design.T
        return -0.5 * (resid**2).sum(-1) / self.noise**2 - self.data_dim * math.log(
            self.noise * math.sqrt(2 * math.pi)
        )

    def log_prior(self, theta: Tensor) -> Tensor:
        theta = torch.as_tensor(theta, dtype=self.dtype).reshape(-1, self.theta_dim)
        solved = torch.linalg.solve(self.prior_cov, theta.T).T
        return -0.5 * (theta * solved).sum(-1)


class MA2(Simulator):
    r"""MA(2): ``y_t = e_t + theta_1 e_{t-1} + theta_2 e_{t-2}``, ``e_t ~ N(0, 1)``.

    The standard likelihood-free benchmark. The prior is uniform on the
    invertibility triangle :math:`\{|\theta_1| < 2,\ \theta_1 + \theta_2 > -1,\
    \theta_1 - \theta_2 < 1\}`, which makes the posterior distinctly
    non-Gaussian and bounded.

    The series is Gaussian with a banded Toeplitz covariance, so the exact
    likelihood *is* computable here even though a simulation-based method would
    not use it. :meth:`grid_posterior` exploits that to produce reference
    posteriors by quadrature.
    """

    theta_dim = 2

    def __init__(self, n_timesteps: int = 50, summaries: bool = True, n_lags: int = 4, **kw):
        super().__init__(**kw)
        self.n_timesteps = int(n_timesteps)
        self.summaries = bool(summaries)
        self.n_lags = int(n_lags)
        self.data_dim = self.n_lags + 1 if summaries else self.n_timesteps

    # --- simulation --------------------------------------------------------
    def sample_prior(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        """Rejection-sample the invertibility triangle from its bounding box."""
        kept = []
        total = 0
        while total < n:
            batch = torch.rand(max(n, 256), 2, generator=generator, dtype=self.dtype)
            cand = torch.stack([4.0 * batch[:, 0] - 2.0, 2.0 * batch[:, 1] - 1.0], dim=-1)
            ok = cand[self.in_support(cand)]
            kept.append(ok)
            total += ok.shape[0]
        return torch.cat(kept)[:n]

    @staticmethod
    def in_support(theta: Tensor) -> Tensor:
        t1, t2 = theta[:, 0], theta[:, 1]
        return (t1.abs() < 2.0) & (t1 + t2 > -1.0) & (t1 - t2 < 1.0)

    def simulate(self, theta: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        n, t = theta.shape[0], self.n_timesteps
        e = torch.randn(n, t + 2, generator=generator, dtype=self.dtype)
        series = e[:, 2:] + theta[:, :1] * e[:, 1:-1] + theta[:, 1:2] * e[:, :-2]
        return self.summarize(series) if self.summaries else series

    def summarize(self, series: Tensor) -> Tensor:
        """Sample variance plus autocovariances at lags 1..``n_lags``.

        The classical ABC summaries for this model. They are not sufficient, but
        they make the point that the method is indifferent to the choice: pass
        ``summaries=False`` to feed the raw series instead.
        """
        centred = series - series.mean(dim=-1, keepdim=True)
        t = centred.shape[-1]
        feats = [(centred**2).mean(dim=-1, keepdim=True)]
        for lag in range(1, self.n_lags + 1):
            feats.append((centred[:, lag:] * centred[:, : t - lag]).mean(dim=-1, keepdim=True))
        return torch.cat(feats, dim=-1)

    # --- exact reference ---------------------------------------------------
    def _covariance(self, theta: Tensor) -> Tensor:
        """Banded Toeplitz covariance of the series, shape ``(n, T, T)``."""
        t1, t2 = theta[:, 0], theta[:, 1]
        gamma = torch.stack([1 + t1**2 + t2**2, t1 + t1 * t2, t2], dim=-1)  # lags 0,1,2
        t = self.n_timesteps
        idx = (torch.arange(t).unsqueeze(0) - torch.arange(t).unsqueeze(1)).abs()
        band = torch.zeros(theta.shape[0], t, t, dtype=theta.dtype)
        for lag in range(3):
            band += (idx == lag).to(theta.dtype).unsqueeze(0) * gamma[:, lag].reshape(-1, 1, 1)
        return band

    def log_likelihood(self, theta: Tensor, y_obs: Tensor, chunk_size: int = 2048) -> Tensor:
        """Exact Gaussian log-likelihood of a raw series, chunked over ``theta``."""
        if self.summaries:
            raise NotImplementedError(
                "the exact likelihood is for the raw series; build the simulator with "
                "summaries=False, or pass the raw series to grid_posterior"
            )
        theta = torch.as_tensor(theta, dtype=torch.float64).reshape(-1, 2)
        y = torch.as_tensor(y_obs, dtype=torch.float64).reshape(self.n_timesteps)
        out = torch.full((theta.shape[0],), -float("inf"), dtype=torch.float64)
        ok = self.in_support(theta)
        for start in range(0, theta.shape[0], chunk_size):
            stop = min(start + chunk_size, theta.shape[0])
            sel = ok[start:stop].nonzero().flatten() + start
            if sel.numel() == 0:
                continue
            cov = self._covariance(theta[sel].to(torch.float64))
            chol = torch.linalg.cholesky(cov)
            solved = torch.cholesky_solve(y.expand(sel.numel(), self.n_timesteps).unsqueeze(-1), chol)
            quad = (y.unsqueeze(0) * solved.squeeze(-1)).sum(-1)
            log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(-1)
            out[sel] = -0.5 * (quad + log_det + self.n_timesteps * math.log(2 * math.pi))
        return out

    def log_prior(self, theta: Tensor) -> Tensor:
        theta = torch.as_tensor(theta, dtype=self.dtype).reshape(-1, 2)
        return torch.where(self.in_support(theta), torch.zeros(theta.shape[0]), torch.full((theta.shape[0],), -math.inf))

    def grid_posterior(
        self, y_obs: Tensor, resolution: int = 160, log_prior: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        r"""Reference posterior on a grid over the triangle, by quadrature.

        Args:
            y_obs: one raw series of length ``n_timesteps``.
            resolution: grid points per axis over the bounding box.
            log_prior: optional ``(resolution^2,)`` log prior replacing the
                uniform one, for prior-sensitivity experiments.

        Returns:
            ``(grid, weights)`` with the grid of shape ``(resolution^2, 2)`` and
            normalised posterior masses summing to one (zero outside the
            triangle).
        """
        axis1 = torch.linspace(-2.0, 2.0, resolution, dtype=self.dtype)
        axis2 = torch.linspace(-1.0, 1.0, resolution, dtype=self.dtype)
        grid = torch.stack(torch.meshgrid(axis1, axis2, indexing="ij"), dim=-1).reshape(-1, 2)
        log_post = self.log_likelihood(grid, y_obs).to(self.dtype)
        log_post = log_post + (self.log_prior(grid) if log_prior is None else log_prior.reshape(-1))
        log_post = torch.where(torch.isfinite(log_post), log_post, torch.full_like(log_post, -float("inf")))
        weights = torch.softmax(log_post.double(), dim=0).to(self.dtype)
        return grid, weights


class SumIdentified(Simulator):
    r"""``Theta ~ N(0, I_2)``, ``y_j ~ N(theta_1 + theta_2, sigma^2)``, ``j = 1..m``.

    A minimal non-identifiability stress test: the data constrain only
    :math:`\theta_1 + \theta_2`, while the orthogonal direction
    :math:`\theta_1 - \theta_2` keeps its prior exactly. A correct operator
    estimate should put a single non-zero singular value on the identified
    direction, so :math:`\hat\sigma_1` -- the Hirschfeld-Gebelein-Renyi maximal
    correlation -- becomes a readable identifiability diagnostic, and
    :math:`\hat\sigma_2 \approx 0`.

    Shrinking ``noise`` also drives the model towards the non-compact regime
    warned about in the theory: as :math:`\sigma \to 0` the data pin the
    identified direction down exactly, the density ratio leaves
    :math:`L^2(\pi \times \mu)` and :math:`\sigma_1 \to 1`.
    """

    theta_dim = 2

    def __init__(self, n_obs: int = 10, noise: float = 1.0, **kw):
        super().__init__(**kw)
        self.n_obs = int(n_obs)
        self.noise = float(noise)
        self.data_dim = self.n_obs

    def sample_prior(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        return torch.randn(n, 2, generator=generator, dtype=self.dtype)

    def simulate(self, theta: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        loc = theta.sum(dim=-1, keepdim=True)
        noise = torch.randn(theta.shape[0], self.n_obs, generator=generator, dtype=self.dtype)
        return loc + self.noise * noise

    # --- exact reference ---------------------------------------------------
    def posterior_mean_cov(self, y_obs: Tensor) -> Tuple[Tensor, Tensor]:
        """Exact Gaussian posterior mean ``(n, 2)`` and its shared covariance ``(2, 2)``."""
        y_obs = torch.as_tensor(y_obs, dtype=self.dtype).reshape(-1, self.n_obs)
        ones = torch.ones(2, 1, dtype=self.dtype)
        precision = torch.eye(2, dtype=self.dtype) + (self.n_obs / self.noise**2) * (ones @ ones.T)
        cov = torch.linalg.inv(precision)
        rhs = (self.n_obs * y_obs.mean(dim=-1, keepdim=True) / self.noise**2) * ones.T  # (n, 2)
        return rhs @ cov.T, cov

    @property
    def identified_direction(self) -> Tensor:
        r"""The unit vector :math:`(1, 1)/\sqrt{2}` that the data constrain."""
        return torch.tensor([1.0, 1.0], dtype=self.dtype) / math.sqrt(2.0)

    def maximal_correlation(self) -> float:
        r"""Exact :math:`\sigma_1`: the top canonical correlation of a Gaussian pair."""
        cov_theta = torch.eye(2, dtype=torch.float64)
        ones_m = torch.ones(self.n_obs, 1, dtype=torch.float64)
        cov_yy = 2.0 * (ones_m @ ones_m.T) + self.noise**2 * torch.eye(self.n_obs, dtype=torch.float64)
        cov_ty = torch.ones(2, self.n_obs, dtype=torch.float64)  # Cov(theta_i, y_j) = 1
        m = _inv_sqrt(cov_theta) @ cov_ty @ _inv_sqrt(cov_yy)
        return float(torch.linalg.svdvals(m)[0])

    def log_likelihood(self, theta: Tensor, y_obs: Tensor) -> Tensor:
        theta = torch.as_tensor(theta, dtype=self.dtype).reshape(-1, 2)
        y_obs = torch.as_tensor(y_obs, dtype=self.dtype).reshape(self.n_obs)
        resid = y_obs.unsqueeze(0) - theta.sum(dim=-1, keepdim=True)
        return -0.5 * (resid**2).sum(-1) / self.noise**2 - self.n_obs * math.log(
            self.noise * math.sqrt(2 * math.pi)
        )

    def log_prior(self, theta: Tensor) -> Tensor:
        theta = torch.as_tensor(theta, dtype=self.dtype).reshape(-1, 2)
        return -0.5 * (theta**2).sum(-1)


def _inv_sqrt(matrix: Tensor) -> Tensor:
    """Symmetric inverse square root of a positive-definite matrix."""
    evals, evecs = torch.linalg.eigh(matrix)
    return (evecs * evals.clamp_min(torch.finfo(matrix.dtype).eps).rsqrt()) @ evecs.T
