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

__all__ = ["Simulator", "GaussianLinear", "MA2", "AR2", "GAndK", "SIR", "SumIdentified"]

_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


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


class AR2(Simulator):
    r"""AR(2): ``y_t = phi_1 y_{t-1} + phi_2 y_{t-2} + e_t``, ``e_t ~ N(0, 1)``.

    The autoregressive companion to :class:`MA2`. The prior is uniform on the
    stationarity triangle :math:`\{\phi_1 + \phi_2 < 1,\ \phi_2 - \phi_1 < 1,\
    |\phi_2| < 1\}`. Unlike MA(2) the stationary covariance is full Toeplitz
    rather than banded -- the autocovariances decay geometrically instead of
    vanishing after lag 2 -- which makes it a useful contrast when asking how
    fast the operator spectrum decays.
    """

    theta_dim = 2

    def __init__(self, n_timesteps: int = 50, summaries: bool = True, n_lags: int = 4, **kw):
        super().__init__(**kw)
        self.n_timesteps = int(n_timesteps)
        self.summaries = bool(summaries)
        self.n_lags = int(n_lags)
        self.data_dim = self.n_lags + 1 if summaries else self.n_timesteps

    @staticmethod
    def in_support(theta: Tensor) -> Tensor:
        p1, p2 = theta[:, 0], theta[:, 1]
        return (p1 + p2 < 1.0) & (p2 - p1 < 1.0) & (p2.abs() < 1.0)

    def sample_prior(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        kept, total = [], 0
        while total < n:
            batch = torch.rand(max(n, 256), 2, generator=generator, dtype=self.dtype)
            cand = torch.stack([4.0 * batch[:, 0] - 2.0, 2.0 * batch[:, 1] - 1.0], dim=-1)
            ok = cand[self.in_support(cand)]
            kept.append(ok)
            total += ok.shape[0]
        return torch.cat(kept)[:n]

    def autocovariance(self, theta: Tensor, n_lags: int) -> Tensor:
        r"""Stationary autocovariances :math:`\gamma_0, \dots, \gamma_{n\_lags}`.

        From the Yule-Walker solution
        :math:`\gamma_0 = (1 - \phi_2) / [(1 + \phi_2)((1 - \phi_2)^2 - \phi_1^2)]`
        and :math:`\gamma_1 = \phi_1 \gamma_0 / (1 - \phi_2)`, then the
        recursion :math:`\gamma_k = \phi_1 \gamma_{k-1} + \phi_2 \gamma_{k-2}`.
        """
        p1, p2 = theta[:, 0], theta[:, 1]
        gamma0 = (1 - p2) / ((1 + p2) * ((1 - p2) ** 2 - p1**2))
        gamma1 = p1 * gamma0 / (1 - p2)
        out = [gamma0, gamma1]
        for _ in range(2, n_lags + 1):
            out.append(p1 * out[-1] + p2 * out[-2])
        return torch.stack(out, dim=-1)

    def simulate(self, theta: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        n, t = theta.shape[0], self.n_timesteps
        # Start from the exact stationary law of (y_1, y_2) so there is no burn-in bias.
        gamma = self.autocovariance(theta, 1)
        g0, g1 = gamma[:, 0], gamma[:, 1]
        z = torch.randn(n, 2, generator=generator, dtype=self.dtype)
        y_prev2 = g0.sqrt() * z[:, 0]
        cond_sd = (g0 - g1**2 / g0).clamp_min(1e-12).sqrt()
        y_prev1 = (g1 / g0) * y_prev2 + cond_sd * z[:, 1]

        noise = torch.randn(n, t, generator=generator, dtype=self.dtype)
        series = [y_prev2, y_prev1]
        for step in range(2, t):
            series.append(theta[:, 0] * series[-1] + theta[:, 1] * series[-2] + noise[:, step])
        out = torch.stack(series[:t], dim=-1)
        return self.summarize(out) if self.summaries else out

    def summarize(self, series: Tensor) -> Tensor:
        """Sample variance plus autocovariances at lags 1..``n_lags``."""
        centred = series - series.mean(dim=-1, keepdim=True)
        t = centred.shape[-1]
        feats = [(centred**2).mean(dim=-1, keepdim=True)]
        for lag in range(1, self.n_lags + 1):
            feats.append((centred[:, lag:] * centred[:, : t - lag]).mean(dim=-1, keepdim=True))
        return torch.cat(feats, dim=-1)

    def log_prior(self, theta: Tensor) -> Tensor:
        theta = torch.as_tensor(theta, dtype=self.dtype).reshape(-1, 2)
        return torch.where(
            self.in_support(theta), torch.zeros(theta.shape[0]), torch.full((theta.shape[0],), -math.inf)
        )

    def log_likelihood(self, theta: Tensor, y_obs: Tensor, chunk_size: int = 1024) -> Tensor:
        """Exact stationary Gaussian log-likelihood of a raw series."""
        if self.summaries:
            raise NotImplementedError("the exact likelihood is for the raw series; use summaries=False")
        theta = torch.as_tensor(theta, dtype=torch.float64).reshape(-1, 2)
        y = torch.as_tensor(y_obs, dtype=torch.float64).reshape(self.n_timesteps)
        t = self.n_timesteps
        idx = (torch.arange(t).unsqueeze(0) - torch.arange(t).unsqueeze(1)).abs()
        out = torch.full((theta.shape[0],), -float("inf"), dtype=torch.float64)
        ok = self.in_support(theta)
        for start in range(0, theta.shape[0], chunk_size):
            stop = min(start + chunk_size, theta.shape[0])
            sel = ok[start:stop].nonzero().flatten() + start
            if sel.numel() == 0:
                continue
            gamma = self.autocovariance(theta[sel].to(torch.float64), t - 1)
            cov = gamma[:, idx.reshape(-1)].reshape(sel.numel(), t, t)
            chol = torch.linalg.cholesky(cov)
            solved = torch.cholesky_solve(y.expand(sel.numel(), t).unsqueeze(-1), chol)
            quad = (y.unsqueeze(0) * solved.squeeze(-1)).sum(-1)
            log_det = 2.0 * torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(-1)
            out[sel] = -0.5 * (quad + log_det + t * math.log(2 * math.pi))
        return out


class GAndK(Simulator):
    r"""The g-and-k distribution, defined by its quantile function.

    .. math::
        Q(z; A, B, g, k) = A + B\big(1 + c\,\tanh(gz/2)\big)\, z\,(1 + z^2)^k,
        \qquad z \sim N(0, 1),\ c = 0.8.

    A standard likelihood-free benchmark: sampling is trivial (push a normal
    draw through :math:`Q`) while the density has no closed form, since it
    requires inverting :math:`Q`. ``A`` and ``B`` set location and scale,
    ``g`` skewness and ``k`` tail weight. The prior is uniform on
    :math:`[0, 10]^4`.

    The density *can* be recovered numerically, because :math:`Q` is monotone:
    :math:`p(y) = \varphi(z) / Q'(z)` at :math:`z = Q^{-1}(y)`. That is what
    :meth:`log_likelihood` does, by bisection -- available for validation, not
    something a simulation-based method would use.
    """

    theta_dim = 4
    C = 0.8

    def __init__(self, n_obs: int = 100, summaries: bool = True, prior_high: float = 10.0, **kw):
        super().__init__(**kw)
        self.n_obs = int(n_obs)
        self.summaries = bool(summaries)
        self.prior_high = float(prior_high)
        self.data_dim = 4 if summaries else self.n_obs

    def sample_prior(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        return self.prior_high * torch.rand(n, 4, generator=generator, dtype=self.dtype)

    def log_prior(self, theta: Tensor) -> Tensor:
        theta = torch.as_tensor(theta, dtype=self.dtype).reshape(-1, 4)
        inside = ((theta >= 0) & (theta <= self.prior_high)).all(dim=-1)
        return torch.where(inside, torch.zeros(theta.shape[0]), torch.full((theta.shape[0],), -math.inf))

    def quantile(self, theta: Tensor, z: Tensor) -> Tensor:
        r""":math:`Q(z)` broadcast over a batch of parameters, shape ``(n, m)``."""
        a, b, g, k = (theta[:, i : i + 1] for i in range(4))
        return a + b * (1 + self.C * torch.tanh(g * z / 2)) * z * (1 + z**2) ** k

    def quantile_derivative(self, theta: Tensor, z: Tensor) -> Tensor:
        r""":math:`Q'(z)`, needed to turn :math:`\varphi(z)` into a density in ``y``."""
        a, b, g, k = (theta[:, i : i + 1] for i in range(4))
        tanh = torch.tanh(g * z / 2)
        tilt = 1 + self.C * tanh
        tilt_derivative = self.C * (g / 2) * (1 - tanh**2)
        base = z * (1 + z**2) ** k
        base_derivative = (1 + z**2) ** (k - 1) * (1 + (2 * k + 1) * z**2)
        return b * (tilt_derivative * base + tilt * base_derivative)

    def simulate(self, theta: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        z = torch.randn(theta.shape[0], self.n_obs, generator=generator, dtype=self.dtype)
        y = self.quantile(theta, z)
        return self.summarize(y) if self.summaries else y

    def summarize(self, y: Tensor) -> Tensor:
        r"""The four robust order-statistic summaries standard for this model.

        Location, scale, skewness and kurtosis read off the octiles
        :math:`E_1, \dots, E_7`: :math:`E_4`, :math:`E_6 - E_2`,
        :math:`(E_6 + E_2 - 2E_4)/(E_6 - E_2)` and
        :math:`(E_7 - E_5 + E_3 - E_1)/(E_6 - E_2)`.
        """
        octiles = torch.quantile(
            y, torch.arange(1, 8, dtype=y.dtype) / 8.0, dim=-1, interpolation="linear"
        ).T  # (n, 7)
        e1, e2, e3, e4, e5, e6, e7 = (octiles[:, i] for i in range(7))
        scale = (e6 - e2).clamp_min(1e-8)
        return torch.stack([e4, scale, (e6 + e2 - 2 * e4) / scale, (e7 - e5 + e3 - e1) / scale], dim=-1)

    def inverse_quantile(self, theta: Tensor, y: Tensor, iterations: int = 80) -> Tensor:
        r""":math:`Q^{-1}(y)` by bisection. ``Q`` is increasing, so this is safe."""
        low = torch.full_like(y, -40.0)
        high = torch.full_like(y, 40.0)
        for _ in range(iterations):
            mid = 0.5 * (low + high)
            too_big = self.quantile(theta, mid) > y
            high = torch.where(too_big, mid, high)
            low = torch.where(too_big, low, mid)
        return 0.5 * (low + high)

    def log_likelihood(self, theta: Tensor, y_obs: Tensor, chunk_size: int = 4096) -> Tensor:
        """Numerically inverted log-likelihood of one observed sample."""
        theta = torch.as_tensor(theta, dtype=self.dtype).reshape(-1, 4)
        y = torch.as_tensor(y_obs, dtype=self.dtype).reshape(1, -1)
        out = torch.full((theta.shape[0],), -float("inf"), dtype=self.dtype)
        valid = (theta[:, 1] > 1e-6) & (theta[:, 3] > -0.5)
        for start in range(0, theta.shape[0], chunk_size):
            stop = min(start + chunk_size, theta.shape[0])
            sel = valid[start:stop].nonzero().flatten() + start
            if sel.numel() == 0:
                continue
            block = theta[sel]
            z = self.inverse_quantile(block, y.expand(sel.numel(), y.shape[1]))
            derivative = self.quantile_derivative(block, z).clamp_min(1e-30)
            log_density = -0.5 * z**2 - _LOG_SQRT_2PI - torch.log(derivative)
            out[sel] = log_density.sum(dim=-1)
        return out


class SIR(Simulator):
    r"""A mechanistic SIR epidemic: infected counts observed with Gaussian noise.

    .. math::
        S' = -\beta S I / N, \qquad I' = \beta S I / N - \gamma I, \qquad R' = \gamma I,

    integrated by fixed-step RK4 from :math:`(S, I, R) = (N - I_0, I_0, 0)`, with
    :math:`I(t_j)` observed at ``n_obs`` equally spaced times under additive
    noise. The parameter :math:`(\beta, \gamma)` is low-dimensional but the data
    are a whole epidemic curve, which is the regime where amortising over both
    observations and functionals pays.

    Because the mean curve is a deterministic ODE solution and the noise is
    additive Gaussian, the likelihood *is* computable: :math:`y \mid \theta
    \sim N(\text{mean\_curve}(\theta), \sigma^2 I)`. A reference posterior is
    therefore available by quadrature via :meth:`grid_posterior`, and posterior
    functionals can be scored against it.

    That makes this a *mechanistic* simulator, not a likelihood-free one. It is
    a tractable stand-in, chosen so that a reference exists; genuinely
    intractable epidemic models are stochastic (a Gillespie / Markov-jump SIR,
    or partial observation of latent compartments), and for those no reference
    posterior is available without expensive ABC or MCMC.

    The default ``noise`` is chosen so the posterior is about five times tighter
    than the prior, which is a realistic reporting-noise regime. Lowering it
    makes the curve nearly pin the parameter down (at ``noise=12`` the posterior
    is over a hundred times tighter than the prior and
    :math:`\hat\sigma_1 > 0.999`), which is the near-deterministic regime where
    the density ratio leaves :math:`L^2` and the low-rank model should not be
    trusted.
    """

    theta_dim = 2

    def __init__(
        self,
        population: float = 1000.0,
        initial_infected: float = 5.0,
        duration: float = 30.0,
        n_obs: int = 30,
        noise: float = 40.0,
        steps_per_unit: int = 4,
        beta_range: Tuple[float, float] = (0.4, 3.0),
        gamma_range: Tuple[float, float] = (0.1, 1.0),
        **kw,
    ):
        super().__init__(**kw)
        self.population = float(population)
        self.initial_infected = float(initial_infected)
        self.duration = float(duration)
        self.n_obs = int(n_obs)
        self.noise = float(noise)
        self.steps_per_unit = int(steps_per_unit)
        self.beta_range = beta_range
        self.gamma_range = gamma_range
        self.data_dim = self.n_obs

    def sample_prior(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        unit = torch.rand(n, 2, generator=generator, dtype=self.dtype)
        lo = torch.tensor([self.beta_range[0], self.gamma_range[0]], dtype=self.dtype)
        hi = torch.tensor([self.beta_range[1], self.gamma_range[1]], dtype=self.dtype)
        return lo + (hi - lo) * unit

    def log_prior(self, theta: Tensor) -> Tensor:
        theta = torch.as_tensor(theta, dtype=self.dtype).reshape(-1, 2)
        lo = torch.tensor([self.beta_range[0], self.gamma_range[0]], dtype=theta.dtype)
        hi = torch.tensor([self.beta_range[1], self.gamma_range[1]], dtype=theta.dtype)
        inside = ((theta >= lo) & (theta <= hi)).all(dim=-1)
        return torch.where(inside, torch.zeros(theta.shape[0]), torch.full((theta.shape[0],), -math.inf))

    def mean_curve(self, theta: Tensor) -> Tensor:
        """Noise-free infected counts at the observation times, shape ``(n, n_obs)``."""
        theta = torch.as_tensor(theta, dtype=self.dtype).reshape(-1, 2)
        beta, gamma = theta[:, 0], theta[:, 1]
        n_pop = self.population
        # Round the step count up to a multiple of n_obs so every observation
        # time lands exactly on an integration step. Without this, two
        # observation times can round to the same step and one of them is
        # silently never recorded, leaving a zero in the curve.
        base = max(1, int(round(self.duration * self.steps_per_unit)))
        total_steps = -(-base // self.n_obs) * self.n_obs
        dt = self.duration / total_steps
        stride = total_steps // self.n_obs
        record_at = {j * stride: j for j in range(1, self.n_obs + 1)}

        state = torch.stack(
            [
                torch.full_like(beta, n_pop - self.initial_infected),
                torch.full_like(beta, self.initial_infected),
                torch.zeros_like(beta),
            ],
            dim=-1,
        )
        observed = torch.zeros(theta.shape[0], self.n_obs, dtype=self.dtype)

        def derivative(z: Tensor) -> Tensor:
            s, i = z[:, 0], z[:, 1]
            infection = beta * s * i / n_pop
            removal = gamma * i
            return torch.stack([-infection, infection - removal, removal], dim=-1)

        for step in range(1, total_steps + 1):
            k1 = derivative(state)
            k2 = derivative(state + 0.5 * dt * k1)
            k3 = derivative(state + 0.5 * dt * k2)
            k4 = derivative(state + dt * k3)
            state = (state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).clamp_min(0.0)
            if step in record_at:
                observed[:, record_at[step] - 1] = state[:, 1]
        return observed

    def simulate(self, theta: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        mean = self.mean_curve(theta)
        noise = torch.randn(mean.shape, generator=generator, dtype=self.dtype)
        return mean + self.noise * noise

    def log_likelihood(self, theta: Tensor, y_obs: Tensor, chunk_size: int = 4096) -> Tensor:
        theta = torch.as_tensor(theta, dtype=self.dtype).reshape(-1, 2)
        y = torch.as_tensor(y_obs, dtype=self.dtype).reshape(1, self.n_obs)
        out = torch.empty(theta.shape[0], dtype=self.dtype)
        for start in range(0, theta.shape[0], chunk_size):
            stop = min(start + chunk_size, theta.shape[0])
            residual = y - self.mean_curve(theta[start:stop])
            out[start:stop] = -0.5 * (residual**2).sum(-1) / self.noise**2 - self.n_obs * math.log(
                self.noise * math.sqrt(2 * math.pi)
            )
        return out

    def grid_posterior(self, y_obs: Tensor, resolution: int = 100) -> Tuple[Tensor, Tensor]:
        """Reference posterior on a grid over the prior box, by quadrature."""
        beta_axis = torch.linspace(*self.beta_range, resolution, dtype=self.dtype)
        gamma_axis = torch.linspace(*self.gamma_range, resolution, dtype=self.dtype)
        grid = torch.stack(torch.meshgrid(beta_axis, gamma_axis, indexing="ij"), dim=-1).reshape(-1, 2)
        log_post = self.log_likelihood(grid, y_obs)
        weights = torch.softmax(log_post.double(), dim=0).to(self.dtype)
        return grid, weights

    def in_support(self, theta: Tensor) -> Tensor:
        """Whether each row lies inside the prior box -- for rejecting NPE leakage."""
        theta = torch.as_tensor(theta, dtype=self.dtype).reshape(-1, 2)
        lo = torch.tensor([self.beta_range[0], self.gamma_range[0]], dtype=theta.dtype)
        hi = torch.tensor([self.beta_range[1], self.gamma_range[1]], dtype=theta.dtype)
        return ((theta >= lo) & (theta <= hi)).all(dim=-1)

    @property
    def basic_reproduction_number_range(self) -> Tuple[float, float]:
        r"""The range of :math:`R_0 = \beta/\gamma` implied by the prior box."""
        return (self.beta_range[0] / self.gamma_range[1], self.beta_range[1] / self.gamma_range[0])
