"""Synthetic conditional distributions with closed-form ground truth.

Each generator exposes the exact conditional mean, standard deviation, density
and CDF wherever they are available in closed form, so an NCP fit can be
checked against the truth instead of against another estimator. The four
families cover the regimes the method is meant to handle: a Gaussian baseline,
input-dependent spread, multimodality, and heavy tails.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "SyntheticConditional",
    "LinearGaussian",
    "Heteroscedastic",
    "BimodalMixture",
    "StudentT",
    "Standardizer",
    "train_val_split",
]

_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def _normal_pdf(y: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    z = (y - mean) / std
    return torch.exp(-0.5 * z**2 - _LOG_SQRT_2PI) / std


def _normal_cdf(y: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf((y - mean) / (std * _SQRT2)))


def _normal_icdf(p: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return mean + std * _SQRT2 * torch.erfinv(2.0 * p - 1.0)


class SyntheticConditional:
    """Base class: a joint law over ``(X, Y)`` with a known conditional."""

    y_dim = 1

    def __init__(self, x_dim: int = 1, dtype: torch.dtype = torch.float32):
        self.x_dim = x_dim
        self.dtype = dtype

    # --- to implement in subclasses -------------------------------------
    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def conditional_mean(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def conditional_std(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def conditional_pdf(self, x: Tensor, y: Tensor) -> Tensor:
        """``(n_x, q)`` densities of ``y`` given each row of ``x``."""
        raise NotImplementedError

    def conditional_cdf(self, x: Tensor, y: Tensor) -> Tensor:
        """``(n_x, q)`` CDF values. Subclasses without a closed form may omit this."""
        raise NotImplementedError

    # --- generic helpers -------------------------------------------------
    def _prep(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.as_tensor(x, dtype=self.dtype)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        y = torch.as_tensor(y, dtype=self.dtype).reshape(1, -1)
        return x, y

    def conditional_quantile(self, x: Tensor, levels: Tensor, grid: Optional[Tensor] = None) -> Tensor:
        """Conditional quantiles, by inverting :meth:`conditional_cdf` on a fine grid.

        Subclasses with an analytic inverse override this.
        """
        if grid is None:
            grid = self.default_grid()
        levels = torch.as_tensor(levels, dtype=self.dtype).reshape(-1)
        cdf = self.conditional_cdf(x, grid)
        idx = torch.searchsorted(cdf.contiguous(), levels.expand(cdf.shape[0], -1).contiguous(), right=False)
        return grid.reshape(-1)[idx.clamp_max(grid.numel() - 1)]

    def default_grid(self, n: int = 1001) -> Tensor:
        """A grid wide enough to carry essentially all of the marginal mass of ``Y``."""
        lo, hi = self._y_range()
        return torch.linspace(lo, hi, n, dtype=self.dtype)

    def _y_range(self) -> Tuple[float, float]:
        return (-10.0, 10.0)


class LinearGaussian(SyntheticConditional):
    r"""``X ~ N(0, I_p)``, ``Y = a·X + b + sigma·eps``.

    Everything is Gaussian, including the marginal of ``Y``, which makes this
    the reference case for checking densities, CDFs and singular values. For a
    jointly Gaussian pair the conditional expectation operator diagonalises in
    the Hermite basis with singular values :math:`\rho^k`, :math:`k \ge 0`,
    where :math:`\rho = \lVert a \rVert / \sqrt{\lVert a \rVert^2 +
    \sigma^2}`; deflation removes the :math:`k = 0` direction, so a fitted
    operator should report :math:`\hat\sigma_k \approx \rho^k` for
    :math:`k = 1, 2, \dots` (see :meth:`true_singular_values`).
    """

    def __init__(self, x_dim: int = 1, a: Optional[Tensor] = None, b: float = 0.0, sigma: float = 0.5, **kw):
        super().__init__(x_dim=x_dim, **kw)
        self.a = torch.ones(x_dim, dtype=self.dtype) if a is None else torch.as_tensor(a, dtype=self.dtype)
        if self.a.shape != (x_dim,):
            raise ValueError(f"a must have shape ({x_dim},), got {tuple(self.a.shape)}")
        self.b = float(b)
        self.sigma = float(sigma)

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor]:
        x = torch.randn(n, self.x_dim, generator=generator, dtype=self.dtype)
        noise = torch.randn(n, 1, generator=generator, dtype=self.dtype)
        y = x @ self.a.unsqueeze(-1) + self.b + self.sigma * noise
        return x, y

    def conditional_mean(self, x: Tensor) -> Tensor:
        x = torch.as_tensor(x, dtype=self.dtype)
        x = x if x.ndim == 2 else x.unsqueeze(-1)
        return x @ self.a.unsqueeze(-1) + self.b

    def conditional_std(self, x: Tensor) -> Tensor:
        return torch.full((torch.as_tensor(x).reshape(-1, self.x_dim).shape[0], 1), self.sigma, dtype=self.dtype)

    def conditional_pdf(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = self._prep(x, y)
        return _normal_pdf(y, self.conditional_mean(x), torch.tensor(self.sigma, dtype=self.dtype))

    def conditional_cdf(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = self._prep(x, y)
        return _normal_cdf(y, self.conditional_mean(x), torch.tensor(self.sigma, dtype=self.dtype))

    def conditional_quantile(self, x: Tensor, levels: Tensor, grid: Optional[Tensor] = None) -> Tensor:
        levels = torch.as_tensor(levels, dtype=self.dtype).reshape(1, -1)
        return _normal_icdf(levels, self.conditional_mean(x), torch.tensor(self.sigma, dtype=self.dtype))

    @property
    def marginal_std(self) -> float:
        return math.sqrt(float(self.a @ self.a) + self.sigma**2)

    def marginal_pdf(self, y: Tensor) -> Tensor:
        r"""Exact marginal :math:`\pi_Y`, needed to turn ratios into densities."""
        y = torch.as_tensor(y, dtype=self.dtype).reshape(-1)
        return _normal_pdf(
            y, torch.tensor(self.b, dtype=self.dtype), torch.tensor(self.marginal_std, dtype=self.dtype)
        )

    @property
    def correlation(self) -> float:
        r""":math:`\rho = \operatorname{corr}(a \cdot X, Y)`."""
        norm_a = math.sqrt(float(self.a @ self.a))
        return norm_a / math.sqrt(norm_a**2 + self.sigma**2)

    def true_singular_values(self, k: int) -> Tensor:
        r"""The top ``k`` singular values of the deflated operator, :math:`\rho^1, \dots, \rho^k`."""
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        powers = torch.arange(1, k + 1, dtype=self.dtype)
        return torch.tensor(self.correlation, dtype=self.dtype) ** powers

    def _y_range(self) -> Tuple[float, float]:
        s = self.marginal_std
        return (self.b - 6 * s, self.b + 6 * s)


class Heteroscedastic(SyntheticConditional):
    r"""``X ~ U(-2, 2)``, ``Y | X ~ N(sin(pi·X/2), (0.2 + 0.6·|X|)^2)``.

    Unimodal but with a spread that varies by a factor of five across the input
    range, so a method that reports a single global predictive variance cannot
    be calibrated everywhere at once.
    """

    def __init__(self, low: float = -2.0, high: float = 2.0, **kw):
        super().__init__(x_dim=1, **kw)
        self.low, self.high = float(low), float(high)

    def _loc_scale(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.as_tensor(x, dtype=self.dtype)
        x = x if x.ndim == 2 else x.unsqueeze(-1)
        return torch.sin(math.pi * x / 2.0), 0.2 + 0.6 * x.abs()

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor]:
        u = torch.rand(n, 1, generator=generator, dtype=self.dtype)
        x = self.low + (self.high - self.low) * u
        loc, scale = self._loc_scale(x)
        y = loc + scale * torch.randn(n, 1, generator=generator, dtype=self.dtype)
        return x, y

    def conditional_mean(self, x: Tensor) -> Tensor:
        return self._loc_scale(x)[0]

    def conditional_std(self, x: Tensor) -> Tensor:
        return self._loc_scale(x)[1]

    def conditional_pdf(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = self._prep(x, y)
        loc, scale = self._loc_scale(x)
        return _normal_pdf(y, loc, scale)

    def conditional_cdf(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = self._prep(x, y)
        loc, scale = self._loc_scale(x)
        return _normal_cdf(y, loc, scale)

    def conditional_quantile(self, x: Tensor, levels: Tensor, grid: Optional[Tensor] = None) -> Tensor:
        loc, scale = self._loc_scale(x)
        levels = torch.as_tensor(levels, dtype=self.dtype).reshape(1, -1)
        return _normal_icdf(levels, loc, scale)

    def _y_range(self) -> Tuple[float, float]:
        return (-9.0, 9.0)


class BimodalMixture(SyntheticConditional):
    r"""``X ~ U(-1, 1)``, ``Y | X`` a two-component Gaussian mixture.

    The component means separate as :math:`\pm(1 + |x|)` while the mixing
    weight sweeps from one mode to the other through
    :math:`w(x) = \mathrm{sigmoid}(4x)`. The conditional mean therefore sits in
    a region of near-zero density for ``x`` near 0, which is exactly where
    mean-plus-variance summaries mislead and a full conditional law does not.
    """

    def __init__(self, scale: float = 0.35, sharpness: float = 4.0, **kw):
        super().__init__(x_dim=1, **kw)
        self.scale = float(scale)
        self.sharpness = float(sharpness)

    def _components(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = torch.as_tensor(x, dtype=self.dtype)
        x = x if x.ndim == 2 else x.unsqueeze(-1)
        sep = 1.0 + x.abs()
        weight = torch.sigmoid(self.sharpness * x)
        return weight, -sep, sep

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor]:
        x = 2.0 * torch.rand(n, 1, generator=generator, dtype=self.dtype) - 1.0
        weight, mu_lo, mu_hi = self._components(x)
        pick = (torch.rand(n, 1, generator=generator, dtype=self.dtype) < weight).to(self.dtype)
        loc = pick * mu_hi + (1.0 - pick) * mu_lo
        y = loc + self.scale * torch.randn(n, 1, generator=generator, dtype=self.dtype)
        return x, y

    def conditional_mean(self, x: Tensor) -> Tensor:
        weight, mu_lo, mu_hi = self._components(x)
        return weight * mu_hi + (1.0 - weight) * mu_lo

    def conditional_std(self, x: Tensor) -> Tensor:
        weight, mu_lo, mu_hi = self._components(x)
        mean = weight * mu_hi + (1.0 - weight) * mu_lo
        second = weight * (mu_hi**2 + self.scale**2) + (1.0 - weight) * (mu_lo**2 + self.scale**2)
        return (second - mean**2).clamp_min(0.0).sqrt()

    def conditional_pdf(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = self._prep(x, y)
        weight, mu_lo, mu_hi = self._components(x)
        sd = torch.tensor(self.scale, dtype=self.dtype)
        return weight * _normal_pdf(y, mu_hi, sd) + (1.0 - weight) * _normal_pdf(y, mu_lo, sd)

    def conditional_cdf(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = self._prep(x, y)
        weight, mu_lo, mu_hi = self._components(x)
        sd = torch.tensor(self.scale, dtype=self.dtype)
        return weight * _normal_cdf(y, mu_hi, sd) + (1.0 - weight) * _normal_cdf(y, mu_lo, sd)

    def _y_range(self) -> Tuple[float, float]:
        return (-5.0, 5.0)


class StudentT(SyntheticConditional):
    r"""``X ~ U(-2, 2)``, ``Y | X = loc(x) + scale(x)·T_nu``.

    Heavy tails: with ``df = 3`` the conditional kurtosis is infinite, so tail
    quantiles are driven by rare observations. A good stress test for whether
    conditional intervals stay calibrated rather than merely narrow.
    """

    def __init__(self, df: float = 3.0, low: float = -2.0, high: float = 2.0, **kw):
        super().__init__(x_dim=1, **kw)
        if df <= 2:
            raise ValueError(f"df must exceed 2 for a finite conditional variance, got {df}")
        self.df = float(df)
        self.low, self.high = float(low), float(high)

    def _loc_scale(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.as_tensor(x, dtype=self.dtype)
        x = x if x.ndim == 2 else x.unsqueeze(-1)
        return 0.5 * x, 0.3 + 0.2 * x.abs()

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor]:
        u = torch.rand(n, 1, generator=generator, dtype=self.dtype)
        x = self.low + (self.high - self.low) * u
        loc, scale = self._loc_scale(x)
        # T_nu = Z / sqrt(W/nu) with W ~ chi2_nu, built from a Gamma draw.
        z = torch.randn(n, 1, generator=generator, dtype=self.dtype)
        w = 2.0 * torch._standard_gamma(torch.full((n, 1), self.df / 2.0, dtype=self.dtype), generator=generator)
        t = z / (w / self.df).sqrt()
        return x, loc + scale * t

    def conditional_mean(self, x: Tensor) -> Tensor:
        return self._loc_scale(x)[0]

    def conditional_std(self, x: Tensor) -> Tensor:
        _, scale = self._loc_scale(x)
        return scale * math.sqrt(self.df / (self.df - 2.0))

    def conditional_pdf(self, x: Tensor, y: Tensor) -> Tensor:
        x, y = self._prep(x, y)
        loc, scale = self._loc_scale(x)
        nu = self.df
        log_norm = (
            math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
        )
        z = (y - loc) / scale
        return torch.exp(log_norm - 0.5 * (nu + 1) * torch.log1p(z**2 / nu)) / scale

    def _y_range(self) -> Tuple[float, float]:
        return (-20.0, 20.0)


class Standardizer:
    """Per-coordinate mean/scale normaliser, fitted on training data only.

    Standardising both ``X`` and ``Y`` matters more than usual here: the
    embeddings are plain MLPs and the whitening step inverts their feature
    covariance, so badly scaled inputs show up as an ill-conditioned inverse.
    """

    def __init__(self):
        self.mean: Optional[Tensor] = None
        self.scale: Optional[Tensor] = None

    def fit(self, data: Tensor) -> "Standardizer":
        data = torch.as_tensor(data)
        data = data if data.ndim == 2 else data.reshape(data.shape[0], -1)
        self.mean = data.mean(dim=0, keepdim=True)
        self.scale = data.std(dim=0, unbiased=True, keepdim=True).clamp_min(torch.finfo(data.dtype).eps)
        return self

    def transform(self, data: Tensor) -> Tensor:
        if self.mean is None or self.scale is None:
            raise RuntimeError("call fit() before transform()")
        data = torch.as_tensor(data)
        data = data if data.ndim == 2 else data.reshape(data.shape[0], -1)
        return (data - self.mean) / self.scale

    def fit_transform(self, data: Tensor) -> Tensor:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: Tensor) -> Tensor:
        if self.mean is None or self.scale is None:
            raise RuntimeError("call fit() before inverse_transform()")
        return torch.as_tensor(data) * self.scale + self.mean

    def inverse_scale(self, data: Tensor) -> Tensor:
        """Undo scaling only -- for standard deviations and interval widths."""
        if self.scale is None:
            raise RuntimeError("call fit() before inverse_scale()")
        return torch.as_tensor(data) * self.scale


def train_val_split(
    x: Tensor, y: Tensor, val_fraction: float = 0.2, generator: Optional[torch.Generator] = None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Random split into ``(x_train, y_train, x_val, y_val)``."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must lie in (0, 1), got {val_fraction}")
    n = x.shape[0]
    n_val = max(2, int(round(n * val_fraction)))
    if n_val >= n - 1:
        raise ValueError(f"val_fraction={val_fraction} leaves too few training points out of {n}")
    perm = torch.randperm(n, generator=generator)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]
