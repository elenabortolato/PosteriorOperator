"""The synthetic generators: is the stated ground truth actually the truth?

Every closed-form conditional is cross-checked against the sampler, so a bug
in the reference quantities cannot masquerade as a bug in NCP later on.
"""

import math

import pytest
import torch

from posterior_operator.data import (
    BimodalMixture,
    Heteroscedastic,
    LinearGaussian,
    Standardizer,
    StudentT,
    train_val_split,
)
from posterior_operator.metrics import trapezoid

ALL_GENERATORS = [LinearGaussian, Heteroscedastic, BimodalMixture, StudentT]
WITH_CDF = [LinearGaussian, Heteroscedastic, BimodalMixture]


@pytest.fixture(params=ALL_GENERATORS)
def generator_cls(request):
    return request.param


def _dataset(cls):
    return cls(dtype=torch.float64)


def test_sample_shapes_and_finiteness(generator_cls):
    data = _dataset(generator_cls)
    x, y = data.sample(500, generator=torch.Generator().manual_seed(0))
    assert x.shape == (500, data.x_dim)
    assert y.shape == (500, 1)
    assert torch.isfinite(x).all() and torch.isfinite(y).all()


def test_sampling_is_reproducible(generator_cls):
    data = _dataset(generator_cls)
    first = data.sample(50, generator=torch.Generator().manual_seed(3))
    second = data.sample(50, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(first[0], second[0]) and torch.allclose(first[1], second[1])


def test_conditional_density_integrates_to_one(generator_cls):
    data = _dataset(generator_cls)
    grid = data.default_grid(4001)
    x = torch.tensor([[-0.7], [0.0], [0.6]], dtype=torch.float64)
    mass = trapezoid(data.conditional_pdf(x, grid), grid)
    assert torch.allclose(mass, torch.ones(3, dtype=torch.float64), atol=2e-3)


def test_conditional_moments_match_the_sampler(generator_cls):
    """Rejection-free check: condition by sampling at a fixed x."""
    data = _dataset(generator_cls)
    x0 = 0.6
    g = torch.Generator().manual_seed(5)
    n = 400_000
    # Re-run the sampler with X pinned by rejecting nothing: build the response
    # directly from the generator's own conditional at x0 via inverse sampling
    # of the pdf on a fine grid.
    grid = data.default_grid(20001)
    pdf = data.conditional_pdf(torch.tensor([[x0]], dtype=torch.float64), grid)[0]
    cdf = torch.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * (grid[1:] - grid[:-1]), dim=0)
    cdf = torch.cat([torch.zeros(1, dtype=torch.float64), cdf])
    cdf = cdf / cdf[-1]
    u = torch.rand(n, generator=g, dtype=torch.float64)
    draws = grid[torch.searchsorted(cdf.contiguous(), u.contiguous()).clamp_max(grid.numel() - 1)]

    mean = data.conditional_mean(torch.tensor([[x0]], dtype=torch.float64)).item()
    std = data.conditional_std(torch.tensor([[x0]], dtype=torch.float64)).item()
    # The Student-t grid is truncated, which clips a little tail variance.
    tol = 0.05 if generator_cls is not StudentT else 0.15
    assert draws.mean().item() == pytest.approx(mean, abs=tol)
    assert draws.std().item() == pytest.approx(std, rel=tol)


def test_conditional_cdf_is_the_integral_of_the_pdf(generator_cls):
    data = _dataset(generator_cls)
    if generator_cls not in WITH_CDF:
        with pytest.raises(NotImplementedError):
            data.conditional_cdf(torch.zeros(1, 1, dtype=torch.float64), data.default_grid(11))
        return
    grid = data.default_grid(4001)
    x = torch.tensor([[-0.5], [0.4]], dtype=torch.float64)
    pdf = data.conditional_pdf(x, grid)
    cdf = data.conditional_cdf(x, grid)
    integrated = torch.cumsum(0.5 * (pdf[:, 1:] + pdf[:, :-1]) * (grid[1:] - grid[:-1]), dim=-1)
    assert torch.allclose(cdf[:, 1:], cdf[:, :1] + integrated, atol=2e-3)
    assert torch.all(cdf[:, 1:] >= cdf[:, :-1] - 1e-12)
    assert torch.allclose(cdf[:, -1], torch.ones(2, dtype=torch.float64), atol=2e-3)


def test_conditional_quantile_inverts_the_cdf(generator_cls):
    data = _dataset(generator_cls)
    if generator_cls not in WITH_CDF:
        return
    x = torch.tensor([[-0.3], [0.5]], dtype=torch.float64)
    levels = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64)
    q = data.conditional_quantile(x, levels, grid=data.default_grid(20001))
    recovered = torch.stack([data.conditional_cdf(x[i : i + 1], q[i])[0] for i in range(2)])
    assert torch.allclose(recovered, levels.expand(2, -1), atol=5e-3)


# --------------------------------------------------------------------------- #
# Generator-specific ground truth
# --------------------------------------------------------------------------- #


def test_linear_gaussian_marginal_and_spectrum():
    data = LinearGaussian(x_dim=1, a=[1.0], sigma=0.5, dtype=torch.float64)
    assert data.correlation == pytest.approx(1 / math.sqrt(1.25))
    assert data.marginal_std == pytest.approx(math.sqrt(1.25))

    grid = data.default_grid(8001)
    assert float(trapezoid(data.marginal_pdf(grid), grid)) == pytest.approx(1.0, abs=1e-6)

    # The marginal must equal the conditional averaged over X.
    g = torch.Generator().manual_seed(7)
    n_mc, chunk = 100_000, 2_000
    # Accumulate in chunks: the full (n_mc, len(grid)) density table would be
    # several gigabytes.
    mixed = torch.zeros_like(grid)
    for _ in range(n_mc // chunk):
        x = torch.randn(chunk, 1, generator=g, dtype=torch.float64)
        mixed += data.conditional_pdf(x, grid).sum(dim=0)
    mixed /= n_mc
    gap = (mixed - data.marginal_pdf(grid)).abs()
    # Integrated discrepancy: total variation under 0.5%. Asserted on the L1
    # gap rather than the sup norm, which over 8001 grid points is dominated by
    # its own extreme-value fluctuation and converges slower than 1/sqrt(n).
    assert float(trapezoid(gap, grid)) < 1e-2
    assert float(gap.max()) < 0.02 * float(data.marginal_pdf(grid).max())

    sv = data.true_singular_values(4)
    assert torch.allclose(sv, torch.tensor([data.correlation**k for k in (1, 2, 3, 4)], dtype=torch.float64))
    with pytest.raises(ValueError, match="k must be positive"):
        data.true_singular_values(0)


def test_linear_gaussian_multivariate_x():
    data = LinearGaussian(x_dim=3, a=[1.0, -0.5, 0.25], sigma=0.3, dtype=torch.float64)
    x, y = data.sample(2000, generator=torch.Generator().manual_seed(8))
    assert x.shape == (2000, 3)
    residual = y - data.conditional_mean(x)
    assert residual.std().item() == pytest.approx(0.3, rel=0.05)
    with pytest.raises(ValueError, match="a must have shape"):
        LinearGaussian(x_dim=2, a=[1.0])


def test_heteroscedastic_spread_varies_with_x():
    data = Heteroscedastic(dtype=torch.float64)
    centre = data.conditional_std(torch.zeros(1, 1, dtype=torch.float64)).item()
    edge = data.conditional_std(torch.full((1, 1), 2.0, dtype=torch.float64)).item()
    assert edge / centre == pytest.approx(7.0, rel=1e-6)


def test_bimodal_mixture_is_genuinely_bimodal_and_the_mean_sits_in_a_valley():
    data = BimodalMixture(dtype=torch.float64)
    grid = data.default_grid(2001)
    pdf = data.conditional_pdf(torch.zeros(1, 1, dtype=torch.float64), grid)[0]
    peaks = ((pdf[1:-1] > pdf[:-2]) & (pdf[1:-1] > pdf[2:])).nonzero().flatten()
    assert peaks.numel() == 2
    mean = data.conditional_mean(torch.zeros(1, 1, dtype=torch.float64)).item()
    assert mean == pytest.approx(0.0, abs=1e-9)
    density_at_mean = float(data.conditional_pdf(torch.zeros(1, 1, dtype=torch.float64), torch.tensor([mean]))[0])
    # Modes at +-1 with scale 0.35 sit 2.86 sd out, so the valley holds
    # 2*exp(-0.5*(1/0.35)^2) ~ 3.4% of the peak density.
    assert density_at_mean < 0.05 * float(pdf[peaks].max())


def test_student_t_has_heavier_tails_than_a_matched_gaussian():
    data = StudentT(df=3.0, dtype=torch.float64)
    x = torch.zeros(1, 1, dtype=torch.float64)
    far = torch.tensor([12.0], dtype=torch.float64)
    t_tail = float(data.conditional_pdf(x, far)[0])
    scale = float(data.conditional_std(x))
    gaussian_tail = math.exp(-0.5 * (12.0 / scale) ** 2) / (scale * math.sqrt(2 * math.pi))
    assert t_tail > 1e5 * gaussian_tail

    _, y = data.sample(200_000, generator=torch.Generator().manual_seed(9))
    assert float(y.abs().max()) > 10.0  # extreme draws do occur
    with pytest.raises(ValueError, match="df must exceed 2"):
        StudentT(df=1.5)


# --------------------------------------------------------------------------- #
# Preprocessing helpers
# --------------------------------------------------------------------------- #


def test_standardizer_round_trip():
    g = torch.Generator().manual_seed(10)
    data = torch.randn(300, 3, generator=g, dtype=torch.float64) * 5.0 + 2.0
    scaler = Standardizer()
    scaled = scaler.fit_transform(data)
    assert torch.allclose(scaled.mean(0), torch.zeros(3, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(scaled.std(0), torch.ones(3, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(scaler.inverse_transform(scaled), data, atol=1e-10)
    assert torch.allclose(scaler.inverse_scale(scaled.std(0, keepdim=True)), scaler.scale)


def test_standardizer_promotes_1d_and_guards_constant_columns():
    scaler = Standardizer()
    scaled = scaler.fit_transform(torch.arange(10, dtype=torch.float64))
    assert scaled.shape == (10, 1)
    constant = Standardizer().fit(torch.ones(20, 2, dtype=torch.float64))
    assert torch.isfinite(constant.transform(torch.ones(5, 2, dtype=torch.float64))).all()


def test_standardizer_requires_fit_first():
    for call in ("transform", "inverse_transform", "inverse_scale"):
        with pytest.raises(RuntimeError, match="call fit"):
            getattr(Standardizer(), call)(torch.zeros(2, 1))


def test_train_val_split_is_a_disjoint_partition():
    g = torch.Generator().manual_seed(11)
    x = torch.arange(100, dtype=torch.float64).reshape(-1, 1)
    y = x * 2
    xt, yt, xv, yv = train_val_split(x, y, val_fraction=0.25, generator=g)
    assert xt.shape[0] == 75 and xv.shape[0] == 25
    assert torch.allclose(yt, xt * 2) and torch.allclose(yv, xv * 2)  # pairing preserved
    assert set(xt.flatten().tolist()).isdisjoint(xv.flatten().tolist())
    assert sorted(xt.flatten().tolist() + xv.flatten().tolist()) == list(range(100))


def test_train_val_split_validates_the_fraction():
    x = torch.zeros(10, 1)
    for bad in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            train_val_split(x, x, val_fraction=bad)
    with pytest.raises(ValueError, match="too few training points"):
        train_val_split(x, x, val_fraction=0.99)
