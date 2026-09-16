"""End-to-end behaviour: train an operator and check it against known truth.

The linear-Gaussian case is the strongest available check, because its
conditional law, marginal *and* operator spectrum are all known in closed form
-- including the Hermite spectrum ``sigma_k = rho^k``, which the whitening step
has to recover for the singular values to mean anything.
"""

import math

import pytest
import torch

from posterior_operator import (
    BimodalMixture,
    GaussianKDE,
    Heteroscedastic,
    LinearGaussian,
    NCPLoss,
    build_ncp,
    train_ncp,
)
from posterior_operator.metrics import coverage, hellinger, kolmogorov_smirnov


def _fit(data, n=6000, latent_dim=32, epochs=400, seed=0, **kwargs):
    g = torch.Generator().manual_seed(seed)
    x, y = data.sample(n, generator=g)
    x_val, y_val = data.sample(n // 4, generator=g)
    torch.manual_seed(seed)
    operator = build_ncp(x_dim=data.x_dim, y_dim=1, latent_dim=latent_dim)
    history = train_ncp(
        operator,
        x,
        y,
        epochs=epochs,
        lr=1e-3,
        validation_data=(x_val, y_val),
        patience=40,
        val_every=5,
        seed=seed,
        **kwargs,
    )
    return operator, history


@pytest.fixture(scope="module")
def gaussian_fit():
    return _fit(LinearGaussian(sigma=0.5))


# --------------------------------------------------------------------------- #
# The training loop itself
# --------------------------------------------------------------------------- #


def test_training_decreases_the_objective_and_fits_statistics(gaussian_fit):
    operator, history = gaussian_fit
    assert operator.is_fitted
    early = sum(history["train_loss"][:10]) / 10
    late = sum(history["train_loss"][-10:]) / 10
    assert late < early
    assert history["best_epoch"] is not None
    assert all(math.isfinite(v) for v in history["train_loss"] + history["val_loss"])


def test_early_stopping_restores_the_best_validation_weights():
    data = LinearGaussian(sigma=0.5)
    g = torch.Generator().manual_seed(1)
    x, y = data.sample(600, generator=g)
    val = data.sample(300, generator=g)
    torch.manual_seed(1)
    operator = build_ncp(1, 1, latent_dim=8)
    history = train_ncp(
        operator, x, y, epochs=300, validation_data=val, patience=3, val_every=1, seed=1, fit_statistics=False
    )
    assert history["best_epoch"] is not None
    # The retained weights must reproduce the recorded best validation loss.
    with torch.no_grad():
        u, v, s = operator.raw_embeddings(*val)
        assert float(NCPLoss()(u, v, s)) == pytest.approx(history["best_val_loss"], rel=1e-6)
    assert len(history["train_loss"]) < 300  # it actually stopped early


def test_minibatch_training_runs_and_reports_per_epoch_losses():
    data = LinearGaussian(sigma=0.5)
    x, y = data.sample(500, generator=torch.Generator().manual_seed(2))
    torch.manual_seed(2)
    operator = build_ncp(1, 1, latent_dim=8)
    history = train_ncp(operator, x, y, epochs=20, batch_size=64, seed=2)
    assert len(history["train_loss"]) == 20
    assert operator.is_fitted


def test_split_mode_also_trains():
    data = LinearGaussian(sigma=0.5)
    x, y = data.sample(2000, generator=torch.Generator().manual_seed(3))
    torch.manual_seed(3)
    operator = build_ncp(1, 1, latent_dim=16)
    history = train_ncp(operator, x, y, epochs=300, loss=NCPLoss(mode="split", gamma=1e-3), seed=3)
    assert history["train_loss"][-1] < history["train_loss"][0]
    # Both estimators target the same objective, so both must find the signal.
    assert float(operator.singular_values[0]) > 0.7


def test_training_is_reproducible_under_a_fixed_seed():
    data = LinearGaussian(sigma=0.5)
    x, y = data.sample(500, generator=torch.Generator().manual_seed(4))
    outputs = []
    for _ in range(2):
        torch.manual_seed(4)
        operator = build_ncp(1, 1, latent_dim=8)
        train_ncp(operator, x, y, epochs=30, seed=4)
        outputs.append(operator.condition(x[:5]).mean())
    assert torch.allclose(outputs[0], outputs[1])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(epochs=0), "epochs must be positive"),
        (dict(val_every=0), "val_every must be positive"),
        (dict(batch_size=1), "batch_size must be at least 2"),
    ],
)
def test_training_validates_arguments(kwargs, message):
    x = torch.randn(40, 1)
    y = torch.randn(40, 1)
    operator = build_ncp(1, 1, latent_dim=4)
    with pytest.raises(ValueError, match=message):
        train_ncp(operator, x, y, **{"epochs": 5, **kwargs})


def test_mismatched_sample_sizes_are_rejected():
    operator = build_ncp(1, 1, latent_dim=4)
    with pytest.raises(ValueError, match="rows"):
        train_ncp(operator, torch.randn(10, 1), torch.randn(11, 1), epochs=2)


# --------------------------------------------------------------------------- #
# Recovering known conditional quantities
# --------------------------------------------------------------------------- #


def test_recovers_the_gaussian_conditional_mean_and_spread(gaussian_fit):
    operator, _ = gaussian_fit
    data = LinearGaussian(sigma=0.5)
    x = torch.tensor([[-1.5], [-0.5], [0.0], [0.5], [1.5]])
    posterior = operator.condition(x)
    assert torch.allclose(posterior.mean(), data.conditional_mean(x), atol=0.12)
    assert torch.allclose(posterior.std(), data.conditional_std(x), rtol=0.2)


def test_recovers_the_hermite_spectrum(gaussian_fit):
    r"""For a jointly Gaussian pair the singular values must be ``rho^k``."""
    operator, _ = gaussian_fit
    data = LinearGaussian(sigma=0.5)
    got = operator.singular_values[:3]
    expected = data.true_singular_values(3)
    assert torch.allclose(got, expected, atol=0.1)


def test_whitening_bias_is_an_estimator_property_not_sample_reuse():
    r"""Pins the behaviour that ``stats_reg`` exists to control.

    The plug-in canonical correlations are biased upward, and the bias is
    driven by ``latent_dim`` rather than by whitening on the fitted data: it
    shows up identically with *untrained* embeddings, which no sample can be
    "in-sample" for. That is why sample splitting does not remove it and
    ``reg`` does.
    """
    data = LinearGaussian(sigma=0.5)
    truth = data.true_singular_values(4)
    g = torch.Generator().manual_seed(1)
    x, y = data.sample(5000, generator=g)

    errors = {}
    for latent_dim in (8, 128):
        torch.manual_seed(0)
        untrained = build_ncp(1, 1, latent_dim=latent_dim)
        untrained.fit_statistics(x, y, reg=1e-6)
        sv = untrained.singular_values[:4]
        # Upward, and worse in the wider latent space.
        assert torch.all(sv >= truth - 0.02)
        errors[latent_dim] = float((sv - truth).abs().max())
    assert errors[128] > errors[8]

    # Regularisation is the lever that shrinks it.
    torch.manual_seed(0)
    wide = build_ncp(1, 1, latent_dim=128)
    wide.fit_statistics(x, y, reg=1e-6)
    unregularised = float((wide.singular_values[:4] - truth).abs().max())
    wide.fit_statistics(x, y, reg=1e-3)
    regularised = float((wide.singular_values[:4] - truth).abs().max())
    assert regularised < 0.5 * unregularised


def test_recovers_the_gaussian_conditional_cdf_and_density(gaussian_fit):
    operator, _ = gaussian_fit
    data = LinearGaussian(sigma=0.5)
    x = torch.tensor([[-1.0], [0.0], [1.0]])
    grid = torch.linspace(-4.0, 4.0, 401)
    posterior = operator.condition(x)

    _, cdf = posterior.cdf(grid=grid)
    assert torch.all(kolmogorov_smirnov(cdf, data.conditional_cdf(x, grid)) < 0.1)

    # With the exact marginal in hand, the density needs no KDE.
    pdf = posterior.density(grid, data.marginal_pdf)
    assert torch.all(hellinger(pdf, data.conditional_pdf(x, grid), grid) < 0.2)


def test_quantiles_track_the_gaussian_truth(gaussian_fit):
    operator, _ = gaussian_fit
    data = LinearGaussian(sigma=0.5)
    x = torch.tensor([[-1.0], [0.0], [1.0]])
    levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    got = operator.condition(x).quantile(levels)
    expected = data.conditional_quantile(x, levels)
    assert torch.allclose(got, expected, atol=0.2)


def test_a_kde_marginal_gives_a_comparable_density(gaussian_fit):
    """Densities should not hinge on knowing the marginal analytically."""
    operator, _ = gaussian_fit
    data = LinearGaussian(sigma=0.5)
    x = torch.tensor([[-1.0], [0.0], [1.0]])
    grid = torch.linspace(-4.0, 4.0, 401)
    posterior = operator.condition(x)
    truth = data.conditional_pdf(x, grid)
    with_kde = hellinger(posterior.density(grid, GaussianKDE(operator.reference_y)), truth, grid)
    assert torch.all(with_kde < 0.25)


def test_rank_truncation_stays_valid_and_costs_accuracy(gaussian_fit):
    """Every truncation must still be a probability law, and fuller must be better.

    A learned singular direction is a mixture of the true ones at finite
    sample, so a low-rank conditional mean is shrunk towards the marginal
    rather than merely noisier -- keeping all directions is what recovers it.
    """
    operator, _ = gaussian_fit
    data = LinearGaussian(sigma=0.5)
    x = torch.tensor([[-1.0], [0.0], [1.0]])
    errors = {}
    for rank in (1, 4, 32):
        posterior = operator.condition(x, rank=rank)
        weights = posterior.weights
        assert torch.all(weights >= 0)
        assert torch.allclose(weights.sum(-1), torch.ones(3), atol=1e-4)
        errors[rank] = float((posterior.mean() - data.conditional_mean(x)).abs().mean())
    assert errors[32] < errors[4] < errors[1]
    assert errors[32] < 0.1


def test_conditional_intervals_are_calibrated_out_of_sample():
    operator, _ = _fit(Heteroscedastic(), n=6000, epochs=500, seed=5)
    data = Heteroscedastic()
    x, y = data.sample(3000, generator=torch.Generator().manual_seed(99))
    for alpha in (0.1, 0.2):
        intervals = operator.condition(x).interval(alpha)
        assert float(coverage(intervals, y)) == pytest.approx(1 - alpha, abs=0.05)


def test_interval_width_follows_the_heteroscedastic_spread():
    """Narrow intervals where the noise is small, wide where it is large."""
    operator, _ = _fit(Heteroscedastic(), n=6000, epochs=500, seed=5)
    x = torch.tensor([[0.0], [1.0], [2.0]])
    widths = operator.condition(x).interval(0.1)
    widths = (widths[:, 1] - widths[:, 0]).tolist()
    assert widths[0] < widths[1] < widths[2]
    # True conditional sd goes 0.2 -> 0.8 -> 1.4, a factor of 7 across the range.
    assert widths[2] / widths[0] > 3.0


def test_captures_bimodality_that_a_mean_and_variance_would_miss():
    operator, _ = _fit(BimodalMixture(), n=8000, epochs=600, latent_dim=48, seed=6)
    data = BimodalMixture()
    x = torch.zeros(1, 1)
    grid = torch.linspace(-3.0, 3.0, 601)
    pdf = operator.condition(x).density(grid, GaussianKDE(operator.reference_y))[0]

    interior = pdf[1:-1]
    peaks = ((interior > pdf[:-2]) & (interior > pdf[2:])).nonzero().flatten() + 1
    # Keep only peaks that are a real mode, not grid-scale ripple.
    peaks = peaks[pdf[peaks] > 0.25 * float(pdf.max())]
    assert peaks.numel() == 2, f"expected two modes, found {peaks.numel()}"
    assert float(grid[peaks[0]]) < -0.4 and float(grid[peaks[1]]) > 0.4
    # The valley between the modes must be much lower than the peaks.
    valley = pdf[int(peaks[0]) : int(peaks[1]) + 1].min()
    assert float(valley) < 0.3 * float(pdf[peaks].max())
    assert float(hellinger(pdf.reshape(1, -1), data.conditional_pdf(x, grid), grid)) < 0.3
