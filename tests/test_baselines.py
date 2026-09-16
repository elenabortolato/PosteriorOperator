"""The comparison baselines: one-regression-per-functional, and NPE.

These have to be correct for the comparisons in the examples to mean anything,
so they are checked against the same closed-form posterior the operator is.
"""

import math

import pytest
import torch

from posterior_operator.baselines import DirectRegression, NeuralPosteriorEstimator
from posterior_operator.simulators import GaussianLinear


@pytest.fixture(scope="module")
def gaussian():
    sim = GaussianLinear(theta_dim=2, data_dim=3, noise=0.5, seed=0)
    g = torch.Generator().manual_seed(0)
    theta, y = sim.sample_joint(12000, generator=g)
    _, y_obs = sim.sample_joint(200, generator=g)
    return sim, theta, y, y_obs


# --------------------------------------------------------------------------- #
# DirectRegression
# --------------------------------------------------------------------------- #


def test_direct_regression_recovers_the_posterior_mean(gaussian):
    """f = identity, so the regression target IS the posterior mean."""
    sim, theta, y, y_obs = gaussian
    model = DirectRegression(data_dim=3, output_dim=2, layer_size=64)
    model.fit(y, theta, epochs=300, seed=0)
    exact = sim.posterior_mean(y_obs)
    error = float(((model.predict(y_obs) - exact) ** 2).mean().sqrt())
    baseline = float((exact**2).mean().sqrt())
    assert error < 0.1 * baseline, f"RMSE {error:.4f} against baseline {baseline:.4f}"


def test_direct_regression_handles_a_nonlinear_functional(gaussian):
    r"""f(theta) = theta_1^2: the target is E[theta_1^2 | y], not (E[theta_1|y])^2."""
    sim, theta, y, y_obs = gaussian
    model = DirectRegression(data_dim=3, output_dim=1, layer_size=64)
    model.fit(y, (theta[:, 0] ** 2).unsqueeze(-1), epochs=300, seed=0)
    # Exact second moment of a Gaussian posterior.
    exact = sim.posterior_mean(y_obs)[:, 0] ** 2 + sim.posterior_cov()[0, 0]
    predicted = model.predict(y_obs).reshape(-1)
    assert float((predicted - exact).abs().mean()) < 0.1
    # And it must differ from the squared posterior mean by the posterior variance.
    assert float((exact - sim.posterior_mean(y_obs)[:, 0] ** 2).mean()) == pytest.approx(
        float(sim.posterior_cov()[0, 0]), rel=1e-5
    )


def test_direct_regression_validates_shapes_and_fit_order():
    model = DirectRegression(data_dim=3, output_dim=2)
    with pytest.raises(RuntimeError, match="call fit"):
        model.predict(torch.zeros(4, 3))
    with pytest.raises(ValueError, match="expected output_dim=2"):
        model.fit(torch.zeros(40, 3), torch.zeros(40, 5), epochs=2)


def test_direct_regression_is_reproducible(gaussian):
    _, theta, y, y_obs = gaussian
    outputs = []
    for _ in range(2):
        model = DirectRegression(data_dim=3, output_dim=2, layer_size=32)
        model.fit(y[:2000], theta[:2000], epochs=30, seed=7)
        outputs.append(model.predict(y_obs[:5]))
    assert torch.allclose(outputs[0], outputs[1])


# --------------------------------------------------------------------------- #
# NeuralPosteriorEstimator
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def npe_fit(gaussian):
    sim, theta, y, y_obs = gaussian
    model = NeuralPosteriorEstimator(theta_dim=2, data_dim=3, n_components=8, layer_size=64)
    model.fit(theta, y, epochs=300, seed=0)
    return sim, model, y_obs


def test_npe_recovers_the_gaussian_posterior(npe_fit):
    sim, model, y_obs = npe_fit
    exact_mean, exact_cov = sim.posterior_mean(y_obs), sim.posterior_cov()
    assert float(((model.mean(y_obs) - exact_mean) ** 2).mean().sqrt()) < 0.05
    assert torch.allclose(model.covariance(y_obs).mean(0), exact_cov, atol=0.02)


def test_npe_density_integrates_to_one(npe_fit):
    """A real density in natural parameter units, Jacobian included."""
    _, model, y_obs = npe_fit
    axis = torch.linspace(-5, 5, 161)
    grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)
    log_density = model.log_prob(y_obs[:1].expand(grid.shape[0], 3), grid)
    cell = float((axis[1] - axis[0]) ** 2)
    assert float(log_density.exp().sum() * cell) == pytest.approx(1.0, abs=0.02)


def test_npe_moments_agree_between_closed_form_and_sampling(npe_fit):
    """The analytic mixture moments must match Monte Carlo from the same fit."""
    _, model, y_obs = npe_fit
    subset = y_obs[:20]
    g = torch.Generator().manual_seed(1)
    draws = model.sample(subset, 60000, generator=g)
    assert draws.shape == (20, 60000, 2)
    assert torch.allclose(draws.mean(dim=1), model.mean(subset), atol=0.02)
    centred = draws - draws.mean(dim=1, keepdim=True)
    empirical_cov = torch.einsum("nsi,nsj->nij", centred, centred) / draws.shape[1]
    assert torch.allclose(empirical_cov, model.covariance(subset), atol=0.03)


def test_npe_functional_matches_its_analytic_mean(npe_fit):
    _, model, y_obs = npe_fit
    g = torch.Generator().manual_seed(2)
    estimate = model.functional(y_obs[:20], lambda t: t, n_samples=60000, generator=g)
    assert torch.allclose(estimate, model.mean(y_obs[:20]), atol=0.02)


def test_npe_quantiles_are_ordered_and_bracket_the_mean(npe_fit):
    _, model, y_obs = npe_fit
    g = torch.Generator().manual_seed(3)
    quantiles = model.quantile(y_obs[:20], [0.05, 0.5, 0.95], coordinate=0, generator=g)
    assert quantiles.shape == (20, 3)
    assert torch.all(quantiles[:, 0] < quantiles[:, 1])
    assert torch.all(quantiles[:, 1] < quantiles[:, 2])
    mean = model.mean(y_obs[:20])[:, 0]
    assert torch.all(quantiles[:, 0] < mean) and torch.all(mean < quantiles[:, 2])


def test_npe_quantiles_match_the_gaussian_truth(npe_fit):
    sim, model, y_obs = npe_fit
    g = torch.Generator().manual_seed(4)
    levels = [0.1, 0.5, 0.9]
    estimated = model.quantile(y_obs[:40], levels, coordinate=0, generator=g)
    sd = math.sqrt(float(sim.posterior_cov()[0, 0]))
    z = math.sqrt(2) * torch.erfinv(2 * torch.tensor(levels) - 1)
    exact = sim.posterior_mean(y_obs[:40])[:, :1] + sd * z.reshape(1, -1)
    assert float((estimated - exact).abs().mean()) < 0.1


def test_npe_requires_fitting_first():
    model = NeuralPosteriorEstimator(theta_dim=2, data_dim=3)
    with pytest.raises(RuntimeError, match="call fit"):
        model.mean(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="expected data_dim=3"):
        model.fit(torch.zeros(60, 2), torch.zeros(60, 4), epochs=2)


def test_npe_mixture_weights_are_normalised(npe_fit):
    _, model, y_obs = npe_fit
    log_weights, means, scales = model._mixture(model._prepare(y_obs[:10]))
    assert torch.allclose(log_weights.exp().sum(-1), torch.ones(10), atol=1e-6)
    assert means.shape == (10, model.n_components, 2)
    assert torch.all(scales >= model.min_scale)


def test_npe_beats_the_prior_baseline_on_a_probability(npe_fit):
    sim, model, y_obs = npe_fit
    g = torch.Generator().manual_seed(5)

    def region(t):
        return (t[:, 0] > 0).to(t.dtype).unsqueeze(-1)

    estimated = model.functional(y_obs, region, n_samples=40000, generator=g).reshape(-1)
    # Exact P(theta_1 > 0 | y) for a Gaussian posterior.
    sd = math.sqrt(float(sim.posterior_cov()[0, 0]))
    exact = 0.5 * (1 + torch.erf(sim.posterior_mean(y_obs)[:, 0] / (sd * math.sqrt(2))))
    assert float((estimated - exact).abs().mean()) < 0.05
