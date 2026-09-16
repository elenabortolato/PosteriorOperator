"""End-to-end accuracy of posterior functionals against exact posteriors.

Slower than the rest of the suite because each test trains an operator. These
pin the statistical claims the examples make, including the two that cut against
the method and should not be allowed to regress silently: that the signed
Eq. (2) estimator beats the clipped one on second moments, and that the
reported spread degrades as the posterior concentrates.
"""

import math

import pytest
import torch

from posterior_operator import PosteriorOperator
from posterior_operator.simulators import MA2, GaussianLinear, SumIdentified


def _fit(simulator, theta, data, rank=32, epochs=400, seed=0, **kw):
    torch.manual_seed(seed)
    operator = PosteriorOperator(
        theta_dim=simulator.theta_dim, data_dim=data.shape[1], rank=rank, layer_size=64
    )
    operator.fit(theta, data, epochs=epochs, lr=1e-3, seed=seed, **kw)
    return operator


# --------------------------------------------------------------------------- #
# Gaussian linear: the spectrum is known exactly
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def gaussian_fit():
    sim = GaussianLinear(theta_dim=3, data_dim=5, noise=0.5, seed=0)
    g = torch.Generator().manual_seed(1)
    theta, y = sim.sample_joint(20000, generator=g)
    operator = _fit(sim, theta, y, rank=32, epochs=500)
    _, y_obs = sim.sample_joint(300, generator=g)
    return sim, operator, y_obs


def test_recovers_the_maximal_correlation(gaussian_fit):
    sim, operator, _ = gaussian_fit
    exact = float(sim.canonical_correlations()[0])
    assert operator.maximal_correlation == pytest.approx(exact, abs=0.01)


def test_recovers_the_exact_hermite_spectrum(gaussian_fit):
    r"""The estimated spectrum should track prod_i rho_i^{a_i}, not the r* canonical rho."""
    sim, operator, _ = gaussian_fit
    exact, _ = sim.exact_spectrum(max_order=30)
    estimated = operator.singular_values[:6]
    assert torch.allclose(estimated, exact[:6], atol=0.05)
    # Specifically: the second singular value is rho_1^2, not rho_2.
    rho = sim.canonical_correlations()
    assert float(rho[0]) ** 2 > float(rho[1]), "test model must have rho_1^2 > rho_2"
    assert abs(float(estimated[1]) - float(rho[0]) ** 2) < abs(float(estimated[1]) - float(rho[1]))


def test_posterior_mean_beats_the_prior_baseline(gaussian_fit):
    sim, operator, y_obs = gaussian_fit
    exact = sim.posterior_mean(y_obs)
    estimated = operator.posterior(y_obs).mean()
    error = float(((estimated - exact) ** 2).mean().sqrt())
    baseline = float((exact**2).mean().sqrt())  # reporting the prior mean, zero
    assert error < 0.25 * baseline, f"RMSE {error:.3f} against a baseline of {baseline:.3f}"


def test_posterior_covariance_is_the_right_order_of_magnitude(gaussian_fit):
    sim, operator, y_obs = gaussian_fit
    exact = sim.posterior_cov()
    estimated = operator.posterior(y_obs).covariance().mean(0)
    ratio = torch.diagonal(estimated) / torch.diagonal(exact)
    assert torch.all(ratio > 0.5) and torch.all(ratio < 3.0), f"variance ratios {ratio.tolist()}"


def test_direct_regression_beats_the_operator_for_a_single_functional(gaussian_fit):
    """The sharp form of the remark that the operator trades accuracy for reuse.

    In this model E[Theta|Y] is linear in y, so least squares is essentially
    exact, while the operator has to reach the linear directions through a
    spectrum crowded with nonlinear ones.
    """
    sim, operator, y_obs = gaussian_fit
    g = torch.Generator().manual_seed(1)
    theta, y = sim.sample_joint(20000, generator=g)
    design = torch.cat([y, torch.ones(y.shape[0], 1)], dim=1)
    coefficients = torch.linalg.lstsq(design, theta).solution
    fitted = torch.cat([y_obs, torch.ones(y_obs.shape[0], 1)], dim=1) @ coefficients

    exact = sim.posterior_mean(y_obs)
    regression_error = float(((fitted - exact) ** 2).mean().sqrt())
    operator_error = float(((operator.posterior(y_obs).mean() - exact) ** 2).mean().sqrt())
    assert regression_error < 0.05
    assert regression_error < operator_error


# --------------------------------------------------------------------------- #
# Non-identifiability: the operator should say which direction is identified
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def sum_identified_fit():
    sim = SumIdentified(n_obs=10, noise=1.0)
    g = torch.Generator().manual_seed(2)
    theta, y = sim.sample_joint(20000, generator=g)
    operator = _fit(sim, theta, y, rank=24, epochs=400, seed=2)
    _, y_obs = sim.sample_joint(200, generator=g)
    return sim, operator, y_obs


def test_maximal_correlation_matches_the_closed_form(sum_identified_fit):
    sim, operator, _ = sum_identified_fit
    assert operator.maximal_correlation == pytest.approx(sim.maximal_correlation(), abs=0.01)


def test_leading_singular_function_recovers_the_identified_direction(sum_identified_fit):
    r"""v_1 should align with (1,1)/sqrt(2) without being told."""
    sim, operator, _ = sum_identified_fit
    g = torch.Generator().manual_seed(3)
    probe = sim.sample_prior(20000, generator=g)
    v1 = operator.singular_function_theta(probe)[:, 0]
    design = torch.cat([probe, torch.ones(probe.shape[0], 1)], dim=1)
    coefficients = torch.linalg.lstsq(design, v1.unsqueeze(-1)).solution[:2, 0]
    direction = coefficients / coefficients.norm()
    assert float((direction @ sim.identified_direction).abs()) > 0.99


def test_unidentified_direction_keeps_its_prior_spread(sum_identified_fit):
    sim, operator, y_obs = sum_identified_fit
    posterior = operator.posterior(y_obs)
    _, exact_cov = sim.posterior_mean_cov(y_obs)
    for vector in (sim.identified_direction, torch.tensor([1.0, -1.0]) / math.sqrt(2)):
        first = posterior.functional(lambda t, v=vector: (t @ v).unsqueeze(-1))
        second = posterior.functional(lambda t, v=vector: ((t @ v) ** 2).unsqueeze(-1))
        estimated_sd = float((second - first**2).clamp_min(0).sqrt().mean())
        exact_sd = math.sqrt(float(vector @ exact_cov @ vector))
        assert estimated_sd == pytest.approx(exact_sd, rel=0.15)


def test_maximal_correlation_approaches_one_as_noise_vanishes():
    """The compactness warning sign, tracked against the closed form."""
    previous = 0.0
    for noise in (1.0, 0.3):
        sim = SumIdentified(n_obs=10, noise=noise)
        g = torch.Generator().manual_seed(4)
        theta, y = sim.sample_joint(12000, generator=g)
        operator = _fit(sim, theta, y, rank=16, epochs=300, seed=4)
        assert operator.maximal_correlation == pytest.approx(sim.maximal_correlation(), abs=0.02)
        assert operator.maximal_correlation > previous
        previous = operator.maximal_correlation


# --------------------------------------------------------------------------- #
# MA(2): a non-Gaussian posterior with an exact reference
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def ma2_fit():
    raw = MA2(n_timesteps=50, summaries=False)
    summaries = MA2(n_timesteps=50, summaries=True, n_lags=3)
    g = torch.Generator().manual_seed(5)
    theta, series = raw.sample_joint(20000, generator=g)
    features = summaries.summarize(series)
    operator = _fit(summaries, theta, features, rank=32, epochs=400, seed=5)

    _, series_obs = raw.sample_joint(12, generator=g)
    references = [raw.grid_posterior(series_obs[i], resolution=110) for i in range(12)]
    return raw, summaries, operator, series_obs, references


def test_ma2_posterior_mean_tracks_the_grid_posterior(ma2_fit):
    _, summaries, operator, series_obs, references = ma2_fit
    posterior = operator.posterior(summaries.summarize(series_obs))
    exact = torch.stack([(w.unsqueeze(-1) * grid).sum(0) for grid, w in references])
    error = float((posterior.mean() - exact).abs().mean())
    prior_spread = float(posterior.theta.std(0).mean())
    assert error < 0.4 * prior_spread, f"mean error {error:.3f} vs prior spread {prior_spread:.3f}"


def test_ma2_event_probabilities_track_the_grid_posterior(ma2_fit):
    _, summaries, operator, series_obs, references = ma2_fit
    posterior = operator.posterior(summaries.summarize(series_obs))
    for coordinate in (0, 1):
        estimated = posterior.probability(lambda t, c=coordinate: t[:, c] > 0)
        exact = torch.stack([w @ (grid[:, coordinate] > 0).to(w.dtype) for grid, w in references])
        assert float((estimated - exact).abs().mean()) < 0.15


def test_ma2_credible_intervals_contain_the_exact_posterior_mean(ma2_fit):
    _, summaries, operator, series_obs, references = ma2_fit
    posterior = operator.posterior(summaries.summarize(series_obs))
    exact = torch.stack([(w.unsqueeze(-1) * grid).sum(0) for grid, w in references])
    interval = posterior.credible_interval(0.1, coordinate=0)
    inside = (exact[:, 0] >= interval[:, 0]) & (exact[:, 0] <= interval[:, 1])
    assert float(inside.to(torch.float32).mean()) > 0.9


def test_signed_estimator_beats_clipping_on_second_moments(ma2_fit):
    """Guards the design choice behind the signed default.

    Clipping discards the negative mass that carves probability away from the
    prior's tails, so it inflates the reported spread. This must not regress
    into a clipped default again.
    """
    _, summaries, operator, series_obs, references = ma2_fit
    features = summaries.summarize(series_obs)
    exact_sd = torch.stack(
        [
            ((w.unsqueeze(-1) * (grid - (w.unsqueeze(-1) * grid).sum(0)) ** 2).sum(0)).sqrt()
            for grid, w in references
        ]
    )
    signed_ratio = float((operator.posterior(features).std() / exact_sd).mean())
    clipped_ratio = float((operator.posterior(features, clip=True).std() / exact_sd).mean())
    assert signed_ratio > 1.0, "the low-rank posterior is expected to be over-dispersed"
    assert signed_ratio < clipped_ratio, f"signed {signed_ratio:.2f} should beat clipped {clipped_ratio:.2f}"


def test_over_dispersion_grows_with_posterior_concentration():
    """The rank-selection finding: informativeness is what strains the truncation."""
    ratios = {}
    for n_timesteps in (5, 50):
        raw = MA2(n_timesteps=n_timesteps, summaries=False)
        summaries = MA2(n_timesteps=n_timesteps, summaries=True, n_lags=3)
        g = torch.Generator().manual_seed(6)
        theta, series = raw.sample_joint(15000, generator=g)
        features = summaries.summarize(series)
        operator = _fit(summaries, theta, features, rank=32, epochs=300, seed=6)

        _, series_obs = raw.sample_joint(10, generator=g)
        exact_sd = []
        for i in range(10):
            grid, w = raw.grid_posterior(series_obs[i], resolution=100)
            mean = (w.unsqueeze(-1) * grid).sum(0)
            exact_sd.append(((w.unsqueeze(-1) * (grid - mean) ** 2).sum(0)).sqrt())
        exact_sd = torch.stack(exact_sd)
        estimated = operator.posterior(summaries.summarize(series_obs)).std()
        ratios[n_timesteps] = float((estimated / exact_sd).mean())

    assert ratios[50] > ratios[5], f"over-dispersion should worsen with information: {ratios}"
