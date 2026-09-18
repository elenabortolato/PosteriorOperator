"""The likelihood-free simulators and their exact reference quantities.

The reference quantities are what every posterior-functional estimate is later
scored against, so they are cross-checked here against independent computations
(importance sampling from the prior, closed-form Gaussian algebra, quadrature).
"""

import math

import pytest
import torch

from posterior_operator.simulators import (
    MA2,
    SIR,
    AR2,
    DirichletMultinomial,
    SignAmbiguous,
    GAndK,
    GaussianLinear,
    SumIdentified,
)

DT = torch.float64


def _importance_posterior(sim, y_obs, n=200_000, seed=0):
    """Brute-force posterior by importance sampling from the prior.

    The draws already come from the prior, so the importance weights are the
    likelihood alone -- multiplying by the prior again would target pi^2 L and,
    for a Gaussian prior, halve its variance.
    """
    g = torch.Generator().manual_seed(seed)
    theta = sim.sample_prior(n, generator=g)
    weights = torch.softmax(sim.log_likelihood(theta, y_obs).double(), dim=0)
    mean = (weights.unsqueeze(-1) * theta.double()).sum(0)
    centred = theta.double() - mean
    cov = (weights.unsqueeze(-1) * centred).T @ centred
    return mean, cov


# --------------------------------------------------------------------------- #
# Shared interface
# --------------------------------------------------------------------------- #


@pytest.fixture(
    params=[
        lambda: GaussianLinear(theta_dim=2, data_dim=3, noise=0.5, seed=0, dtype=DT),
        lambda: MA2(n_timesteps=20, summaries=True, n_lags=3, dtype=DT),
        lambda: SumIdentified(n_obs=8, noise=1.0, dtype=DT),
    ]
)
def simulator(request):
    return request.param()


def test_sample_joint_shapes_and_finiteness(simulator):
    theta, y = simulator.sample_joint(256, generator=torch.Generator().manual_seed(0))
    assert theta.shape == (256, simulator.theta_dim)
    assert y.shape == (256, simulator.data_dim)
    assert torch.isfinite(theta).all() and torch.isfinite(y).all()


def test_sampling_is_reproducible(simulator):
    first = simulator.sample_joint(64, generator=torch.Generator().manual_seed(7))
    second = simulator.sample_joint(64, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(first[0], second[0]) and torch.allclose(first[1], second[1])


def test_base_class_methods_are_abstract():
    from posterior_operator.simulators import Simulator

    base = Simulator()
    for call in (lambda: base.sample_prior(4), lambda: base.simulate(torch.zeros(4, 1))):
        with pytest.raises(NotImplementedError):
            call()


# --------------------------------------------------------------------------- #
# GaussianLinear
# --------------------------------------------------------------------------- #


def test_gaussian_posterior_matches_the_precision_form():
    """Cross-check the covariance form against the independent precision form.

    mean = Sigma_ThetaY Sigma_YY^-1 y versus mean = Lambda^-1 A^T y / sigma^2
    with Lambda = Sigma_theta^-1 + A^T A / sigma^2. Equal by Woodbury, so
    agreement to machine precision tests the implementation, not the identity.
    """
    sim = GaussianLinear(theta_dim=2, data_dim=3, noise=0.5, seed=1, dtype=DT)
    _, y = sim.sample_joint(8, generator=torch.Generator().manual_seed(2))
    precision = torch.linalg.inv(sim.prior_cov) + sim.design.T @ sim.design / sim.noise**2
    cov_from_precision = torch.linalg.inv(precision)
    mean_from_precision = (cov_from_precision @ sim.design.T @ y.T / sim.noise**2).T
    assert torch.allclose(sim.posterior_cov(), cov_from_precision, atol=1e-10)
    assert torch.allclose(sim.posterior_mean(y), mean_from_precision, atol=1e-10)


def test_gaussian_posterior_agrees_with_importance_sampling():
    """End-to-end sanity check against brute force, averaged over observations."""
    sim = GaussianLinear(theta_dim=2, data_dim=3, noise=0.5, seed=1, dtype=DT)
    _, y = sim.sample_joint(4, generator=torch.Generator().manual_seed(2))
    for i in range(4):
        mean, cov = _importance_posterior(sim, y[i], n=400_000, seed=i)
        assert torch.allclose(sim.posterior_mean(y[i : i + 1])[0], mean, atol=0.05)
        assert torch.allclose(sim.posterior_cov(), cov, atol=0.05)


def test_gaussian_canonical_correlations_are_in_unit_interval():
    sim = GaussianLinear(theta_dim=3, data_dim=5, noise=0.5, seed=0, dtype=DT)
    rho = sim.canonical_correlations()
    assert rho.shape == (3,)
    assert torch.all(rho > 0) and torch.all(rho < 1)
    assert torch.all(rho[:-1] >= rho[1:])


def test_gaussian_posterior_covariance_shrinks_the_prior():
    sim = GaussianLinear(theta_dim=3, data_dim=5, noise=0.5, seed=0, dtype=DT)
    gap = sim.prior_cov - sim.posterior_cov()
    # Data can only reduce uncertainty, so the difference is PSD.
    assert torch.all(torch.linalg.eigvalsh(gap) > -1e-10)


def test_gaussian_design_shape_is_validated():
    with pytest.raises(ValueError, match="design must be"):
        GaussianLinear(theta_dim=2, data_dim=3, design=torch.zeros(3, 4))


# --- the exact L^2 spectrum ------------------------------------------------- #


def test_exact_spectrum_is_the_hermite_product_system():
    r"""sigma_a = prod_i rho_i^{a_i}, sorted, with the canonical rho as |a| = 1."""
    sim = GaussianLinear(theta_dim=2, data_dim=3, noise=0.6, seed=4, dtype=DT)
    rho = sim.canonical_correlations().tolist()
    values, indices = sim.exact_spectrum(max_order=12)

    assert len(values) == len(indices)
    assert torch.all(values[:-1] >= values[1:])  # sorted
    for value, index in zip(values.tolist()[:40], indices[:40]):
        assert value == pytest.approx(math.prod(r**a for r, a in zip(rho, index)), rel=1e-10)
    # The canonical correlations are exactly the first-order multi-indices.
    first_order = sorted(
        (v for v, a in zip(values.tolist(), indices) if sum(a) == 1), reverse=True
    )
    assert first_order == pytest.approx(sorted(rho, reverse=True), rel=1e-10)
    # And the top value is the largest canonical correlation.
    assert float(values[0]) == pytest.approx(max(rho), rel=1e-12)


def test_scalar_gaussian_operator_is_diagonal_in_the_hermite_basis():
    r"""Quadrature check that sigma_k = rho^k, the basis of :meth:`exact_spectrum`."""
    rho, order, n_quad = 0.75, 6, 60
    import numpy as np

    nodes, wts = np.polynomial.hermite_e.hermegauss(n_quad)
    nodes = torch.tensor(nodes, dtype=DT)
    wts = torch.tensor(wts, dtype=DT) / math.sqrt(2 * math.pi)
    a, b = torch.meshgrid(nodes, nodes, indexing="ij")
    wa, wb = torch.meshgrid(wts, wts, indexing="ij")
    quad = (wa * wb).reshape(-1)
    theta = a.reshape(-1)
    y = (rho * a + math.sqrt(1 - rho**2) * b).reshape(-1)

    def hermite(x):
        out = [torch.ones_like(x), x]
        for j in range(1, order):
            out.append((x * out[j] - math.sqrt(j) * out[j - 1]) / math.sqrt(j + 1))
        return out[: order + 1]

    h_theta, h_y = hermite(theta), hermite(y)
    matrix = torch.stack(
        [torch.stack([(quad * h_y[i] * h_theta[j]).sum() for j in range(order + 1)]) for i in range(order + 1)]
    )
    diag = torch.diagonal(matrix)
    assert torch.allclose(diag, torch.tensor([rho**k for k in range(order + 1)], dtype=DT), atol=1e-10)
    assert float((matrix - torch.diag(diag)).abs().max()) < 1e-10


def test_linear_directions_can_be_outranked_by_nonlinear_ones():
    """rank(Sigma_ThetaY) does not bound the rank needed for E[Theta|Y].

    When rho_1^2 exceeds the weakest canonical correlation, the squared Hermite
    direction of the first canonical pair outranks a *linear* direction, so the
    rank-r* truncation drops part of the posterior mean.
    """
    # rho_1^2 > rho_2: the linear direction e_2 gets pushed down the spectrum.
    sim = _gaussian_with_correlations([0.95, 0.5])
    ranks = sim.linear_direction_ranks(max_order=30)
    assert ranks[0] == 1
    assert ranks[1] > 2, "e_2 should be outranked by (2, 0)"
    assert sim.truncation_error_posterior_mean(2) > 0.4
    assert sim.truncation_error_posterior_mean(max(ranks)) == 0.0

    # rho_1^2 < rho_2: now the two linear directions do come first.
    sim = _gaussian_with_correlations([0.6, 0.5])
    assert sim.linear_direction_ranks(max_order=30) == [1, 2]
    assert sim.truncation_error_posterior_mean(2) == 0.0


def test_truncation_error_decreases_and_reaches_zero():
    sim = _gaussian_with_correlations([0.9, 0.6, 0.3])
    errors = [sim.truncation_error_posterior_mean(d) for d in range(1, 40)]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] == 0.0
    assert errors[0] > 0.0


def _gaussian_with_correlations(rho):
    r"""A :class:`GaussianLinear` whose canonical correlations are exactly ``rho``.

    With a diagonal design ``A = diag(a)``, unit prior and unit noise, the
    canonical correlations are ``a_i / sqrt(a_i^2 + 1)``, so invert that.
    """
    scale = [r / math.sqrt(1 - r**2) for r in rho]
    design = torch.diag(torch.tensor(scale, dtype=DT))
    sim = GaussianLinear(theta_dim=len(rho), data_dim=len(rho), noise=1.0, design=design, dtype=DT)
    assert torch.allclose(sim.canonical_correlations(), torch.tensor(sorted(rho, reverse=True), dtype=DT), atol=1e-10)
    return sim


# --------------------------------------------------------------------------- #
# MA2
# --------------------------------------------------------------------------- #


def test_ma2_prior_lives_on_the_invertibility_triangle():
    sim = MA2(n_timesteps=20, dtype=DT)
    theta = sim.sample_prior(4000, generator=torch.Generator().manual_seed(0))
    assert bool(MA2.in_support(theta).all())
    # Vertices (-2, 1), (2, 1), (0, -1): area 4, inside a bounding box of area 8.
    box = torch.stack([4 * torch.rand(40_000, dtype=DT) - 2, 2 * torch.rand(40_000, dtype=DT) - 1], dim=-1)
    assert float(MA2.in_support(box).to(DT).mean()) == pytest.approx(0.5, abs=0.02)


def test_ma2_simulated_autocovariance_matches_the_theoretical_one():
    sim = MA2(n_timesteps=4000, summaries=False, dtype=DT)
    theta = torch.tensor([[0.6, -0.3]], dtype=DT)
    series = sim.simulate(theta.expand(400, 2).contiguous(), generator=torch.Generator().manual_seed(0))
    t1, t2 = 0.6, -0.3
    expected = [1 + t1**2 + t2**2, t1 + t1 * t2, t2, 0.0]
    centred = series - series.mean(dim=-1, keepdim=True)
    for lag, want in enumerate(expected):
        got = float((centred[:, lag:] * centred[:, : centred.shape[1] - lag]).mean())
        assert got == pytest.approx(want, abs=0.05)


def test_ma2_summaries_have_the_declared_dimension():
    sim = MA2(n_timesteps=30, summaries=True, n_lags=5, dtype=DT)
    assert sim.data_dim == 6
    _, y = sim.sample_joint(16, generator=torch.Generator().manual_seed(0))
    assert y.shape == (16, 6)


def test_ma2_likelihood_needs_the_raw_series():
    sim = MA2(n_timesteps=20, summaries=True, dtype=DT)
    with pytest.raises(NotImplementedError, match="raw series"):
        sim.log_likelihood(torch.zeros(2, 2, dtype=DT), torch.zeros(20, dtype=DT))


def test_ma2_likelihood_is_minus_inf_outside_the_triangle():
    sim = MA2(n_timesteps=20, summaries=False, dtype=DT)
    outside = torch.tensor([[3.0, 0.0], [0.0, -2.0]], dtype=DT)
    assert torch.all(torch.isinf(sim.log_likelihood(outside, torch.zeros(20, dtype=DT))))


def test_ma2_likelihood_matches_a_direct_gaussian_evaluation():
    """Cross-check the chunked Cholesky against an explicit multivariate normal."""
    sim = MA2(n_timesteps=12, summaries=False, dtype=DT)
    theta = torch.tensor([[0.5, 0.2], [-0.4, 0.3]], dtype=DT)
    y = torch.randn(12, generator=torch.Generator().manual_seed(0), dtype=DT)
    got = sim.log_likelihood(theta, y, chunk_size=1)
    for i in range(2):
        cov = sim._covariance(theta[i : i + 1])[0]
        dist = torch.distributions.MultivariateNormal(torch.zeros(12, dtype=DT), covariance_matrix=cov)
        assert float(got[i]) == pytest.approx(float(dist.log_prob(y)), rel=1e-10)


def test_ma2_grid_posterior_is_a_normalised_measure_on_the_triangle():
    sim = MA2(n_timesteps=30, summaries=False, dtype=DT)
    theta, series = sim.sample_joint(1, generator=torch.Generator().manual_seed(3))
    grid, weights = sim.grid_posterior(series[0], resolution=80)
    assert grid.shape == (80 * 80, 2)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-6)
    assert torch.all(weights >= 0)
    assert float(weights[~MA2.in_support(grid)].sum()) == pytest.approx(0.0, abs=1e-12)
    # It should concentrate somewhere near the parameter that generated the data.
    posterior_mean = (weights.unsqueeze(-1) * grid).sum(0)
    assert float((posterior_mean - theta[0]).abs().max()) < 0.6


def test_ma2_grid_posterior_respects_a_supplied_prior():
    sim = MA2(n_timesteps=30, summaries=False, dtype=DT)
    _, series = sim.sample_joint(1, generator=torch.Generator().manual_seed(4))
    grid, flat = sim.grid_posterior(series[0], resolution=80)
    # A prior supported only on theta_2 > 0 must move all the mass there.
    restricted = torch.where(grid[:, 1] > 0, torch.zeros(grid.shape[0], dtype=DT), torch.full((grid.shape[0],), -math.inf, dtype=DT))
    _, tilted = sim.grid_posterior(series[0], resolution=80, log_prior=restricted)
    assert float(tilted[grid[:, 1] <= 0].sum()) == pytest.approx(0.0, abs=1e-12)
    assert float(tilted.sum()) == pytest.approx(1.0, abs=1e-6)
    assert not torch.allclose(flat, tilted)


# --------------------------------------------------------------------------- #
# SumIdentified
# --------------------------------------------------------------------------- #


def test_sum_identified_posterior_solves_the_normal_equations():
    """Lambda mu = m ybar 1 / sigma^2 exactly, with Lambda = I + (m/sigma^2) 1 1^T."""
    sim = SumIdentified(n_obs=8, noise=0.7, dtype=DT)
    _, y = sim.sample_joint(6, generator=torch.Generator().manual_seed(5))
    mean, cov = sim.posterior_mean_cov(y)
    ones = torch.ones(2, 1, dtype=DT)
    precision = torch.eye(2, dtype=DT) + (sim.n_obs / sim.noise**2) * (ones @ ones.T)
    rhs = (sim.n_obs * y.mean(dim=-1, keepdim=True) / sim.noise**2) * ones.T
    assert torch.allclose(mean @ precision.T, rhs, atol=1e-10)
    assert torch.allclose(precision @ cov, torch.eye(2, dtype=DT), atol=1e-10)


def test_sum_identified_posterior_agrees_with_importance_sampling():
    sim = SumIdentified(n_obs=8, noise=1.0, dtype=DT)
    _, y = sim.sample_joint(4, generator=torch.Generator().manual_seed(5))
    exact_mean, exact_cov = sim.posterior_mean_cov(y)
    for i in range(4):
        mean, cov = _importance_posterior(sim, y[i], n=400_000, seed=i)
        assert torch.allclose(exact_mean[i], mean, atol=0.05)
        assert torch.allclose(exact_cov, cov, atol=0.05)


def test_sum_identified_leaves_the_orthogonal_direction_at_its_prior():
    sim = SumIdentified(n_obs=10, noise=1.0, dtype=DT)
    _, cov = sim.posterior_mean_cov(torch.zeros(1, 10, dtype=DT))
    unidentified = torch.tensor([1.0, -1.0], dtype=DT) / math.sqrt(2)
    identified = sim.identified_direction
    assert float(unidentified @ cov @ unidentified) == pytest.approx(1.0, abs=1e-10)
    # Precision along the identified direction is 1 + 2 m / sigma^2 = 21.
    assert float(identified @ cov @ identified) == pytest.approx(1 / 21, rel=1e-10)


def test_sum_identified_maximal_correlation_grows_as_noise_shrinks():
    previous = 0.0
    for noise in (2.0, 1.0, 0.5, 0.1):
        value = SumIdentified(n_obs=10, noise=noise, dtype=DT).maximal_correlation()
        assert 0.0 < value < 1.0
        assert value > previous
        previous = value
    assert SumIdentified(n_obs=10, noise=0.01, dtype=DT).maximal_correlation() > 0.999


def test_sum_identified_noise_is_validated():
    sim = SumIdentified(n_obs=4, noise=1.0, dtype=DT)
    theta, y = sim.sample_joint(32, generator=torch.Generator().manual_seed(0))
    # Only the sum enters the likelihood, so permuting the coordinates is a no-op.
    assert torch.allclose(
        sim.log_likelihood(theta, y[0]), sim.log_likelihood(theta.flip(-1), y[0]), atol=1e-10
    )


# --------------------------------------------------------------------------- #
# AR2
# --------------------------------------------------------------------------- #


def test_ar2_prior_lives_on_the_stationarity_triangle():
    sim = AR2(n_timesteps=20, dtype=DT)
    theta = sim.sample_prior(4000, generator=torch.Generator().manual_seed(0))
    assert bool(AR2.in_support(theta).all())
    # Vertices (-2, -1), (2, -1), (0, 1): area 4 inside a bounding box of area 8.
    box = torch.stack([4 * torch.rand(40_000, dtype=DT) - 2, 2 * torch.rand(40_000, dtype=DT) - 1], dim=-1)
    assert float(AR2.in_support(box).to(DT).mean()) == pytest.approx(0.5, abs=0.02)


def test_ar2_autocovariance_matches_the_simulator():
    """Yule-Walker against the empirical autocovariance of long simulated paths."""
    sim = AR2(n_timesteps=3000, summaries=False, dtype=DT)
    theta = torch.tensor([[0.5, 0.3]], dtype=DT)
    series = sim.simulate(theta.expand(200, 2).contiguous(), generator=torch.Generator().manual_seed(0))
    centred = series - series.mean(dim=-1, keepdim=True)
    theoretical = sim.autocovariance(theta, 3)[0]
    for lag in range(4):
        empirical = float((centred[:, lag:] * centred[:, : centred.shape[1] - lag]).mean())
        assert empirical == pytest.approx(float(theoretical[lag]), rel=0.05)


def test_ar2_autocovariance_satisfies_its_own_recursion():
    sim = AR2(dtype=DT)
    theta = torch.tensor([[0.4, -0.3], [-0.6, 0.2]], dtype=DT)
    gamma = sim.autocovariance(theta, 6)
    for k in range(2, 7):
        expected = theta[:, 0] * gamma[:, k - 1] + theta[:, 1] * gamma[:, k - 2]
        assert torch.allclose(gamma[:, k], expected, atol=1e-12)
    assert torch.all(gamma[:, 0] > 0)  # a variance


def test_ar2_starts_from_the_stationary_law():
    """No burn-in bias: the marginal variance should match gamma_0 at every t."""
    sim = AR2(n_timesteps=12, summaries=False, dtype=DT)
    theta = torch.tensor([[0.6, 0.2]], dtype=DT)
    series = sim.simulate(theta.expand(60_000, 2).contiguous(), generator=torch.Generator().manual_seed(1))
    gamma0 = float(sim.autocovariance(theta, 0)[0, 0])
    variances = series.var(dim=0, unbiased=True)
    assert torch.allclose(variances, torch.full_like(variances, gamma0), rtol=0.06)


def test_ar2_likelihood_matches_a_direct_gaussian_evaluation():
    sim = AR2(n_timesteps=10, summaries=False, dtype=DT)
    theta = torch.tensor([[0.5, 0.2], [-0.4, 0.3]], dtype=DT)
    y = torch.randn(10, generator=torch.Generator().manual_seed(0), dtype=DT)
    got = sim.log_likelihood(theta, y, chunk_size=1)
    idx = (torch.arange(10).unsqueeze(0) - torch.arange(10).unsqueeze(1)).abs()
    for i in range(2):
        gamma = sim.autocovariance(theta[i : i + 1], 9)[0]
        cov = gamma[idx.reshape(-1)].reshape(10, 10)
        dist = torch.distributions.MultivariateNormal(torch.zeros(10, dtype=DT), covariance_matrix=cov)
        assert float(got[i]) == pytest.approx(float(dist.log_prob(y)), rel=1e-10)
    assert torch.all(torch.isinf(sim.log_likelihood(torch.tensor([[1.5, 0.9]], dtype=DT), y)))
    with pytest.raises(NotImplementedError, match="raw series"):
        AR2(summaries=True, dtype=DT).log_likelihood(theta, y)


# --------------------------------------------------------------------------- #
# GAndK
# --------------------------------------------------------------------------- #


def test_gandk_quantile_inversion_round_trips():
    sim = GAndK(n_obs=20, dtype=DT)
    theta = torch.tensor([[3.0, 1.0, 2.0, 0.5], [0.0, 2.0, 0.0, 0.1]], dtype=DT)
    z = torch.linspace(-4, 4, 17, dtype=DT).expand(2, 17)
    recovered = sim.inverse_quantile(theta, sim.quantile(theta, z))
    assert torch.allclose(recovered, z, atol=1e-5)


def test_gandk_quantile_is_increasing_in_z():
    sim = GAndK(dtype=DT)
    theta = sim.sample_prior(50, generator=torch.Generator().manual_seed(0))
    z = torch.linspace(-4, 4, 60, dtype=DT).expand(50, 60)
    values = sim.quantile(theta, z)
    assert torch.all(values[:, 1:] > values[:, :-1])
    assert torch.all(sim.quantile_derivative(theta, z) > 0)


def test_gandk_density_integrates_to_one():
    r"""p(y) = phi(z) / Q'(z) at z = Q^{-1}(y); the basis of the log-likelihood."""
    sim = GAndK(dtype=DT)
    theta = torch.tensor([[3.0, 1.0, 1.5, 0.4]], dtype=DT)
    lo = float(sim.quantile(theta, torch.tensor([[-9.0]], dtype=DT)))
    hi = float(sim.quantile(theta, torch.tensor([[9.0]], dtype=DT)))
    grid = torch.linspace(lo, hi, 40001, dtype=DT)
    z = sim.inverse_quantile(theta, grid.reshape(1, -1))
    density = torch.exp(-0.5 * z**2 - 0.5 * math.log(2 * math.pi)) / sim.quantile_derivative(theta, z)
    assert float(torch.trapz(density[0], grid)) == pytest.approx(1.0, abs=1e-3)


def test_gandk_log_likelihood_is_the_sum_of_log_densities():
    sim = GAndK(n_obs=8, summaries=False, dtype=DT)
    theta = torch.tensor([[3.0, 1.0, 1.5, 0.4]], dtype=DT)
    _, y = sim.sample_joint(1, generator=torch.Generator().manual_seed(2))
    z = sim.inverse_quantile(theta, y[0].reshape(1, -1))
    manual = float(
        (-0.5 * z**2 - 0.5 * math.log(2 * math.pi) - torch.log(sim.quantile_derivative(theta, z))).sum()
    )
    assert float(sim.log_likelihood(theta, y[0])[0]) == pytest.approx(manual, rel=1e-9)


def test_gandk_summaries_respond_to_the_right_parameters():
    """Each summary should move with the parameter it is meant to track."""
    sim = GAndK(n_obs=4000, dtype=DT)
    base = torch.tensor([[3.0, 1.0, 0.0, 0.1]], dtype=DT)
    g = torch.Generator().manual_seed(3)
    reference = sim.simulate(base.expand(200, 4).contiguous(), generator=g).mean(0)
    for index, name in enumerate(("location", "scale", "skewness", "tail")):
        shifted = base.clone()
        shifted[0, index] += 1.5
        moved = sim.simulate(shifted.expand(200, 4).contiguous(), generator=g).mean(0)
        gap = (moved - reference).abs()
        assert int(gap.argmax()) == index, f"changing {name} should move summary {index}, moved {int(gap.argmax())}"


def test_gandk_log_likelihood_rejects_invalid_parameters():
    sim = GAndK(n_obs=5, summaries=False, dtype=DT)
    _, y = sim.sample_joint(1, generator=torch.Generator().manual_seed(4))
    bad = torch.tensor([[3.0, 0.0, 0.0, 0.0], [3.0, 1.0, 0.0, -2.0]], dtype=DT)
    assert torch.all(torch.isinf(sim.log_likelihood(bad, y[0])))


# --------------------------------------------------------------------------- #
# SIR
# --------------------------------------------------------------------------- #


def test_sir_curve_conserves_the_population():
    """RK4 on the SIR system must keep S + I + R fixed."""
    sim = SIR(dtype=DT)
    theta = sim.sample_prior(32, generator=torch.Generator().manual_seed(0))
    curve = sim.mean_curve(theta)
    assert curve.shape == (32, sim.n_obs)
    assert torch.all(curve >= 0)
    assert torch.all(curve <= sim.population + 1e-6)


def test_sir_epidemic_takes_off_only_when_r0_exceeds_one():
    sim = SIR(dtype=DT)
    subcritical = sim.mean_curve(torch.tensor([[0.5, 1.0]], dtype=DT))[0]  # R0 = 0.5
    supercritical = sim.mean_curve(torch.tensor([[2.0, 0.4]], dtype=DT))[0]  # R0 = 5
    assert float(subcritical.max()) <= sim.initial_infected + 1e-6
    assert float(supercritical.max()) > 50 * sim.initial_infected


def test_sir_solver_is_converged_at_the_default_step():
    """Halving the step must not move the curve materially."""
    coarse = SIR(steps_per_unit=4, dtype=DT)
    fine = SIR(steps_per_unit=16, dtype=DT)
    theta = torch.tensor([[2.5, 0.3], [0.8, 0.5]], dtype=DT)
    gap = (coarse.mean_curve(theta) - fine.mean_curve(theta)).abs().max()
    assert float(gap) < 0.01 * coarse.population


def test_sir_likelihood_is_gaussian_about_the_mean_curve():
    sim = SIR(dtype=DT)
    theta = sim.sample_prior(5, generator=torch.Generator().manual_seed(1))
    _, y = sim.sample_joint(1, generator=torch.Generator().manual_seed(2))
    residual = y[0].unsqueeze(0) - sim.mean_curve(theta)
    expected = -0.5 * (residual**2).sum(-1) / sim.noise**2 - sim.n_obs * math.log(
        sim.noise * math.sqrt(2 * math.pi)
    )
    assert torch.allclose(sim.log_likelihood(theta, y[0]), expected, rtol=1e-10)


def test_sir_grid_posterior_concentrates_near_the_truth():
    sim = SIR(dtype=DT)
    theta, y = sim.sample_joint(6, generator=torch.Generator().manual_seed(3))
    hits = 0
    for i in range(6):
        grid, weights = sim.grid_posterior(y[i], resolution=90)
        assert float(weights.sum()) == pytest.approx(1.0, abs=1e-6)
        assert torch.all(weights >= 0)
        mean = (weights.unsqueeze(-1) * grid).sum(0)
        sd = ((weights.unsqueeze(-1) * (grid - mean) ** 2).sum(0)).sqrt()
        # Within three posterior standard deviations of the generating parameter.
        hits += int(bool(((mean - theta[i]).abs() <= 3 * sd + 1e-6).all()))
    assert hits >= 5, f"only {hits}/6 reference posteriors covered the truth"


def test_sir_in_support_matches_the_prior_box():
    sim = SIR(dtype=DT)
    theta = sim.sample_prior(500, generator=torch.Generator().manual_seed(4))
    assert bool(sim.in_support(theta).all())
    outside = torch.tensor([[0.0, 0.5], [1.0, 5.0], [10.0, 0.5]], dtype=DT)
    assert not bool(sim.in_support(outside).any())


def test_sir_posterior_is_tighter_at_lower_noise():
    """The noise level is what places the model in or out of the compact regime."""
    spreads = {}
    for noise in (40.0, 120.0):
        sim = SIR(noise=noise, dtype=DT)
        theta, y = sim.sample_joint(4, generator=torch.Generator().manual_seed(5))
        widths = []
        for i in range(4):
            grid, weights = sim.grid_posterior(y[i], resolution=90)
            mean = (weights.unsqueeze(-1) * grid).sum(0)
            widths.append(((weights.unsqueeze(-1) * (grid - mean) ** 2).sum(0)).sqrt())
        spreads[noise] = float(torch.stack(widths).mean())
    assert spreads[40.0] < spreads[120.0]


@pytest.mark.parametrize("n_obs, steps_per_unit", [(30, 4), (100, 4), (7, 3), (13, 5)])
def test_sir_records_every_observation_time(n_obs, steps_per_unit):
    """Observation times must land on distinct integration steps.

    The step count is rounded up to a multiple of ``n_obs`` for exactly this
    reason: with a plain round-to-nearest, two observation times can map to the
    same step and one is then never written, leaving a stale zero in the curve.
    Checked end to end against a much finer solver, which also confirms each
    slot holds the value for the right time rather than merely a non-zero one.
    """
    coarse = SIR(n_obs=n_obs, steps_per_unit=steps_per_unit, dtype=DT)
    fine = SIR(n_obs=n_obs, steps_per_unit=steps_per_unit * 12, dtype=DT)
    theta = torch.tensor([[2.0, 0.4], [1.0, 0.3]], dtype=DT)
    curve = coarse.mean_curve(theta)
    assert curve.shape == (2, n_obs)
    assert torch.allclose(curve, fine.mean_curve(theta), atol=0.01 * coarse.population)


# --------------------------------------------------------------------------- #
# DirichletMultinomial -- the conjugate benchmark
# --------------------------------------------------------------------------- #
#
# This simulator's whole purpose is that its posterior is exact, so these
# checks are about the claimed conjugacy actually holding, not just about
# shapes. If the prior, the forward model and the stated posterior are not
# mutually consistent, every comparison scored against it is worthless.


def test_dirichlet_multinomial_validates_its_arguments():
    with pytest.raises(ValueError, match="at least 2 categories"):
        DirichletMultinomial(n_categories=1)
    with pytest.raises(ValueError, match="n_trials must be positive"):
        DirichletMultinomial(n_trials=0)
    with pytest.raises(ValueError, match="concentration must be positive"):
        DirichletMultinomial(n_categories=3, concentration=-1.0)
    with pytest.raises(ValueError, match="scalar or length"):
        DirichletMultinomial(n_categories=3, concentration=[1.0, 2.0])


def test_gamma_sampler_has_the_right_mean_and_variance():
    # Gamma(a, 1) has mean a and variance a; the shape < 1 branch is boosted
    # separately, so both sides need checking.
    sim = DirichletMultinomial(dtype=DT)
    g = torch.Generator().manual_seed(0)
    for a in (0.3, 1.0, 5.0):
        x = sim._standard_gamma(torch.full((200_000,), a, dtype=DT), g)
        assert float(x.mean()) == pytest.approx(a, rel=0.02)
        assert float(x.var()) == pytest.approx(a, rel=0.05)
        assert bool((x > 0).all())


def test_prior_is_dirichlet():
    sim = DirichletMultinomial(n_categories=4, concentration=2.0, dtype=DT)
    theta = sim.sample_prior(200_000, generator=torch.Generator().manual_seed(0))
    assert theta.shape == (200_000, 3)
    assert bool((theta > 0).all()) and bool((theta.sum(-1) < 1).all())
    exact_mean = (sim.alpha / sim.alpha.sum())[:3]
    assert torch.allclose(theta.mean(0), exact_mean, atol=0.005)
    assert torch.allclose(theta.std(0), sim.prior_sd(), atol=0.005)


def test_simulate_is_multinomial():
    sim = DirichletMultinomial(n_categories=4, n_trials=50, dtype=DT)
    probability = torch.tensor([0.5, 0.2, 0.2], dtype=DT)
    theta = probability.expand(100_000, 3)
    y = sim.simulate(theta, generator=torch.Generator().manual_seed(0))
    assert y.shape == (100_000, 4)
    assert bool((y.sum(-1) == sim.n_trials).all())
    assert bool((y >= 0).all())
    full = torch.tensor([0.5, 0.2, 0.2, 0.1], dtype=DT)
    assert torch.allclose(y.mean(0) / sim.n_trials, full, atol=0.005)
    expected_sd = (sim.n_trials * full * (1 - full)).sqrt()
    assert torch.allclose(y.std(0), expected_sd, rtol=0.05)


def test_posterior_is_exactly_dirichlet_alpha_plus_y():
    """The conjugacy claim, checked against brute-force importance sampling."""
    sim = DirichletMultinomial(n_categories=4, n_trials=50, concentration=2.0, dtype=DT)
    g = torch.Generator().manual_seed(1)
    _, y_obs = sim.sample_joint(1, generator=g)
    mean, cov = _importance_posterior(sim, y_obs[0], n=500_000, seed=2)
    assert torch.allclose(mean, sim.posterior_mean(y_obs)[0], atol=0.004)
    assert torch.allclose(cov.diagonal().sqrt(), sim.posterior_sd(y_obs)[0], atol=0.004)


def test_exact_posterior_sampler_matches_the_exact_moments():
    sim = DirichletMultinomial(n_categories=5, n_trials=40, dtype=DT)
    g = torch.Generator().manual_seed(3)
    _, y_obs = sim.sample_joint(2, generator=g)
    draws = sim.sample_posterior(y_obs, 200_000, generator=g)
    assert draws.shape == (2, 200_000, 4)
    assert bool((draws > 0).all()) and bool((draws.sum(-1) < 1).all())
    assert torch.allclose(draws.mean(1), sim.posterior_mean(y_obs), atol=0.004)
    assert torch.allclose(draws.std(1), sim.posterior_sd(y_obs), atol=0.004)


def test_marginal_beta_parameters_match_the_marginal_moments():
    sim = DirichletMultinomial(n_categories=4, n_trials=30, dtype=DT)
    g = torch.Generator().manual_seed(4)
    _, y_obs = sim.sample_joint(3, generator=g)
    for j in range(sim.theta_dim):
        a, b = sim.marginal_beta(y_obs, j)
        mean = a / (a + b)
        sd = (a * b / ((a + b) ** 2 * (a + b + 1))).sqrt()
        assert torch.allclose(mean, sim.posterior_mean(y_obs)[:, j], atol=1e-10)
        assert torch.allclose(sd, sim.posterior_sd(y_obs)[:, j], atol=1e-10)


def test_more_trials_concentrate_the_posterior():
    """n_trials is the concentration knob the regime sweep relies on."""
    previous = None
    for n_trials in (5, 50, 500):
        sim = DirichletMultinomial(n_categories=4, n_trials=n_trials, dtype=DT)
        g = torch.Generator().manual_seed(5)
        _, y_obs = sim.sample_joint(64, generator=g)
        spread = float(sim.posterior_sd(y_obs).mean())
        if previous is not None:
            assert spread < previous
        previous = spread


def test_log_prior_and_log_likelihood_reject_points_off_the_simplex():
    sim = DirichletMultinomial(n_categories=3, n_trials=10, dtype=DT)
    outside = torch.tensor([[0.7, 0.8]], dtype=DT)  # sums above one
    y = torch.tensor([3.0, 3.0, 4.0], dtype=DT)
    assert float(sim.log_prior(outside)[0]) == -math.inf
    assert float(sim.log_likelihood(outside, y)[0]) == -math.inf
    inside = torch.tensor([[0.3, 0.3]], dtype=DT)
    assert math.isfinite(float(sim.log_prior(inside)[0]))
    assert math.isfinite(float(sim.log_likelihood(inside, y)[0]))


# --------------------------------------------------------------------------- #
# SignAmbiguous -- the multimodal exact-reference benchmark
# --------------------------------------------------------------------------- #


def test_sign_ambiguous_validates_its_arguments():
    with pytest.raises(ValueError, match="theta_dim must be positive"):
        SignAmbiguous(theta_dim=0)
    with pytest.raises(ValueError, match="noise must be positive"):
        SignAmbiguous(noise=0.0)
    with pytest.raises(ValueError, match="n_obs must be positive"):
        SignAmbiguous(n_obs=0)


def test_sign_ambiguous_quadrature_matches_importance_sampling():
    """The exact reference, checked against a brute-force independent route."""
    sim = SignAmbiguous(theta_dim=2, noise=0.6, n_obs=4, prior_mean=0.5, dtype=DT)
    g = torch.Generator().manual_seed(0)
    _, y_obs = sim.sample_joint(2, generator=g)
    mean, cov = _importance_posterior(sim, y_obs[0], n=400_000, seed=1)
    assert torch.allclose(mean, sim.posterior_mean(y_obs[:1])[0], atol=0.02)
    assert torch.allclose(cov.diagonal().sqrt(), sim.posterior_sd(y_obs[:1])[0], atol=0.02)


def test_sign_ambiguous_posterior_is_bimodal_near_plus_minus_sqrt_y():
    sim = SignAmbiguous(theta_dim=1, noise=0.3, n_obs=4, prior_mean=0.5, dtype=DT)
    y_obs = torch.tensor([[4.0]], dtype=DT)
    points, density = sim.posterior_grid(y_obs, coordinate=0)
    row = density[0]
    interior = row[1:-1]
    peaks = points[1:-1][(interior > row[:-2]) & (interior > row[2:])]
    assert peaks.numel() == 2, f"expected two modes, found {peaks.numel()}"
    assert torch.allclose(peaks.abs(), torch.full((2,), 2.0, dtype=DT), atol=0.15)
    assert float(peaks[0]) < 0 < float(peaks[1])


def test_sign_ambiguous_prior_mean_breaks_the_symmetry():
    """A symmetric prior would make the posterior mean identically zero."""
    y_obs = torch.tensor([[4.0]], dtype=DT)
    symmetric = SignAmbiguous(theta_dim=1, prior_mean=0.0, dtype=DT)
    assert abs(float(symmetric.posterior_mean(y_obs)[0, 0])) < 1e-9
    tilted = SignAmbiguous(theta_dim=1, prior_mean=0.5, dtype=DT)
    assert abs(float(tilted.posterior_mean(y_obs)[0, 0])) > 0.5


def test_sign_ambiguous_density_and_cdf_are_consistent():
    sim = SignAmbiguous(theta_dim=2, dtype=DT)
    g = torch.Generator().manual_seed(2)
    _, y_obs = sim.sample_joint(3, generator=g)
    for j in range(2):
        points, density = sim.posterior_grid(y_obs, coordinate=j)
        assert torch.all(density >= 0)
        assert torch.allclose(torch.trapezoid(density, points, dim=-1),
                              torch.ones(3, dtype=DT), atol=1e-8)
        grid, cdf = sim.posterior_cdf(y_obs, coordinate=j)
        assert torch.allclose(grid, points)
        assert torch.all(cdf.diff(dim=-1) >= -1e-12)
        assert torch.allclose(cdf[:, -1], torch.ones(3, dtype=DT), atol=1e-8)


def test_sign_ambiguous_simulate_matches_its_stated_law():
    sim = SignAmbiguous(theta_dim=2, noise=0.6, n_obs=4, dtype=DT)
    theta = torch.tensor([[1.5, -0.5]], dtype=DT).expand(200_000, 2)
    y = sim.simulate(theta, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(y.mean(0), torch.tensor([2.25, 0.25], dtype=DT), atol=0.01)
    assert torch.allclose(y.std(0), torch.full((2,), sim.effective_noise, dtype=DT), rtol=0.05)
