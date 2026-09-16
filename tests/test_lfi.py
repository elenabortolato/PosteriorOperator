"""Posterior-functional inference: the wrapper, the query surface, retargeting.

Most of these use a small fit so they stay fast; the point is the algebra of
the query surface (that the estimator really is Eq. (2), that reweighting
implements the self-normalised identity, that order statistics see a valid
measure), not statistical accuracy. Accuracy against exact posteriors is
checked in `test_lfi_accuracy.py`.
"""

import math
import warnings

import pytest
import torch

from posterior_operator import PosteriorOperator, PosteriorSample, reference_posterior_moments
from posterior_operator.simulators import GaussianLinear, SumIdentified


@pytest.fixture(scope="module")
def fitted():
    """A small fitted operator on a linear-Gaussian simulator."""
    sim = GaussianLinear(theta_dim=2, data_dim=3, noise=0.5, seed=0)
    g = torch.Generator().manual_seed(0)
    theta, y = sim.sample_joint(4000, generator=g)
    torch.manual_seed(0)
    operator = PosteriorOperator(theta_dim=2, data_dim=3, rank=12, layer_size=32)
    operator.fit(theta, y, epochs=120, lr=1e-3, seed=0)
    _, y_obs = sim.sample_joint(6, generator=g)
    return sim, operator, y_obs


def _hand_made(weights, atoms):
    return PosteriorSample(
        weights=torch.as_tensor(weights, dtype=torch.float64),
        atoms=torch.as_tensor(atoms, dtype=torch.float64).reshape(-1, 1),
    )


# --------------------------------------------------------------------------- #
# The wrapper
# --------------------------------------------------------------------------- #


def test_posterior_before_fit_is_refused():
    operator = PosteriorOperator(theta_dim=2, data_dim=3, rank=8)
    with pytest.raises(RuntimeError, match="call fit"):
        operator.posterior(torch.zeros(1, 3))
    assert "unfitted" in repr(operator)


def test_fit_validates_the_data_dimension():
    operator = PosteriorOperator(theta_dim=2, data_dim=3, rank=8)
    with pytest.raises(ValueError, match="expected data_dim=3"):
        operator.fit(torch.zeros(50, 2), torch.zeros(50, 4), epochs=2)


def test_fit_validates_the_validation_split():
    operator = PosteriorOperator(theta_dim=2, data_dim=3, rank=8)
    with pytest.raises(ValueError, match="validation_split"):
        operator.fit(torch.zeros(50, 2), torch.zeros(50, 3), epochs=2, validation_split=1.5)


def test_posterior_is_reported_in_natural_parameter_units(fitted):
    """Standardisation happens inside; the atoms must come back unscaled."""
    sim, operator, y_obs = fitted
    draws = operator.posterior(y_obs).theta
    g = torch.Generator().manual_seed(11)
    prior = sim.sample_prior(4000, generator=g)
    assert torch.allclose(draws.mean(0), prior.mean(0), atol=0.15)
    assert torch.allclose(draws.std(0), prior.std(0), atol=0.15)


def test_shapes_and_repr(fitted):
    _, operator, y_obs = fitted
    posterior = operator.posterior(y_obs)
    assert len(posterior) == y_obs.shape[0]
    assert posterior.theta.shape[1] == 2
    assert posterior.mean().shape == (y_obs.shape[0], 2)
    assert posterior.covariance().shape == (y_obs.shape[0], 2, 2)
    assert "n_draws" in repr(posterior)
    assert "fitted" in repr(operator)
    assert "sigma_1" in operator.spectrum_report()


def test_a_single_flat_observation_is_accepted(fitted):
    _, operator, y_obs = fitted
    assert len(operator.posterior(y_obs[0])) == 1


def test_diagnostics_are_consistent_with_the_spectrum(fitted):
    _, operator, _ = fitted
    sv = operator.singular_values
    assert operator.maximal_correlation == pytest.approx(float(sv[0]))
    assert operator.chi2_divergence == pytest.approx(float((sv**2).sum()))
    assert 0.0 <= operator.maximal_correlation <= 1.0


def test_singular_functions_have_the_right_shapes(fitted):
    _, operator, y_obs = fitted
    g = torch.Generator().manual_seed(1)
    theta = torch.randn(40, 2, generator=g)
    assert operator.singular_function_theta(theta).shape == (40, operator.rank)
    assert operator.singular_function_theta(theta, rank=3).shape == (40, 3)
    assert operator.singular_function_data(y_obs).shape == (y_obs.shape[0], operator.rank)


def test_alternative_theta_draws_are_used(fitted):
    _, operator, y_obs = fitted
    g = torch.Generator().manual_seed(2)
    draws = torch.randn(300, 2, generator=g)
    posterior = operator.posterior(y_obs, theta_draws=draws)
    assert posterior.n_atoms == 300
    assert torch.allclose(posterior.theta, draws)


# --------------------------------------------------------------------------- #
# The estimator is Eq. (2)
# --------------------------------------------------------------------------- #


def test_functional_equals_the_explicit_spectral_formula(fitted):
    r"""T_f(y) = mean f + sum_k sigma_k u_k(y) * mean(v_k f), written out by hand."""
    _, operator, y_obs = fitted
    posterior = operator.posterior(y_obs)
    draws = posterior.theta

    def f(t):
        return (t[:, 0] ** 2 - 0.5 * t[:, 1]).unsqueeze(-1)

    values = f(draws).reshape(-1)
    u = operator.singular_function_data(y_obs)  # (n_obs, d)
    v = operator.singular_function_theta(draws)  # (n_draws, d)
    sigma = operator.singular_values
    explicit = values.mean() + (u * sigma) @ (v * values.unsqueeze(-1)).mean(dim=0)

    assert torch.allclose(posterior.functional(f).reshape(-1), explicit, atol=1e-4)


def test_masses_sum_to_one_and_are_signed_by_default(fitted):
    _, operator, y_obs = fitted
    posterior = operator.posterior(y_obs)
    assert torch.allclose(posterior.weights.sum(-1), torch.ones(len(posterior)), atol=1e-4)
    # Default is the unclipped estimator, so negatives are allowed through.
    assert bool((posterior.weights < 0).any())
    assert torch.all(operator.posterior(y_obs, clip=True).weights >= 0)


def test_functional_agrees_with_mean_and_probability(fitted):
    _, operator, y_obs = fitted
    posterior = operator.posterior(y_obs)
    assert torch.allclose(posterior.functional(lambda t: t), posterior.mean(), atol=1e-6)
    assert torch.allclose(
        posterior.probability(lambda t: t[:, 0] > 0),
        posterior.functional(lambda t: (t[:, 0] > 0).to(t.dtype).unsqueeze(-1)).reshape(-1),
        atol=1e-6,
    )


def test_probability_is_in_the_unit_interval_on_the_clipped_measure(fitted):
    _, operator, y_obs = fitted
    probability = operator.posterior(y_obs).as_probability()
    values = probability.probability(lambda t: t[:, 0] > 0)
    assert torch.all(values >= -1e-6) and torch.all(values <= 1 + 1e-6)
    complement = probability.probability(lambda t: t[:, 0] <= 0)
    assert torch.allclose(values + complement, torch.ones_like(values), atol=1e-5)


def test_probability_validates_its_region(fitted):
    _, operator, y_obs = fitted
    with pytest.raises(ValueError, match="region returned"):
        operator.posterior(y_obs).probability(lambda t: torch.ones(3))


def test_rank_truncation_changes_the_answer(fitted):
    _, operator, y_obs = fitted
    full = operator.posterior(y_obs).mean()
    low = operator.posterior(y_obs, rank=1).mean()
    assert not torch.allclose(full, low, atol=1e-3)


# --------------------------------------------------------------------------- #
# Signed vs clipped
# --------------------------------------------------------------------------- #


def test_as_probability_produces_a_valid_measure():
    signed = _hand_made([[0.5, -0.2, 0.7]], [0.0, 1.0, 2.0])
    probability = signed.as_probability()
    assert torch.all(probability.weights >= 0)
    assert float(probability.weights.sum()) == pytest.approx(1.0)
    # Renormalised from the surviving 0.5 and 0.7.
    assert torch.allclose(probability.weights[0], torch.tensor([0.5, 0.0, 0.7], dtype=torch.float64) / 1.2)
    # A non-negative posterior is returned unchanged.
    already = _hand_made([[0.25, 0.25, 0.5]], [0.0, 1.0, 2.0])
    assert already.as_probability() is already


def test_order_statistics_use_the_clipped_measure_even_on_signed_weights():
    signed = _hand_made([[0.6, -0.3, 0.7]], [0.0, 1.0, 2.0])
    _, cdf = signed.cdf()
    assert torch.all(cdf[:, 1:] >= cdf[:, :-1] - 1e-12)  # monotone despite the negative
    assert float(cdf[0, -1]) == pytest.approx(1.0)
    interval = signed.credible_interval(0.1)
    assert float(interval[0, 1]) >= float(interval[0, 0])
    draws = signed.sample(200, generator=torch.Generator().manual_seed(0))
    assert torch.isin(draws, signed.atoms).all()
    # The atom with negative mass is never drawn.
    assert not bool((draws == 1.0).any())


def test_moments_use_the_signed_weights():
    signed = _hand_made([[0.6, -0.3, 0.7]], [0.0, 1.0, 2.0])
    expected = 0.6 * 0.0 + (-0.3) * 1.0 + 0.7 * 2.0
    assert float(signed.mean()[0, 0]) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# The isotonic projection
# --------------------------------------------------------------------------- #


def test_with_projection_validates_its_argument():
    signed = _hand_made([[0.6, -0.3, 0.7]], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="clip.*isotonic"):
        signed.with_projection("nearest")


def test_with_projection_leaves_the_default_and_the_moments_alone():
    signed = _hand_made([[0.6, -0.3, 0.7]], [0.0, 1.0, 2.0])
    isotonic = signed.with_projection("isotonic")
    # Moment functionals keep the signed masses whatever the projection is.
    assert float(isotonic.mean()[0, 0]) == pytest.approx(float(signed.mean()[0, 0]))
    # The original is untouched, and "clip" restores the default behaviour.
    assert torch.allclose(signed.credible_interval(0.1), _hand_made(
        [[0.6, -0.3, 0.7]], [0.0, 1.0, 2.0]).credible_interval(0.1))
    assert torch.allclose(
        isotonic.with_projection("clip").cdf()[1], signed.cdf()[1]
    )


def test_isotonic_projection_gives_a_valid_cdf():
    signed = _hand_made([[0.6, -0.3, 0.7], [-0.4, 0.9, 0.5]], [0.0, 1.0, 2.0])
    values, cdf = signed.with_projection("isotonic").cdf()
    assert torch.all(values[1:] >= values[:-1])
    assert torch.all(cdf[:, 1:] >= cdf[:, :-1] - 1e-12)
    assert torch.all((cdf >= 0) & (cdf <= 1))
    assert torch.allclose(cdf[:, -1], torch.ones(2, dtype=cdf.dtype))


def test_isotonic_projection_keeps_mass_the_clipped_one_discards():
    # Masses 0.6, -0.3, 0.7 accumulate to 0.6, 0.3, 1.0. That is already
    # non-monotone at the middle atom, so PAVA pools the first two to 0.45 and
    # the projected CDF is (0.45, 0.45, 1.0) -- the middle atom keeps zero mass
    # but the FIRST atom is pulled down, which clipping never does.
    signed = _hand_made([[0.6, -0.3, 0.7]], [0.0, 1.0, 2.0])
    _, isotonic = signed.with_projection("isotonic").cdf()
    _, clipped = signed.cdf()
    assert float(isotonic[0, 0]) == pytest.approx(0.45)
    assert float(clipped[0, 0]) == pytest.approx(0.6 / 1.3)
    assert float(isotonic[0, 0]) < float(clipped[0, 0])


def test_isotonic_projection_is_the_identity_on_a_probability_measure():
    already = _hand_made([[0.25, 0.25, 0.5]], [0.0, 1.0, 2.0])
    _, isotonic = already.with_projection("isotonic").cdf()
    _, plain = already.cdf()
    assert torch.allclose(isotonic, plain)


def test_isotonic_quantiles_are_monotone_in_the_level():
    signed = _hand_made([[0.9, -0.5, 0.2, 0.4]], [0.0, 1.0, 2.0, 3.0])
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantiles = signed.with_projection("isotonic").quantile(levels)
    assert torch.all(quantiles.diff(dim=-1) >= 0)
    assert float(signed.mean()[0, 0]) != pytest.approx(float(signed.as_probability().mean()[0, 0]))


def test_effective_sample_size_uses_the_clipped_measure():
    uniform = _hand_made([[0.25] * 4], [0.0, 1.0, 2.0, 3.0])
    assert float(uniform.effective_sample_size()) == pytest.approx(4.0)
    peaked = _hand_made([[0.97, 0.01, 0.01, 0.01]], [0.0, 1.0, 2.0, 3.0])
    assert float(peaked.effective_sample_size()) < 1.1
    signed = _hand_made([[0.6, -0.3, 0.7]], [0.0, 1.0, 2.0])
    assert float(signed.effective_sample_size()) > 0  # finite, not NaN


# --------------------------------------------------------------------------- #
# Decision-theoretic queries
# --------------------------------------------------------------------------- #


def test_bayes_action_under_absolute_loss_is_the_median(fitted):
    _, operator, y_obs = fitted
    posterior = operator.posterior(y_obs)
    actions = torch.linspace(-3, 3, 241).reshape(-1, 1)

    def absolute(a, t):
        return (a - t[:, 0].unsqueeze(0)).abs()

    # The identity holds for a probability measure. `median` reads the clipped
    # measure, so the risk must be minimised over the same one -- minimising the
    # SIGNED risk is a different problem with a different solution.
    probability = posterior.as_probability()
    chosen = probability.bayes_action(absolute, actions).reshape(-1)
    assert torch.allclose(chosen, probability.median(observable=0), atol=0.05)  # grid step 0.025


def test_bayes_action_under_squared_loss_is_the_mean(fitted):
    _, operator, y_obs = fitted
    posterior = operator.posterior(y_obs)
    actions = torch.linspace(-3, 3, 481).reshape(-1, 1)

    def squared(a, t):
        return (a - t[:, 0].unsqueeze(0)) ** 2

    # Squared loss is minimised at the mean under the same measure the risk uses,
    # so here the signed weights are consistent on both sides.
    chosen = posterior.bayes_action(squared, actions).reshape(-1)
    assert torch.allclose(chosen, posterior.mean()[:, 0], atol=0.05)


def test_posterior_risk_shapes_and_validation(fitted):
    _, operator, y_obs = fitted
    posterior = operator.posterior(y_obs)
    actions = torch.linspace(-1, 1, 7).reshape(-1, 1)
    risk = posterior.posterior_risk(lambda a, t: (a - t[:, 0].unsqueeze(0)) ** 2, actions)
    assert risk.shape == (len(posterior), 7)
    with pytest.raises(ValueError, match="loss must return"):
        posterior.posterior_risk(lambda a, t: torch.zeros(3, 3), actions)


def test_marginal_histogram_is_a_probability_vector(fitted):
    _, operator, y_obs = fitted
    posterior = operator.posterior(y_obs)
    edges, probabilities = posterior.marginal_histogram(coordinate=0, bins=10)
    assert edges.numel() == 11
    assert probabilities.shape == (len(posterior), 10)
    assert torch.all(probabilities >= -1e-9)
    assert torch.allclose(probabilities.sum(-1), torch.ones(len(posterior)), atol=1e-5)
    # Explicit edges are honoured.
    custom = torch.tensor([-5.0, 0.0, 5.0])
    edges2, probabilities2 = posterior.marginal_histogram(0, bins=custom)
    assert torch.allclose(edges2, custom)
    assert probabilities2.shape == (len(posterior), 2)
    with pytest.raises(ValueError, match="bins must be positive"):
        posterior.marginal_histogram(0, bins=0)
    with pytest.raises(ValueError, match="at least 2 entries"):
        posterior.marginal_histogram(0, bins=torch.tensor([1.0]))


def test_credible_interval_narrows_as_alpha_grows(fitted):
    _, operator, y_obs = fitted
    posterior = operator.posterior(y_obs)
    wide = posterior.credible_interval(0.05)
    narrow = posterior.credible_interval(0.5)
    assert torch.all((narrow[:, 1] - narrow[:, 0]) <= (wide[:, 1] - wide[:, 0]) + 1e-9)


# --------------------------------------------------------------------------- #
# Prior retargeting
# --------------------------------------------------------------------------- #


def test_reweighting_by_a_constant_is_the_identity(fitted):
    _, operator, y_obs = fitted
    posterior = operator.posterior(y_obs)
    for constant in (0.0, 3.5, -2.0):
        same = posterior.reweight(torch.full((posterior.n_atoms,), constant))
        assert torch.allclose(same.weights, posterior.weights, atol=1e-6)


def test_reweighting_implements_the_self_normalised_identity():
    r"""E_pi[f|y] = E_q[f w|y] / E_q[w|y], computed both ways on a hand-made law."""
    posterior = _hand_made([[0.2, 0.3, 0.5]], [0.0, 1.0, 2.0])
    log_w = torch.tensor([0.0, math.log(2.0), math.log(4.0)], dtype=torch.float64)
    retargeted = posterior.reweight(log_w)

    w = torch.exp(log_w)
    values = posterior.atoms[:, 0]
    numerator = float((posterior.weights[0] * w * values).sum())
    denominator = float((posterior.weights[0] * w).sum())
    assert float(retargeted.mean()[0, 0]) == pytest.approx(numerator / denominator)
    assert float(retargeted.weights.sum()) == pytest.approx(1.0)
    # Weights are only defined up to scale.
    shifted = posterior.reweight(log_w + 7.0)
    assert torch.allclose(shifted.weights, retargeted.weights, atol=1e-12)


def test_reweighting_can_delete_draws_with_minus_inf():
    posterior = _hand_made([[0.2, 0.3, 0.5]], [0.0, 1.0, 2.0])
    log_w = torch.tensor([0.0, -math.inf, 0.0], dtype=torch.float64)
    retargeted = posterior.reweight(log_w)
    assert float(retargeted.weights[0, 1]) == pytest.approx(0.0)
    assert float(retargeted.weights.sum()) == pytest.approx(1.0)
    # Only the surviving atoms carry mass, in their original proportions.
    assert float(retargeted.weights[0, 0]) == pytest.approx(0.2 / 0.7)


def test_reweighting_falls_back_to_the_clipped_measure_when_signed_mass_cancels():
    """A truncating prior can annihilate a signed measure; that must warn, not crash."""
    # Signed masses -0.5 and +0.5 survive and cancel exactly to zero.
    signed = _hand_made([[-0.5, 1.0, 0.5]], [0.0, 1.0, 2.0])
    log_w = torch.tensor([0.0, -math.inf, 0.0], dtype=torch.float64)
    with pytest.warns(UserWarning, match="signed masses cancelled"):
        retargeted = signed.reweight(log_w)
    assert float(retargeted.weights.sum()) == pytest.approx(1.0)
    assert torch.all(retargeted.weights >= 0)
    # The clipped measure puts all surviving mass on the last atom.
    assert float(retargeted.weights[0, 2]) == pytest.approx(1.0)
    # Opting in explicitly is silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        signed.as_probability().reweight(log_w)


def test_reweighting_raises_when_no_mass_survives_at_all():
    """Even the clipped fallback can be empty; that is an error, not a warning."""
    signed = _hand_made([[-0.5, 1.0, 0.5]], [0.0, 1.0, 2.0])
    only_negative = torch.tensor([0.0, -math.inf, -math.inf], dtype=torch.float64)
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="no posterior mass"):
        signed.reweight(only_negative)


def test_reweighting_validates_its_input():
    posterior = _hand_made([[0.2, 0.3, 0.5]], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="expected 3 log weights"):
        posterior.reweight(torch.zeros(4))
    with pytest.raises(ValueError, match="no support"):
        posterior.reweight(torch.full((3,), -math.inf))


def test_reweighting_reduces_the_effective_sample_size(fitted):
    _, operator, y_obs = fitted
    posterior = operator.posterior(y_obs)
    base = posterior.effective_sample_size().mean()
    # A prior concentrated far from the bulk of the draws.
    log_w = -0.5 * (((posterior.theta - 2.5) / 0.3) ** 2).sum(-1)
    assert float(posterior.reweight(log_w).effective_sample_size().mean()) < float(base)


def test_reweighting_recovers_the_target_posterior_from_an_exact_proposal_fit():
    r"""End-to-end check of the retargeting identity against a closed-form posterior.

    Sidesteps the quality of the neural fit: the proposal posterior is built
    exactly, by importance sampling from the proposal with likelihood weights.
    Reweighting it by pi/q must then reproduce ``SumIdentified``'s closed-form
    posterior under pi.

    (With a *fitted* operator this need not improve accuracy -- the weights
    multiply the estimate, so its error interacts with them rather than
    cancelling. That is measured in examples/lfi/03_prior_retargeting.py, and it
    is why the claim tested here is the identity, not an error reduction.)
    """
    sim = SumIdentified(n_obs=6, noise=1.0)
    g = torch.Generator().manual_seed(4)
    scale = 1.8  # proposal q = N(0, scale^2 I), target pi = N(0, I)
    draws = sim.sample_prior(200_000, generator=g) * scale
    _, y_obs = sim.sample_joint(5, generator=g)

    # Exact posterior under q, as a weighted set of proposal draws.
    log_like = torch.stack([sim.log_likelihood(draws, y_obs[i]) for i in range(5)])
    proposal = PosteriorSample(
        weights=torch.softmax(log_like.double(), dim=-1), atoms=draws.double()
    )
    # log w = log pi - log q, up to a constant.
    log_w = -0.5 * (draws.double() ** 2).sum(-1) * (1 - 1 / scale**2)
    retargeted = proposal.reweight(log_w)

    exact_mean, exact_cov = sim.posterior_mean_cov(y_obs)
    assert torch.allclose(retargeted.mean(), exact_mean.double(), atol=0.02)
    assert torch.allclose(retargeted.covariance().mean(0), exact_cov.double(), atol=0.02)
    # And the un-retargeted proposal posterior is genuinely different.
    assert float((proposal.mean() - exact_mean.double()).abs().mean()) > 0.02


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def test_reference_posterior_moments():
    grid = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]], dtype=torch.float64)
    weights = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
    mean, cov = reference_posterior_moments(grid, weights)
    assert torch.allclose(mean, torch.tensor([1.0, 2.0], dtype=torch.float64))
    centred = grid - mean
    expected = (weights.unsqueeze(-1) * centred).T @ centred
    assert torch.allclose(cov, expected)
    assert cov.shape == (2, 2)


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals for the functional
# --------------------------------------------------------------------------- #


def test_bootstrap_interval_brackets_the_estimate(fitted):
    _, operator, y_obs = fitted
    posterior = operator.posterior(y_obs)
    estimate, interval = posterior.bootstrap_functional(
        lambda t: t, n_resamples=200, alpha=0.1, generator=torch.Generator().manual_seed(0)
    )
    assert estimate.shape == (len(posterior), 2)
    assert interval.shape == (len(posterior), 2, 2)
    assert torch.all(interval[..., 0] <= interval[..., 1])
    # The point estimate should sit inside its own resampling interval.
    assert torch.all(estimate >= interval[..., 0] - 1e-6)
    assert torch.all(estimate <= interval[..., 1] + 1e-6)
    assert torch.allclose(estimate, posterior.functional(lambda t: t), atol=1e-6)


def test_bootstrap_interval_narrows_with_more_draws():
    r"""The Monte Carlo part of the error shrinks like the number of draws."""
    widths = {}
    for n_draws in (500, 8000):
        g = torch.Generator().manual_seed(0)
        atoms = torch.randn(n_draws, 1, generator=g, dtype=torch.float64)
        weights = torch.full((1, n_draws), 1.0 / n_draws, dtype=torch.float64)
        posterior = PosteriorSample(weights=weights, atoms=atoms)
        _, interval = posterior.bootstrap_functional(
            lambda t: t, n_resamples=300, alpha=0.1, generator=torch.Generator().manual_seed(1)
        )
        widths[n_draws] = float(interval[..., 1] - interval[..., 0])
    ratio = widths[500] / widths[8000]
    assert ratio == pytest.approx(math.sqrt(8000 / 500), rel=0.35), f"widths {widths}"


def test_bootstrap_interval_is_calibrated_for_the_pure_monte_carlo_part():
    """With the weights exact, the only error IS Monte Carlo, so coverage should hold.

    Uniform weights over standard-normal draws make the target the population
    mean (zero), and the sole error is the sample average -- the case the
    bootstrap is actually designed for.
    """
    alpha, covered, trials = 0.1, 0, 200
    for r in range(trials):
        g = torch.Generator().manual_seed(r)
        atoms = torch.randn(400, 1, generator=g, dtype=torch.float64)
        weights = torch.full((1, 400), 1.0 / 400, dtype=torch.float64)
        posterior = PosteriorSample(weights=weights, atoms=atoms)
        _, interval = posterior.bootstrap_functional(
            lambda t: t, n_resamples=200, alpha=alpha, generator=torch.Generator().manual_seed(1000 + r)
        )
        covered += int(bool((interval[0, 0, 0] <= 0.0) and (0.0 <= interval[0, 0, 1])))
    assert covered / trials == pytest.approx(1 - alpha, abs=0.06), f"coverage {covered / trials}"


def test_bootstrap_validates_its_arguments(fitted):
    _, operator, y_obs = fitted
    posterior = operator.posterior(y_obs)
    with pytest.raises(ValueError, match="n_resamples"):
        posterior.bootstrap_functional(lambda t: t, n_resamples=1)
    for bad in (0.0, 1.0, -0.2):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            posterior.bootstrap_functional(lambda t: t, alpha=bad)
