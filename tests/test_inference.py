"""Inference functionals of the discrete conditional law.

These are exact statements about weighted statistics, so they are tested
against brute-force computations on hand-built weight vectors rather than
against a trained model.
"""

import math

import pytest
import torch

from posterior_operator.inference import ConditionalDistribution, GaussianKDE, isotonic_regression
from posterior_operator.metrics import trapezoid


def _law(weights, atoms):
    return ConditionalDistribution(
        weights=torch.as_tensor(weights, dtype=torch.float64),
        atoms=torch.as_tensor(atoms, dtype=torch.float64).reshape(-1, 1),
    )


@pytest.fixture
def simple_law():
    """Two conditioning values over four ordered atoms."""
    return _law([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], [-1.0, 0.0, 1.0, 2.0])


@pytest.fixture
def random_law():
    """Atoms kept in increasing order, so `weights[i]` lines up with `cdf()`."""
    g = torch.Generator().manual_seed(0)
    w = torch.rand(6, 40, generator=g, dtype=torch.float64)
    w /= w.sum(dim=-1, keepdim=True)
    atoms = torch.randn(40, 1, generator=g, dtype=torch.float64).sort(dim=0).values
    return ConditionalDistribution(weights=w, atoms=atoms)


# --------------------------------------------------------------------------- #
# Moments
# --------------------------------------------------------------------------- #


def test_mean_and_variance_match_direct_sums(simple_law):
    atoms = simple_law.atoms[:, 0]
    w = simple_law.weights
    expected_mean = (w * atoms).sum(dim=-1, keepdim=True)
    assert torch.allclose(simple_law.mean(), expected_mean)
    expected_var = (w * atoms**2).sum(dim=-1, keepdim=True) - expected_mean**2
    assert torch.allclose(simple_law.variance(), expected_var)
    assert torch.allclose(simple_law.std(), expected_var.sqrt())


def test_covariance_is_the_multivariate_second_central_moment():
    g = torch.Generator().manual_seed(1)
    atoms = torch.randn(25, 3, generator=g, dtype=torch.float64)
    w = torch.rand(4, 25, generator=g, dtype=torch.float64)
    w /= w.sum(dim=-1, keepdim=True)
    law = ConditionalDistribution(weights=w, atoms=atoms)
    cov = law.covariance()
    assert cov.shape == (4, 3, 3)
    for i in range(4):
        mu = (w[i, :, None] * atoms).sum(0)
        expected = (w[i, :, None, None] * torch.einsum("mi,mj->mij", atoms, atoms)).sum(0) - torch.outer(mu, mu)
        assert torch.allclose(cov[i], expected, atol=1e-12)
    # Diagonal of the covariance must agree with the per-coordinate variance.
    assert torch.allclose(torch.diagonal(cov, dim1=1, dim2=2), law.variance(), atol=1e-12)


def test_expectation_handles_scalar_and_vector_observables(simple_law):
    scalar = simple_law.expectation(lambda a: a[:, 0] ** 2)
    assert scalar.shape == (2,)
    assert torch.allclose(scalar, simple_law.moment(2))

    vector = simple_law.expectation(lambda a: torch.cat([a, a**2], dim=-1))
    assert vector.shape == (2, 2)
    assert torch.allclose(vector[:, :1], simple_law.mean())


def test_central_moments(simple_law):
    atoms = simple_law.atoms[:, 0]
    mu = simple_law.mean()
    expected = (simple_law.weights * (atoms - mu) ** 3).sum(dim=-1)
    assert torch.allclose(simple_law.moment(3, central=True), expected)
    assert torch.allclose(simple_law.moment(1, central=True), torch.zeros(2, dtype=torch.float64), atol=1e-14)


def test_observable_selects_a_coordinate_or_applies_a_callable():
    g = torch.Generator().manual_seed(2)
    atoms = torch.randn(20, 3, generator=g, dtype=torch.float64)
    w = torch.full((1, 20), 1 / 20, dtype=torch.float64)
    law = ConditionalDistribution(weights=w, atoms=atoms)
    assert torch.allclose(law.moment(1, observable=1), atoms[:, 1].mean().reshape(1))
    assert torch.allclose(law.moment(1, observable=lambda a: a.sum(-1)), atoms.sum(-1).mean().reshape(1))
    with pytest.raises(ValueError, match="scalar observable is required"):
        law.moment(1)
    with pytest.raises(ValueError, match="out of range"):
        law.moment(1, observable=5)


# --------------------------------------------------------------------------- #
# CDF and quantiles
# --------------------------------------------------------------------------- #


def test_cdf_on_atoms_is_the_cumulative_weight(simple_law):
    points, cdf = simple_law.cdf()
    assert torch.allclose(points, simple_law.atoms[:, 0])
    assert torch.allclose(cdf, simple_law.weights.cumsum(-1))
    assert torch.allclose(cdf[:, -1], torch.ones(2, dtype=torch.float64))


def test_cdf_is_a_valid_distribution_function(random_law):
    _, cdf = random_law.cdf()
    assert torch.all(cdf >= -1e-12) and torch.all(cdf <= 1 + 1e-12)
    assert torch.all(cdf[:, 1:] >= cdf[:, :-1] - 1e-12)
    assert torch.allclose(cdf[:, -1], torch.ones(len(random_law), dtype=torch.float64))


def test_cdf_on_a_grid_is_the_right_step_function(simple_law):
    grid = torch.tensor([-2.0, -1.0, -0.5, 0.0, 1.5, 2.0, 5.0], dtype=torch.float64)
    _, cdf = simple_law.cdf(grid=grid)
    # P(Y <= t) read off the atom masses [0.1, 0.2, 0.3, 0.4] at {-1, 0, 1, 2}.
    expected = torch.tensor(
        [[0.0, 0.1, 0.1, 0.3, 0.6, 1.0, 1.0], [0.0, 0.4, 0.4, 0.7, 0.9, 1.0, 1.0]], dtype=torch.float64
    )
    assert torch.allclose(cdf, expected)


def test_cdf_respects_atoms_out_of_input_order():
    """Atoms need not arrive sorted; the CDF must sort them."""
    law = _law([[0.5, 0.2, 0.3]], [2.0, -1.0, 0.5])
    points, cdf = law.cdf()
    assert torch.allclose(points, torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64))
    assert torch.allclose(cdf[0], torch.tensor([0.2, 0.5, 1.0], dtype=torch.float64))


def test_quantile_is_the_generalised_inverse_cdf(simple_law):
    # Masses 0.1/0.2/0.3/0.4 at -1/0/1/2, so F = 0.1, 0.3, 0.6, 1.0.
    got = simple_law.quantile([0.05, 0.1, 0.25, 0.6, 0.95, 1.0])[0]
    expected = torch.tensor([-1.0, -1.0, 0.0, 1.0, 2.0, 2.0], dtype=torch.float64)
    assert torch.allclose(got, expected)


def test_quantile_agrees_with_the_cdf_everywhere(random_law):
    levels = torch.linspace(0.01, 0.99, 25, dtype=torch.float64)
    q = random_law.quantile(levels)
    points, cdf = random_law.cdf()
    for i in range(len(random_law)):
        for j, level in enumerate(levels):
            # The returned atom is the smallest one whose CDF reaches the level.
            idx = int((points == q[i, j]).nonzero()[0])
            assert cdf[i, idx] >= level - 1e-12
            if idx > 0:
                assert cdf[i, idx - 1] < level + 1e-12


def test_scalar_and_vector_levels_differ_in_shape(random_law):
    assert random_law.quantile(0.5).shape == (6,)
    assert random_law.quantile([0.5]).shape == (6, 1)
    assert torch.allclose(random_law.median(), random_law.quantile(0.5))


def test_quantile_levels_are_validated(simple_law):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        simple_law.quantile([0.5, 1.5])


# --------------------------------------------------------------------------- #
# Confidence regions
# --------------------------------------------------------------------------- #


def test_interval_attains_nominal_coverage_under_the_estimated_law(random_law):
    for alpha in (0.05, 0.1, 0.25):
        iv = random_law.interval(alpha)
        points, _ = random_law.cdf()
        inside = (points.unsqueeze(0) >= iv[:, :1]) & (points.unsqueeze(0) <= iv[:, 1:])
        mass = (random_law.weights * inside).sum(-1)
        assert torch.all(mass >= 1 - alpha - 1e-9)


def test_interval_is_the_shortest_one_with_that_coverage(random_law):
    """Compare against an exhaustive search over all atom pairs."""
    alpha = 0.1
    iv = random_law.interval(alpha)
    points, _ = random_law.cdf()
    m = points.numel()
    for i in range(len(random_law)):
        w = random_law.weights[i]
        best = math.inf
        for lo in range(m):
            for hi in range(lo, m):
                if float(w[lo : hi + 1].sum()) >= 1 - alpha - 1e-12:
                    best = min(best, float(points[hi] - points[lo]))
                    break
        assert float(iv[i, 1] - iv[i, 0]) == pytest.approx(best, abs=1e-9)


def test_interval_is_narrower_on_a_concentrated_law():
    tight = _law([[0.01, 0.98, 0.01]], [-5.0, 0.0, 5.0])
    diffuse = _law([[1 / 3, 1 / 3, 1 / 3]], [-5.0, 0.0, 5.0])
    assert float(tight.interval(0.1)[0, 1] - tight.interval(0.1)[0, 0]) < float(
        diffuse.interval(0.1)[0, 1] - diffuse.interval(0.1)[0, 0]
    )


def test_interval_alpha_is_validated(simple_law):
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            simple_law.interval(bad)


def test_highest_density_region_carries_the_nominal_mass():
    """A bimodal target: the region should be disjoint and hold ~1-alpha mass."""
    grid = torch.linspace(-6, 6, 601, dtype=torch.float64)
    density = 0.5 * torch.exp(-0.5 * (grid - 2.5) ** 2) + 0.5 * torch.exp(-0.5 * (grid + 2.5) ** 2)
    density /= trapezoid(density, grid)

    class _Stub:
        """Stand in for an operator whose conditional density is `density`."""

        def deflated_ratio(self, x, y, rank=None):
            return (density / (1 / 12.0)).reshape(1, -1) - 1.0

    law = ConditionalDistribution(
        weights=torch.full((1, 601), 1 / 601, dtype=torch.float64),
        atoms=grid.reshape(-1, 1),
        operator=_Stub(),
        x=torch.zeros(1, 1, dtype=torch.float64),
    )
    mask, level = law.highest_density_region(grid, lambda pts: torch.full_like(pts.reshape(-1), 1 / 12.0), alpha=0.1)
    covered = trapezoid(density * mask[0], grid)
    assert float(covered) == pytest.approx(0.9, abs=0.02)
    assert float(level) > 0
    # Disjoint: the mask must switch on and off more than once.
    switches = int((mask[0, 1:] != mask[0, :-1]).sum())
    assert switches == 4
    # And it must exclude the low-density valley between the modes.
    assert not bool(mask[0, 300])


def test_density_requires_an_originating_operator(simple_law):
    with pytest.raises(RuntimeError, match="NCPOperator.condition"):
        simple_law.density(torch.linspace(-1, 1, 5), lambda p: torch.ones_like(p.reshape(-1)))


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def test_sampling_reproduces_the_weighted_mean(random_law):
    draws = random_law.sample(40_000, generator=torch.Generator().manual_seed(4))
    assert draws.shape == (6, 40_000, 1)
    assert torch.allclose(draws.mean(dim=1), random_law.mean(), atol=0.02)


def test_sampling_only_returns_atoms(simple_law):
    draws = simple_law.sample(200, generator=torch.Generator().manual_seed(5))
    assert torch.isin(draws, simple_law.atoms).all()


def test_sample_count_is_validated(simple_law):
    with pytest.raises(ValueError, match="must be positive"):
        simple_law.sample(0)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def test_isotonic_regression_matches_a_known_pooling():
    got = isotonic_regression(torch.tensor([[1.0, 3.0, 2.0, 4.0]], dtype=torch.float64))
    assert torch.allclose(got, torch.tensor([[1.0, 2.5, 2.5, 4.0]], dtype=torch.float64))


def test_isotonic_regression_is_the_l2_projection():
    g = torch.Generator().manual_seed(6)
    y = torch.randn(1, 12, generator=g, dtype=torch.float64)
    fit = isotonic_regression(y)
    assert torch.all(fit[0, 1:] >= fit[0, :-1] - 1e-12)
    base = float(((y - fit) ** 2).sum())
    # No monotone perturbation of the fit does better.
    for _ in range(200):
        step = torch.randn(1, 12, generator=g, dtype=torch.float64) * 0.05
        candidate = torch.cummax(fit + step, dim=-1).values
        assert float(((y - candidate) ** 2).sum()) >= base - 1e-12
    # Already-monotone input is left alone.
    monotone = torch.arange(6, dtype=torch.float64).reshape(1, -1)
    assert torch.allclose(isotonic_regression(monotone), monotone)


def test_isotonic_repairs_a_cdf_built_from_signed_weights():
    law = _law([[0.5, -0.2, 0.4, 0.3]], [0.0, 1.0, 2.0, 3.0])
    _, raw = law.cdf()
    assert bool((raw[:, 1:] < raw[:, :-1]).any())  # non-monotone, as set up
    _, fixed = law.cdf(monotone=True)
    assert torch.all(fixed[:, 1:] >= fixed[:, :-1] - 1e-12)
    assert torch.all((fixed >= 0) & (fixed <= 1))


def test_gaussian_kde_integrates_to_one_and_tracks_the_true_density():
    g = torch.Generator().manual_seed(7)
    samples = torch.randn(4000, 1, generator=g, dtype=torch.float64)
    kde = GaussianKDE(samples)
    grid = torch.linspace(-6, 6, 1201, dtype=torch.float64).reshape(-1, 1)
    est = kde(grid)
    assert float(trapezoid(est, grid[:, 0])) == pytest.approx(1.0, abs=1e-3)
    truth = torch.exp(-0.5 * grid[:, 0] ** 2) / math.sqrt(2 * math.pi)
    # Smoothing bias is O(h^2) with h ~ n^(-1/5) ~ 0.19 here, so a couple of
    # percent at the peak is expected; the average error is far smaller.
    assert float((est - truth).abs().max()) < 0.03
    assert float((est - truth).abs().mean()) < 5e-3


def test_gaussian_kde_handles_multivariate_samples():
    g = torch.Generator().manual_seed(8)
    kde = GaussianKDE(torch.randn(500, 3, generator=g, dtype=torch.float64))
    assert kde(torch.randn(11, 3, generator=g, dtype=torch.float64)).shape == (11,)
    with pytest.raises(ValueError, match="at least 2 samples"):
        GaussianKDE(torch.zeros(1, 2, dtype=torch.float64))


def test_constructor_validates_shapes():
    with pytest.raises(ValueError, match="weights must be 2D"):
        ConditionalDistribution(weights=torch.ones(3), atoms=torch.zeros(3, 1))
    with pytest.raises(ValueError, match="atoms must be 2D"):
        ConditionalDistribution(weights=torch.ones(1, 3), atoms=torch.zeros(3))
    with pytest.raises(ValueError, match="columns"):
        ConditionalDistribution(weights=torch.ones(1, 3), atoms=torch.zeros(4, 1))


def test_repr_and_shape_accessors(random_law):
    assert len(random_law) == 6
    assert random_law.n_atoms == 40
    assert random_law.y_dim == 1
    assert "n_atoms=40" in repr(random_law)
