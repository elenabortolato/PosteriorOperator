"""Evaluation metrics, checked against closed-form values."""

import math

import pytest
import torch

from posterior_operator.metrics import (
    coverage,
    hellinger,
    interval_width,
    js_divergence,
    kl_divergence,
    kolmogorov_smirnov,
    pinball_loss,
    total_variation,
    trapezoid,
    wasserstein1,
)

DT = torch.float64


def _normal(grid, mean, std):
    return torch.exp(-0.5 * ((grid - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))


@pytest.fixture
def grid():
    return torch.linspace(-12, 12, 4001, dtype=DT)


def test_trapezoid_integrates_known_functions(grid):
    assert float(trapezoid(_normal(grid, 0.0, 1.0), grid)) == pytest.approx(1.0, abs=1e-9)
    x = torch.linspace(0, 1, 2001, dtype=DT)
    assert float(trapezoid(x**2, x)) == pytest.approx(1 / 3, abs=1e-6)
    # Batched over leading axes.
    stacked = torch.stack([_normal(grid, 0.0, 1.0), _normal(grid, 2.0, 0.5)])
    assert torch.allclose(trapezoid(stacked, grid), torch.ones(2, dtype=DT), atol=1e-9)


def test_trapezoid_validates_the_grid(grid):
    with pytest.raises(ValueError, match="last axis"):
        trapezoid(torch.ones(5, dtype=DT), grid)
    with pytest.raises(ValueError, match="at least 2 grid points"):
        trapezoid(torch.ones(1, dtype=DT), torch.zeros(1, dtype=DT))


def test_distances_vanish_between_identical_densities(grid):
    p = _normal(grid, 0.3, 1.2)
    assert float(hellinger(p, p, grid)) == pytest.approx(0.0, abs=1e-9)
    assert float(total_variation(p, p, grid)) == pytest.approx(0.0, abs=1e-9)
    assert float(kl_divergence(p, p, grid)) == pytest.approx(0.0, abs=1e-9)
    assert float(js_divergence(p, p, grid)) == pytest.approx(0.0, abs=1e-9)


def test_hellinger_matches_the_closed_form_for_two_gaussians(grid):
    """H^2 = 1 - sqrt(2 s1 s2 / (s1^2 + s2^2)) exp(-(m1-m2)^2 / (4(s1^2+s2^2)))."""
    m1, s1, m2, s2 = 0.0, 1.0, 1.5, 2.0
    got = float(hellinger(_normal(grid, m1, s1), _normal(grid, m2, s2), grid))
    coefficient = math.sqrt(2 * s1 * s2 / (s1**2 + s2**2))
    expected = math.sqrt(1 - coefficient * math.exp(-((m1 - m2) ** 2) / (4 * (s1**2 + s2**2))))
    assert got == pytest.approx(expected, rel=1e-6)


def test_kl_matches_the_closed_form_for_two_gaussians(grid):
    m1, s1, m2, s2 = 0.0, 1.0, 0.5, 1.3
    got = float(kl_divergence(_normal(grid, m1, s1), _normal(grid, m2, s2), grid))
    expected = math.log(s2 / s1) + (s1**2 + (m1 - m2) ** 2) / (2 * s2**2) - 0.5
    assert got == pytest.approx(expected, rel=1e-4)


def test_distances_saturate_on_disjoint_supports():
    grid = torch.linspace(0, 20, 4001, dtype=DT)
    p, q = _normal(grid, 3.0, 0.3), _normal(grid, 17.0, 0.3)
    assert float(hellinger(p, q, grid)) == pytest.approx(1.0, abs=1e-6)
    assert float(total_variation(p, q, grid)) == pytest.approx(1.0, abs=1e-6)
    assert float(js_divergence(p, q, grid)) == pytest.approx(math.log(2.0), rel=1e-3)


def test_distances_grow_with_separation(grid):
    p = _normal(grid, 0.0, 1.0)
    previous = {"h": -1.0, "tv": -1.0}
    for shift in (0.0, 0.5, 1.0, 2.0, 4.0):
        q = _normal(grid, shift, 1.0)
        h, tv = float(hellinger(p, q, grid)), float(total_variation(p, q, grid))
        assert h > previous["h"] and tv > previous["tv"]
        previous = {"h": h, "tv": tv}


def test_kl_is_asymmetric_but_js_is_not(grid):
    p, q = _normal(grid, 0.0, 1.0), _normal(grid, 1.0, 2.0)
    assert float(kl_divergence(p, q, grid)) != pytest.approx(float(kl_divergence(q, p, grid)), rel=1e-3)
    assert float(js_divergence(p, q, grid)) == pytest.approx(float(js_divergence(q, p, grid)), rel=1e-9)


def test_unnormalised_densities_are_rescaled(grid):
    p = _normal(grid, 0.0, 1.0)
    assert float(hellinger(p, 7.0 * p, grid)) == pytest.approx(0.0, abs=1e-9)
    assert float(hellinger(p, 7.0 * p, grid, normalise=False)) > 0.1


def test_cdf_distances(grid):
    p_cdf = 0.5 * (1 + torch.erf(grid / math.sqrt(2)))
    q_cdf = 0.5 * (1 + torch.erf((grid - 1.0) / math.sqrt(2)))
    assert float(kolmogorov_smirnov(p_cdf, p_cdf)) == pytest.approx(0.0, abs=1e-12)
    # Two unit normals one apart: sup gap is at the midpoint, 2*Phi(0.5) - 1.
    expected_ks = 2 * float(0.5 * (1 + torch.erf(torch.tensor(0.5 / math.sqrt(2), dtype=DT)))) - 1
    assert float(kolmogorov_smirnov(p_cdf, q_cdf)) == pytest.approx(expected_ks, abs=1e-3)
    # A location shift of delta has W1 exactly delta.
    assert float(wasserstein1(p_cdf, q_cdf, grid)) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="shape mismatch"):
        kolmogorov_smirnov(p_cdf, p_cdf[:-1])


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_coverage_counts_inclusive_membership():
    intervals = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=DT)
    y = torch.tensor([0.5, 0.0, 1.0, 1.5], dtype=DT)  # inside, on both edges, outside
    assert float(coverage(intervals, y)) == pytest.approx(0.75)
    assert float(coverage(intervals, y.reshape(-1, 1))) == pytest.approx(0.75)


def test_coverage_validates_shapes():
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        coverage(torch.zeros(3, 3), torch.zeros(3))
    with pytest.raises(ValueError, match="responses"):
        coverage(torch.zeros(3, 2), torch.zeros(4))


def test_interval_width_reports_mean_and_spread():
    intervals = torch.tensor([[0.0, 2.0], [1.0, 5.0]], dtype=DT)
    mean, std = interval_width(intervals)
    assert float(mean) == pytest.approx(3.0)
    assert float(std) == pytest.approx(math.sqrt(2.0))
    # A single interval must not produce NaN from an unbiased variance.
    single_mean, single_std = interval_width(intervals[:1])
    assert float(single_mean) == pytest.approx(2.0) and float(single_std) == pytest.approx(0.0)


def test_pinball_loss_is_minimised_at_the_true_quantile():
    g = torch.Generator().manual_seed(0)
    y = torch.randn(20_000, generator=g, dtype=DT)
    level = 0.9
    truth = math.sqrt(2) * torch.erfinv(torch.tensor(2 * level - 1, dtype=DT))
    best = float(pinball_loss(torch.full((20_000, 1), float(truth), dtype=DT), y, [level]))
    for offset in (-0.4, -0.1, 0.1, 0.4):
        worse = float(pinball_loss(torch.full((20_000, 1), float(truth) + offset, dtype=DT), y, [level]))
        assert worse > best


def test_pinball_loss_matches_the_definition():
    pred = torch.tensor([[1.0, 2.0]], dtype=DT)
    y = torch.tensor([3.0], dtype=DT)
    levels = [0.1, 0.9]
    # Errors are +2 and +1, both positive, so the loss is p * e.
    assert float(pinball_loss(pred, y, levels)) == pytest.approx((0.1 * 2 + 0.9 * 1) / 2)
    with pytest.raises(ValueError, match="levels"):
        pinball_loss(pred, y, [0.5])
