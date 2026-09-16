"""The training objective: does it estimate what the derivation says it does?

The tests pin the estimators to an analytically known population objective on
a finite joint distribution, where the deflated ratio ``r`` can be written down
exactly and every expectation is a finite sum.
"""

import math

import pytest
import torch

from posterior_operator.losses import (
    NCPLoss,
    log_fro_penalty,
    orthonormality_penalty,
    split_objective,
    ustat_objective,
)


# --------------------------------------------------------------------------- #
# A finite joint distribution, where everything is a finite sum.
# --------------------------------------------------------------------------- #

K, L = 4, 5


def _random_joint(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    joint = torch.rand(K, L, generator=g, dtype=torch.float64) + 0.2
    joint /= joint.sum()
    return joint


def _marginals(joint):
    return joint.sum(dim=1), joint.sum(dim=0)


def _deflated_ratio(joint):
    px, py = _marginals(joint)
    return joint / torch.outer(px, py) - 1.0


def _population_objective(h, joint):
    r"""``E_prod[h^2] - 2(E_joint[h] - E_prod[h])`` by direct summation."""
    px, py = _marginals(joint)
    prod = torch.outer(px, py)
    return (prod * h**2).sum() - 2.0 * ((joint * h).sum() - (prod * h).sum())


def _embed(h, x_idx, y_idx):
    """Express an arbitrary ``h`` in the ``(u, v, s)`` parametrisation.

    One-hot ``u`` and ``v = h[:, y]`` reproduce any table exactly with ``s = 1``,
    so the estimators can be checked against the population value of *that*
    table rather than of whatever a network happens to learn.
    """
    u = torch.nn.functional.one_hot(x_idx, K).to(h.dtype)
    v = h.T[y_idx]  # (n, K); row y of h^T is h[:, y]
    s = torch.ones(K, dtype=h.dtype)
    return u, v, s


def _sample(joint, n, generator):
    flat = torch.multinomial(joint.reshape(-1), n, replacement=True, generator=generator)
    return flat // L, flat % L


def test_objective_equals_squared_distance_to_the_deflated_ratio():
    """The population objective is ||h - r||^2 minus a constant independent of h."""
    joint = _random_joint()
    px, py = _marginals(joint)
    prod = torch.outer(px, py)
    r = _deflated_ratio(joint)
    norm_r = (prod * r**2).sum()

    g = torch.Generator().manual_seed(3)
    for _ in range(5):
        h = torch.randn(K, L, generator=g, dtype=torch.float64)
        expected = (prod * (h - r) ** 2).sum() - norm_r
        assert torch.allclose(_population_objective(h, joint), expected, atol=1e-10)


def test_deflated_ratio_is_the_unique_minimiser():
    """No perturbation of ``h = r`` lowers the objective."""
    joint = _random_joint()
    r = _deflated_ratio(joint)
    best = _population_objective(r, joint)
    g = torch.Generator().manual_seed(5)
    for _ in range(20):
        step = torch.randn(K, L, generator=g, dtype=torch.float64) * 0.3
        assert _population_objective(r + step, joint) > best + 1e-12


@pytest.mark.parametrize("estimator", [ustat_objective, split_objective])
def test_estimators_are_unbiased_for_the_population_objective(estimator):
    joint = _random_joint()
    g = torch.Generator().manual_seed(11)
    h = torch.randn(K, L, generator=g, dtype=torch.float64)
    target = _population_objective(h, joint)

    values = []
    for _ in range(600):
        x_idx, y_idx = _sample(joint, 64, g)
        values.append(estimator(*_embed(h, x_idx, y_idx)))
    estimate = torch.stack(values)
    # Monte Carlo standard error of the mean; unbiasedness means the gap is
    # within a few of those, and nothing larger.
    stderr = estimate.std() / math.sqrt(estimate.numel())
    assert abs(float(estimate.mean() - target)) < 4.0 * float(stderr) + 1e-9


def test_ustat_matches_a_brute_force_double_loop():
    """Guards the Gram-free identities used to avoid an n-by-n matrix."""
    g = torch.Generator().manual_seed(17)
    n, d = 9, 3
    u = torch.randn(n, d, generator=g, dtype=torch.float64)
    v = torch.randn(n, d, generator=g, dtype=torch.float64)
    s = torch.rand(d, generator=g, dtype=torch.float64)

    prod_h2 = prod_h = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            h_ij = float((s * u[i] * v[j]).sum())
            prod_h2 += h_ij**2
            prod_h += h_ij
    pairs = n * (n - 1)
    joint_h = sum(float((s * u[i] * v[i]).sum()) for i in range(n)) / n
    expected = prod_h2 / pairs - 2.0 * (joint_h - prod_h / pairs)

    assert ustat_objective(u, v, s).item() == pytest.approx(expected, rel=1e-10)


def test_ustat_has_lower_variance_than_split():
    """The all-pairs estimator should be the tighter of the two, as advertised."""
    joint = _random_joint()
    g = torch.Generator().manual_seed(23)
    h = torch.randn(K, L, generator=g, dtype=torch.float64)

    spread = {}
    for name, estimator in (("ustat", ustat_objective), ("split", split_objective)):
        vals = []
        for seed in range(200):
            gg = torch.Generator().manual_seed(1000 + seed)
            x_idx, y_idx = _sample(joint, 64, gg)
            vals.append(estimator(*_embed(h, x_idx, y_idx)))
        spread[name] = float(torch.stack(vals).std())
    assert spread["ustat"] < spread["split"]


# --------------------------------------------------------------------------- #
# Penalties
# --------------------------------------------------------------------------- #


def test_orthonormality_penalty_is_unbiased():
    g = torch.Generator().manual_seed(29)
    d = 3
    root = torch.randn(d, d, generator=g, dtype=torch.float64)
    second_moment = root @ root.T + torch.eye(d, dtype=torch.float64)
    chol = torch.linalg.cholesky(second_moment)
    target = ((second_moment - torch.eye(d, dtype=torch.float64)) ** 2).sum()

    vals = torch.stack(
        [
            orthonormality_penalty(torch.randn(48, d, generator=g, dtype=torch.float64) @ chol.T)
            for _ in range(800)
        ]
    )
    stderr = vals.std() / math.sqrt(vals.numel())
    assert abs(float(vals.mean() - target)) < 4.0 * float(stderr)


def test_orthonormality_penalty_is_minimised_at_orthonormal_features():
    g = torch.Generator().manual_seed(31)
    d, n = 4, 4000
    z = torch.randn(n, d, generator=g, dtype=torch.float64)
    # Whiten so the empirical second moment is exactly the identity.
    moment = z.T @ z / n
    evals, evecs = torch.linalg.eigh(moment)
    white = z @ (evecs * evals.rsqrt()) @ evecs.T
    assert abs(float(orthonormality_penalty(white))) < 1e-2
    assert float(orthonormality_penalty(white * 2.0)) > 1.0
    assert float(orthonormality_penalty(white * 0.2)) > 1.0


def test_log_fro_penalty_is_minimised_at_the_identity():
    g = torch.Generator().manual_seed(37)
    d, n = 4, 4000
    z = torch.randn(n, d, generator=g, dtype=torch.float64)
    moment = z.T @ z / n
    evals, evecs = torch.linalg.eigh(moment)
    white = z @ (evecs * evals.rsqrt()) @ evecs.T
    base = float(log_fro_penalty(white))
    assert base == pytest.approx(0.0, abs=1e-6)
    assert float(log_fro_penalty(white * 1.5)) > base
    # Diverges towards a collapsed subspace, unlike the Frobenius penalty,
    # growing like -log(lambda_min)/d as the direction is squeezed out.
    previous = base
    for shrink in (1e-2, 1e-4, 1e-6):
        collapsed = white.clone()
        collapsed[:, 0] *= shrink
        value = float(log_fro_penalty(collapsed))
        assert value == pytest.approx(-2 * math.log(shrink) / d, rel=1e-3)
        assert value > previous
        previous = value


# --------------------------------------------------------------------------- #
# The loss wrapper
# --------------------------------------------------------------------------- #


def test_loss_combines_fit_and_penalty_with_gamma():
    g = torch.Generator().manual_seed(41)
    u = torch.randn(32, 5, generator=g, dtype=torch.float64)
    v = torch.randn(32, 5, generator=g, dtype=torch.float64)
    s = torch.rand(5, generator=g, dtype=torch.float64)

    loss = NCPLoss(mode="ustat", gamma=0.25)
    fit, pen = loss.parts(u, v, s)
    assert loss(u, v, s).item() == pytest.approx(float(fit + 0.25 * pen), rel=1e-12)
    assert NCPLoss(gamma=0.0)(u, v, s).item() == pytest.approx(float(fit), rel=1e-12)


def test_loss_gradients_reach_every_parameter_group():
    torch.manual_seed(0)
    u = torch.randn(40, 6, dtype=torch.float64, requires_grad=True)
    v = torch.randn(40, 6, dtype=torch.float64, requires_grad=True)
    s = torch.rand(6, dtype=torch.float64, requires_grad=True)
    NCPLoss(gamma=1e-2)(u, v, s).backward()
    for name, tensor in (("u", u), ("v", v), ("s", s)):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.abs().sum() > 0, name


def test_split_mode_is_reproducible_with_a_generator():
    u = torch.randn(40, 6, dtype=torch.float64)
    v = torch.randn(40, 6, dtype=torch.float64)
    s = torch.rand(6, dtype=torch.float64)
    first = split_objective(u, v, s, generator=torch.Generator().manual_seed(7))
    second = split_objective(u, v, s, generator=torch.Generator().manual_seed(7))
    assert first.item() == pytest.approx(second.item(), rel=1e-12)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(mode="nope"), "unknown mode"),
        (dict(penalty="nope"), "unknown penalty"),
        (dict(gamma=-1.0), "non-negative"),
    ],
)
def test_loss_rejects_bad_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        NCPLoss(**kwargs)


def test_objectives_validate_shapes():
    u = torch.randn(10, 4)
    with pytest.raises(ValueError, match="same shape"):
        ustat_objective(u, torch.randn(10, 3), torch.ones(4))
    with pytest.raises(ValueError, match="singular values"):
        ustat_objective(u, u, torch.ones(3))
    with pytest.raises(ValueError, match="2D"):
        ustat_objective(torch.randn(10), torch.randn(10), torch.ones(1))
    with pytest.raises(ValueError, match="at least 2 samples"):
        ustat_objective(u[:1], u[:1], torch.ones(4))
