"""The operator and its post-training whitening step.

The whitening step is pure linear algebra, so it can be checked exactly:
against a hand-written least-squares projection, and against the properties it
is supposed to establish (centered, orthonormal singular functions; singular
values that are genuine canonical correlations).
"""

import pytest
import torch

from posterior_operator import MLP, NCPOperator, build_ncp
from posterior_operator.operator import _inv_sqrt_psd


@pytest.fixture
def operator():
    torch.manual_seed(0)
    return build_ncp(x_dim=2, y_dim=1, latent_dim=5, n_hidden=1, layer_size=16)


@pytest.fixture
def joint_sample():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(400, 2, generator=g)
    y = (x[:, :1] + 0.5 * torch.randn(400, 1, generator=g)).contiguous()
    return x, y


# --------------------------------------------------------------------------- #
# Guard rails
# --------------------------------------------------------------------------- #


def test_inference_requires_fitted_statistics(operator, joint_sample):
    x, y = joint_sample
    assert not operator.is_fitted
    for call in (
        lambda: operator.embed_x(x),
        lambda: operator.embed_y(y),
        lambda: operator.deflated_ratio(x, y),
        lambda: operator.condition(x),
        lambda: operator.singular_values,
    ):
        with pytest.raises(RuntimeError, match="fit_statistics"):
            call()


def test_mismatched_latent_dims_are_rejected():
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        NCPOperator(MLP(1, 4), MLP(1, 4), latent_dim=0)


def test_fit_statistics_needs_more_samples_than_latent_dimensions():
    op = build_ncp(1, 1, latent_dim=8)
    g = torch.Generator().manual_seed(2)
    with pytest.raises(ValueError, match="more samples than latent dimensions"):
        op.fit_statistics(torch.randn(5, 1, generator=g), torch.randn(5, 1, generator=g))


def test_rank_is_validated(operator, joint_sample):
    operator.fit_statistics(*joint_sample)
    with pytest.raises(ValueError, match=r"rank must be in \[1, 5\]"):
        operator.embed_x(joint_sample[0], rank=6)
    with pytest.raises(ValueError, match=r"rank must be in \[1, 5\]"):
        operator.embed_x(joint_sample[0], rank=0)


def test_one_dimensional_inputs_are_promoted(joint_sample):
    op = build_ncp(1, 1, latent_dim=4)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(200, generator=g)  # no feature axis
    y = torch.randn(200, generator=g)
    op.fit_statistics(x, y)
    assert op.embed_x(x).shape == (200, 4)


# --------------------------------------------------------------------------- #
# What whitening establishes
# --------------------------------------------------------------------------- #


def test_whitened_embeddings_are_centered_and_orthonormal(operator, joint_sample):
    x, y = joint_sample
    operator.fit_statistics(x, y, reg=0.0)
    for embed, data in ((operator.embed_x, x), (operator.embed_y, y)):
        z = embed(data).double()
        assert torch.allclose(z.mean(dim=0), torch.zeros(5, dtype=torch.float64), atol=1e-4)
        cov = (z - z.mean(0)).T @ (z - z.mean(0)) / (z.shape[0] - 1)
        assert torch.allclose(cov, torch.eye(5, dtype=torch.float64), atol=1e-4)


def test_singular_values_are_sorted_canonical_correlations(operator, joint_sample):
    operator.fit_statistics(*joint_sample)
    sv = operator.singular_values
    assert sv.shape == (5,)
    assert torch.all(sv >= 0) and torch.all(sv <= 1)
    assert torch.all(sv[:-1] >= sv[1:] - 1e-6)


def test_deflated_ratio_equals_the_least_squares_projection(operator, joint_sample):
    """Reproduce phi_c^T C_phi^-1 C_phi_psi C_psi^-1 psi_c by hand."""
    x, y = joint_sample
    operator.fit_statistics(x, y, reg=0.0)

    with torch.no_grad():
        sqrt_s = operator.singular_layer.values.sqrt()
        phi = (operator.x_embedding(x) * sqrt_s).double()
        psi = (operator.y_embedding(y) * sqrt_s).double()
    n = phi.shape[0]
    phi_c = phi - phi.mean(0)
    psi_c = psi - psi.mean(0)
    cov_phi = phi_c.T @ phi_c / (n - 1)
    cov_psi = psi_c.T @ psi_c / (n - 1)
    cov_cross = phi_c.T @ psi_c / (n - 1)
    coefficients = torch.linalg.solve(cov_phi, cov_cross) @ torch.linalg.inv(cov_psi)
    expected = phi_c @ coefficients @ psi_c.T

    got = operator.deflated_ratio(x, y).double()
    assert torch.allclose(got, expected, atol=1e-3, rtol=1e-3)


def test_rank_truncation_is_the_top_rank_approximation(operator, joint_sample):
    x, y = joint_sample
    operator.fit_statistics(x, y)
    full_u = operator.embed_x(x[:20], rank=5)
    full_v = operator.embed_y(y[:30], rank=5)
    sv = operator.singular_values
    for rank in (1, 3, 5):
        expected = (full_u[:, :rank] * sv[:rank]) @ full_v[:, :rank].T
        assert torch.allclose(operator.deflated_ratio(x[:20], y[:30], rank=rank), expected, atol=1e-5)


def test_deflated_ratio_shapes(operator, joint_sample):
    x, y = joint_sample
    operator.fit_statistics(x, y)
    assert operator.deflated_ratio(x[:7], y[:13]).shape == (7, 13)


# --------------------------------------------------------------------------- #
# Conditional weights
# --------------------------------------------------------------------------- #


def test_conditional_weights_form_a_probability_vector(operator, joint_sample):
    x, y = joint_sample
    operator.fit_statistics(x, y)
    weights, atoms = operator.conditional_weights(x[:10])
    assert weights.shape == (10, atoms.shape[0])
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(10), atol=1e-5)


def test_unclipped_weights_sum_to_one_but_may_be_negative(operator, joint_sample):
    """Without clipping the masses still integrate to one, by construction."""
    x, y = joint_sample
    operator.fit_statistics(x, y)
    raw, _ = operator.conditional_weights(x[:10], clip=False)
    assert torch.allclose(raw.sum(dim=-1), torch.ones(10), atol=1e-4)


def test_conditional_weights_track_the_conditioning_value(operator, joint_sample):
    """Different x must give different laws, or nothing has been learned."""
    x, y = joint_sample
    operator.fit_statistics(x, y)
    weights, _ = operator.conditional_weights(torch.tensor([[-2.0, 0.0], [2.0, 0.0]]))
    assert (weights[0] - weights[1]).abs().max() > 1e-6


def test_explicit_reference_sample_is_used(operator, joint_sample):
    x, y = joint_sample
    operator.fit_statistics(x, y)
    atoms = torch.linspace(-2, 2, 17).reshape(-1, 1)
    weights, used = operator.conditional_weights(x[:4], y_reference=atoms)
    assert weights.shape == (4, 17)
    assert torch.allclose(used, atoms)


def test_reference_sample_is_capped(joint_sample):
    op = build_ncp(2, 1, latent_dim=4)
    x, y = joint_sample
    op.fit_statistics(x, y, max_reference=50)
    assert op.reference_y.shape == (50, 1)


def test_missing_reference_sample_is_reported(joint_sample):
    op = build_ncp(2, 1, latent_dim=4)
    op.fit_statistics(*joint_sample, store_reference=False)
    with pytest.raises(RuntimeError, match="no reference Y sample"):
        op.condition(joint_sample[0][:3])


# --------------------------------------------------------------------------- #
# Serialisation and numerics
# --------------------------------------------------------------------------- #


def test_state_dict_round_trip_preserves_inference(operator, joint_sample):
    x, y = joint_sample
    operator.fit_statistics(x, y)
    before = operator.condition(x[:8]).mean()

    clone = build_ncp(x_dim=2, y_dim=1, latent_dim=5, n_hidden=1, layer_size=16)
    clone.load_state_dict(operator.state_dict())
    assert clone.is_fitted
    assert torch.allclose(clone.condition(x[:8]).mean(), before, atol=1e-6)
    assert torch.allclose(clone.reference_y, operator.reference_y)


def test_chunked_moment_accumulation_matches_single_pass(operator, joint_sample):
    x, y = joint_sample
    operator.fit_statistics(x, y, chunk_size=10_000)
    one_pass = operator.singular_values.clone()
    operator.fit_statistics(x, y, chunk_size=37)
    assert torch.allclose(operator.singular_values, one_pass, atol=1e-5)


def test_inv_sqrt_psd_inverts_the_square_root():
    g = torch.Generator().manual_seed(5)
    root = torch.randn(6, 6, generator=g, dtype=torch.float64)
    cov = root @ root.T + torch.eye(6, dtype=torch.float64)
    inv_sqrt = _inv_sqrt_psd(cov, reg=0.0)
    assert torch.allclose(inv_sqrt @ cov @ inv_sqrt, torch.eye(6, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(inv_sqrt, inv_sqrt.T, atol=1e-10)


def test_inv_sqrt_psd_survives_a_singular_matrix():
    """A rank-deficient feature covariance must not produce NaNs or infinities."""
    g = torch.Generator().manual_seed(7)
    root = torch.randn(6, 3, generator=g, dtype=torch.float64)
    cov = root @ root.T  # rank 3 out of 6
    out = _inv_sqrt_psd(cov, reg=1e-8)
    assert torch.isfinite(out).all()


def test_collinear_embeddings_still_fit(joint_sample):
    """Latent dimension far above the intrinsic rank: regularisation must hold."""
    x, y = joint_sample
    op = build_ncp(2, 1, latent_dim=40, n_hidden=1, layer_size=8)
    op.fit_statistics(x, y, reg=1e-4)
    assert torch.isfinite(op.singular_values).all()
    weights, _ = op.conditional_weights(x[:5])
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-4)


def test_summary_reports_fit_state(operator, joint_sample):
    assert "fitted=False" in operator.parameters_summary()
    operator.fit_statistics(*joint_sample)
    assert "top_singular_values" in operator.parameters_summary()
