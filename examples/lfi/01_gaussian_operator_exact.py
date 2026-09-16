"""The Gaussian toy model, checked against exact theory.

This is the first item of the experimental protocol: the jointly Gaussian case,
where the conditional expectation operator is known in closed form, so a fitted
operator can be checked against the truth rather than against another
estimator.

It runs four checks:

1. Canonical correlation analysis reconstructs the posterior mean
   Sigma_{ThetaY} Sigma_{YY}^{-1} y_0 to floating-point precision.
2. The exact L^2 spectrum of the deflated operator is the Hermite tensor system
   with sigma_a = prod_i rho_i^{a_i}, verified by quadrature. A Gaussian pair
   therefore has infinitely many non-zero singular values even when
   rank(Sigma_{ThetaY}) is 1, and the canonical correlations are only the
   |a| = 1 subset of the spectrum.
3. Consequence for truncation: a nonlinear direction of a strongly correlated
   canonical pair can outrank a weakly correlated *linear* one, so the rank-r*
   truncation can drop a linear direction and miss part of E[Theta | Y].
   The script reports the rank actually needed.
4. The learned operator against that exact spectrum as n grows, and against
   direct regression for the single functional f = id.

Run:  python examples/lfi/01_gaussian_operator_exact.py
"""

import math

import numpy as np
import torch

from posterior_operator import PosteriorOperator
from posterior_operator.simulators import GaussianLinear

torch.set_printoptions(precision=4, sci_mode=False)


# --------------------------------------------------------------------------- #
# 1. CCA reconstructs the posterior mean exactly
# --------------------------------------------------------------------------- #


def check_cca_reconstruction() -> None:
    print("=" * 78)
    print("1. CCA reconstruction of the posterior mean")
    print("=" * 78)
    sim = GaussianLinear(theta_dim=3, data_dim=5, noise=0.5, seed=0, dtype=torch.float64)
    _, y = sim.sample_joint(64, generator=torch.Generator().manual_seed(1))

    # Textbook Gaussian conditioning.
    textbook = sim.posterior_mean(y)

    # Reconstruction through the canonical system: whiten both sides, take the
    # SVD, and sum sigma_k * u_k(y) over the canonical directions.
    inv_sqrt_theta = _inv_sqrt(sim.prior_cov)
    inv_sqrt_y = _inv_sqrt(sim.cov_yy)
    left, svals, right_h = torch.linalg.svd(
        inv_sqrt_theta @ sim.cov_theta_y @ inv_sqrt_y, full_matrices=False
    )
    v_coef = inv_sqrt_theta @ left  # theta -> canonical variates
    u_coef = inv_sqrt_y @ right_h.T  # y -> canonical variates
    canonical = (y @ u_coef * svals) @ torch.linalg.inv(v_coef)

    err = float((canonical - textbook).abs().max())
    print(f"canonical correlations: {svals.numpy().round(6)}")
    print(f"max |CCA reconstruction - Sigma_ThetaY Sigma_YY^-1 y_0| = {err:.2e}")
    print("=> the canonical system does reproduce the posterior mean exactly.\n")


def _inv_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(matrix)
    return (evecs * evals.rsqrt()) @ evecs.T


# --------------------------------------------------------------------------- #
# 2. The exact L^2 spectrum is the Hermite system
# --------------------------------------------------------------------------- #


def _orthonormal_hermite(x: torch.Tensor, order: int) -> list:
    """Hermite polynomials orthonormal under the standard normal."""
    out = [torch.ones_like(x), x]
    for j in range(1, order):
        out.append((x * out[j] - math.sqrt(j) * out[j - 1]) / math.sqrt(j + 1))
    return out[: order + 1]


def check_hermite_spectrum(rho: float = 0.8, order: int = 7, n_quad: int = 60) -> None:
    print("=" * 78)
    print("2. The exact spectrum of the deflated operator (scalar Gaussian pair)")
    print("=" * 78)
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_quad)
    nodes = torch.tensor(nodes, dtype=torch.float64)
    weights = torch.tensor(weights, dtype=torch.float64) / math.sqrt(2 * math.pi)

    a, b = torch.meshgrid(nodes, nodes, indexing="ij")
    wa, wb = torch.meshgrid(weights, weights, indexing="ij")
    quad_w = (wa * wb).reshape(-1)
    theta = a.reshape(-1)
    y = (rho * a + math.sqrt(1 - rho**2) * b).reshape(-1)

    h_theta = _orthonormal_hermite(theta, order)
    h_y = _orthonormal_hermite(y, order)
    matrix = torch.stack(
        [torch.stack([(quad_w * h_y[i] * h_theta[j]).sum() for j in range(order + 1)]) for i in range(order + 1)]
    )
    diag = torch.diagonal(matrix)
    off = float((matrix - torch.diag(diag)).abs().max())

    print(f"rho = {rho};  rank(Sigma_ThetaY) = 1")
    print(f"  operator diagonal in the Hermite basis: {[round(v, 5) for v in diag.tolist()]}")
    print(f"  rho^k                                 : {[round(rho**k, 5) for k in range(order + 1)]}")
    print(f"  max off-diagonal entry: {off:.2e}")
    print(f"  chi^2 from the truncated spectrum: {float((diag[1:] ** 2).sum()):.4f}")
    print(f"  chi^2 in closed form, rho^2/(1-rho^2): {rho**2 / (1 - rho**2):.4f}")
    print("=> sigma_k = rho^k for every k >= 1: a scalar Gaussian pair has a rank-1")
    print("   cross-covariance but INFINITELY many non-zero singular values. The gap")
    print("   between the two chi^2 values above is the tail k > 7 that was dropped.\n")


# --------------------------------------------------------------------------- #
# 3. What that means for rank-d truncation of E[Theta | Y]
# --------------------------------------------------------------------------- #


def check_truncation_of_the_posterior_mean() -> None:
    print("=" * 78)
    print("3. Rank-d truncation of the posterior mean")
    print("=" * 78)
    for label, sim in [
        ("well separated", GaussianLinear(theta_dim=2, data_dim=2, noise=0.7, seed=3, dtype=torch.float64)),
        ("r* = 3", GaussianLinear(theta_dim=3, data_dim=5, noise=0.5, seed=0, dtype=torch.float64)),
    ]:
        rho = sim.canonical_correlations()
        values, indices = sim.exact_spectrum(max_order=30)
        ranks = sim.linear_direction_ranks(max_order=30)
        print(f"\n[{label}]  canonical correlations rho = {rho.numpy().round(4)}   (r* = {rho.numel()})")
        print(f"  top 8 exact singular values : {values[:8].numpy().round(4)}")
        print(f"  their multi-indices         : {indices[:8]}")
        for i, (r, position) in enumerate(zip(rho.tolist(), ranks), start=1):
            print(f"  linear direction e_{i} (rho = {r:.4f}) sits at spectral position {position}")
        needed = max(ranks)
        verdict = "HOLDS" if needed == rho.numel() else "FAILS"
        print(f"  rank-{rho.numel()} truncation keeps {indices[: rho.numel()]}")
        print(f"  condition rho_1^2 < rho_r*:  {float(rho[0]) ** 2:.4f} < {float(rho[-1]):.4f}"
              f"  =>  exactness at d = r* {verdict}")
        print("  exact RMS truncation error of E[Theta|Y] by rank:")
        for d in sorted({1, 2, rho.numel(), needed - 1, needed, needed + 2}):
            if d < 1:
                continue
            err = sim.truncation_error_posterior_mean(d)
            flag = "  <- exact" if err == 0.0 else ""
            print(f"      d = {d:>3}: {err:.4f}{flag}")
        if needed == rho.numel():
            print(f"  => exactness at d >= r* = {rho.numel()}, as the proposition claims.")
        else:
            print(f"  => exactness needs d >= {needed}, NOT d >= r* = {rho.numel()}.")
    print()
    print("  The reason: the top-d directions are ordered by singular value, and a")
    print("  nonlinear Hermite direction of a strongly correlated canonical pair")
    print("  (sigma = rho_1^2, rho_1^3, ...) can outrank a weakly correlated LINEAR")
    print("  direction (sigma = rho_j). Truncating at r* then drops a linear")
    print("  direction that E[Theta|Y] needs.")
    print()
    print("  The sharp statement: rank-d truncation recovers E[Theta|Y] exactly iff")
    print("      d >= #{ a != 0 : prod_i rho_i^{a_i} >= rho_{r*} },")
    print("  which collapses to d >= r* exactly when rho_1^2 < rho_{r*} -- the first")
    print("  case above. Otherwise the requirement is strictly larger, as in the")
    print("  second, and it grows quickly as rho_1 -> 1.\n")


# --------------------------------------------------------------------------- #
# 4. The learned operator against the exact spectrum
# --------------------------------------------------------------------------- #


def check_learned_operator(sizes=(5000, 20000, 80000), rank: int = 32) -> None:
    print("=" * 78)
    print("4. The learned operator vs the exact spectrum, as n grows")
    print("=" * 78)
    sim = GaussianLinear(theta_dim=3, data_dim=5, noise=0.5, seed=0)
    exact_values, _ = sim.exact_spectrum(max_order=30)
    exact_top = exact_values[:8]
    rho = sim.canonical_correlations()
    print(f"exact top-8 spectrum : {exact_top.numpy().round(4)}")
    print(f"exact sigma_1 (HGR maximal correlation) = {float(rho[0]):.4f}\n")

    g = torch.Generator().manual_seed(0)
    theta_test, y_test = sim.sample_joint(500, generator=g)
    exact_mean = sim.posterior_mean(y_test)

    print(f"{'n':>8}  {'sigma_1':>8}  {'err(sigma_1)':>12}  {'err(top 8)':>11}  {'chi^2':>7}  {'mean RMSE':>10}")
    print("-" * 68)
    for n in sizes:
        gg = torch.Generator().manual_seed(1)
        theta, y = sim.sample_joint(n, generator=gg)
        torch.manual_seed(0)
        op = PosteriorOperator(theta_dim=3, data_dim=5, rank=rank, layer_size=64)
        op.fit(theta, y, epochs=600, lr=1e-3, seed=0)
        sv = op.singular_values[:8]
        rmse = float(((op.posterior(y_test).mean() - exact_mean) ** 2).mean().sqrt())
        print(
            f"{n:>8}  {op.maximal_correlation:>8.4f}  {abs(op.maximal_correlation - float(rho[0])):>12.4f}"
            f"  {float((sv - exact_top).abs().max()):>11.4f}  {op.chi2_divergence:>7.2f}  {rmse:>10.4f}"
        )

    # Direct regression for the one functional f = id (Section 3). In this model
    # the posterior mean is linear in y, so least squares is essentially exact.
    gg = torch.Generator().manual_seed(1)
    theta, y = sim.sample_joint(sizes[-1], generator=gg)
    design = torch.cat([y, torch.ones(y.shape[0], 1)], dim=1)
    coef = torch.linalg.lstsq(design, theta).solution
    fitted = torch.cat([y_test, torch.ones(y_test.shape[0], 1)], dim=1) @ coef
    print("-" * 68)
    print(f"direct regression of theta on y (Section 3, f = id): RMSE {float(((fitted - exact_mean) ** 2).mean().sqrt()):.4f}")
    print(f"prior-mean baseline:                                 RMSE {float((exact_mean**2).mean().sqrt()):.4f}")
    print()
    print("  The spectrum is recovered well, and sigma_1 to a few parts in a thousand.")
    print("  The posterior mean is not: direct regression wins by a wide margin here,")
    print("  which is the sharp form of the remark that rank-one truncation and direct")
    print("  regression differ. On a Gaussian model the Hermite tail crowds the linear")
    print("  directions out of the top of the spectrum, so the rank needed for the")
    print("  posterior mean is far above r* (see check 3). When only one functional is")
    print("  ever wanted, regress on it directly; the operator earns its cost when many")
    print("  functionals are queried after training.\n")


if __name__ == "__main__":
    check_cca_reconstruction()
    check_hermite_spectrum()
    check_truncation_of_the_posterior_mean()
    check_learned_operator()
