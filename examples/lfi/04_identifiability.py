"""Reading identifiability off the spectrum, and where the low-rank model strains.

Three diagnostics that come out of the fit itself, with no extra machinery:

1. sigma_1 is the Hirschfeld-Gebelein-Renyi maximal correlation between Theta
   and Y -- the largest correlation achievable between any square-integrable
   feature of the parameter and any feature of the data. It is a model-free
   replacement for eyeballing scatterplots of simulated (theta_i, y_i) pairs.
   On SumIdentified, where only theta_1 + theta_2 is identified, the value is
   known exactly, and the leading singular function should recover the
   identified direction.

2. sum_k sigma_k^2 = chi^2(rho || pi x mu) is the squared Hilbert-Schmidt norm.
   Finiteness of the population version is the compactness condition the
   expansion relies on, and it fails for near-deterministic simulators. Shrink
   the observation noise and watch the diagnostic fire.

3. The truncation rank needed grows with how much the posterior concentrates
   relative to the prior. This is measured here rather than assumed, because it
   determines how far the low-rank summary can be pushed.

Run:  python examples/lfi/04_identifiability.py
"""

import math

import torch

from posterior_operator import PosteriorOperator
from posterior_operator.simulators import MA2, SumIdentified

SEED = 4
N_SIM = 30000


# --------------------------------------------------------------------------- #
# 1. sigma_1 as a maximal correlation, and v_1 as the identified direction
# --------------------------------------------------------------------------- #


def check_maximal_correlation() -> None:
    print("=" * 80)
    print("1. sigma_1 = maximal correlation; v_1 = the identified direction")
    print("=" * 80)
    sim = SumIdentified(n_obs=10, noise=1.0)
    g = torch.Generator().manual_seed(SEED)
    theta, y = sim.sample_joint(N_SIM, generator=g)

    torch.manual_seed(SEED)
    operator = PosteriorOperator(theta_dim=2, data_dim=sim.data_dim, rank=32, layer_size=64)
    operator.fit(theta, y, epochs=500, lr=1e-3, seed=SEED)

    exact = sim.maximal_correlation()
    print(f"  model: y_j ~ N(theta_1 + theta_2, {sim.noise}^2), j = 1..{sim.n_obs}; theta ~ N(0, I_2)")
    print("  only theta_1 + theta_2 is identified; theta_1 - theta_2 keeps its prior exactly.")
    print(f"\n  sigma_1 estimated : {operator.maximal_correlation:.4f}")
    print(f"  sigma_1 exact     : {exact:.4f}   (error {abs(operator.maximal_correlation - exact):.4f})")
    # The second singular value is NOT zero even though only one linear feature
    # is identified: the operator is diagonal in the Hermite basis, so the
    # squared Hermite polynomial of the same identified feature contributes
    # sigma = rho_1^2. Only the second *linear* direction is absent.
    print(f"  sigma_2 estimated : {float(operator.singular_values[1]):.4f}"
          f"   (exact rho_1^2 = {exact**2:.4f}, the SQUARED Hermite direction")
    print("                       of the same identified feature -- not a second one)")

    # The leading singular function of a Gaussian pair is linear, so its
    # direction can be read off by least squares against theta.
    probe = sim.sample_prior(20000, generator=g)
    v1 = operator.singular_function_theta(probe)[:, 0]
    design = torch.cat([probe, torch.ones(probe.shape[0], 1)], dim=1)
    coefficients = torch.linalg.lstsq(design, v1.unsqueeze(-1)).solution[:2, 0]
    direction = coefficients / coefficients.norm()
    identified = sim.identified_direction
    alignment = float((direction @ identified).abs())
    residual = float(1 - (design @ torch.linalg.lstsq(design, v1.unsqueeze(-1)).solution).squeeze().var() / v1.var())
    print(f"\n  v_1 direction recovered by least squares : [{direction[0]:+.4f}, {direction[1]:+.4f}]")
    print(f"  identified direction (1,1)/sqrt(2)       : [{identified[0]:+.4f}, {identified[1]:+.4f}]")
    print(f"  |cos angle| between them                 : {alignment:.4f}")
    print(f"  fraction of v_1 not explained by a linear function of theta: {abs(residual):.4f}")

    # The posterior must update along the identified direction only.
    _, y_obs = sim.sample_joint(200, generator=g)
    posterior = operator.posterior(y_obs)
    exact_mean, exact_cov = sim.posterior_mean_cov(y_obs)
    unidentified = torch.tensor([1.0, -1.0]) / math.sqrt(2.0)
    print("\n  posterior spread along each direction (200 observations):")
    for label, vec in (("identified   (1, 1)/sqrt2", identified), ("unidentified (1,-1)/sqrt2", unidentified)):
        projected = posterior.functional(lambda t, v=vec: (t @ v).unsqueeze(-1))
        second = posterior.functional(lambda t, v=vec: ((t @ v) ** 2).unsqueeze(-1))
        sd = (second - projected**2).clamp_min(0).sqrt().mean()
        exact_sd = math.sqrt(float(vec @ exact_cov @ vec))
        print(f"    {label}: estimated sd {float(sd):.4f}   exact {exact_sd:.4f}")
    print("\n  => the unidentified direction keeps its prior sd of 1, and the fit says so")
    print("     without being told which direction was identified.\n")


# --------------------------------------------------------------------------- #
# 2. The compactness diagnostic
# --------------------------------------------------------------------------- #


def check_compactness() -> None:
    print("=" * 80)
    print("2. chi^2 and the near-deterministic-simulator regime")
    print("=" * 80)
    print("  As the observation noise shrinks, the data pin the identified direction")
    print("  down exactly, the joint law concentrates near a lower-dimensional")
    print("  manifold, the density ratio leaves L^2(pi x mu) and sigma_1 -> 1.\n")
    print(f"{'noise sigma':>12}{'exact sigma_1':>15}{'est sigma_1':>13}{'sum sigma_k^2':>15}{'ESS':>9}")
    print("-" * 64)
    for noise in (2.0, 1.0, 0.5, 0.2, 0.05):
        sim = SumIdentified(n_obs=10, noise=noise)
        g = torch.Generator().manual_seed(SEED)
        theta, y = sim.sample_joint(N_SIM, generator=g)
        torch.manual_seed(SEED)
        operator = PosteriorOperator(theta_dim=2, data_dim=sim.data_dim, rank=32, layer_size=64)
        operator.fit(theta, y, epochs=400, lr=1e-3, seed=SEED)
        _, y_obs = sim.sample_joint(100, generator=g)
        ess = float(operator.posterior(y_obs).effective_sample_size().mean())
        print(
            f"{noise:>12.2f}{sim.maximal_correlation():>15.4f}{operator.maximal_correlation:>13.4f}"
            f"{operator.chi2_divergence:>15.2f}{ess:>9.0f}"
        )
    print("-" * 64)
    print("  sigma_1 climbing towards 1 while chi^2 keeps growing with the rank is the")
    print("  warning sign. The remedy in that regime is to add observation noise or")
    print("  coarsen y before fitting, not to raise the rank.\n")


# --------------------------------------------------------------------------- #
# 3. Required rank vs posterior concentration
# --------------------------------------------------------------------------- #


def check_concentration(rank: int = 48) -> None:
    print("=" * 80)
    print("3. How the low-rank posterior degrades as the data get more informative")
    print("=" * 80)
    print("  The same MA(2) model with a longer series gives a tighter posterior. The")
    print("  posterior location stays accurate; the reported spread does not.\n")
    print(f"{'series length T':>16}{'concentration':>15}{'chi^2':>9}{'mean RMSE':>12}"
          f"{'sd ratio':>10}{'sd ratio':>11}")
    print(f"{'':>16}{'(prior/post sd)':>15}{'':>9}{'':>12}{'signed':>10}{'clipped':>11}")
    print("-" * 74)

    for n_timesteps in (5, 15, 50):
        raw = MA2(n_timesteps=n_timesteps, summaries=False)
        summ = MA2(n_timesteps=n_timesteps, summaries=True, n_lags=3)
        g = torch.Generator().manual_seed(SEED)
        theta, series = raw.sample_joint(N_SIM, generator=g)
        features = summ.summarize(series)
        _, series_obs = raw.sample_joint(15, generator=g)
        features_obs = summ.summarize(series_obs)

        reference_mean, reference_sd = [], []
        for i in range(15):
            grid, weights = raw.grid_posterior(series_obs[i], resolution=120)
            m = (weights.unsqueeze(-1) * grid).sum(0)
            reference_mean.append(m)
            reference_sd.append(((weights.unsqueeze(-1) * (grid - m) ** 2).sum(0)).sqrt())
        reference_mean = torch.stack(reference_mean)
        reference_sd = torch.stack(reference_sd)

        torch.manual_seed(SEED)
        operator = PosteriorOperator(theta_dim=2, data_dim=features.shape[1], rank=rank, layer_size=64)
        operator.fit(theta, features, epochs=400, lr=1e-3, seed=SEED)
        signed = operator.posterior(features_obs)
        clipped = operator.posterior(features_obs, clip=True)

        concentration = float((theta.std(0) / reference_sd.mean(0)).mean())
        print(
            f"{n_timesteps:>16}{concentration:>14.1f}x{operator.chi2_divergence:>9.2f}"
            f"{float(((signed.mean() - reference_mean) ** 2).mean().sqrt()):>12.4f}"
            f"{float((signed.std() / reference_sd).mean()):>10.2f}"
            f"{float((clipped.std() / reference_sd).mean()):>11.2f}"
        )
    print("-" * 74)
    print("  A sd ratio of 1.00 would be exact. Two things are visible:")
    print("   * the ratio grows with concentration -- a posterior much tighter than the")
    print("     prior needs a large chi^2, so the truncated spectrum falls short;")
    print("   * clipping negative masses makes it markedly worse, because the negative")
    print("     mass is what carves probability away from the prior's tails. Moment")
    print("     functionals should therefore use the signed estimator, which is what")
    print("     Eq. (2) actually specifies.")
    print("\n  Practical reading: the low-rank operator is at its best for moderately")
    print("  informative experiments and for low-order functionals. For tail quantiles")
    print("  under highly informative data, either raise the rank substantially or fall")
    print("  back to direct regression on the one functional wanted.\n")


if __name__ == "__main__":
    check_maximal_correlation()
    check_compactness()
    check_concentration()
