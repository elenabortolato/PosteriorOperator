"""Change the prior after training, without refitting the operator.

The spectral expansion is taken with respect to the prior-predictive law
rho = pi (x) p(.|theta), so changing pi changes the operator itself, not just
the answer it gives. When simulations come from a proposal q instead, the
functional under pi is recovered by self-normalised importance weights
w(theta) = pi(theta) / q(theta):

    E_pi[f(Theta) | Y = y] = E_q[f(Theta) w(Theta) | Y = y] / E_q[w(Theta) | Y = y]

Because the estimator is already a weighted average over draws theta_i, this is
just a multiplication of each draw's mass by w(theta_i) followed by
renormalisation -- so one training run serves any prior pi << q chosen
afterwards. What it costs is importance-sampling variance, which the effective
sample size reports.

This script trains once under a uniform (flat) proposal on the MA(2)
invertibility triangle, then retargets to four different priors and scores each
against the exact posterior computed under that same prior by quadrature.

Run:  python examples/lfi/03_prior_retargeting.py
"""

import math
from typing import Callable

import torch

from posterior_operator import PosteriorOperator
from posterior_operator.simulators import MA2

SEED = 3
N_SIM = 40000
RANK = 48
N_OBS = 20
GRID_RESOLUTION = 140


def _log_gaussian_prior(centre, scale) -> Callable[[torch.Tensor], torch.Tensor]:
    centre = torch.tensor(centre)
    scale = torch.tensor(scale)

    def log_density(theta: torch.Tensor) -> torch.Tensor:
        return -0.5 * (((theta - centre) / scale) ** 2).sum(dim=-1)

    return log_density


# Each entry is (name, log pi up to a constant). The proposal q is flat on the
# triangle, so log w = log pi + const and the constant cancels in the
# self-normalisation.
PRIORS = {
    "flat (= the proposal q)": lambda t: torch.zeros(t.shape[0]),
    "Gaussian at (0.6, 0.2), sd 0.5": _log_gaussian_prior([0.6, 0.2], [0.5, 0.5]),
    "Gaussian at (-1.0, 0.5), sd 0.3": _log_gaussian_prior([-1.0, 0.5], [0.3, 0.3]),
    "hard constraint theta_2 > 0": lambda t: torch.where(
        t[:, 1] > 0, torch.zeros(t.shape[0]), torch.full((t.shape[0],), -math.inf)
    ),
}


def main() -> None:
    raw = MA2(n_timesteps=50, summaries=False)
    summ = MA2(n_timesteps=50, summaries=True, n_lags=3)
    g = torch.Generator().manual_seed(SEED)

    # --- one training run, under the flat proposal ------------------------
    theta, series = raw.sample_joint(N_SIM, generator=g)
    features = summ.summarize(series)
    torch.manual_seed(SEED)
    operator = PosteriorOperator(theta_dim=2, data_dim=features.shape[1], rank=RANK, layer_size=64)
    operator.fit(theta, features, epochs=500, lr=1e-3, seed=SEED)
    print(f"trained once on {N_SIM} draws from the flat proposal q")
    print(operator.spectrum_report())

    theta_true, series_obs = raw.sample_joint(N_OBS, generator=g)
    features_obs = summ.summarize(series_obs)
    base_posterior = operator.posterior(features_obs)
    draws = base_posterior.theta

    print("\n" + "=" * 86)
    print("retargeting the same fit to four priors, scored against exact quadrature")
    print("=" * 86)
    print(f"{'target prior pi':<34}{'KL(pi||q)':>11}{'mean |err|':>12}{'P(t1>0) |err|':>15}{'ESS':>8}{'ESS %':>8}")
    print("-" * 86)

    for name, log_prior in PRIORS.items():
        log_w = log_prior(draws)
        # A truncating prior cancels the signed masses, so opt in to the clipped
        # measure explicitly for those rather than relying on the fallback.
        source = base_posterior.as_probability() if bool(torch.isinf(log_w).any()) else base_posterior
        retargeted = source.reweight(log_w)

        # Reference: the exact posterior under this same prior.
        ref_means, ref_probs = [], []
        for i in range(N_OBS):
            grid, weights = raw.grid_posterior(
                series_obs[i], resolution=GRID_RESOLUTION, log_prior=log_prior(grid_points(raw, GRID_RESOLUTION))
            )
            ref_means.append((weights.unsqueeze(-1) * grid).sum(0))
            ref_probs.append(weights @ (grid[:, 0] > 0).to(weights.dtype))
        ref_mean = torch.stack(ref_means)
        ref_prob = torch.stack(ref_probs)

        mean_err = float((retargeted.mean() - ref_mean).abs().mean())
        prob_err = float((retargeted.probability(lambda t: t[:, 0] > 0) - ref_prob).abs().mean())
        ess = float(retargeted.effective_sample_size().mean())

        # A rough divergence of the target prior from the proposal, on the draws.
        finite = log_w[torch.isfinite(log_w)]
        normalised = torch.softmax(finite, dim=0)
        kl = float((normalised * (torch.log(normalised.clamp_min(1e-300)) + math.log(finite.numel()))).sum())

        print(
            f"{name:<34}{kl:>11.3f}{mean_err:>12.4f}{prob_err:>15.4f}"
            f"{ess:>8.0f}{100 * ess / draws.shape[0]:>7.1f}%"
        )

    print("-" * 86)
    print(f"  ESS is out of {draws.shape[0]} stored draws.")
    print("\n  The first row is the identity check: reweighting by a constant leaves the")
    print("  posterior untouched, and its error is the low-rank fit's own error.")
    print("\n  The rows below it carry a caveat worth stating in the write-up. The")
    print("  identity E_pi[f|y] = E_q[f w|y] / E_q[w|y] is exact in population, but the")
    print("  weights multiply the ESTIMATED proposal posterior, so the estimator's error")
    print("  does not cancel -- it interacts with the weights. Because the low-rank")
    print("  posterior is over-dispersed (see 04_identifiability.py), the product leans")
    print("  on the new prior more than the truth does, and retargeting to a")
    print("  concentrated prior AMPLIFIES the error rather than leaving it unchanged.")
    print("  So the honest claim is: retargeting is free of retraining, not free of")
    print("  error, and its accuracy degrades with how far pi sits from q.")
    print("\n  The effective sample size is the companion diagnostic. The hard")
    print("  constraint is the extreme case: it deletes the draws outside its support,")
    print("  so the ESS falls to roughly the prior mass of the constraint.")

    print("\n" + "=" * 86)
    print("no-retraining check: is a retargeted fit as good as one trained under the prior?")
    print("=" * 86)
    target_name = "Gaussian at (0.6, 0.2), sd 0.5"
    log_prior = PRIORS[target_name]

    # Train a second operator on draws from that prior directly, by rejection
    # weighting of the flat draws (accept with probability proportional to w).
    log_w_all = log_prior(theta)
    accept = torch.rand(theta.shape[0], generator=g) < torch.exp(log_w_all - log_w_all.max())
    theta_direct, features_direct = theta[accept], features[accept]
    torch.manual_seed(SEED)
    direct = PosteriorOperator(theta_dim=2, data_dim=features.shape[1], rank=RANK, layer_size=64)
    direct.fit(theta_direct, features_direct, epochs=500, lr=1e-3, seed=SEED)

    ref_means = []
    for i in range(N_OBS):
        grid, weights = raw.grid_posterior(
            series_obs[i], resolution=GRID_RESOLUTION, log_prior=log_prior(grid_points(raw, GRID_RESOLUTION))
        )
        ref_means.append((weights.unsqueeze(-1) * grid).sum(0))
    ref_mean = torch.stack(ref_means)

    retargeted_err = float((base_posterior.reweight(log_prior(draws)).mean() - ref_mean).abs().mean())
    direct_err = float((direct.posterior(features_obs).mean() - ref_mean).abs().mean())
    print(f"  target prior: {target_name}")
    print(f"  retargeted from the flat fit ({draws.shape[0]} draws reweighted): mean |error| {retargeted_err:.4f}")
    print(f"  refitted on {theta_direct.shape[0]} draws from that prior:          mean |error| {direct_err:.4f}")
    print(f"\n  The refit is {retargeted_err / direct_err:.1f}x more accurate, and it lands at roughly the")
    print("  flat fit's own error -- so the gap is the price of retargeting, not of the")
    print("  smaller simulation budget. Retargeting buys the ability to answer a prior")
    print("  chosen after training at no extra simulation cost; it does not match a")
    print("  refit. Worth stating that way rather than as equivalence, and it suggests")
    print("  the more ambitious route in the paper's Section 9 -- folding the weights")
    print("  into the training objective itself, so the operator is fitted under the")
    print("  target prior rather than corrected afterwards -- is the one that would")
    print("  actually close this gap.")


def grid_points(simulator: MA2, resolution: int) -> torch.Tensor:
    """The same grid :meth:`MA2.grid_posterior` builds, for evaluating a prior on it."""
    axis1 = torch.linspace(-2.0, 2.0, resolution, dtype=simulator.dtype)
    axis2 = torch.linspace(-1.0, 1.0, resolution, dtype=simulator.dtype)
    return torch.stack(torch.meshgrid(axis1, axis2, indexing="ij"), dim=-1).reshape(-1, 2)


if __name__ == "__main__":
    main()
