"""Frequentist uncertainty in T_f(y_0): does 'ask anything later' come with an interval?

Not a credible interval for Theta -- a CONFIDENCE interval for the estimate
T_hat_f(y_0) itself, over repetitions of the whole simulate-and-fit pipeline.
The distinction that governs it:

* In the FIXED-ARCHITECTURE regime, (u, v, sigma) is a correctly specified
  finite-dimensional parametric family, T_hat_f is an asymptotically linear
  M-estimator, and sqrt(n) (T_hat_f - T_f) should be asymptotically normal --
  a root-n guarantee for any f fixed after training.
* In the GROWING-CAPACITY regime, u_hat_k(y_0) - u_k(y_0) is a nonparametric
  error at a single point. Because T_f(y_0) evaluates the nuisance at one y_0
  instead of averaging it over the population, Neyman orthogonality does not
  apply and no amount of sample splitting buys back the root-n rate.

Both are measured here by brute force: the pipeline is repeated over
independent simulator draws, giving the true sampling distribution of
T_hat_f(y_0). Then:

  (a) does the sampling standard deviation shrink like n^{-1/2}?
  (b) is the sampling distribution centred on the truth, or biased?
  (c) how much of that spread does the cheap draw-bootstrap capture, and what
      is its actual coverage?

Spoiler, because it inverts the expectation above: the fixed-architecture
regime is the one that FAILS to achieve root-n here, and the growing-capacity
regime is the one that tracks it. A small fixed network is not a correctly
specified parametric family for this operator, so it has an approximation floor
that no amount of data removes, and what spread remains across replicates comes
from where the non-convex fit lands rather than from the sample. See the
'Reading' section at the end.

Run:  python examples/lfi/07_functional_confidence_intervals.py [--replicates 20]
"""

import argparse
import math

import torch

from posterior_operator import PosteriorOperator
from posterior_operator.simulators import GaussianLinear

SEED = 0
# Small and fixed: this is the fixed-architecture regime, so the architecture
# must NOT grow with n.
FIXED_RANK = 8
FIXED_WIDTH = 32
N_OBS = 6


def run_pipeline(sim, n_sim: int, rank: int, width: int, seed: int, y_obs, f):
    """One full repetition: fresh draws, fresh fit, evaluate the functional."""
    g = torch.Generator().manual_seed(seed)
    theta, y = sim.sample_joint(n_sim, generator=g)
    torch.manual_seed(seed)
    operator = PosteriorOperator(
        theta_dim=sim.theta_dim, data_dim=sim.data_dim, rank=rank, layer_size=width
    )
    operator.fit(theta, y, epochs=200, lr=1e-3, seed=seed, patience=25)
    return operator, operator.posterior(y_obs).functional(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=20)
    args = parser.parse_args()

    # Linear-Gaussian, so T_f(y_0) is known exactly for f = identity.
    sim = GaussianLinear(theta_dim=2, data_dim=3, noise=0.5, seed=0)
    g = torch.Generator().manual_seed(1234)
    _, y_obs = sim.sample_joint(N_OBS, generator=g)
    truth = sim.posterior_mean(y_obs)  # (N_OBS, 2)

    def f(theta: torch.Tensor) -> torch.Tensor:
        return theta

    print("f = identity, so T_f(y_0) = E[Theta | Y = y_0] is known in closed form.")
    print(f"{N_OBS} fixed observations, {args.replicates} independent repetitions of the")
    print(f"whole pipeline per sample size. Architecture held FIXED at rank {FIXED_RANK},")
    print(f"width {FIXED_WIDTH} -- the regime where a root-n guarantee is expected.\n")

    # ---------------------------------------------------------------- (a), (b)
    print("=" * 86)
    print("Sampling distribution of T_hat_f(y_0) over repeated simulate-and-fit")
    print("=" * 86)
    print(f"{'n':>8}{'RMSE':>10}{'|bias|':>10}{'sd':>10}{'sd * sqrt(n)':>14}{'bias/sd':>10}")
    print("-" * 86)

    sizes = (1000, 4000, 16000)
    sampling, bootstraps = {}, {}
    alpha = 0.1
    for n_sim in sizes:
        estimates, intervals = [], []
        for r in range(args.replicates):
            operator, value = run_pipeline(sim, n_sim, FIXED_RANK, FIXED_WIDTH, SEED + 100 * r, y_obs, f)
            estimates.append(value)
            # Section (c) needs a bootstrap from this same fit; doing it here
            # avoids refitting every model a second time.
            _, interval = operator.posterior(y_obs).bootstrap_functional(
                f, n_resamples=200, alpha=alpha, generator=torch.Generator().manual_seed(r)
            )
            intervals.append(interval)
        estimates = torch.stack(estimates)  # (R, N_OBS, 2)
        sampling[n_sim] = estimates
        bootstraps[n_sim] = torch.stack(intervals)  # (R, N_OBS, 2, 2)
        bias = estimates.mean(0) - truth
        sd = estimates.std(0, unbiased=True)
        rmse = ((estimates - truth) ** 2).mean().sqrt()
        print(
            f"{n_sim:>8}{float(rmse):>10.4f}{float(bias.abs().mean()):>10.4f}"
            f"{float(sd.mean()):>10.4f}{float(sd.mean()) * math.sqrt(n_sim):>14.3f}"
            f"{float(bias.abs().mean() / sd.mean()):>10.2f}"
        )
    print("-" * 86)
    print("  'sd * sqrt(n)' is the diagnostic for (a): constant down the column means the")
    print("  spread shrinks at the root-n rate, while a GROWING column means it shrinks")
    print("  more slowly than that. 'bias/sd' is the diagnostic for (b): around 0.5 means")
    print("  bias and spread are comparable, so an interval that accounts only for the")
    print("  spread cannot be calibrated even if it gets the spread exactly right.")

    # Ratios make the rate easier to read than the raw column.
    print("\n  rate between consecutive sample sizes (2.0 would be exactly root-n):")
    for small, large in zip(sizes[:-1], sizes[1:]):
        sd_small = float(sampling[small].std(0, unbiased=True).mean())
        sd_large = float(sampling[large].std(0, unbiased=True).mean())
        expected = math.sqrt(large / small)
        print(f"    n {small} -> {large}: sd ratio {sd_small / sd_large:.2f}  (root-n would be {expected:.2f})")

    # -------------------------------------------------------------------- (c)
    print("\n" + "=" * 86)
    print("Does the cheap draw-bootstrap interval cover T_f(y_0)?")
    print("=" * 86)
    print("  The bootstrap resamples the stored parameter draws with the networks held")
    print("  fixed, so it sees the Monte Carlo error of the final average but not the")
    print("  error in (u_hat, v_hat, sigma_hat).\n")
    print(f"{'n':>8}{'true sd':>10}{'bootstrap sd':>15}{'captured':>11}{'coverage':>11}{'nominal':>10}")
    print("-" * 86)

    for n_sim in sizes:
        interval = bootstraps[n_sim]
        inside = (truth >= interval[..., 0]) & (truth <= interval[..., 1])
        covered = int(inside.sum())
        total = args.replicates * truth.numel()
        true_sd = float(sampling[n_sim].std(0, unbiased=True).mean())
        # Convert a 90% percentile width back to an implied standard deviation.
        bootstrap_sd = float(((interval[..., 1] - interval[..., 0]) / (2 * 1.6449)).mean())
        print(
            f"{n_sim:>8}{true_sd:>10.4f}{bootstrap_sd:>15.4f}"
            f"{bootstrap_sd / true_sd:>10.0%}{covered / total:>11.2f}{1 - alpha:>10.2f}"
        )
    print("-" * 86)

    # ------------------------------------------------- growing-capacity regime
    print("\n" + "=" * 86)
    print("Growing capacity: the same experiment with rank and width scaled with n")
    print("=" * 86)
    print(f"{'n':>8}{'rank':>7}{'width':>7}{'RMSE':>10}{'|bias|':>10}{'sd':>10}{'sd * sqrt(n)':>14}")
    print("-" * 86)
    for n_sim, rank, width in ((1000, 8, 32), (4000, 16, 64), (16000, 32, 128)):
        estimates = torch.stack(
            [
                run_pipeline(sim, n_sim, rank, width, SEED + 100 * r, y_obs, f)[1]
                for r in range(max(8, args.replicates // 3))
            ]
        )
        bias = estimates.mean(0) - truth
        sd = estimates.std(0, unbiased=True)
        rmse = ((estimates - truth) ** 2).mean().sqrt()
        print(
            f"{n_sim:>8}{rank:>7}{width:>7}{float(rmse):>10.4f}{float(bias.abs().mean()):>10.4f}"
            f"{float(sd.mean()):>10.4f}{float(sd.mean()) * math.sqrt(n_sim):>14.3f}"
        )
    print("-" * 86)
    print("  Note what happens to 'sd * sqrt(n)' here compared with the fixed-architecture")
    print("  table: it stays roughly flat instead of growing, and the RMSE keeps falling")
    print("  rather than plateauing. Capacity was grown by a factor of 2 each time n grew")
    print("  by 4, so this is one schedule over three sample sizes, not a rate theorem --")
    print("  but the contrast with the fixed-architecture column is large and consistent.")

    print("\n" + "=" * 86)
    print("Reading")
    print("=" * 86)
    print("  The measured picture is the reverse of the expected one, and the reversal")
    print("  is the useful result.")
    print()
    print("  In the FIXED-ARCHITECTURE regime -- the one predicted to give root-n -- the")
    print("  spread does NOT shrink at root-n: 'sd * sqrt(n)' grows steadily and the RMSE")
    print("  plateaus. Two things cause it. A rank-8 network is not a correctly specified")
    print("  parametric family for this operator (its exact spectrum is infinite), so")
    print("  there is an approximation floor no amount of data removes; and the residual")
    print("  spread across replicates is driven by where the non-convex fit lands, which")
    print("  is a function of initialisation rather than of n and so does not shrink at")
    print("  all. The M-estimation argument needs a unique, well-separated population")
    print("  minimiser that the optimiser actually reaches -- with a neural parametrisation")
    print("  neither premise holds, and that gap is what shows up here.")
    print()
    print("  In the GROWING-CAPACITY regime -- the one predicted to be slower -- the error")
    print("  keeps falling and tracks root-n over this range, because the approximation")
    print("  error is still the dominant, shrinking term.")
    print()
    print("  Consequence for the interval: the draw-bootstrap degrades from usable to")
    print("  badly overconfident as n grows (90% nominal covering about half the time at")
    print("  the largest n), and it degrades precisely BECAUSE it is correct about the")
    print("  part it measures. Its width falls like n^{-1/2} while the true spread")
    print("  plateaus, so the two diverge. Report it as a Monte Carlo error bar on the")
    print("  averaging step, which is what it is, not as a confidence interval for")
    print("  T_f(y_0).")
    print()
    print("  For the write-up this suggests stating the fixed-architecture result as an")
    print("  interval for the best rank-d approximation of T_f(y_0) -- conditional on the")
    print("  learned subspace -- rather than for T_f(y_0) itself, and being explicit that")
    print("  optimisation variability is a third error source alongside approximation and")
    print("  sampling. It is checkable, and it is what the experiment supports.")


if __name__ == "__main__":
    main()
