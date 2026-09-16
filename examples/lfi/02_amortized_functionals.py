"""Train once, ask whatever next: many posterior functionals from one fit.

The amortization claim, on the MA(2) benchmark. One operator is trained on
prior-predictive draws, and then a whole list of posterior functionals is read
off it at several observations with no retraining and no likelihood evaluation:

  posterior mean and covariance      f(theta) = theta, theta theta^T
  event probabilities                f(theta) = 1{theta in B}
  marginal histograms                f(theta) = 1{theta_j in B_{j,k}}
  CDF, quantiles, credible interval  f_t(theta) = 1{theta <= t} over a grid of t
  Bayes action under a loss          f(theta) = l(a, theta)

MA(2) has a banded Gaussian covariance, so its exact posterior is available by
quadrature on the invertibility triangle. Every functional below is scored
against that reference -- which a simulation-based method would not have.

Run:  python examples/lfi/02_amortized_functionals.py
"""

import time
from typing import Callable

import torch

from posterior_operator import PosteriorOperator
from posterior_operator.simulators import MA2

SEED = 0
N_SIM = 40000
RANK = 48
N_OBS = 25
GRID_RESOLUTION = 140


def main() -> None:
    raw = MA2(n_timesteps=50, summaries=False)
    summ = MA2(n_timesteps=50, summaries=True, n_lags=3)
    g = torch.Generator().manual_seed(SEED)

    # --- step 1-4 of the training procedure, run once ---------------------
    theta, series = raw.sample_joint(N_SIM, generator=g)
    features = summ.summarize(series)
    torch.manual_seed(SEED)
    operator = PosteriorOperator(theta_dim=2, data_dim=features.shape[1], rank=RANK, layer_size=64)
    start = time.perf_counter()
    operator.fit(theta, features, epochs=500, lr=1e-3, seed=SEED)
    train_seconds = time.perf_counter() - start

    print(f"trained once on {N_SIM} simulator draws in {train_seconds:.0f}s")
    print(operator.spectrum_report())

    # --- observations, plus exact reference posteriors ---------------------
    theta_true, series_obs = raw.sample_joint(N_OBS, generator=g)
    features_obs = summ.summarize(series_obs)
    references = [raw.grid_posterior(series_obs[i], resolution=GRID_RESOLUTION) for i in range(N_OBS)]

    start = time.perf_counter()
    posterior = operator.posterior(features_obs)
    condition_seconds = time.perf_counter() - start
    print(f"\nconditioning on all {N_OBS} observations took {condition_seconds * 1e3:.1f} ms")

    def reference(f: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """The same functional under the exact grid posterior."""
        out = []
        for grid, weights in references:
            values = f(grid)
            values = values.reshape(values.shape[0], -1)
            out.append(weights @ values)
        return torch.stack(out)

    # --- step 5, repeated for every functional of interest ----------------
    print("\n" + "=" * 78)
    print("posterior functionals, all from the same fit")
    print("=" * 78)
    print(f"{'functional':<40}{'mean |error|':>14}{'max |error|':>13}{'ms':>7}")
    print("-" * 78)
    query_times = []

    functionals = {
        "mean                 f = theta": lambda t: t,
        "second moment        f = theta^2": lambda t: t**2,
        "cross moment         f = t1*t2": lambda t: (t[:, 0] * t[:, 1]).unsqueeze(-1),
        "P(theta_1 > 0)": lambda t: (t[:, 0] > 0).to(t.dtype).unsqueeze(-1),
        "P(theta_2 > 0)": lambda t: (t[:, 1] > 0).to(t.dtype).unsqueeze(-1),
        "P(theta in [0,1] x [-.5,.5])": lambda t: (
            ((t[:, 0] > 0) & (t[:, 0] < 1) & (t[:, 1] > -0.5) & (t[:, 1] < 0.5)).to(t.dtype).unsqueeze(-1)
        ),
        "posterior risk, abs loss at a=0": lambda t: t[:, 0].abs().unsqueeze(-1),
    }
    repeats = 20
    for name, f in functionals.items():
        estimate = posterior.functional(f)
        start = time.perf_counter()  # timed over repeats; a single call is sub-millisecond
        for _ in range(repeats):
            posterior.functional(f)
        elapsed = (time.perf_counter() - start) * 1e3 / repeats
        query_times.append(elapsed)
        truth = reference(f)
        gap = (estimate.reshape(truth.shape) - truth).abs()
        print(f"{name:<40}{float(gap.mean()):>14.4f}{float(gap.max()):>13.4f}{elapsed:>7.1f}")

    # Derived quantities that are functions of several functionals.
    est_cov = posterior.covariance()
    ref_mean = reference(lambda t: t)
    ref_second = torch.stack(
        [(w.unsqueeze(-1) * grid).T @ grid for grid, w in references]
    )
    ref_cov = ref_second - torch.einsum("ni,nj->nij", ref_mean, ref_mean)
    print(f"{'covariance (from the two above)':<40}{float((est_cov - ref_cov).abs().mean()):>14.4f}"
          f"{float((est_cov - ref_cov).abs().max()):>13.4f}")

    print("\n" + "=" * 78)
    print("order statistics: CDF over a grid of t, then quantiles and an interval")
    print("=" * 78)
    for coordinate in (0, 1):
        levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        estimate = posterior.quantile(levels, observable=coordinate)
        truth = torch.stack([_grid_quantile(grid[:, coordinate], w, levels) for grid, w in references])
        interval = posterior.credible_interval(0.1, coordinate=coordinate)
        inside = (
            (theta_true[:, coordinate] >= interval[:, 0]) & (theta_true[:, coordinate] <= interval[:, 1])
        ).to(torch.float32)
        ref_interval = torch.stack(
            [_grid_quantile(grid[:, coordinate], w, [0.05, 0.95]) for grid, w in references]
        )
        width = float((interval[:, 1] - interval[:, 0]).mean())
        exact_width = float((ref_interval[:, 1] - ref_interval[:, 0]).mean())
        print(
            f"  theta_{coordinate + 1}: quantile mean |error| {float((estimate - truth).abs().mean()):.4f}"
            f" | 90% interval width {width:.3f} vs exact {exact_width:.3f}"
            f" ({width / exact_width:.1f}x too wide)"
            f" | covers the truth {float(inside.mean()):.2f} of the time"
        )
    print("\n  The intervals are valid but conservative, by the factor shown: the")
    print("  quantiles inherit the over-dispersion of the low-rank posterior, whereas")
    print("  the moments and probabilities above do not. Tail queries are where the")
    print("  truncation rank bites hardest.")

    edges, probabilities = posterior.marginal_histogram(coordinate=0, bins=8)
    print(f"\n  marginal histogram of theta_1 over {edges.numel() - 1} bins, "
          f"rows sum to {float(probabilities.sum(-1).mean()):.4f}")

    print("\n" + "=" * 78)
    print("a decision: Bayes action under an asymmetric (check) loss")
    print("=" * 78)
    # Over-prediction penalised three times as heavily as under-prediction, so
    # the Bayes action is the 0.25 quantile rather than the mean.
    tau = 0.25
    actions = torch.linspace(-2.0, 2.0, 161).reshape(-1, 1)

    def check_loss(a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        err = t[:, 0].unsqueeze(0) - a  # (n_actions, n_draws)
        return torch.maximum(tau * err, (tau - 1.0) * err)

    chosen = posterior.bayes_action(check_loss, actions).reshape(-1)
    reference_action = torch.stack(
        [_grid_quantile(grid[:, 0], w, [tau]) for grid, w in references]
    ).reshape(-1)
    print(f"  loss weights (tau = {tau}) make the Bayes action the {tau} quantile")
    print(f"  mean |NCP action - exact {tau} quantile| = {float((chosen - reference_action).abs().mean()):.4f}")
    print(f"  (the posterior mean would be off by {float((posterior.mean()[:, 0] - reference_action).abs().mean()):.4f})")

    # --- the amortization argument ---------------------------------------
    print("\n" + "=" * 78)
    print("what amortization buys")
    print("=" * 78)
    n_queried = len(functionals) + 4
    print(f"  one training run:                     {train_seconds:>8.0f} s")
    print(f"  conditioning on {N_OBS} observations:      {condition_seconds * 1e3:>8.1f} ms")
    print(f"  each additional functional:      {min(query_times):>6.2f}-{max(query_times):<5.2f} ms")
    print(f"  {n_queried} functionals answered from one fit.")
    print(f"  ratio: the slowest functional query costs "
          f"{train_seconds * 1e3 / max(query_times):,.0f}x less than the fit it reuses.")
    print("  A separate regression per functional would repeat the training cost")
    print("  every time, and could not answer a functional chosen after the fact.")
    print("\n  Caveat worth carrying into the write-up: the posterior *location* and")
    print("  probabilities above are accurate, but the reported spread is wider than")
    print("  the truth, and the gap grows with how much the posterior concentrates")
    print("  relative to the prior -- see 04_identifiability.py for the measurement.")


def _grid_quantile(values: torch.Tensor, weights: torch.Tensor, levels) -> torch.Tensor:
    """Quantiles of a discrete reference posterior."""
    order = torch.argsort(values)
    sorted_values, cumulative = values[order], weights[order].cumsum(0)
    lv = torch.as_tensor(levels, dtype=values.dtype)
    idx = torch.searchsorted(cumulative.contiguous(), lv.contiguous()).clamp_max(values.numel() - 1)
    return sorted_values[idx]


if __name__ == "__main__":
    main()
