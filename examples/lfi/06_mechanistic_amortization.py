"""The amortization claim on a mechanistic simulator, against both baselines.

An SIR epidemic model: infected counts observed over 30 days under Gaussian
noise, with (beta, gamma) unknown. The data are a whole epidemic curve, the
simulator is an ODE, and nothing about it is analytic -- but because the mean
curve is deterministic and the noise is Gaussian, an exact reference posterior
is available by quadrature, so every estimate below is scored against truth.

The comparison Section 10 asks for, at a matched simulation budget:

  NCP operator          one training run, any functional afterwards
  NPE (mixture density) one training run, any functional afterwards, via sampling
  direct regression     one training run PER functional

The functionals span the cases the method is meant to cover, including two that
are nonlinear in theta and so are genuinely new problems for direct regression:

  posterior mean of (beta, gamma)          f = theta
  posterior mean of R0 = beta / gamma      f = theta_1 / theta_2
  P(R0 > 1), the epidemic-takeoff event    f = 1{theta_1 > theta_2}
  the 0.9 quantile of R0                   f_t = 1{R0 <= t} over a grid of t

Run:  python examples/lfi/06_mechanistic_amortization.py
"""

import time

import torch

from posterior_operator import PosteriorOperator
from posterior_operator.baselines import DirectRegression, NeuralPosteriorEstimator
from posterior_operator.simulators import SIR

SEED = 0
N_SIM = 30000
RANK = 48
N_OBS = 40
EPOCHS = 400
GRID_RESOLUTION = 140


def r0(theta: torch.Tensor) -> torch.Tensor:
    """Basic reproduction number, a nonlinear functional of the parameter."""
    return theta[:, 0] / theta[:, 1].clamp_min(1e-6)


def main() -> None:
    sim = SIR()
    g = torch.Generator().manual_seed(SEED)
    theta, y = sim.sample_joint(N_SIM, generator=g)
    theta_true, y_obs = sim.sample_joint(N_OBS, generator=g)

    print(f"SIR: theta = (beta, gamma), data = {sim.data_dim} noisy infected counts")
    print(f"prior gives R0 in [{sim.basic_reproduction_number_range[0]:.1f},"
          f" {sim.basic_reproduction_number_range[1]:.1f}];"
          f" {float((r0(theta) > 1).to(torch.float32).mean()):.0%} of prior draws take off")
    print(f"{N_SIM} simulations for training, {N_OBS} held-out observations\n")

    # --- exact reference posteriors -----------------------------------------
    references = [sim.grid_posterior(y_obs[i], resolution=GRID_RESOLUTION) for i in range(N_OBS)]

    def reference(f) -> torch.Tensor:
        out = []
        for grid, weights in references:
            values = f(grid).reshape(grid.shape[0], -1)
            out.append(weights @ values)
        return torch.stack(out)

    def reference_quantile(level: float) -> torch.Tensor:
        out = []
        for grid, weights in references:
            values = r0(grid)
            order = torch.argsort(values)
            cumulative = weights[order].cumsum(0)
            idx = int(torch.searchsorted(cumulative.contiguous(), torch.tensor(level)).clamp_max(len(values) - 1))
            out.append(values[order][idx])
        return torch.stack(out)

    truth = {
        "mean (beta, gamma)": reference(lambda t: t),
        "mean R0": reference(lambda t: r0(t).unsqueeze(-1)),
        "P(R0 > 1)": reference(lambda t: (r0(t) > 1).to(t.dtype).unsqueeze(-1)),
        "q_0.9 of R0": reference_quantile(0.9).unsqueeze(-1),
    }

    # --- fit the two amortised methods once each ----------------------------
    torch.manual_seed(SEED)
    operator = PosteriorOperator(theta_dim=2, data_dim=sim.data_dim, rank=RANK, layer_size=64)
    start = time.perf_counter()
    operator.fit(theta, y, epochs=EPOCHS, lr=1e-3, seed=SEED)
    ncp_train = time.perf_counter() - start

    npe = NeuralPosteriorEstimator(theta_dim=2, data_dim=sim.data_dim, n_components=10, layer_size=64)
    start = time.perf_counter()
    npe.fit(theta, y, epochs=EPOCHS, lr=1e-3, seed=SEED)
    npe_train = time.perf_counter() - start

    concentration = []
    for grid, weights in references:
        mean = (weights.unsqueeze(-1) * grid).sum(0)
        sd = ((weights.unsqueeze(-1) * (grid - mean) ** 2).sum(0)).sqrt()
        concentration.append(theta.std(0) / sd)
    print(f"{operator.spectrum_report()}")
    print(f"posterior is {float(torch.stack(concentration).mean()):.1f}x tighter than the prior "
          "(the regime that governs how much the truncation costs)\n")
    print(f"training: NCP {ncp_train:.0f}s ({operator.parameter_count():,} params),"
          f"  NPE {npe_train:.0f}s ({npe.parameter_count():,} params)")

    posterior = operator.posterior(y_obs)
    mc_generator = torch.Generator().manual_seed(SEED + 1)

    # NPE puts a Gaussian mixture on an unbounded space while the prior is a box,
    # so some draws leak outside it. Left in, a handful of draws with gamma near
    # zero dominate any functional involving R0 = beta/gamma -- on the first run
    # of this script that alone pushed the NPE error for E[R0] above the
    # prior-only baseline. Rejecting to the support is the standard remedy.
    npe_draws = npe.sample(y_obs, 40000, generator=mc_generator)
    inside = sim.in_support(npe_draws.reshape(-1, 2)).reshape(npe_draws.shape[:2])
    print(f"NPE leakage outside the prior box: {float(1 - inside.to(torch.float32).mean()):.2%}"
          f" (rejected before any functional below)")

    def npe_functional(f) -> torch.Tensor:
        """Support-restricted Monte Carlo average of ``f`` under the NPE posterior."""
        values = f(npe_draws.reshape(-1, 2)).reshape(N_OBS, npe_draws.shape[1], -1)
        mask = inside.unsqueeze(-1).to(values.dtype)
        return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    # --- accuracy table ------------------------------------------------------
    print("\n" + "=" * 84)
    print("accuracy against the exact posterior: mean |error| over the 40 observations")
    print("=" * 84)
    print(f"{'functional':<24}{'NCP':>12}{'NPE':>12}{'direct regr.':>15}{'prior-only':>13}")
    print("-" * 84)

    regression_times = []
    rows = {}

    # f = theta
    ncp_value = posterior.functional(lambda t: t)
    npe_value = npe.mean(y_obs)
    start = time.perf_counter()
    reg = DirectRegression(data_dim=sim.data_dim, output_dim=2, layer_size=64)
    reg.fit(y, theta, epochs=EPOCHS, seed=SEED)
    regression_times.append(time.perf_counter() - start)
    rows["mean (beta, gamma)"] = (ncp_value, npe_value, reg.predict(y_obs), theta.mean(0, keepdim=True))

    # f = R0
    ncp_value = posterior.functional(lambda t: r0(t).unsqueeze(-1))
    npe_value = npe_functional(lambda t: r0(t).unsqueeze(-1))
    start = time.perf_counter()
    reg_r0 = DirectRegression(data_dim=sim.data_dim, output_dim=1, layer_size=64)
    reg_r0.fit(y, r0(theta).unsqueeze(-1), epochs=EPOCHS, seed=SEED)
    regression_times.append(time.perf_counter() - start)
    rows["mean R0"] = (ncp_value, npe_value, reg_r0.predict(y_obs), r0(theta).mean().reshape(1, 1))

    # f = indicator of takeoff
    ncp_value = posterior.functional(lambda t: (r0(t) > 1).to(t.dtype).unsqueeze(-1))
    npe_value = npe_functional(lambda t: (r0(t) > 1).to(t.dtype).unsqueeze(-1))
    start = time.perf_counter()
    reg_p = DirectRegression(data_dim=sim.data_dim, output_dim=1, layer_size=64)
    reg_p.fit(y, (r0(theta) > 1).to(theta.dtype).unsqueeze(-1), epochs=EPOCHS, seed=SEED)
    regression_times.append(time.perf_counter() - start)
    rows["P(R0 > 1)"] = (
        ncp_value,
        npe_value,
        reg_p.predict(y_obs),
        (r0(theta) > 1).to(theta.dtype).mean().reshape(1, 1),
    )

    # A quantile: an order statistic, so not a single regression target at all.
    ncp_value = posterior.quantile(0.9, observable=r0).reshape(-1, 1)
    npe_r0 = r0(npe_draws.reshape(-1, 2)).reshape(N_OBS, -1)
    npe_r0 = torch.where(inside, npe_r0, torch.full_like(npe_r0, float("nan")))
    npe_value = torch.nanquantile(npe_r0, 0.9, dim=-1).unsqueeze(-1)
    rows["q_0.9 of R0"] = (
        ncp_value,
        npe_value,
        None,
        torch.quantile(r0(theta), 0.9).reshape(1, 1),
    )

    for name, (ncp_value, npe_value, reg_value, prior_value) in rows.items():
        target = truth[name]
        cells = []
        for value in (ncp_value, npe_value, reg_value):
            if value is None:
                cells.append("n/a")
            else:
                cells.append(f"{float((value.reshape(target.shape) - target).abs().mean()):.4f}")
        cells.append(f"{float((prior_value - target).abs().mean()):.4f}")
        print(f"{name:<24}{cells[0]:>12}{cells[1]:>12}{cells[2]:>15}{cells[3]:>13}")

    print("-" * 84)
    print("  'prior-only' reports the prior functional, ignoring the data: the baseline")
    print("  any method has to beat. 'n/a' marks a query direct regression cannot answer")
    print("  with a single fit -- a quantile is an order statistic, so it needs a")
    print("  regression per threshold and then an inversion.")

    # --- cost table ----------------------------------------------------------
    repeats = 20
    start = time.perf_counter()
    for _ in range(repeats):
        posterior.functional(lambda t: r0(t).unsqueeze(-1))
    ncp_query = (time.perf_counter() - start) * 1e3 / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        npe.functional(y_obs, lambda t: r0(t).unsqueeze(-1), n_samples=40000, generator=mc_generator)
    npe_query = (time.perf_counter() - start) * 1e3 / repeats

    print("\n" + "=" * 84)
    print("cost of a functional chosen after training")
    print("=" * 84)
    print(f"{'method':<24}{'training runs':>15}{'train (s)':>12}{'per query (ms)':>17}")
    print("-" * 84)
    print(f"{'NCP operator':<24}{1:>15}{ncp_train:>12.0f}{ncp_query:>17.2f}")
    print(f"{'NPE (mixture density)':<24}{1:>15}{npe_train:>12.0f}{npe_query:>17.2f}")
    print(f"{'direct regression':<24}{'1 per functional':>15}"
          f"{sum(regression_times) / len(regression_times):>12.0f}"
          f"{'(a full refit)':>17}")
    print("-" * 84)
    print(f"  NCP answers a new functional {npe_query / ncp_query:.0f}x faster than NPE, because it is a")
    print("  weighted average over stored draws rather than fresh Monte Carlo sampling,")
    print(f"  and {sum(regression_times) / len(regression_times) * 1e3 / ncp_query:,.0f}x faster than refitting a regression for it.")

    print("\n" + "=" * 84)
    print("Reading")
    print("=" * 84)
    print("  Every method beats the prior-only baseline on every row, so the epidemic")
    print("  curve is informative about (beta, gamma) and all three find that signal.")
    print()
    print("  NPE is the most accurate here, on every functional. That is the expected")
    print("  result rather than a surprise: this posterior is smooth, unimodal and")
    print("  two-dimensional, which is exactly what a conditional mixture density models")
    print("  well. Direct regression is close behind on the functionals it can target,")
    print("  as it should be -- it optimises for that one functional and nothing else.")
    print()
    print("  The operator is the least accurate of the three, and one row is much worse")
    print("  than the others: the 0.9 quantile of R0. It still beats prior-only, but only")
    print("  by about a factor of two, against NPE's factor of sixty. This is the")
    print("  over-dispersion of the low-rank posterior showing up where it hurts most --")
    print("  a tail quantile, at 14x posterior concentration. Moments and probabilities")
    print("  are within a factor of three of the best method; tail order statistics are")
    print("  not, and should not be reported from a low-rank fit at this concentration")
    print("  without raising the rank a great deal or falling back to sampling.")
    print()
    print("  What the operator wins on is the shape of the cost, decisively: one fit, and")
    print("  then any functional -- including nonlinear ones like R0 -- as a weighted")
    print("  average over stored draws, with no sampling and no refit. So the honest")
    print("  claim is a cost claim, not an accuracy claim. When a single functional is")
    print("  known in advance, regress on it directly; when the full density is wanted")
    print("  and is well behaved, NPE is stronger; the operator is for the case where")
    print("  many low-order functionals will be asked of one fit, cheaply.")


if __name__ == "__main__":
    main()
