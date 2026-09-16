"""g-and-k: the operator against NPE on a genuinely likelihood-free model.

The third item of the experimental protocol. Unlike SIR -- a deterministic ODE
with additive Gaussian noise, whose likelihood is perfectly tractable and which
is therefore only a *stand-in* for an intractable model -- the g-and-k
distribution is defined by its quantile function and has no closed-form
density. Sampling is one line; evaluating p(y | theta) requires inverting Q
numerically. It is the standard likelihood-free benchmark, and it is the model
on which the comparison with NPE actually means something.

The setting: n_obs = 100 iid draws per dataset, reduced to the four standard
robust octile summaries (location, spread, skewness, kurtosis), with the usual
uniform prior on [0, 10]^4.

A NOTE ON THE SUMMARIES, because it changes the result. Two of the four
summaries are positive and extremely heavy-tailed under this prior: `spread`
has a standard deviation of about 694 with a 0.1% quantile of 0.044, because
k up to 10 makes the tails astronomically heavy. Both methods standardise
their inputs by mean and standard deviation, so on the raw scale both are
handed a coordinate that is numerically invisible for 99.9% of datasets --
and the ABC reference metric has the same problem. Everything below therefore
applies log to those two coordinates first, uniformly to NCP, to NPE and to
the reference. Without it the experiment measures which method degrades more
gracefully on badly scaled inputs, which is not the question.

THE REFERENCE. There is no exact posterior here -- that is the point of the
model. The reference is a large-budget rejection ABC on the transformed
summaries: 2,000,000 prior-predictive draws, the closest 0.1% retained. Its
tolerance sensitivity is reported, so the reader can see how much of the
disagreement between methods is reference error. Note the reference targets
p(theta | s_obs), the posterior given the summaries, which is what both
methods condition on -- not p(theta | y_1..y_100).

What is measured, at a matched simulation budget:

  * marginal accuracy against the reference, as a 1-Wasserstein distance in
    units of the prior standard deviation -- a whole-shape criterion, not just
    the mean, since g is weakly identified here and a method can get the mean
    right while being badly overconfident;
  * calibration: empirical coverage of nominal 90% credible intervals over
    many observations, which needs no reference at all;
  * cost: training once, then the price of one more functional.

Run:  python examples/lfi/08_gandk_npe.py
"""

import time
from pathlib import Path

import torch

from posterior_operator import PosteriorOperator
from posterior_operator.baselines import NeuralPosteriorEstimator
from posterior_operator.simulators import GAndK

SEED = 0
N_SIM = 60000  # training budget, shared by both methods
N_REFERENCE = 2_000_000  # prior-predictive table for the ABC reference
ABC_KEEP = 2000  # retained draws per observation (0.1%)
N_SCORED = 24  # observations given a full ABC reference
N_COVERAGE = 300  # observations used for the coverage check
RANK = 64
LAYER_SIZE = 128
NAMES = ("A", "B", "g", "k")


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #


def transform(summaries: torch.Tensor) -> torch.Tensor:
    """Variance-stabilise the two positive, heavy-tailed summary coordinates.

    Applied identically to both methods and to the reference metric, so it
    cannot favour either one.
    """
    spread = summaries[:, 1].clamp_min(1e-8).log()
    kurtosis = summaries[:, 3].clamp_min(1e-8).log()
    return torch.stack([summaries[:, 0], spread, summaries[:, 2], kurtosis], dim=-1)


# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #


def build_reference_table(sim, n: int, seed: int):
    """Simulate the prior-predictive table the ABC reference draws from."""
    generator = torch.Generator().manual_seed(seed)
    thetas, summaries = [], []
    remaining = n
    while remaining > 0:
        block = min(200_000, remaining)
        theta, s = sim.sample_joint(block, generator=generator)
        thetas.append(theta)
        summaries.append(transform(s))
        remaining -= block
    return torch.cat(thetas), torch.cat(summaries)


def abc_posterior(table_theta, table_z, scale, z_obs, keep: int):
    """Rejection-ABC draws for one observation: the `keep` nearest summaries."""
    distance = (((table_z - z_obs) / scale) ** 2).sum(dim=-1).sqrt()
    index = torch.topk(-distance, keep).indices
    return table_theta[index], float(distance[index].max())


# --------------------------------------------------------------------------- #
# Comparison metrics
# --------------------------------------------------------------------------- #


def wasserstein1(sample_a, weights_a, sample_b):
    """W_1 between a weighted atomic measure and an equally weighted sample.

    Both are one-dimensional, so this is the integral of |F_a - F_b|, computed
    on the merged support.
    """
    order_a = torch.argsort(sample_a)
    values_a, mass_a = sample_a[order_a].contiguous(), weights_a[order_a]
    cdf_a = torch.cumsum(mass_a, 0) / mass_a.sum()
    values_b = torch.sort(sample_b).values.contiguous()
    cdf_b = torch.arange(1, values_b.numel() + 1, dtype=values_b.dtype) / values_b.numel()

    def evaluate(values, cdf, grid):
        """Right-continuous F(x) = mass of atoms <= x, on the merged grid."""
        count = torch.searchsorted(values, grid.contiguous(), right=True)
        out = cdf[(count - 1).clamp(min=0)]
        return torch.where(count == 0, torch.zeros_like(out), out)

    grid = torch.cat([values_a, values_b]).sort().values
    gap = grid.diff()
    difference = (evaluate(values_a, cdf_a, grid) - evaluate(values_b, cdf_b, grid)).abs()
    return float((difference[:-1] * gap).sum())


def interval_from_sample(draws, alpha: float):
    """Equal-tailed 1 - alpha interval of a plain sample, shape (p, 2)."""
    lo = torch.quantile(draws, alpha / 2, dim=0)
    hi = torch.quantile(draws, 1 - alpha / 2, dim=0)
    return torch.stack([lo, hi], dim=-1)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    torch.manual_seed(SEED)
    sim = GAndK(n_obs=100, summaries=True, prior_high=10.0)

    print("=" * 84)
    print("g-and-k: no closed-form density, so this is a real likelihood-free model")
    print("=" * 84)
    print(f"  prior            uniform on [0, {sim.prior_high:.0f}]^4 over (A, B, g, k)")
    print(f"  data             {sim.n_obs} iid draws -> 4 octile summaries (log on spread, kurtosis)")
    print(f"  training budget  {N_SIM} simulations, shared by NCP and NPE")
    print(f"  reference        rejection ABC, {N_REFERENCE:,} draws, closest {ABC_KEEP} kept\n")

    # ------------------------------------------------------------- reference
    started = time.perf_counter()
    table_theta, table_z = build_reference_table(sim, N_REFERENCE, seed=SEED + 11)
    scale = table_z.std(dim=0)
    prior_sd = table_theta.std(dim=0)
    print(f"reference table simulated in {time.perf_counter() - started:.1f}s")

    # Observations to score, drawn from the joint so theta_true is known.
    g = torch.Generator().manual_seed(SEED + 99)
    theta_true, y_raw = sim.sample_joint(N_COVERAGE, generator=g)
    y_obs = transform(y_raw)

    references, tolerances = [], []
    for i in range(N_SCORED):
        draws, eps = abc_posterior(table_theta, table_z, scale, y_obs[i], ABC_KEEP)
        references.append(draws)
        tolerances.append(eps)
    print(f"ABC tolerance over the {N_SCORED} scored observations: "
          f"median {torch.tensor(tolerances).median():.3f}, max {max(tolerances):.3f}")

    # How much of what follows is reference error? Halve the acceptance rate
    # and see how far the reference itself moves.
    uniform = torch.full((ABC_KEEP,), 1.0 / ABC_KEEP)
    drift = []
    for i in range(N_SCORED):
        tight, _ = abc_posterior(table_theta, table_z, scale, y_obs[i], ABC_KEEP // 4)
        drift.append(
            torch.tensor([wasserstein1(references[i][:, j], uniform, tight[:, j]) for j in range(4)])
        )
    drift = torch.stack(drift).mean(0) / prior_sd
    print("reference self-drift when the tolerance is tightened 4x (W1 / prior sd):")
    print("   " + "  ".join(f"{n}={v:.3f}" for n, v in zip(NAMES, drift.tolist())))
    print("   -> differences below this floor are reference error, not method error.\n")

    # -------------------------------------------------------------- training
    gt = torch.Generator().manual_seed(SEED + 1)
    theta_train, raw_train = sim.sample_joint(N_SIM, generator=gt)
    z_train = transform(raw_train)

    print("=" * 84)
    print("Training, on the same simulations")
    print("=" * 84)

    torch.manual_seed(SEED)
    operator = PosteriorOperator(theta_dim=4, data_dim=4, rank=RANK, layer_size=LAYER_SIZE)
    started = time.perf_counter()
    operator.fit(theta_train, z_train, epochs=400, lr=1e-3, seed=SEED)
    ncp_fit_time = time.perf_counter() - started

    torch.manual_seed(SEED)
    npe = NeuralPosteriorEstimator(theta_dim=4, data_dim=4, n_components=10, layer_size=LAYER_SIZE)
    started = time.perf_counter()
    npe.fit(theta_train, z_train, epochs=400, lr=1e-3, seed=SEED)
    npe_fit_time = time.perf_counter() - started

    print(f"  NCP  rank {RANK:>3}  {operator.parameter_count():>8,} parameters  fitted in {ncp_fit_time:6.1f}s")
    print(f"  NPE  10 cmp    {npe.parameter_count():>8,} parameters  fitted in {npe_fit_time:6.1f}s")
    print(f"  sigma_1 = {operator.maximal_correlation:.4f},  chi^2 = {operator.chi2_divergence:.2f}")
    print(f"  top singular values: {operator.singular_values[:6].numpy().round(3)}\n")

    # --------------------------------------------------- accuracy vs the ABC
    posterior = operator.posterior(y_obs[:N_SCORED])
    probability = posterior.as_probability()
    npe_draws = npe.sample(y_obs[:N_SCORED], 4000, generator=torch.Generator().manual_seed(3))

    ncp_mean = posterior.mean()
    npe_mean = npe.mean(y_obs[:N_SCORED])
    reference_mean = torch.stack([r.mean(0) for r in references])

    ncp_w1 = torch.zeros(N_SCORED, 4)
    npe_w1 = torch.zeros(N_SCORED, 4)
    for i in range(N_SCORED):
        for j in range(4):
            ncp_w1[i, j] = wasserstein1(
                probability.atoms[:, j], probability.weights[i], references[i][:, j]
            )
            npe_w1[i, j] = wasserstein1(
                npe_draws[i, :, j],
                torch.full((npe_draws.shape[1],), 1.0 / npe_draws.shape[1]),
                references[i][:, j],
            )

    print("=" * 84)
    print(f"Accuracy against the ABC reference, averaged over {N_SCORED} observations")
    print("=" * 84)
    print("  All errors in units of the prior standard deviation, so 1.0 means 'as")
    print("  wrong as answering with the prior'.\n")
    print(f"{'':>6}{'|mean err| NCP':>16}{'NPE':>9}{'   ':>4}{'W1 NCP':>10}{'NPE':>9}{'  ref floor':>13}")
    print("-" * 84)
    for j, name in enumerate(NAMES):
        me_ncp = float((ncp_mean[:, j] - reference_mean[:, j]).abs().mean() / prior_sd[j])
        me_npe = float((npe_mean[:, j] - reference_mean[:, j]).abs().mean() / prior_sd[j])
        w_ncp = float(ncp_w1[:, j].mean() / prior_sd[j])
        w_npe = float(npe_w1[:, j].mean() / prior_sd[j])
        print(f"{name:>6}{me_ncp:>16.4f}{me_npe:>9.4f}{'':>4}{w_ncp:>10.4f}{w_npe:>9.4f}{float(drift[j]):>13.4f}")
    print("-" * 84)
    print(f"{'all':>6}{float((ncp_mean - reference_mean).abs().mean(0).div(prior_sd).mean()):>16.4f}"
          f"{float((npe_mean - reference_mean).abs().mean(0).div(prior_sd).mean()):>9.4f}"
          f"{'':>4}{float((ncp_w1.mean(0) / prior_sd).mean()):>10.4f}"
          f"{float((npe_w1.mean(0) / prior_sd).mean()):>9.4f}{float(drift.mean()):>13.4f}")

    # ----------------------------------------------------------- calibration
    print("\n" + "=" * 84)
    print(f"Calibration: nominal 90% credible intervals over {N_COVERAGE} observations")
    print("=" * 84)
    print("  This needs no reference -- it asks whether the stated intervals contain the")
    print("  theta that actually generated the data, at the rate they claim.\n")

    all_posterior = operator.posterior(y_obs)
    all_npe = npe.sample(y_obs, 4000, generator=torch.Generator().manual_seed(5))
    print(f"{'':>6}{'coverage NCP':>15}{'NPE':>9}{'   ':>4}{'width NCP':>12}{'NPE':>9}{'  ABC(24 obs)':>15}")
    print("-" * 84)
    ncp_cov, npe_cov = [], []
    for j, name in enumerate(NAMES):
        interval_ncp = all_posterior.credible_interval(alpha=0.10, coordinate=j)
        inside_ncp = (theta_true[:, j] >= interval_ncp[:, 0]) & (theta_true[:, j] <= interval_ncp[:, 1])
        q = torch.tensor([0.05, 0.95])
        marginal = all_npe[:, :, j]
        lo = torch.quantile(marginal, q[0], dim=1)
        hi = torch.quantile(marginal, q[1], dim=1)
        inside_npe = (theta_true[:, j] >= lo) & (theta_true[:, j] <= hi)
        abc_width = torch.stack(
            [torch.quantile(r[:, j], q[1]) - torch.quantile(r[:, j], q[0]) for r in references]
        ).mean()
        ncp_cov.append(float(inside_ncp.float().mean()))
        npe_cov.append(float(inside_npe.float().mean()))
        print(f"{name:>6}{ncp_cov[-1]:>15.3f}{npe_cov[-1]:>9.3f}{'':>4}"
              f"{float((interval_ncp[:, 1] - interval_ncp[:, 0]).mean()):>12.3f}"
              f"{float((hi - lo).mean()):>9.3f}{float(abc_width):>15.3f}")
    print("-" * 84)
    print(f"{'mean':>6}{sum(ncp_cov) / 4:>15.3f}{sum(npe_cov) / 4:>9.3f}"
          f"     nominal 0.900")

    # Why the operator's intervals come out wide: order-statistic queries need a
    # genuine probability measure, so they clip the negative masses and
    # renormalise. If the surviving measure is close to uniform over the stored
    # prior draws, every interval reverts towards the prior.
    negative = float((all_posterior.weights < 0).float().mean())
    ess = all_posterior.effective_sample_size().mean() / all_posterior.n_atoms
    print("\n  diagnostics on the clipped measure used for intervals:")
    print(f"    fraction of negative masses            {negative:.3f}")
    print(f"    effective sample size / n_atoms        {float(ess):.3f}   (1.0 = the prior itself)")

    # ------------------------------------------------------------------ cost
    print("\n" + "=" * 84)
    print("Cost of one more functional, after training")
    print("=" * 84)

    def new_functional(theta):
        """Something nobody declared at training time."""
        return torch.stack([theta[:, 2], (theta[:, 1] * theta[:, 3]).clamp_max(50.0)], dim=-1)

    batch = y_obs
    n_batch = batch.shape[0]
    n_draws = 4000

    # Regime (a): everything from scratch, the price of answering a new
    # functional on a new batch of observations.
    started = time.perf_counter()
    for _ in range(10):
        operator.posterior(batch).functional(new_functional)
    ncp_cold = (time.perf_counter() - started) / (10 * n_batch) * 1000
    started = time.perf_counter()
    for _ in range(10):
        npe.functional(batch, new_functional, n_samples=n_draws,
                       generator=torch.Generator().manual_seed(0))
    npe_cold = (time.perf_counter() - started) / (10 * n_batch) * 1000

    # Regime (b): the conditioning work already done and cached, so this is the
    # marginal price of the SECOND functional on the same observations.
    cached_posterior = operator.posterior(batch)
    cached_draws = npe.sample(batch, n_draws, generator=torch.Generator().manual_seed(0))
    started = time.perf_counter()
    for _ in range(10):
        cached_posterior.functional(new_functional)
    ncp_warm = (time.perf_counter() - started) / (10 * n_batch) * 1000
    started = time.perf_counter()
    for _ in range(10):
        values = new_functional(cached_draws.reshape(-1, 4))
        values.reshape(n_batch, n_draws, -1).mean(dim=1)
    npe_warm = (time.perf_counter() - started) / (10 * n_batch) * 1000

    print(f"  {n_batch} observations, NPE using {n_draws} draws each.\n")
    print(f"{'':>6}{'cold (ms/obs)':>16}{'warm (ms/obs)':>16}{'f evaluated at':>18}")
    print("-" * 84)
    print(f"{'NCP':>6}{ncp_cold:>16.4f}{ncp_warm:>16.4f}"
          f"{cached_posterior.n_atoms:>13,} pts")
    print(f"{'NPE':>6}{npe_cold:>16.4f}{npe_warm:>16.4f}"
          f"{n_batch * n_draws:>13,} pts")
    print("-" * 84)
    print(f"  cold: {npe_cold / max(ncp_cold, 1e-9):.0f}x in NCP's favour, "
          f"warm: {npe_warm / max(ncp_warm, 1e-9):.0f}x")
    print("  The structural reason is the last column. The operator's atoms are the same")
    print("  stored prior draws for every observation, so a new f is evaluated once for")
    print("  the whole batch and each observation costs one dot product. NPE's draws are")
    print("  per-observation, so f must be evaluated n_obs x n_draws times and the count")
    print("  grows with the batch. The gap widens with the number of observations queried,")
    print("  not with the number of functionals.")

    _plot(sim, references, probability, npe_draws, theta_true, prior_sd,
          ncp_w1, npe_w1, drift, ncp_cov, npe_cov)

    print("\n" + "=" * 84)
    print("Reading")
    print("=" * 84)
    print("  Read the W1 column, not the mean column. Under this prior g is only weakly")
    print("  identified by 100 observations, so a method can land the posterior mean and")
    print("  still be badly wrong about the shape; W1 against the reference catches that")
    print("  and the mean error does not.")
    print()
    print("  Everything here is scored against an ABC reference, which is itself an")
    print("  estimate. The 'ref floor' column says how far that reference moves when its")
    print("  own tolerance is tightened fourfold. A method-to-method gap smaller than the")
    print("  floor is not evidence of anything.")


def _plot(sim, references, probability, npe_draws, theta_true, prior_sd,
          ncp_w1, npe_w1, drift, ncp_cov, npe_cov) -> None:
    """Two figures: the marginals for one observation, and the summary scores."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n(matplotlib not installed; skipping the figures)")
        return

    figures = Path(__file__).resolve().parents[2] / "figures"
    figures.mkdir(exist_ok=True)
    which = 0

    # ------------------------------------------------ marginals, one dataset
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    edges = np.linspace(0, sim.prior_high, 61)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for j, (ax, name) in enumerate(zip(axes, NAMES)):
        ax.hist(references[which][:, j].numpy(), bins=edges, density=True,
                color="0.82", edgecolor="none", label="ABC reference")
        # NCP is a weighted atomic measure: bin the masses rather than resample.
        index = np.clip(np.digitize(probability.atoms[:, j].numpy(), edges) - 1, 0, len(centres) - 1)
        heights = np.bincount(index, weights=probability.weights[which].numpy(), minlength=len(centres))
        ax.plot(centres, heights / np.diff(edges), color="C0", lw=1.8, label="NCP")
        density, _ = np.histogram(npe_draws[which, :, j].numpy(), bins=edges, density=True)
        ax.plot(centres, density, color="C3", lw=1.8, label="NPE")
        ax.axvline(float(theta_true[which, j]), color="k", ls="--", lw=1.2, label=r"$\theta_{true}$")
        ax.axhline(1.0 / sim.prior_high, color="0.5", ls=":", lw=1.0, label="prior")
        ax.set_xlabel(name)
        ax.set_xlim(0, sim.prior_high)
        if j == 0:
            ax.set_ylabel("posterior density")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("g-and-k marginal posteriors for one dataset (100 observations, 4 octile summaries)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(figures / "gandk_posteriors.png", dpi=150)
    plt.close(fig)

    # --------------------------------------------------------- summary panel
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(4)
    ax = axes[0]
    ax.bar(x - 0.2, (ncp_w1.mean(0) / prior_sd).numpy(), 0.4, color="C0", label="NCP")
    ax.bar(x + 0.2, (npe_w1.mean(0) / prior_sd).numpy(), 0.4, color="C3", label="NPE")
    for k, value in enumerate(drift.tolist()):
        ax.hlines(value, k - 0.42, k + 0.42, color="k", ls="--", lw=1.2,
                  label="reference floor" if k == 0 else None)
    ax.set_xticks(x, NAMES)
    ax.set_ylabel(r"$W_1$ to ABC / prior sd")
    ax.set_title("marginal accuracy (lower is better)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    ax.bar(x - 0.2, ncp_cov, 0.4, color="C0", label="NCP")
    ax.bar(x + 0.2, npe_cov, 0.4, color="C3", label="NPE")
    ax.axhline(0.9, color="k", ls="--", lw=1.2, label="nominal 0.90")
    ax.set_xticks(x, NAMES)
    ax.set_ylim(0.8, 1.02)  # zoomed: everything of interest sits above 0.85
    ax.set_ylabel("coverage of 90% intervals")
    ax.set_title("calibration (closer to the line is better)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(figures / "gandk_npe_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nfigures written to {figures}/gandk_posteriors.png and "
          f"{figures}/gandk_npe_comparison.png")


if __name__ == "__main__":
    main()
