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
N_SIM = 240000  # training budget, shared by both methods
N_REFERENCE = 2_000_000  # prior-predictive table for the ABC reference
ABC_KEEP = 2000  # retained draws per observation (0.1%)
N_SCORED = 24  # observations given a full ABC reference
N_COVERAGE = 300  # observations used for the coverage check
# Rank and budget, both measured rather than guessed. A sweep at n = 60000 on
# the CLIPPED measure gave a flat picture -- W1 between 0.287 and 0.310 for
# ranks 32 to 512, coverage pinned near 0.98 -- which is what the clipping does
# rather than what the rank does. Under the monotone projection the same axis
# looks different (examples/lfi/12_gandk_budget.py):
#
#   rank    W1 @ n=15k   W1 @ n=60k   W1 @ n=240k
#      8        0.3310       0.3243        0.3276
#     32        0.1435       0.1407        0.1381
#    128        0.1239       0.1022        0.0939
#
# Read down: budget is worth ~1% at rank 8 and ~24% at rank 128, because low
# rank is truncation-limited and high rank is estimation-limited. Read across:
# rank is worth 63% at a fixed budget, far more than sixteen times the data.
# So rank first, then budget -- and both are set generously below. The
# reference's own self-drift is 0.0378, so differences under that are noise.
RANK = 128
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

    # The third estimate: the same signed masses, projected onto a probability
    # measure monotonically instead of by clipping. See the signed-versus-
    # clipped section below for why this is the interesting comparison.
    isotonic = posterior.with_projection("isotonic")
    iso_masses, iso_atoms = [], []
    for j in range(4):
        values, cumulative = isotonic._sorted_cdf(j)
        iso_atoms.append(values)
        iso_masses.append(
            torch.diff(cumulative, dim=-1, prepend=torch.zeros(cumulative.shape[0], 1))
        )

    ncp_w1 = torch.zeros(N_SCORED, 4)
    iso_w1 = torch.zeros(N_SCORED, 4)
    npe_w1 = torch.zeros(N_SCORED, 4)
    uniform_draws = torch.full((npe_draws.shape[1],), 1.0 / npe_draws.shape[1])
    for i in range(N_SCORED):
        for j in range(4):
            ncp_w1[i, j] = wasserstein1(
                probability.atoms[:, j], probability.weights[i], references[i][:, j]
            )
            iso_w1[i, j] = wasserstein1(iso_atoms[j], iso_masses[j][i], references[i][:, j])
            npe_w1[i, j] = wasserstein1(npe_draws[i, :, j], uniform_draws, references[i][:, j])

    print("=" * 84)
    print(f"Accuracy against the ABC reference, averaged over {N_SCORED} observations")
    print("=" * 84)
    print("  All errors in units of the prior standard deviation, so 1.0 means 'as")
    print("  wrong as answering with the prior'. 'NCP iso' is the same fit with the")
    print("  signed masses projected monotonically rather than clipped.\n")
    print(f"{'':>6}{'|mean err| NCP':>16}{'NPE':>9}{'   ':>4}"
          f"{'W1 NCP clip':>13}{'NCP iso':>10}{'NPE':>9}{'  ref floor':>13}")
    print("-" * 84)
    for j, name in enumerate(NAMES):
        me_ncp = float((ncp_mean[:, j] - reference_mean[:, j]).abs().mean() / prior_sd[j])
        me_npe = float((npe_mean[:, j] - reference_mean[:, j]).abs().mean() / prior_sd[j])
        print(f"{name:>6}{me_ncp:>16.4f}{me_npe:>9.4f}{'':>4}"
              f"{float(ncp_w1[:, j].mean() / prior_sd[j]):>13.4f}"
              f"{float(iso_w1[:, j].mean() / prior_sd[j]):>10.4f}"
              f"{float(npe_w1[:, j].mean() / prior_sd[j]):>9.4f}{float(drift[j]):>13.4f}")
    print("-" * 84)
    print(f"{'all':>6}{float((ncp_mean - reference_mean).abs().mean(0).div(prior_sd).mean()):>16.4f}"
          f"{float((npe_mean - reference_mean).abs().mean(0).div(prior_sd).mean()):>9.4f}"
          f"{'':>4}{float((ncp_w1.mean(0) / prior_sd).mean()):>13.4f}"
          f"{float((iso_w1.mean(0) / prior_sd).mean()):>10.4f}"
          f"{float((npe_w1.mean(0) / prior_sd).mean()):>9.4f}{float(drift.mean()):>13.4f}")

    # ----------------------------------------------------------- calibration
    print("\n" + "=" * 84)
    print(f"Calibration: nominal 90% credible intervals over {N_COVERAGE} observations")
    print("=" * 84)
    print("  This needs no reference -- it asks whether the stated intervals contain the")
    print("  theta that actually generated the data, at the rate they claim.\n")

    all_posterior = operator.posterior(y_obs)
    all_isotonic = all_posterior.with_projection("isotonic")
    all_npe = npe.sample(y_obs, 4000, generator=torch.Generator().manual_seed(5))
    print(f"{'':>6}{'coverage clip':>15}{'iso':>7}{'NPE':>8}{'  ':>3}"
          f"{'width clip':>12}{'iso':>8}{'NPE':>8}{'  ABC':>8}")
    print("-" * 84)
    ncp_cov, iso_cov, npe_cov = [], [], []
    for j, name in enumerate(NAMES):
        interval_ncp = all_posterior.credible_interval(alpha=0.10, coordinate=j)
        inside_ncp = (theta_true[:, j] >= interval_ncp[:, 0]) & (theta_true[:, j] <= interval_ncp[:, 1])
        interval_iso = all_isotonic.credible_interval(alpha=0.10, coordinate=j)
        inside_iso = (theta_true[:, j] >= interval_iso[:, 0]) & (theta_true[:, j] <= interval_iso[:, 1])
        q = torch.tensor([0.05, 0.95])
        marginal = all_npe[:, :, j]
        lo = torch.quantile(marginal, q[0], dim=1)
        hi = torch.quantile(marginal, q[1], dim=1)
        inside_npe = (theta_true[:, j] >= lo) & (theta_true[:, j] <= hi)
        abc_width = torch.stack(
            [torch.quantile(r[:, j], q[1]) - torch.quantile(r[:, j], q[0]) for r in references]
        ).mean()
        iso_cov.append(float(inside_iso.float().mean()))
        ncp_cov.append(float(inside_ncp.float().mean()))
        npe_cov.append(float(inside_npe.float().mean()))
        print(f"{name:>6}{ncp_cov[-1]:>15.3f}{iso_cov[-1]:>7.3f}{npe_cov[-1]:>8.3f}{'':>3}"
              f"{float((interval_ncp[:, 1] - interval_ncp[:, 0]).mean()):>12.3f}"
              f"{float((interval_iso[:, 1] - interval_iso[:, 0]).mean()):>8.3f}"
              f"{float((hi - lo).mean()):>8.3f}{float(abc_width):>8.3f}")
    print("-" * 84)
    print(f"{'mean':>6}{sum(ncp_cov) / 4:>15.3f}{sum(iso_cov) / 4:>7.3f}{sum(npe_cov) / 4:>8.3f}"
          f"     nominal 0.900")

    # ------------------------------------------------- signed versus clipped
    # The interval story above and the mean story earlier use DIFFERENT
    # measures. Moment functionals use the masses as they come, negative ones
    # included, which is Eq. (2) verbatim. Order-statistic queries need a
    # genuine probability measure, so they clip at zero and renormalise. If the
    # negative mass is doing real work -- carving probability out of the prior's
    # tails -- then discarding it must inflate the spread, and the two measures
    # will disagree. That is a testable prediction, so test it.
    negative = float((all_posterior.weights < 0).float().mean())
    ess = all_posterior.effective_sample_size().mean() / all_posterior.n_atoms

    print("\n" + "=" * 84)
    print("Signed versus clipped masses: where the shape error actually comes from")
    print("=" * 84)
    print(f"  fraction of negative masses      {negative:.3f}")
    print(f"  ESS / n_atoms after clipping     {float(ess):.3f}   (1.0 would be the prior itself)\n")

    signed = posterior.covariance().diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
    clipped_sd = probability.covariance().diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
    reference_sd = torch.stack([r.std(0) for r in references])
    print("  posterior standard deviation, averaged over the scored observations:")
    print(f"{'':>6}{'ABC':>9}{'NCP signed':>12}{'NCP clip':>10}{'NCP iso':>9}{'NPE':>9}{'prior':>8}")
    print("-" * 84)
    for j, name in enumerate(NAMES):
        centre = (iso_masses[j] * iso_atoms[j]).sum(-1, keepdim=True)
        iso_sd = (iso_masses[j] * (iso_atoms[j] - centre) ** 2).sum(-1).clamp_min(0).sqrt()
        print(f"{name:>6}{float(reference_sd[:, j].mean()):>9.3f}"
              f"{float(signed[:, j].mean()):>12.3f}{float(clipped_sd[:, j].mean()):>10.3f}"
              f"{float(iso_sd.mean()):>9.3f}"
              f"{float(npe_draws[:, :, j].std(dim=1).mean()):>9.3f}{float(prior_sd[j]):>8.3f}")
    print("-" * 84)
    print("  The signed column is the estimator of Eq. (2) verbatim; the clip and iso")
    print("  columns are two ways of turning the same numbers into a probability measure.")
    print("  If signed tracks the ABC while clipped runs towards the prior, the negative")
    print("  mass is not numerical noise to be discarded: it is carrying the information")
    print("  that makes the posterior tighter than the prior, and clipping throws it out.")
    print("  The isotonic projection keeps it -- it accumulates the signed masses into a")
    print("  CDF and projects THAT onto monotone functions, so a negative mass still")
    print("  pulls probability off the atoms before it instead of being deleted.")

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

    _plot(sim, references, probability, iso_atoms, iso_masses, npe_draws, theta_true,
          prior_sd, ncp_w1, iso_w1, npe_w1, drift, ncp_cov, iso_cov, npe_cov)

    print("\n" + "=" * 84)
    print("Reading")
    print("=" * 84)
    print("  Read the W1 column, not the mean column. Under this prior g is only weakly")
    print("  identified by 100 observations, so a method can land the posterior mean and")
    print("  still be badly wrong about the shape; W1 against the reference catches that")
    print("  and the mean error does not.")
    print()
    print("  On the clipped measure -- what every order-statistic query uses by default --")
    print("  NPE is several times better on shape, and the rank sweep at the top of this")
    print("  file rules out capacity as the explanation. But the posterior MEANS tie, and")
    print("  the signed standard deviations tie, which says the operator has estimated the")
    print("  dependence perfectly well and is losing the information afterwards, in the")
    print("  step that makes the masses non-negative. Swapping clipping for the monotone")
    print("  projection closes essentially the whole gap. So the finding is not 'NPE beats")
    print("  the operator on this model'; it is 'clip-and-renormalise is the wrong")
    print("  projection, and it was costing the operator the comparison'.")
    print()
    print("  Everything here is scored against an ABC reference, which is itself an")
    print("  estimate. The 'ref floor' column says how far that reference moves when its")
    print("  own tolerance is tightened fourfold. A method-to-method gap smaller than the")
    print("  floor is not evidence of anything -- which is why the iso-versus-NPE")
    print("  difference should be read as a tie rather than as a win for either.")


def _plot(sim, references, probability, iso_atoms, iso_masses, npe_draws, theta_true,
          prior_sd, ncp_w1, iso_w1, npe_w1, drift, ncp_cov, iso_cov, npe_cov) -> None:
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
        # Both NCP curves are weighted atomic measures: bin the masses rather
        # than resample, so no kernel width enters the picture.
        def binned(values, masses):
            index = np.clip(np.digitize(values, edges) - 1, 0, len(centres) - 1)
            return np.bincount(index, weights=masses, minlength=len(centres)) / np.diff(edges)

        ax.plot(centres, binned(probability.atoms[:, j].numpy(), probability.weights[which].numpy()),
                color="0.45", lw=1.5, ls="--", label="NCP (clipped)")
        ax.plot(centres, binned(iso_atoms[j].numpy(), iso_masses[j][which].numpy()),
                color="#1f77b4", lw=2.0, label="NCP (isotonic)")
        density, _ = np.histogram(npe_draws[which, :, j].numpy(), bins=edges, density=True)
        ax.plot(centres, density, color="#ff7f0e", lw=2.0, ls=":", label="NPE")
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
    ax.bar(x - 0.26, (ncp_w1.mean(0) / prior_sd).numpy(), 0.26, color="0.45", label="NCP (clipped)")
    ax.bar(x, (iso_w1.mean(0) / prior_sd).numpy(), 0.26, color="#1f77b4", label="NCP (isotonic)")
    ax.bar(x + 0.26, (npe_w1.mean(0) / prior_sd).numpy(), 0.26, color="#ff7f0e", label="NPE")
    for k, value in enumerate(drift.tolist()):
        ax.hlines(value, k - 0.42, k + 0.42, color="k", ls="--", lw=1.2,
                  label="reference floor" if k == 0 else None)
    ax.set_xticks(x, NAMES)
    ax.set_ylabel(r"$W_1$ to ABC / prior sd")
    ax.set_title("marginal accuracy (lower is better)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    ax.bar(x - 0.26, ncp_cov, 0.26, color="0.45", label="NCP (clipped)")
    ax.bar(x, iso_cov, 0.26, color="#1f77b4", label="NCP (isotonic)")
    ax.bar(x + 0.26, npe_cov, 0.26, color="#ff7f0e", label="NPE")
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
