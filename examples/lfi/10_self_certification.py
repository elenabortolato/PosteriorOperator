"""Is the spectrum self-certifying? And what does its accuracy look like?

The claim worth making for this method is not that it beats density estimation
on accuracy -- example 09 shows that depends on the regime. It is that the fit
REPORTS ITS OWN ERROR. The operator approximates the deflated density ratio
r = p(y|theta)/mu(y) - 1 by a rank-d truncation, and Parseval says the error it
leaves behind is the tail of the spectrum:

        || r_d - r ||^2_{L^2(pi x mu)}  =  sum_{k>d} sigma_k^2 .

The right-hand side is computable from the fitted sigma_hat alone, with no
reference posterior. If it tracks the left-hand side, a user can tell whether
to trust an answer before having anything to check it against -- which NPE,
NLE and NRE cannot do, since they return a density with no internal error
signal.

WHY THIS MODEL MAKES THE TEST EXACT. On Dirichlet-Multinomial the deflated
ratio is available in closed form. The multinomial coefficient cancels between
likelihood and marginal, and the marginal is a Dirichlet-multinomial integral:

    r(theta, y) = prod_j theta_j^{y_j} * B(alpha) / B(alpha + y)  -  1 .

So the left-hand side above can be evaluated directly rather than estimated
against a reference, and the calibration plot has no reference error in it at
all. (Checked in the tests: E_{pi x mu}[r] = 0 and E[r^2] agrees with chi^2
computed the independent way, as E_rho[R] - 1.)

Two things are measured:

  1. CALIBRATION -- the reported tail against the true L^2 error, over a
     truncation sweep of one fit. This is the self-certification claim itself.
  2. USEFULNESS -- the reported tail against the downstream error in posterior
     marginals. A diagnostic that tracks the ratio error but not the answer's
     error would be of theoretical interest only.

Then the figures: what the accuracy actually looks like, marginal by marginal,
as the parameter dimension grows, in the regime where the diagnostic says the
method is sound.

Run:  python examples/lfi/10_self_certification.py
"""

import time
from pathlib import Path

import torch

from posterior_operator import PosteriorOperator
from posterior_operator.baselines import NeuralPosteriorEstimator
from posterior_operator.simulators import DirichletMultinomial

SEED = 0
N_SIM = 40000
FIT_RANK = 128
LAYER_SIZE = 128
EPOCHS = 300
N_RATIO = 1500  # draws per side for the L^2(pi x mu) error
N_EVAL = 40  # observations scored for the functional error
N_EXACT = 3000  # exact posterior draws
SEEDS = (0, 1, 2)
TRUNCATIONS = (1, 2, 4, 8, 16, 32, 64, 128)

# Validated for colour-vision deficiency (blue/orange, CVD dE 24.6); the
# green/red pair used in the earlier figures fails at dE 3.9 for deuteranopia.
# Line style carries the same distinction, so identity is never colour alone.
NCP_COLOUR, NPE_COLOUR, TRUTH_COLOUR = "#1f77b4", "#ff7f0e", "0.80"
# chi^2 is an ordered magnitude, so its three levels get a single-hue ramp
# (light -> dark) rather than three categorical hues, with marker shape as the
# secondary encoding so the ordering survives a greyscale print.
CHI2_RAMP = ("#9ecae1", "#4292c6", "#08306b")


# --------------------------------------------------------------------------- #
# The exact deflated ratio
# --------------------------------------------------------------------------- #


def log_marginal(sim, y):
    r"""Closed-form :math:`\log E_\pi[\prod_j \theta_j^{y_j}]`.

    The Dirichlet-multinomial integral ``B(alpha + y) / B(alpha)``.
    """
    alpha, total = sim.alpha, sim.alpha.sum()
    return (
        torch.lgamma(alpha + y).sum(-1)
        - torch.lgamma(total + y.sum(-1))
        - torch.lgamma(alpha).sum()
        + torch.lgamma(total)
    )


def exact_deflated_ratio(sim, theta, y):
    r"""``r(theta, y)`` exactly, shape ``(n_y, n_theta)``."""
    full = sim._full_simplex(theta)
    log_likelihood = y @ torch.log(full.clamp_min(1e-300)).T
    return torch.exp(log_likelihood - log_marginal(sim, y).unsqueeze(-1)) - 1.0


def wasserstein1(atoms, masses, sample):
    """W_1 between a weighted atomic measure and an equally weighted sample."""
    order = torch.argsort(atoms)
    values, mass = atoms[order].contiguous(), masses[order]
    cdf_a = torch.cumsum(mass, 0) / mass.sum()
    values_b = torch.sort(sample).values.contiguous()
    cdf_b = torch.arange(1, values_b.numel() + 1, dtype=values_b.dtype) / values_b.numel()

    def evaluate(v, cdf, grid):
        count = torch.searchsorted(v, grid.contiguous(), right=True)
        out = cdf[(count - 1).clamp(min=0)]
        return torch.where(count == 0, torch.zeros_like(out), out)

    grid = torch.cat([values, values_b]).sort().values
    difference = (evaluate(values, cdf_a, grid) - evaluate(values_b, cdf_b, grid)).abs()
    return float((difference[:-1] * grid.diff()).sum())


# --------------------------------------------------------------------------- #
# 1 + 2. Calibration of the reported tail
# --------------------------------------------------------------------------- #


def calibrate(sim, seed):
    """One fit, then read its error off the spectrum at every truncation."""
    g = torch.Generator().manual_seed(seed + 99)
    theta_fit, y_fit = sim.sample_joint(N_SIM, generator=torch.Generator().manual_seed(seed + 1))
    torch.manual_seed(seed)
    operator = PosteriorOperator(
        theta_dim=sim.theta_dim, data_dim=sim.data_dim, rank=FIT_RANK, layer_size=LAYER_SIZE
    )
    operator.fit(theta_fit, y_fit, epochs=EPOCHS, lr=1e-3, seed=seed)

    # An independent product sample: theta ~ pi, y ~ mu.
    theta_ratio = sim.sample_prior(N_RATIO, generator=g)
    _, y_ratio = sim.sample_joint(N_RATIO, generator=g)
    truth = exact_deflated_ratio(sim, theta_ratio, y_ratio)
    norm = float((truth**2).mean()).__pow__(0.5)  # ||r|| = sqrt(chi^2)

    data_scaled = operator._prepare_data(y_ratio)
    theta_scaled = (
        operator._theta_scaler.transform(theta_ratio)
        if operator._theta_scaler is not None
        else theta_ratio
    )

    theta_obs, y_obs = sim.sample_joint(N_EVAL, generator=g)
    exact = sim.sample_posterior(y_obs, N_EXACT, generator=g)
    prior_sd = sim.prior_sd()

    spectrum = operator.singular_values
    rows = []
    for d in TRUNCATIONS:
        if d > spectrum.numel():
            continue
        reported = float(spectrum[d:].pow(2).sum()).__pow__(0.5)
        estimate = operator.operator.deflated_ratio(data_scaled, theta_scaled, rank=d)
        realised = float(((estimate - truth) ** 2).mean()).__pow__(0.5)

        posterior = operator.posterior(y_obs, rank=d).with_projection("isotonic")
        total = 0.0
        for j in range(sim.theta_dim):
            values, cumulative = posterior._sorted_cdf(j)
            mass = torch.diff(cumulative, dim=-1, prepend=torch.zeros(cumulative.shape[0], 1))
            for i in range(N_EVAL):
                total += wasserstein1(values, mass[i], exact[i, :, j]) / prior_sd[j]
        rows.append(
            {
                "rank": d,
                "reported": reported,
                "realised": realised,
                "w1": total / (sim.theta_dim * N_EVAL),
            }
        )
    return rows, float(spectrum.pow(2).sum()), norm**2


def part_calibration():
    print("=" * 88)
    print("1. Does the reported tail equal the true truncation error?")
    print("=" * 88)
    print("  reported = sqrt(sum_{k>d} sigma_hat_k^2), from the fit alone.")
    print("  realised = || r_d - r ||_{L2(pi x mu)}, against the CLOSED-FORM ratio.")
    print("  Parseval says these are the same number. Any gap is estimation error")
    print("  in sigma_hat, not a modelling choice.\n")

    collected = {}
    for label, (k, n_trials) in {"chi2 ~ 1": (5, 2), "chi2 ~ 5": (5, 8), "chi2 ~ 130": (5, 50)}.items():
        sim = DirichletMultinomial(n_categories=k, n_trials=n_trials, concentration=2.0)
        per_seed = []
        started = time.perf_counter()
        for seed in SEEDS:
            rows, captured, true_chi2 = calibrate(sim, seed)
            per_seed.append((rows, captured, true_chi2))
        collected[label] = per_seed

        captured_mean = sum(c for _, c, _ in per_seed) / len(per_seed)
        true_mean = sum(t for _, _, t in per_seed) / len(per_seed)
        print(f"[{label}]  K={k}, N={n_trials}, dim={sim.theta_dim}   "
              f"({time.perf_counter() - started:.0f}s, {len(SEEDS)} seeds)")
        print(f"  true chi^2 = {true_mean:7.3f}   sum sigma_hat^2 = {captured_mean:7.3f}"
              f"   captured {captured_mean / max(true_mean, 1e-9):5.1%}")
        print(f"  {'rank':>6}{'reported':>11}{'realised':>11}{'ratio':>8}{'W1 to exact':>14}")
        print("  " + "-" * 50)
        for index, d in enumerate(TRUNCATIONS):
            reported = sum(s[0][index]["reported"] for s in per_seed) / len(per_seed)
            realised = sum(s[0][index]["realised"] for s in per_seed) / len(per_seed)
            w1 = sum(s[0][index]["w1"] for s in per_seed) / len(per_seed)
            print(f"  {d:>6}{reported:>11.4f}{realised:>11.4f}"
                  f"{reported / max(realised, 1e-9):>8.2f}{w1:>14.4f}")
        print()
    return collected


# --------------------------------------------------------------------------- #
# 3. What the accuracy looks like as the dimension grows
# --------------------------------------------------------------------------- #


def accuracy_by_dimension():
    """Fit both methods in the regime the diagnostic calls sound, and keep the
    marginals so the figure can show what the error numbers mean."""
    print("=" * 88)
    print("2. What that accuracy looks like, as the dimension grows")
    print("=" * 88)
    print("  chi^2 held near 5 throughout (n_trials tuned per K), so only the")
    print("  dimension moves. These are the settings the diagnostic calls sound.\n")
    print(f"{'K':>4}{'dim':>5}{'chi^2 rep':>11}{'prior W1':>10}{'NCP W1':>9}{'NPE W1':>9}"
          f"{'NCP cov':>9}{'NPE cov':>9}")
    print("-" * 70)

    panels = []
    for k, n_trials in ((3, 15), (5, 8), (10, 5), (20, 5)):
        sim = DirichletMultinomial(n_categories=k, n_trials=n_trials, concentration=2.0)
        g = torch.Generator().manual_seed(SEED + 99)
        theta_true, y_obs = sim.sample_joint(N_EVAL, generator=g)
        exact = sim.sample_posterior(y_obs, N_EXACT, generator=g)
        prior_draws = sim.sample_prior(N_EXACT, generator=g)
        prior_sd = sim.prior_sd()

        theta_fit, y_fit = sim.sample_joint(N_SIM, generator=torch.Generator().manual_seed(SEED + 1))
        torch.manual_seed(SEED)
        operator = PosteriorOperator(
            theta_dim=sim.theta_dim, data_dim=sim.data_dim, rank=64, layer_size=LAYER_SIZE
        )
        operator.fit(theta_fit, y_fit, epochs=EPOCHS, lr=1e-3, seed=SEED)
        posterior = operator.posterior(y_obs).with_projection("isotonic")
        torch.manual_seed(SEED)
        npe = NeuralPosteriorEstimator(
            theta_dim=sim.theta_dim, data_dim=sim.data_dim, n_components=10, layer_size=LAYER_SIZE
        )
        npe.fit(theta_fit, y_fit, epochs=EPOCHS, lr=1e-3, seed=SEED)
        npe_draws = npe.sample(y_obs, N_EXACT, generator=torch.Generator().manual_seed(SEED + 3))

        uniform = torch.full((N_EXACT,), 1.0 / N_EXACT)
        ncp_w1 = npe_w1 = prior_w1 = 0.0
        ncp_cov = npe_cov = 0
        marginals = []
        for j in range(sim.theta_dim):
            values, cumulative = posterior._sorted_cdf(j)
            mass = torch.diff(cumulative, dim=-1, prepend=torch.zeros(cumulative.shape[0], 1))
            interval = posterior.credible_interval(alpha=0.10, coordinate=j)
            lo = torch.quantile(npe_draws[:, :, j], 0.05, dim=1)
            hi = torch.quantile(npe_draws[:, :, j], 0.95, dim=1)
            ncp_cov += int(((theta_true[:, j] >= interval[:, 0])
                            & (theta_true[:, j] <= interval[:, 1])).sum())
            npe_cov += int(((theta_true[:, j] >= lo) & (theta_true[:, j] <= hi)).sum())
            for i in range(N_EVAL):
                ncp_w1 += wasserstein1(values, mass[i], exact[i, :, j]) / prior_sd[j]
                npe_w1 += wasserstein1(npe_draws[i, :, j], uniform, exact[i, :, j]) / prior_sd[j]
                prior_w1 += wasserstein1(prior_draws[:, j], uniform, exact[i, :, j]) / prior_sd[j]
            if j < 4:
                marginals.append((values, mass[0], npe_draws[0, :, j], exact[0, :, j]))

        scale = sim.theta_dim * N_EVAL
        print(f"{k:>4}{sim.theta_dim:>5}{operator.chi2_divergence:>11.2f}"
              f"{prior_w1 / scale:>10.4f}{ncp_w1 / scale:>9.4f}{npe_w1 / scale:>9.4f}"
              f"{ncp_cov / scale:>9.3f}{npe_cov / scale:>9.3f}")
        panels.append(
            {
                "dim": sim.theta_dim,
                "marginals": marginals,
                "ncp_w1": ncp_w1 / scale,
                "npe_w1": npe_w1 / scale,
                "prior_w1": prior_w1 / scale,
            }
        )
    print()
    return panels


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def make_figures(collected, panels):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("(matplotlib not installed; skipping the figures)")
        return

    figures = Path(__file__).resolve().parents[2] / "figures"
    figures.mkdir(exist_ok=True)

    # ------------------------------------------------ figure 1: calibration
    # Once the spectrum is exhausted the reported tail is ~0, which on a log
    # axis would swallow the entire informative range. Those points are real
    # and are shown, but pinned to the axis floor and labelled, rather than
    # allowed to set the scale.
    def split(reported, other):
        alive = reported > 1e-3
        return alive, ~alive

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    markers = {"chi2 ~ 1": "o", "chi2 ~ 5": "s", "chi2 ~ 130": "^"}
    colours = dict(zip(collected.keys(), CHI2_RAMP))

    series = {}
    for label, per_seed in collected.items():
        series[label] = (
            np.array([[r["reported"] for r in s_[0]] for s_ in per_seed]).mean(0),
            np.array([[r["realised"] for r in s_[0]] for s_ in per_seed]).mean(0),
            np.array([[r["w1"] for r in s_[0]] for s_ in per_seed]).mean(0),
        )

    live = np.concatenate([r[r > 1e-3] for r, _, _ in series.values()])
    real_all = np.concatenate([x for _, x, _ in series.values()])
    low = min(live.min(), real_all.min()) * 0.45
    high = max(live.max(), real_all.max()) * 1.8

    ax = axes[0]
    ax.plot([low, high], [low, high], color="0.35", lw=1.2, ls=":", zorder=1,
            label="perfect calibration")
    for label, (reported, realised, _) in series.items():
        alive, dead = split(reported, realised)
        ax.plot(realised[alive], reported[alive], marker=markers[label], ms=8, lw=1.6,
                color=colours[label], label=label, zorder=3)
        if dead.any():
            ax.scatter(realised[dead], np.full(dead.sum(), low * 1.35), marker="v", s=55,
                       color=colours[label], edgecolor="white", linewidth=0.8, zorder=4)
    ax.axhspan(low, low * 1.9, color="0.93", zorder=0)
    ax.text(high * 0.92, low * 2.2, "spectrum exhausted:\nreported tail $\\to$ 0",
            ha="right", va="bottom", fontsize=8, color="0.35")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(r"true truncation error  $\|\hat r_d - r\|_{L^2(\pi\times\mu)}$")
    ax.set_ylabel(r"reported from the fit  $(\sum_{k>d}\hat\sigma_k^2)^{1/2}$")
    ax.set_title("the spectrum reporting its own error", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, lw=0.6)

    ax = axes[1]
    for label, (reported, _, w1) in series.items():
        alive, dead = split(reported, w1)
        ax.plot(reported[alive], w1[alive], marker=markers[label], ms=8, lw=1.6,
                color=colours[label], label=label, zorder=3)
        if dead.any():
            ax.scatter(np.full(dead.sum(), low * 1.35), w1[dead], marker="<", s=55,
                       color=colours[label], edgecolor="white", linewidth=0.8, zorder=4)
    ax.axvspan(low, low * 1.9, color="0.93", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_xlabel(r"reported from the fit  $(\sum_{k>d}\hat\sigma_k^2)^{1/2}$")
    ax.set_ylabel(r"posterior marginal error  $W_1$ / prior sd")
    ax.set_title("and ordering the answer's error", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(figures / "self_certification.png", dpi=150)
    plt.close(fig)

    # ----------------------------------- figure 2: accuracy by dimension
    rows = len(panels)
    fig, axes = plt.subplots(rows, 4, figsize=(13, 2.5 * rows), squeeze=False)
    for row, panel in enumerate(panels):
        for column in range(4):
            ax = axes[row][column]
            if column >= len(panel["marginals"]):
                ax.axis("off")
                continue
            values, mass, npe_sample, exact_sample = panel["marginals"][column]
            lo = float(min(exact_sample.min(), npe_sample.min()))
            hi = float(max(exact_sample.max(), npe_sample.max()))
            pad = 0.15 * (hi - lo) + 1e-6
            edges = np.linspace(max(0.0, lo - pad), hi + pad, 48)
            centres = 0.5 * (edges[:-1] + edges[1:])
            ax.hist(exact_sample.numpy(), bins=edges, density=True,
                    color=TRUTH_COLOUR, edgecolor="none",
                    label="exact posterior" if row == column == 0 else None)
            index = np.clip(np.digitize(values.numpy(), edges) - 1, 0, len(centres) - 1)
            height = np.bincount(index, weights=mass.numpy(), minlength=len(centres))
            ax.plot(centres, height / np.diff(edges), color=NCP_COLOUR, lw=2.0,
                    label="NCP" if row == column == 0 else None)
            density, _ = np.histogram(npe_sample.numpy(), bins=edges, density=True)
            ax.plot(centres, density, color=NPE_COLOUR, lw=2.0, ls="--",
                    label="NPE" if row == column == 0 else None)
            ax.set_yticks([])
            ax.tick_params(labelsize=8)
            if column == 0:
                ax.set_ylabel(f"dim {panel['dim']}\n"
                              f"NCP {panel['ncp_w1']:.3f} / NPE {panel['npe_w1']:.3f}",
                              fontsize=9)
            ax.set_xlabel(rf"$\theta_{{{column + 1}}}$", fontsize=9)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("One dataset per row: exact posterior marginals against both estimates, "
                 "as the parameter dimension grows at fixed $\\chi^2$\n"
                 "(row label gives the mean $W_1$ over all coordinates and 40 datasets, "
                 "in units of the prior sd)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figures / "accuracy_by_dimension.png", dpi=150)
    plt.close(fig)
    print(f"figures written to {figures}/self_certification.png and "
          f"{figures}/accuracy_by_dimension.png")


def main() -> None:
    import sys

    cache = Path(__file__).resolve().parent / ".10_self_certification_cache.pt"
    if "--figures-only" in sys.argv and cache.exists():
        payload = torch.load(cache, weights_only=False)
        collected, panels = payload["collected"], payload["panels"]
    else:
        collected = part_calibration()
        panels = accuracy_by_dimension()
        torch.save({"collected": collected, "panels": panels}, cache)
    make_figures(collected, panels)

    print("=" * 88)
    print("Reading")
    print("=" * 88)
    print("  The left panel of the first figure is the claim. If the points sit on the")
    print("  diagonal, the fitted spectrum is an honest report of the fit's own error,")
    print("  and a user can read it off without owning a reference posterior. Points")
    print("  BELOW the diagonal mean the fit is flattering itself -- the tail it can")
    print("  see stops at its own rank, so whatever mass the truncation never captured")
    print("  is invisible to it. The 'captured' percentage above says how much of the")
    print("  true chi^2 the fit accounts for, and that is the honest ceiling on how")
    print("  much of its error it can possibly report.")
    print()
    print("  The right panel is what makes it useful rather than merely true: the same")
    print("  reported number ordering the actual error in the posterior marginals.")
    print()
    print("  The second figure answers what the error numbers mean. A W_1 of 0.07 in")
    print("  units of the prior standard deviation is hard to interpret in the")
    print("  abstract; the marginals show it directly, and show NPE's curves drifting")
    print("  off the exact posterior as the dimension grows while the operator's stay")
    print("  on it.")


if __name__ == "__main__":
    main()
