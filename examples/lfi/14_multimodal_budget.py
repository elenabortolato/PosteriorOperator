"""Can budget and rank buy the operator its way out of multimodality?

Example 13 found the operator losing to NPE on a bimodal posterior, and the
figure diagnosed why: it places a spurious peak in the VALLEY between the two
modes. The proposed explanation is smoothing. A rank-d truncation approximates
the density ratio by d smooth spectral terms, and a sharp bimodal needs many of
them, so the estimate cannot turn quickly enough and fills the gap.

That explanation makes a falsifiable prediction, which is what this script
tests: the binding constraint should be RANK, not simulation budget, because
the defect is approximation rather than estimation. Example 11 adds the rider
that rank has to be paid for with data, since the whitening's plug-in bias
grows like sqrt(d/n), so the two are swept together.

If the prediction holds, the valley fills in less as the rank rises and the
operator closes on NPE. If the error plateaus with rank, sharp multimodality
is a genuine limitation of a low-rank ratio model rather than a budget
problem, and the paper should say so.

The posterior is exact (quadrature on independent coordinates), so nothing
here is limited by a reference.

Run:  python examples/lfi/14_multimodal_budget.py
"""

import time
from pathlib import Path

import torch

from posterior_operator import PosteriorOperator
from posterior_operator.baselines import NeuralPosteriorEstimator
from posterior_operator.simulators import SignAmbiguous

SEED = 0
THETA_DIM = 4  # 16 joint modes
LAYER_SIZE = 128
EPOCHS = 200
N_EVAL = 60
N_DRAWS = 4000
RANKS = (16, 64, 256)
BUDGETS = (60_000, 240_000)
# Rank alone is capped by the embedding width: a layer_size-wide network emits
# into a subspace of at most that dimension, so asking for rank 256 at width
# 128 yields 128 usable singular values and 128 numerical zeros (measured:
# sigma_128 = 6.8e-4, sigma_200 = 4.8e-9, against 0.277 and 0.078 at width
# 512, where the captured chi^2 doubles from 31.5 to 63.8). The second table
# therefore raises the two together. The budget is smaller there because the
# first table establishes that data is irrelevant at fixed rank.
PAIRED_RANKS = (32, 64, 128, 256, 512, 1024)
PAIRED_BUDGET = 30_000
PAIRED_EPOCHS = 150
# The budget has to grow with the rank here, unlike in the first table. The
# whitening's plug-in bias goes like sqrt(d/n), which is 0.03 at rank 32 and
# n = 30000 but 0.18 at rank 1024 -- large enough that a degradation there
# would be estimation noise rather than a statement about approximation. Held
# at roughly sqrt(d/n) = 0.09 by scaling n with d.
PAIRED_PER_RANK = 120
NPE_COMPONENTS = 10

NCP_COLOUR, NPE_COLOUR, TRUTH_COLOUR = "#1f77b4", "#ff7f0e", "0.80"


def w1_against_exact(grid, exact_cdf, atoms, masses):
    """W_1 between a weighted atomic measure and the exact posterior CDF."""
    order = torch.argsort(atoms)
    values, mass = atoms[order].contiguous(), masses[:, order]
    cumulative = torch.cumsum(mass, dim=-1) / mass.sum(dim=-1, keepdim=True)
    index = torch.searchsorted(values, grid.contiguous(), right=True) - 1
    estimated = torch.where(
        (index < 0).unsqueeze(0),
        torch.zeros(1, dtype=grid.dtype),
        cumulative[:, index.clamp(min=0)],
    )
    return torch.trapezoid((estimated - exact_cdf).abs(), grid, dim=-1)


def valley_excess(grid, exact_density, atoms, mass):
    """How much mass the estimate puts where the truth has a valley.

    The defect example 13 found is specifically a peak BETWEEN the modes, which
    an aggregate distance blurs together with everything else. This isolates
    it: the estimated probability of the interval around the density minimum
    between the two modes, minus the exact probability of the same interval.

    Returns NaN for a coordinate whose posterior is unimodal, so that those are
    excluded rather than silently counted as zero excess.

    The second mode has to be found as a genuine LOCAL maximum. Taking an
    argmax over a truncated range instead returns the shoulder of the dominant
    peak, which is adjacent to it, and every downstream quantity then collapses
    -- an earlier version of this function did exactly that and reported 0.0
    for every input, including a measure with all its mass piled in the valley.
    """
    row = exact_density
    interior = row[1:-1]
    maxima = torch.nonzero((interior > row[:-2]) & (interior > row[2:])).flatten() + 1
    if maxima.numel() < 2:
        return float("nan")
    top_two = maxima[torch.argsort(row[maxima], descending=True)[:2]]
    low, high = int(top_two.min()), int(top_two.max())
    trough = low + int(row[low : high + 1].argmin())
    width = max((high - low) // 6, 1)
    left = float(grid[max(trough - width, 0)])
    right = float(grid[min(trough + width, len(grid) - 1)])
    inside = (grid >= left) & (grid <= right)
    exact_mass = float(torch.trapezoid(row[inside], grid[inside]))
    estimated_mass = float(mass[(atoms >= left) & (atoms <= right)].sum())
    return estimated_mass - exact_mass


def main() -> None:
    import sys

    cache = Path(__file__).resolve().parent / ".14_multimodal_cache.pt"
    if "--figures-only" in sys.argv and cache.exists():
        saved = torch.load(cache, weights_only=False)
        _plot(saved["panels"], saved["npe_w1"], saved["npe_first"])
        _plot(saved["paired"], saved["npe_w1"], saved["npe_first"],
              name="multimodal_width.png", title="Rank AND width raised together")
        return
    sim = SignAmbiguous(theta_dim=THETA_DIM, noise=0.6, n_obs=4, prior_mean=0.5)
    g = torch.Generator().manual_seed(SEED + 99)
    _, y_obs = sim.sample_joint(N_EVAL, generator=g)
    prior_sd = sim.prior_sd()
    exact = [sim.posterior_cdf(y_obs, coordinate=j) for j in range(THETA_DIM)]
    exact_density = [sim.posterior_grid(y_obs, coordinate=j)[1] for j in range(THETA_DIM)]

    print("=" * 96)
    print(f"Can rank and budget fix the operator's multimodality failure? "
          f"(p = {THETA_DIM}, {2 ** THETA_DIM} modes)")
    print("=" * 96)
    print("  The posterior is exact, so nothing here is limited by a reference.")
    print("  'valley' is the extra mass placed between the two modes, where the truth")
    print("  has a minimum -- the specific defect, isolated from the aggregate error.\n")

    print(f"{'n_sim':>9}{'rank':>7}{'chi2_hat':>10}{'sigma_1':>9}{'NCP W1':>9}"
          f"{'valley':>9}{'fit s':>8}")
    print("-" * 96)

    panels = []
    for n_sim in BUDGETS:
        theta_fit, y_fit = sim.sample_joint(n_sim, generator=torch.Generator().manual_seed(SEED + 1))
        for rank in RANKS:
            started = time.perf_counter()
            torch.manual_seed(SEED)
            operator = PosteriorOperator(
                theta_dim=THETA_DIM, data_dim=THETA_DIM, rank=rank, layer_size=LAYER_SIZE
            )
            operator.fit(theta_fit, y_fit, epochs=EPOCHS, lr=1e-3, seed=SEED)
            elapsed = time.perf_counter() - started
            posterior = operator.posterior(y_obs).with_projection("isotonic")

            total = 0.0
            valley_terms = []
            first = None
            for j in range(THETA_DIM):
                grid, exact_cdf = exact[j]
                values, cumulative = posterior._sorted_cdf(j)
                mass = torch.diff(cumulative, dim=-1, prepend=torch.zeros(cumulative.shape[0], 1))
                total += float(w1_against_exact(grid, exact_cdf, values, mass).mean()) / prior_sd[j]
                excess = valley_excess(grid, exact_density[j][0], values, mass[0])
                if excess == excess:  # skip NaN: that coordinate is unimodal
                    valley_terms.append(excess)
                if j == 0:
                    first = (grid, exact_density[j][0], values, mass[0])
            print(f"{n_sim:>9}{rank:>7}{operator.chi2_divergence:>10.2f}"
                  f"{operator.maximal_correlation:>9.4f}{total / THETA_DIM:>9.4f}"
                  f"{(sum(valley_terms) / len(valley_terms)) if valley_terms else float('nan'):>9.4f}"
                  f"{elapsed:>8.0f}", flush=True)
            if n_sim == BUDGETS[-1]:
                panels.append({"rank": rank, "w1": total / THETA_DIM, "first": first})

    # ------------------------------------------------------------------ #
    # Rank and width raised together, which is the only way rank rises at all
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 96)
    print("Rank AND width together (rank alone is capped by the embedding width)")
    print("=" * 96)
    print(f"{'rank=width':>12}{'n_sim':>9}{'sqrt(d/n)':>11}{'chi2_hat':>10}{'alive':>8}"
          f"{'sigma_1':>9}{'NCP W1':>9}{'valley':>9}{'fit s':>8}")
    print("-" * 96)
    paired_panels = []
    for rank in PAIRED_RANKS:
        width = rank
        n_paired = max(PAIRED_BUDGET, PAIRED_PER_RANK * rank)
        theta_p, y_p = sim.sample_joint(
            n_paired, generator=torch.Generator().manual_seed(SEED + 1)
        )
        started = time.perf_counter()
        torch.manual_seed(SEED)
        op = PosteriorOperator(theta_dim=THETA_DIM, data_dim=THETA_DIM, rank=rank, layer_size=width)
        op.fit(theta_p, y_p, epochs=PAIRED_EPOCHS, lr=1e-3, seed=SEED)
        elapsed = time.perf_counter() - started
        spectrum = op.singular_values
        alive = int((spectrum > 1e-6 * spectrum[0]).sum())
        post = op.posterior(y_obs).with_projection("isotonic")
        total = 0.0
        terms = []
        first = None
        for j in range(THETA_DIM):
            grid, exact_cdf = exact[j]
            values, cumulative = post._sorted_cdf(j)
            mass = torch.diff(cumulative, dim=-1, prepend=torch.zeros(cumulative.shape[0], 1))
            total += float(w1_against_exact(grid, exact_cdf, values, mass).mean()) / prior_sd[j]
            excess = valley_excess(grid, exact_density[j][0], values, mass[0])
            if excess == excess:
                terms.append(excess)
            if j == 0:
                first = (grid, exact_density[j][0], values, mass[0])
        print(f"{f'{rank}':>12}{n_paired:>9}{(rank / n_paired) ** 0.5:>11.3f}"
              f"{op.chi2_divergence:>10.2f}{alive:>8}"
              f"{op.maximal_correlation:>9.4f}{total / THETA_DIM:>9.4f}"
              f"{(sum(terms) / len(terms)) if terms else float('nan'):>9.4f}"
              f"{elapsed:>8.0f}", flush=True)
        paired_panels.append({"rank": rank, "w1": total / THETA_DIM, "first": first})

    # NPE at the largest budget, as the target to close on.
    theta_fit, y_fit = sim.sample_joint(BUDGETS[-1], generator=torch.Generator().manual_seed(SEED + 1))
    torch.manual_seed(SEED)
    npe = NeuralPosteriorEstimator(theta_dim=THETA_DIM, data_dim=THETA_DIM,
                                   n_components=NPE_COMPONENTS, layer_size=LAYER_SIZE)
    started = time.perf_counter()
    npe.fit(theta_fit, y_fit, epochs=EPOCHS, lr=1e-3, seed=SEED)
    npe_elapsed = time.perf_counter() - started
    draws = npe.sample(y_obs, N_DRAWS, generator=torch.Generator().manual_seed(SEED + 3))
    flat = torch.full((1, N_DRAWS), 1.0 / N_DRAWS)
    npe_total = 0.0
    for j in range(THETA_DIM):
        grid, exact_cdf = exact[j]
        npe_total += float(torch.stack([
            w1_against_exact(grid, exact_cdf[i : i + 1], draws[i, :, j], flat)[0]
            for i in range(N_EVAL)
        ]).mean()) / prior_sd[j]
    npe_w1 = npe_total / THETA_DIM
    print("-" * 96)
    print(f"{BUDGETS[-1]:>9}{'NPE':>7}{'':>10}{'':>9}{npe_w1:>9.4f}{'':>9}{npe_elapsed:>8.0f}")

    cache = Path(__file__).resolve().parent / ".14_multimodal_cache.pt"
    torch.save({"panels": panels, "paired": paired_panels, "npe_w1": npe_w1,
                "npe_first": draws[0, :, 0]}, cache)
    _plot(panels, npe_w1, draws[0, :, 0])
    _plot(paired_panels, npe_w1, draws[0, :, 0], name="multimodal_width.png",
          title="Rank AND width raised together")

    print("\n" + "=" * 96)
    print("Reading")
    print("=" * 96)
    print("  Rank is the lever, budget is not: at fixed width, rank 16 to 256 moves")
    print("  W1 from 0.3262 to 0.2011 while four times the simulations moves it from")
    print("  0.3262 to 0.3290. The defect is approximation, not estimation.")
    print()
    print("  Raising rank AND width together cures the specific defect completely.")
    print("  Valley mass falls 0.0815, 0.1075, 0.0465, 0.0386, 0.0208, 0.0021 across")
    print("  ranks 32 to 1024 -- a fortyfold reduction, and essentially zero at the")
    print("  top. The smoothing account of example 13 was right, and spectral terms")
    print("  are its cure.")
    print()
    print("  BUT THE AGGREGATE ERROR PLATEAUS ANYWAY. W1 falls 0.3413, 0.2885, 0.2049,")
    print("  0.1742, 0.1357, 0.1312: a steady factor of about 0.80 per doubling that")
    print("  stops at the last step, where it is 0.97. It is levelling off near 0.13,")
    print("  more than three times NPE's 0.0386, so more rank will NOT close the gap")
    print("  on this model. An extrapolation of the 0.80 rate -- which this script")
    print("  originally reported -- predicted that rank 27000 would match NPE. That")
    print("  was wrong, because the rate stops holding exactly where it mattered.")
    print()
    print("  The figure shows why. The error changes character rather than shrinking.")
    print("  At low rank the estimate over-smooths and fills the valley; at rank 1024")
    print("  both modes are resolved and the valley is empty, but the curve is jagged,")
    print("  with high-frequency wiggles and an over-tall mode -- ringing from the")
    print("  truncated expansion plus noise in the newly-added directions. One error")
    print("  source is cured and another grows to take its place.")
    print()
    print("  chi^2_hat doubles from 9.3 to 113 and only then slows, reaching 173.7 at")
    print("  rank 1024. It never saturates, so this ratio is simply not low-rank, and")
    print("  the diagnostic says so from the fit alone.")
    print()
    print("  Cost makes the verdict unambiguous. Rank 1024 takes 1914s against NPE's")
    print("  132s -- fourteen times the training time for three and a half times the")
    print("  error. Sharp multimodality is not an intrinsic limitation of the")
    print("  operator, but on this model it is not worth buying out of.")


def _plot(panels, npe_w1, npe_first, name="multimodal_budget.png",
          title="Does more rank let the operator resolve the valley between modes?") -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("(matplotlib not installed; skipping the figure)")
        return

    figures = Path(__file__).resolve().parents[2] / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.0 * (len(panels) + 1), 3.6),
                             squeeze=False)
    for k, panel in enumerate(panels):
        ax = axes[0][k]
        grid, density, atoms, mass = panel["first"]
        edges = np.linspace(float(grid.min()), float(grid.max()), 90)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.fill_between(grid.numpy(), density.numpy(), color=TRUTH_COLOUR, label="exact")
        index = np.clip(np.digitize(atoms.numpy(), edges) - 1, 0, len(centres) - 1)
        height = np.bincount(index, weights=mass.numpy(), minlength=len(centres))
        ax.plot(centres, height / np.diff(edges), color=NCP_COLOUR, lw=2.0,
                label=f"NCP rank {panel['rank']}")
        ax.set_title(f"rank {panel['rank']}  ($W_1$ {panel['w1']:.3f})", fontsize=10)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_yticks([])
        ax.set_xlim(-4, 4)
        ax.legend(fontsize=8, frameon=False)
    ax = axes[0][-1]
    grid, density, _, _ = panels[0]["first"]
    edges = np.linspace(float(grid.min()), float(grid.max()), 90)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax.fill_between(grid.numpy(), density.numpy(), color=TRUTH_COLOUR, label="exact")
    npe_density, _ = np.histogram(npe_first.numpy(), bins=edges, density=True)
    ax.plot(centres, npe_density, color=NPE_COLOUR, lw=2.0, ls="--", label="NPE")
    ax.set_title(f"NPE ($W_1$ {npe_w1:.3f})", fontsize=10)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_yticks([])
    ax.set_xlim(-4, 4)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{title}  (p = {THETA_DIM}, {2 ** THETA_DIM} modes)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(figures / name, dpi=150)
    plt.close(fig)
    print(f"\nfigure written to {figures}/{name}")


if __name__ == "__main__":
    main()
