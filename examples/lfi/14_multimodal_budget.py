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
NPE_COMPONENTS = 10

NCP_COLOUR, NPE_COLOUR, TRUTH_COLOUR = "#1f77b4", "#ff7f0e", "0.80"
RANK_RAMP = ("#9ecae1", "#4292c6", "#08306b")


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

    _plot(panels, npe_w1, draws[0, :, 0])

    print("\n" + "=" * 96)
    print("Reading")
    print("=" * 96)
    print("  Read the 'valley' column down each budget block. It is the mass the")
    print("  estimate puts between the modes, where the truth has a minimum, so it")
    print("  measures the actual defect rather than an aggregate. If it shrinks with")
    print("  rank, the smoothing explanation is right and the fix is spectral terms.")
    print("  If it does not, sharp multimodality is a limitation of a low-rank ratio")
    print("  model and no budget removes it.")
    print()
    print("  Compare the NCP W1 column against the NPE row. Closing the gap is what")
    print("  'achievable by spending more' would mean; note also the fit-time column,")
    print("  since rank costs O(n d^2) and that is the price of the fix if it works.")


def _plot(panels, npe_w1, npe_first) -> None:
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
        ax.plot(centres, height / np.diff(edges), color=RANK_RAMP[k], lw=2.0,
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
    fig.suptitle("Does more rank let the operator resolve the valley between modes? "
                 f"(p = {THETA_DIM}, largest budget)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(figures / "multimodal_budget.png", dpi=150)
    plt.close(fig)
    print(f"\nfigure written to {figures}/multimodal_budget.png")


if __name__ == "__main__":
    main()
