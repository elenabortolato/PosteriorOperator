"""Multimodality, against an exact posterior: where the two families differ.

Every other exact-reference model in this directory has a unimodal posterior,
and on those the operator and NPE differ quantitatively -- one is a bit better
or worse depending on chi^2. Multimodality is where they differ STRUCTURALLY,
and it is the case the operator's representation should handle for free.

The model. Theta ~ N(mu, I_p), and each coordinate is observed only through
its square, Y_j ~ N(Theta_j^2, sigma^2/m). The sign is unrecoverable, so every
marginal posterior is bimodal near +-sqrt(y_j) and the joint has up to 2^p
modes. The prior mean is non-zero on purpose: a symmetric prior would give the
two modes equal mass and make the posterior mean identically zero, which would
render every mean comparison vacuous.

The posterior is EXACT. Coordinates are independent given the data, so each
marginal is a one-dimensional integral evaluated by quadrature on a grid that
holds the whole prior mass -- machine precision, not an ABC tolerance. The
test suite checks it against brute-force importance sampling.

WHY THIS SHOULD FAVOUR THE OPERATOR, and it is a prediction, not a result. The
operator reweights prior draws, so a second mode costs it nothing: if prior
draws sit near both modes, the weights simply place mass on both. NPE must
spend mixture components, and with 2^p modes a fixed budget of ten components
is outrun by p = 4. The sweep below takes p = 2, 4 and 6 -- four, sixteen and
sixty-four modes -- and the question is whether NPE degrades on that axis while
the operator does not.

The honest alternative outcome: the operator's atoms are prior draws, and in
p dimensions the share of them landing near ANY posterior mode falls with p,
so it may degrade for a different reason at the same time. The prior-only
column separates the two.

Run:  python examples/lfi/13_multimodal_exact.py
"""

import time
from pathlib import Path

import torch

from posterior_operator import PosteriorOperator
from posterior_operator.baselines import NeuralPosteriorEstimator
from posterior_operator.simulators import SignAmbiguous

SEED = 0
N_SIM = 60000
RANK = 64
LAYER_SIZE = 128
EPOCHS = 300
N_EVAL = 100
N_DRAWS = 4000
DIMENSIONS = (2, 4, 6)
NPE_COMPONENTS = 10

# Validated for colour-vision deficiency: blue/orange separate at dE 24.6,
# where the green/red pair used in earlier figures fails at 3.9 for
# deuteranopia. Line style repeats the distinction.
NCP_COLOUR, NPE_COLOUR, TRUTH_COLOUR = "#1f77b4", "#ff7f0e", "0.80"


def w1_against_exact(grid, exact_cdf, atoms, masses):
    """W_1 between a weighted atomic measure and the exact posterior.

    The exact side is a CDF on a grid, so this is the integral of the absolute
    CDF difference evaluated on that grid -- no sampling on the truth side.
    """
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


def main() -> None:
    print("=" * 94)
    print("Multimodality with an exact posterior: 2^p modes as p grows")
    print("=" * 94)
    print("  Theta ~ N(mu, I_p),  Y_j ~ N(Theta_j^2, sigma^2/m):  the sign is lost,")
    print("  so each marginal is bimodal and the joint has up to 2^p modes.")
    print(f"  {N_SIM} simulations shared, rank {RANK}, NPE with {NPE_COMPONENTS} components,")
    print(f"  {N_EVAL} observations scored. Errors are W1 to the EXACT marginal, in")
    print("  units of the prior sd (which is 1 here by construction).\n")

    print(f"{'p':>4}{'modes':>7}{'prior-only':>12}{'NCP':>9}{'NPE':>9}{'ratio':>8}"
          f"{'NCP mean':>10}{'NPE mean':>10}{'sigma_1':>9}{'chi2':>8}")
    print("-" * 94)

    panels = []
    for p in DIMENSIONS:
        started = time.perf_counter()
        sim = SignAmbiguous(theta_dim=p, noise=0.6, n_obs=4, prior_mean=0.5)
        g = torch.Generator().manual_seed(SEED + 99)
        theta_true, y_obs = sim.sample_joint(N_EVAL, generator=g)
        exact_mean = sim.posterior_mean(y_obs)
        prior_sd = sim.prior_sd()

        theta_fit, y_fit = sim.sample_joint(N_SIM, generator=torch.Generator().manual_seed(SEED + 1))
        torch.manual_seed(SEED)
        operator = PosteriorOperator(theta_dim=p, data_dim=p, rank=RANK, layer_size=LAYER_SIZE)
        operator.fit(theta_fit, y_fit, epochs=EPOCHS, lr=1e-3, seed=SEED)
        posterior = operator.posterior(y_obs).with_projection("isotonic")

        torch.manual_seed(SEED)
        npe = NeuralPosteriorEstimator(theta_dim=p, data_dim=p,
                                       n_components=NPE_COMPONENTS, layer_size=LAYER_SIZE)
        npe.fit(theta_fit, y_fit, epochs=EPOCHS, lr=1e-3, seed=SEED)
        npe_draws = npe.sample(y_obs, N_DRAWS, generator=torch.Generator().manual_seed(SEED + 3))
        prior_draws = sim.sample_prior(N_DRAWS, generator=g)

        flat = torch.full((1, N_DRAWS), 1.0 / N_DRAWS)
        ncp_total = npe_total = prior_total = 0.0
        marginals = []
        for j in range(p):
            grid, exact_cdf = sim.posterior_cdf(y_obs, coordinate=j)
            values, cumulative = posterior._sorted_cdf(j)
            mass = torch.diff(cumulative, dim=-1, prepend=torch.zeros(cumulative.shape[0], 1))
            ncp_total += float(w1_against_exact(grid, exact_cdf, values, mass).mean()) / prior_sd[j]
            # The operator's atoms are shared across observations, so its whole
            # batch scores in one call. NPE draws a fresh sample per observation,
            # so those are scored one at a time.
            per_observation = torch.stack([
                w1_against_exact(grid, exact_cdf[i : i + 1], npe_draws[i, :, j], flat)[0]
                for i in range(N_EVAL)
            ])
            npe_total += float(per_observation.mean()) / prior_sd[j]
            prior_total += float(
                w1_against_exact(grid, exact_cdf, prior_draws[:, j],
                                 flat.expand(N_EVAL, N_DRAWS)).mean()
            ) / prior_sd[j]
            if j < 3:
                marginals.append((grid, sim.posterior_grid(y_obs, coordinate=j)[1][0],
                                  values, mass[0], npe_draws[0, :, j]))

        ncp_w1, npe_w1, prior_w1 = ncp_total / p, npe_total / p, prior_total / p
        print(f"{p:>4}{2**p:>7}{prior_w1:>12.4f}{ncp_w1:>9.4f}{npe_w1:>9.4f}"
              f"{ncp_w1 / max(npe_w1, 1e-12):>8.2f}"
              f"{float((posterior.mean() - exact_mean).abs().mean()):>10.4f}"
              f"{float((npe.mean(y_obs) - exact_mean).abs().mean()):>10.4f}"
              f"{operator.maximal_correlation:>9.4f}{operator.chi2_divergence:>8.2f}"
              f"   [{time.perf_counter() - started:.0f}s]", flush=True)
        panels.append({"p": p, "marginals": marginals, "ncp": ncp_w1, "npe": npe_w1})

    _plot(panels)

    print("\n" + "=" * 94)
    print("Reading")
    print("=" * 94)
    print("  The 'ratio' column is the whole experiment: NCP error over NPE error, as")
    print("  the number of modes goes 4, 16, 64 against a fixed ten-component mixture.")
    print("  Below 1 means the operator is ahead. If the ratio FALLS down the column,")
    print("  the prediction holds -- multimodality costs a mixture density network and")
    print("  costs an atom reweighting nothing.")
    print()
    print("  The prior-only column is the control. Both methods must beat it, and at")
    print("  larger p the posterior is closer to the prior, so an estimator can look")
    print("  good by doing very little; the margin over that column is the real signal.")
    print()
    print("  The figure shows why, one dataset per row: the exact posterior has two")
    print("  humps per coordinate, and the question is which estimator puts mass on")
    print("  both of them.")


def _plot(panels) -> None:
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
    rows = len(panels)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 2.7 * rows), squeeze=False)
    for r, panel in enumerate(panels):
        for c in range(3):
            ax = axes[r][c]
            if c >= len(panel["marginals"]):
                ax.axis("off")
                continue
            grid, exact_density, atoms, mass, npe_sample = panel["marginals"][c]
            edges = np.linspace(float(grid.min()), float(grid.max()), 90)
            centres = 0.5 * (edges[:-1] + edges[1:])
            ax.fill_between(grid.numpy(), exact_density.numpy(), color=TRUTH_COLOUR,
                            label="exact posterior" if r == c == 0 else None)
            index = np.clip(np.digitize(atoms.numpy(), edges) - 1, 0, len(centres) - 1)
            height = np.bincount(index, weights=mass.numpy(), minlength=len(centres))
            ax.plot(centres, height / np.diff(edges), color=NCP_COLOUR, lw=2.0,
                    label="NCP" if r == c == 0 else None)
            density, _ = np.histogram(npe_sample.numpy(), bins=edges, density=True)
            ax.plot(centres, density, color=NPE_COLOUR, lw=2.0, ls="--",
                    label="NPE" if r == c == 0 else None)
            ax.set_yticks([])
            ax.tick_params(labelsize=8)
            ax.set_xlabel(rf"$\theta_{{{c + 1}}}$", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"p = {panel['p']}  ({2 ** panel['p']} modes)\n"
                              f"NCP {panel['ncp']:.3f} / NPE {panel['npe']:.3f}", fontsize=9)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("Bimodal marginals from a squared observation: does each estimator "
                 "find both humps?\n(row label gives mean $W_1$ to the exact posterior "
                 "over all coordinates and 100 datasets)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figures / "multimodal_exact.png", dpi=150)
    plt.close(fig)
    print(f"\nfigure written to {figures}/multimodal_exact.png")


if __name__ == "__main__":
    main()
