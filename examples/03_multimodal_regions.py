"""Multimodal conditionals: where a mean and a variance actively mislead.

The target is a two-component mixture whose modes separate as x moves away
from zero and whose mixing weight sweeps from one mode to the other. At x = 0
the conditional mean sits in a low-density valley *between* the modes, so the
usual mean-plus-interval summary reports a centre where almost no probability
mass lives.

This script contrasts two conditional regions at level 1 - alpha:

  * the shortest interval, which must bridge the valley to reach its coverage;
  * the highest-density region, which is allowed to be disjoint and so
    reports the two modes separately, at a fraction of the total width.

Writes a figure to `figures/multimodal.png` when matplotlib is available.

Run:  python examples/03_multimodal_regions.py
"""

from pathlib import Path

import torch

from posterior_operator import BimodalMixture, GaussianKDE, build_ncp, train_ncp
from posterior_operator.metrics import coverage, hellinger

SEED = 6
N_TRAIN, N_VAL, N_TEST = 12000, 3000, 5000
ALPHA = 0.1
GRID = torch.linspace(-3.5, 3.5, 701)


def main() -> None:
    data = BimodalMixture()
    g = torch.Generator().manual_seed(SEED)
    x_train, y_train = data.sample(N_TRAIN, generator=g)
    val = data.sample(N_VAL, generator=g)
    x_test, y_test = data.sample(N_TEST, generator=g)

    torch.manual_seed(SEED)
    operator = build_ncp(x_dim=1, y_dim=1, latent_dim=48, layer_size=96)
    train_ncp(
        operator,
        x_train,
        y_train,
        epochs=800,
        lr=1e-3,
        validation_data=val,
        patience=50,
        val_every=5,
        seed=SEED,
        verbose=True,
        log_every=200,
    )
    print(f"\n{operator.parameters_summary()}")

    marginal = GaussianKDE(operator.reference_y)
    x_probe = torch.tensor([[-0.8], [-0.3], [0.0], [0.3], [0.8]])
    posterior = operator.condition(x_probe)
    density = posterior.density(GRID, marginal)
    truth = data.conditional_pdf(x_probe, GRID)

    print("\nDensity accuracy and the mean/mode mismatch")
    print(f"{'x':>6}  {'hellinger':>10}  {'mean':>15}  {'density at mean':>16}  {'peak density':>12}")
    for i, xv in enumerate(x_probe.flatten().tolist()):
        mean_est = float(posterior.mean()[i])
        mean_true = float(data.conditional_mean(x_probe[i : i + 1]))
        at_mean = float(data.conditional_pdf(x_probe[i : i + 1], torch.tensor([mean_est]))[0])
        print(
            f"{xv:>6.2f}  {float(hellinger(density[i : i + 1], truth[i : i + 1], GRID)):>10.4f}  "
            f"{mean_est:>7.3f} ({mean_true:>5.2f})  {at_mean:>16.4f}  {float(truth[i].max()):>12.4f}"
        )

    # --- shortest interval vs highest-density region -----------------------
    intervals = posterior.interval(ALPHA)
    mask, level = posterior.highest_density_region(GRID, marginal, alpha=ALPHA)
    cell = _cell_widths(GRID)
    print(f"\nConditional regions at level {1 - ALPHA:.2f}: total width and true mass covered")
    print(f"{'x':>6}  {'interval':>22}  {'width':>7}  {'HDR width':>10}  {'HDR pieces':>11}  {'true mass':>10}")
    for i, xv in enumerate(x_probe.flatten().tolist()):
        lo, hi = intervals[i].tolist()
        hdr_width = float((cell * mask[i]).sum())
        pieces = _count_runs(mask[i])
        true_mass = float((truth[i] * cell * mask[i]).sum())
        print(
            f"{xv:>6.2f}  [{lo:>9.3f}, {hi:>8.3f}]  {hi - lo:>7.3f}  "
            f"{hdr_width:>10.3f}  {pieces:>11d}  {true_mass:>10.3f}"
        )

    print("\nAt x = 0 the two modes are far apart, so the shortest interval pays for the valley;")
    print("the highest-density region skips it and reports the modes separately.")

    # --- calibration --------------------------------------------------------
    test_posterior = operator.condition(x_test)
    print(f"\nOut-of-sample interval coverage: {float(coverage(test_posterior.interval(ALPHA), y_test)):.3f}")
    print(f"Mean Hellinger distance over the probes: {float(hellinger(density, truth, GRID).mean()):.4f}")

    _plot(data, operator, marginal)


def _cell_widths(grid: torch.Tensor) -> torch.Tensor:
    edges = torch.cat([grid[:1], 0.5 * (grid[1:] + grid[:-1]), grid[-1:]])
    return edges[1:] - edges[:-1]


def _count_runs(mask: torch.Tensor) -> int:
    padded = torch.cat([torch.zeros(1, dtype=torch.bool), mask, torch.zeros(1, dtype=torch.bool)])
    return int(((~padded[:-1]) & padded[1:]).sum())


def _plot(data, operator, marginal) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping the figure)")
        return

    x_probe = torch.tensor([[-0.8], [0.0], [0.8]])
    sub = operator.condition(x_probe)
    density = sub.density(GRID, marginal)
    truth = data.conditional_pdf(x_probe, GRID)
    intervals = sub.interval(ALPHA)
    hdr_mask, _ = sub.highest_density_region(GRID, marginal, alpha=ALPHA)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for i, ax in enumerate(axes):
        ax.fill_between(GRID, 0, truth[i], color="#e2e8f0", label="true density")
        ax.plot(GRID, density[i], color="#1d4ed8", lw=2, label="NCP density")
        ax.axvspan(
            float(intervals[i, 0]),
            float(intervals[i, 1]),
            color="#f59e0b",
            alpha=0.18,
            label=f"shortest {int((1 - ALPHA) * 100)}% interval",
        )
        top = float(max(truth[i].max(), density[i].max())) * 1.05
        ax.fill_between(
            GRID,
            0,
            top,
            where=hdr_mask[i],
            color="#10b981",
            alpha=0.22,
            linewidth=0,
            label=f"{int((1 - ALPHA) * 100)}% HDR",
        )
        ax.axvline(float(sub.mean()[i]), color="#b91c1c", ls="--", lw=1.4, label="conditional mean")
        ax.set_title(f"x = {float(x_probe[i]):.1f}")
        ax.set_xlabel("y")
        ax.set_ylim(0, top)
    axes[0].set_ylabel("conditional density")
    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle("Multimodal conditionals: the mean falls in the valley, the HDR does not", y=1.02)
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    path = out / "multimodal.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
