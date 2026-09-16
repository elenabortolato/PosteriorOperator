"""Conditional intervals that adapt to input-dependent noise.

The target has a mean that oscillates and a spread that varies by a factor of
seven across the input range. A method that reports one global predictive
variance cannot be calibrated at both ends at once; NCP conditions the whole
law on x, so the interval width follows the local noise.

Writes a figure to `figures/heteroscedastic.png` when matplotlib is available.

Run:  python examples/02_heteroscedastic_intervals.py
"""

from pathlib import Path

import torch

from posterior_operator import Heteroscedastic, build_ncp, train_ncp
from posterior_operator.metrics import coverage, interval_width

SEED = 5
N_TRAIN, N_VAL, N_TEST = 8000, 2000, 5000
ALPHA = 0.1


def main() -> None:
    data = Heteroscedastic()
    g = torch.Generator().manual_seed(SEED)
    x_train, y_train = data.sample(N_TRAIN, generator=g)
    val = data.sample(N_VAL, generator=g)
    x_test, y_test = data.sample(N_TEST, generator=g)

    torch.manual_seed(SEED)
    operator = build_ncp(x_dim=1, y_dim=1, latent_dim=32)
    train_ncp(
        operator,
        x_train,
        y_train,
        epochs=600,
        lr=1e-3,
        validation_data=val,
        patience=40,
        val_every=5,
        seed=SEED,
        verbose=True,
        log_every=150,
    )
    print(f"\n{operator.parameters_summary()}")

    # --- the interval must widen where the noise does ----------------------
    x_probe = torch.linspace(-2.0, 2.0, 9).reshape(-1, 1)
    posterior = operator.condition(x_probe)
    intervals = posterior.interval(ALPHA)
    print(f"\n{int((1 - ALPHA) * 100)}% conditional intervals along the input range")
    print(f"{'x':>6}  {'mean':>15}  {'std':>15}  {'width':>15}  {'oracle width':>12}")
    lo = data.conditional_quantile(x_probe, [ALPHA / 2])
    hi = data.conditional_quantile(x_probe, [1 - ALPHA / 2])
    for i, xv in enumerate(x_probe.flatten().tolist()):
        mean_true = float(data.conditional_mean(x_probe[i : i + 1]))
        std_true = float(data.conditional_std(x_probe[i : i + 1]))
        width = float(intervals[i, 1] - intervals[i, 0])
        print(
            f"{xv:>6.2f}  {float(posterior.mean()[i]):>7.3f} ({mean_true:>5.2f})  "
            f"{float(posterior.std()[i]):>7.3f} ({std_true:>5.2f})  "
            f"{width:>15.3f}  {float(hi[i] - lo[i]):>12.3f}"
        )

    widths = (intervals[:, 1] - intervals[:, 0]).tolist()
    print(f"\nWidth at the quietest x vs the noisiest: {min(widths):.3f} -> {max(widths):.3f}")

    # --- calibration, globally and locally ---------------------------------
    test_posterior = operator.condition(x_test)
    test_intervals = test_posterior.interval(ALPHA)
    print(f"\nOverall coverage: {float(coverage(test_intervals, y_test)):.3f} (nominal {1 - ALPHA:.2f})")
    print("Coverage by region of the input range -- the real test of conditional calibration")
    print(f"{'region':>16}  {'n':>6}  {'coverage':>9}  {'mean width':>11}")
    edges = torch.linspace(-2.0, 2.0, 5)
    for left, right in zip(edges[:-1], edges[1:]):
        mask = ((x_test[:, 0] >= left) & (x_test[:, 0] < right)).nonzero().flatten()
        print(
            f"  [{float(left):>5.2f}, {float(right):>5.2f})  {mask.numel():>6}  "
            f"{float(coverage(test_intervals[mask], y_test[mask])):>9.3f}  "
            f"{float(interval_width(test_intervals[mask])[0]):>11.3f}"
        )

    _plot(data, operator, x_train, y_train)


def _plot(data, operator, x_train, y_train) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping the figure)")
        return

    grid_x = torch.linspace(-2.0, 2.0, 200).reshape(-1, 1)
    posterior = operator.condition(grid_x)
    bands = {a: posterior.interval(a) for a in (0.5, 0.1)}

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(x_train[:2000, 0], y_train[:2000, 0], s=3, alpha=0.15, color="#94a3b8", label="training data")
    for alpha, color in ((0.1, "#bfdbfe"), (0.5, "#60a5fa")):
        iv = bands[alpha]
        ax.fill_between(
            grid_x[:, 0],
            iv[:, 0],
            iv[:, 1],
            color=color,
            alpha=0.65,
            linewidth=0,
            label=f"NCP {int((1 - alpha) * 100)}% interval",
        )
    ax.plot(grid_x[:, 0], posterior.mean()[:, 0], color="#1d4ed8", lw=2, label="NCP conditional mean")
    ax.plot(grid_x[:, 0], data.conditional_mean(grid_x)[:, 0], "--", color="#111827", lw=1.6, label="true mean")
    for sign in (-1, 1):
        oracle = data.conditional_quantile(grid_x, [0.95 if sign > 0 else 0.05])[:, 0]
        ax.plot(grid_x[:, 0], oracle, ":", color="#111827", lw=1.3, label="true 90% band" if sign > 0 else None)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("NCP conditional intervals under input-dependent noise")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_ylim(-5, 5)
    fig.tight_layout()

    out = Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(exist_ok=True)
    path = out / "heteroscedastic.png"
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
