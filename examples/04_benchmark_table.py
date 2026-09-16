"""Benchmark NCP across the four synthetic regimes, with an oracle column.

Reports distributional accuracy against the exact conditional law and
calibration on held-out data. The oracle columns use the true conditional
quantiles, so they show how much of the remaining gap is irreducible.

Run:  python examples/04_benchmark_table.py [--seeds 3] [--epochs 600]
"""

import argparse
import statistics
from typing import Dict, List

import torch

from posterior_operator import (
    BimodalMixture,
    GaussianKDE,
    Heteroscedastic,
    LinearGaussian,
    StudentT,
    build_ncp,
    train_ncp,
)
from posterior_operator.metrics import coverage, hellinger, interval_width, kolmogorov_smirnov, pinball_loss

ALPHA = 0.1
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
# Conditioning values inside each generator's own support.
PROBES = {
    "LinearGaussian": [-1.5, -0.5, 0.5, 1.5],
    "Heteroscedastic": [-1.5, -0.5, 0.5, 1.5],
    "BimodalMixture": [-0.8, -0.3, 0.3, 0.8],
    "StudentT": [-1.5, -0.5, 0.5, 1.5],
}


def evaluate(generator_cls, seed: int, n_train: int, epochs: int) -> Dict[str, float]:
    data = generator_cls()
    name = generator_cls.__name__
    g = torch.Generator().manual_seed(seed)
    x_train, y_train = data.sample(n_train, generator=g)
    val = data.sample(n_train // 4, generator=g)
    x_test, y_test = data.sample(5000, generator=g)

    torch.manual_seed(seed)
    operator = build_ncp(x_dim=data.x_dim, y_dim=1, latent_dim=32)
    train_ncp(
        operator,
        x_train,
        y_train,
        epochs=epochs,
        lr=1e-3,
        validation_data=val,
        patience=40,
        val_every=5,
        seed=seed,
    )

    x_probe = torch.tensor(PROBES[name], dtype=torch.float32).reshape(-1, 1)
    grid = data.default_grid(801)
    posterior = operator.condition(x_probe)

    density = posterior.density(grid, GaussianKDE(operator.reference_y))
    result = {"hellinger": float(hellinger(density, data.conditional_pdf(x_probe, grid), grid).mean())}
    try:
        _, cdf = posterior.cdf(grid=grid)
        result["ks"] = float(kolmogorov_smirnov(cdf, data.conditional_cdf(x_probe, grid)).mean())
    except NotImplementedError:
        result["ks"] = float("nan")  # no closed-form CDF for the Student-t case

    test_posterior = operator.condition(x_test)
    intervals = test_posterior.interval(ALPHA)
    result["coverage"] = float(coverage(intervals, y_test))
    result["width"] = float(interval_width(intervals)[0])
    result["pinball"] = float(pinball_loss(test_posterior.quantile(QUANTILE_LEVELS), y_test, QUANTILE_LEVELS))

    # Oracle: the same summaries computed from the true conditional law.
    try:
        lo = data.conditional_quantile(x_test, [ALPHA / 2])
        hi = data.conditional_quantile(x_test, [1 - ALPHA / 2])
        result["oracle_width"] = float((hi - lo).mean())
        result["oracle_pinball"] = float(
            pinball_loss(data.conditional_quantile(x_test, QUANTILE_LEVELS), y_test, QUANTILE_LEVELS)
        )
    except NotImplementedError:
        result["oracle_width"] = float("nan")
        result["oracle_pinball"] = float("nan")
    result["sigma_1"] = float(operator.singular_values[0])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--n-train", type=int, default=8000)
    args = parser.parse_args()

    generators = [LinearGaussian, Heteroscedastic, BimodalMixture, StudentT]
    columns = ["hellinger", "ks", "coverage", "width", "oracle_width", "pinball", "oracle_pinball", "sigma_1"]
    header = (
        f"{'dataset':<17}{'hellinger':>10}{'KS':>8}{'cov90':>8}{'width':>8}"
        f"{'oracle':>8}{'pinball':>9}{'oracle':>8}{'sigma_1':>9}"
    )
    print(f"NCP benchmark: {args.seeds} seeds, n_train={args.n_train}, {args.epochs} epochs, alpha={ALPHA}\n")
    print(header)
    print("-" * len(header))

    for generator_cls in generators:
        runs: List[Dict[str, float]] = [
            evaluate(generator_cls, seed, args.n_train, args.epochs) for seed in range(args.seeds)
        ]
        cells = []
        for key in columns:
            values = [r[key] for r in runs]
            cells.append("n/a" if any(v != v for v in values) else f"{statistics.mean(values):.3f}")
        widths = [10, 8, 8, 8, 8, 9, 8, 9]
        print(f"{generator_cls.__name__:<17}" + "".join(f"{c:>{w}}" for c, w in zip(cells, widths)))

    print(
        "\nhellinger/KS: distance to the exact conditional law at four probe values (lower is better)."
        "\ncov90: out-of-sample coverage of the shortest 90% interval; the target is 0.900."
        "\nwidth/pinball: compare against the adjacent oracle column, not against zero."
        "\noracle width is the *equal-tailed* true interval, so on BimodalMixture the shortest"
        "\n  interval is legitimately narrower -- that column is a reference point, not a lower bound."
        "\nStudentT has no closed-form CDF or quantile, hence the missing entries."
    )


if __name__ == "__main__":
    main()
