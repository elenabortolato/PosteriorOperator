"""Validate NCP against a conditional law that is known in closed form.

The linear-Gaussian model is the one case where the *operator* itself is known,
not just the conditional distribution: for a jointly Gaussian pair with
correlation rho, the conditional expectation operator diagonalises in the
Hermite basis with singular values rho^k. So this script checks the learned
singular values against rho^k as well as the usual conditional quantities.

Run:  python examples/01_gaussian_validation.py
"""

import torch

from posterior_operator import GaussianKDE, LinearGaussian, build_ncp, train_ncp
from posterior_operator.metrics import coverage, hellinger, interval_width, kolmogorov_smirnov, pinball_loss

SEED = 0
N_TRAIN, N_VAL, N_TEST = 8000, 2000, 4000
LATENT_DIM = 32


def main() -> None:
    data = LinearGaussian(x_dim=1, a=[1.0], sigma=0.5)
    g = torch.Generator().manual_seed(SEED)
    x_train, y_train = data.sample(N_TRAIN, generator=g)
    val = data.sample(N_VAL, generator=g)
    x_test, y_test = data.sample(N_TEST, generator=g)

    torch.manual_seed(SEED)
    operator = build_ncp(x_dim=1, y_dim=1, latent_dim=LATENT_DIM)
    history = train_ncp(
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
        log_every=100,
    )
    print(f"\n{operator.parameters_summary()}")
    print(f"best epoch {history['best_epoch']}, validation loss {history['best_val_loss']:+.5f}")

    # --- the operator spectrum, which only this model lets us check ---------
    print(f"\nSingular values vs the Hermite spectrum rho^k (rho = {data.correlation:.4f})")
    print(f"{'k':>3}  {'estimated':>10}  {'rho^k':>10}  {'error':>8}")
    estimated = operator.singular_values[:6]
    expected = data.true_singular_values(6)
    for k, (got, want) in enumerate(zip(estimated.tolist(), expected.tolist()), start=1):
        print(f"{k:>3}  {got:>10.4f}  {want:>10.4f}  {got - want:>+8.4f}")

    # --- conditional summaries ----------------------------------------------
    x_probe = torch.tensor([[-1.5], [-0.5], [0.0], [0.5], [1.5]])
    posterior = operator.condition(x_probe)
    print("\nConditional mean and standard deviation (truth in parentheses)")
    print(f"{'x':>6}  {'mean':>16}  {'std':>16}")
    for i, xv in enumerate(x_probe.flatten().tolist()):
        mean_true = float(data.conditional_mean(x_probe[i : i + 1]))
        std_true = float(data.conditional_std(x_probe[i : i + 1]))
        print(
            f"{xv:>6.1f}  {float(posterior.mean()[i]):>7.3f} ({mean_true:>5.2f})  "
            f"{float(posterior.std()[i]):>7.3f} ({std_true:>5.2f})"
        )

    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    got_q = posterior.quantile(levels)
    want_q = data.conditional_quantile(x_probe, levels)
    print(f"\nConditional quantiles, max absolute error: {float((got_q - want_q).abs().max()):.4f}")

    # --- distributional accuracy on a grid ----------------------------------
    grid = torch.linspace(-4.0, 4.0, 601)
    _, cdf = posterior.cdf(grid=grid)
    ks = kolmogorov_smirnov(cdf, data.conditional_cdf(x_probe, grid))
    print(f"Kolmogorov-Smirnov distance to the true CDF: max {float(ks.max()):.4f}")

    pdf_exact_marginal = posterior.density(grid, data.marginal_pdf)
    pdf_kde_marginal = posterior.density(grid, GaussianKDE(operator.reference_y))
    truth = data.conditional_pdf(x_probe, grid)
    print(
        f"Hellinger distance to the true density: "
        f"{float(hellinger(pdf_exact_marginal, truth, grid).max()):.4f} with the exact marginal, "
        f"{float(hellinger(pdf_kde_marginal, truth, grid).max()):.4f} with a KDE marginal"
    )

    # --- calibration on held-out data ---------------------------------------
    print("\nOut-of-sample calibration of the shortest conditional interval")
    print(f"{'nominal':>8}  {'coverage':>9}  {'mean width':>11}  {'oracle width':>13}")
    test_posterior = operator.condition(x_test)
    for alpha in (0.05, 0.1, 0.2):
        intervals = test_posterior.interval(alpha)
        lo = data.conditional_quantile(x_test, [alpha / 2])
        hi = data.conditional_quantile(x_test, [1 - alpha / 2])
        print(
            f"{1 - alpha:>8.2f}  {float(coverage(intervals, y_test)):>9.3f}  "
            f"{float(interval_width(intervals)[0]):>11.3f}  {float((hi - lo).mean()):>13.3f}"
        )

    q_levels = [0.1, 0.5, 0.9]
    print(
        f"\nPinball loss at {q_levels}: "
        f"{float(pinball_loss(test_posterior.quantile(q_levels), y_test, q_levels)):.4f} (NCP) vs "
        f"{float(pinball_loss(data.conditional_quantile(x_test, q_levels), y_test, q_levels)):.4f} (oracle)"
    )


if __name__ == "__main__":
    main()
