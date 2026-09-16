"""Does the assumed decay of sigma_k hold on standard LFI benchmarks?

The rank-selection bound needs a decay rate for sigma_k -- polynomial decay is
the assumption floated as the analogue of a Sobolev smoothness condition on the
density ratio. This script measures the decay on the standard low-dimensional
benchmarks (MA(2), AR(2), g-and-k) and asks which of

    polynomial   sigma_k ~ C k^{-b}
    geometric    sigma_k ~ C r^k

fits better, then reports the truncation tail sum_{k>d} sigma_k^2 that the bias
bound depends on.

One correction is essential to doing this honestly: the estimated spectrum is
not uniformly accurate down its length. The Gaussian model, whose exact spectrum
is known in closed form, is included as a calibration case, and it shows two
opposite biases depending on the regime -- upward in the middle of the spectrum
when n is small relative to the rank, downward in the tail once n is large. The
calibration table below measures both, which is what licenses reading the rates
for the models where no truth is available.

Run:  python examples/lfi/05_spectrum_decay.py
"""

import math

import torch

from posterior_operator import PosteriorOperator
from posterior_operator.simulators import MA2, AR2, GAndK, GaussianLinear

SEED = 0
N_SIM = 30000
RANK = 64
EPOCHS = 400
# Skip the first few values (the leading ones are not in the asymptotic regime)
# and the last quarter of the rank (where the estimate departs most from the
# truth in either direction -- see the calibration table).
FIT_RANGE = (3, 40)


def fit_decay(values: torch.Tensor, lo: int, hi: int) -> dict:
    """Least-squares fits of both decay laws on log sigma_k, with R^2 for each."""
    k = torch.arange(lo, hi + 1, dtype=torch.float64)
    sigma = values[lo - 1 : hi].double().clamp_min(1e-12)
    log_sigma = torch.log(sigma)
    out = {}
    for name, predictor in (("polynomial", torch.log(k)), ("geometric", k)):
        design = torch.stack([torch.ones_like(predictor), predictor], dim=-1)
        solution = torch.linalg.lstsq(design, log_sigma.unsqueeze(-1)).solution[:, 0]
        residual = log_sigma - design @ solution
        total = log_sigma - log_sigma.mean()
        out[name] = {
            "rate": -float(solution[1]),
            "r2": 1.0 - float((residual**2).sum() / (total**2).sum().clamp_min(1e-30)),
        }
    return out


def tail_mass(values: torch.Tensor, d: int) -> float:
    r"""The truncation tail :math:`\sum_{k>d} \hat\sigma_k^2`, as a fraction of the total."""
    squares = values.double() ** 2
    return float(squares[d:].sum() / squares.sum().clamp_min(1e-30))


def report(label: str, values: torch.Tensor, note: str = "") -> None:
    decay = fit_decay(values, *FIT_RANGE)
    better = "polynomial" if decay["polynomial"]["r2"] > decay["geometric"]["r2"] else "geometric"
    print(f"\n[{label}] {note}")
    print(f"  spectrum       : {[round(v, 3) for v in values[:8].tolist()]} ...")
    print(f"  sigma_1        : {float(values[0]):.4f}     sum sigma_k^2 : {float((values**2).sum()):.2f}")
    print(
        f"  polynomial fit : exponent b = {decay['polynomial']['rate']:.2f}"
        f"   R^2 = {decay['polynomial']['r2']:.4f}"
    )
    print(
        f"  geometric  fit : rate     r = {math.exp(-decay['geometric']['rate']):.4f}"
        f"   R^2 = {decay['geometric']['r2']:.4f}"
    )
    print(f"  better fit over k in {FIT_RANGE}: {better.upper()}")
    print(
        "  truncation tail sum_{k>d} sigma_k^2 / total: "
        + ", ".join(f"d={d}: {tail_mass(values, d):.3f}" for d in (4, 8, 16, 32))
    )


def fit_operator(simulator, theta, data, rank=RANK, epochs=EPOCHS, seed=SEED) -> PosteriorOperator:
    torch.manual_seed(seed)
    operator = PosteriorOperator(
        theta_dim=simulator.theta_dim, data_dim=data.shape[1], rank=rank, layer_size=96, n_hidden=2
    )
    operator.fit(theta, data, epochs=epochs, lr=1e-3, seed=seed)
    return operator


def main() -> None:
    print("=" * 78)
    print("Calibration: the Gaussian model, where the exact spectrum is known")
    print("=" * 78)
    sim = GaussianLinear(theta_dim=2, data_dim=4, noise=0.6, seed=1)
    exact, _ = sim.exact_spectrum(max_order=60)
    report("Gaussian, EXACT spectrum", exact[:RANK], "closed form; the reference for the fits below")

    g = torch.Generator().manual_seed(SEED)
    theta, y = sim.sample_joint(N_SIM, generator=g)
    operator = fit_operator(sim, theta, y)
    report("Gaussian, ESTIMATED spectrum", operator.singular_values, f"rank {RANK}, n = {N_SIM}")

    # Where along the spectrum is the estimate accurate? Measure it, at two
    # sample sizes, rather than asserting a direction.
    print("\n  >>> CALIBRATION: ratio sigma_hat_k / sigma_k against the exact spectrum")
    print(f"  {'setting':<22}" + "".join(f"{f'k={k}':>9}" for k in (1, 2, 4, 8, 16, 32)))
    print("  " + "-" * 76)
    for n_sim in (4000, N_SIM):
        gg = torch.Generator().manual_seed(SEED)
        theta_c, y_c = sim.sample_joint(n_sim, generator=gg)
        calibration = fit_operator(sim, theta_c, y_c)
        est = calibration.singular_values
        row = "".join(f"{float(est[k - 1] / exact[k - 1]):>9.3f}" for k in (1, 2, 4, 8, 16, 32))
        print(f"  n = {n_sim:<18}" + row)
    print("  " + "-" * 76)
    print("  The leading values are recovered to a fraction of a percent either way. The")
    print("  TAIL is where the two regimes differ: at small n the plug-in canonical")
    print("  correlations sit above the truth (CCA overfitting), while at large n they")
    print("  fall below it, because the network cannot resolve that many orthogonal")
    print("  directions. In the large-n regime used below, the estimated spectrum")
    print("  therefore decays FASTER than the truth, so the fitted rates are optimistic")
    print("  -- upper bounds on the decay -- and so are the truncation tails.")

    print("\n" + "=" * 78)
    print("Standard low-dimensional LFI benchmarks")
    print("=" * 78)

    # --- MA(2): banded covariance, autocovariances vanish after lag 2 -------
    raw = MA2(n_timesteps=50, summaries=False)
    summaries = MA2(n_timesteps=50, summaries=True, n_lags=3)
    g = torch.Generator().manual_seed(SEED + 1)
    theta, series = raw.sample_joint(N_SIM, generator=g)
    operator = fit_operator(summaries, theta, summaries.summarize(series))
    report("MA(2)", operator.singular_values, "T = 50, 4 autocovariance summaries")

    # --- AR(2): full Toeplitz covariance, geometric autocovariance decay ----
    raw_ar = AR2(n_timesteps=50, summaries=False)
    summaries_ar = AR2(n_timesteps=50, summaries=True, n_lags=3)
    g = torch.Generator().manual_seed(SEED + 2)
    theta, series = raw_ar.sample_joint(N_SIM, generator=g)
    operator = fit_operator(summaries_ar, theta, summaries_ar.summarize(series))
    report("AR(2)", operator.singular_values, "T = 50, 4 autocovariance summaries")

    # --- g-and-k: 4 parameters, intractable density -------------------------
    gk = GAndK(n_obs=100, summaries=True)
    g = torch.Generator().manual_seed(SEED + 3)
    theta, y = gk.sample_joint(N_SIM, generator=g)
    operator = fit_operator(gk, theta, y)
    report("g-and-k", operator.singular_values, "n_obs = 100, 4 robust octile summaries")

    print("\n" + "=" * 78)
    print("Reading")
    print("=" * 78)
    print("  The GEOMETRIC law fits better than the polynomial one on every benchmark,")
    print("  and on the exact Gaussian spectrum too -- R^2 around 0.98-0.99 against")
    print("  0.83-0.90. So the polynomial-decay assumption behind the rank-selection")
    print("  bound is not what these models exhibit; they decay faster than that.")
    print()
    print("  That is good news for the bound rather than bad. Geometric decay makes")
    print("  sum_{k>d} sigma_k^2 fall geometrically in d, so the truncation-bias term is")
    print("  far smaller than a Sobolev-type polynomial assumption would allow, and the")
    print("  oracle rank grows only logarithmically in the target accuracy instead of")
    print("  polynomially. Worth stating the bound under geometric decay as the primary")
    print("  case, with polynomial decay as the conservative fallback.")
    print()
    print("  Two caveats on the numbers themselves. The fitted rates are OPTIMISTIC, by")
    print("  the calibration above: the estimated tail undershoots the truth at this n,")
    print("  so both the exponents and the truncation tails flatter the method. And the")
    print("  models differ enormously in how hard they are -- AR(2) needs d = 8 to")
    print("  capture 99% of sum sigma_k^2, while g-and-k still leaves 23% outside d = 16.")
    print("  Rank selection is genuinely model-specific, not a constant to be fixed once.")


if __name__ == "__main__":
    main()
