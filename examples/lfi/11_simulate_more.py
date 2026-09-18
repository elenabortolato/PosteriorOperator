"""If the diagnostic says the error is large, does simulating more help?

The natural next question once the spectrum reports its own error. The answer
is mostly no, and the reason is worth stating precisely, because it turns the
diagnostic into a decision rule rather than just a warning light.

The error splits in two:

  * TRUNCATION, sum_{k>d} sigma_k^2. Fixed by the rank d and the model's true
    spectrum. More simulations do not shrink it at all.
  * ESTIMATION, the error in sigma_hat and in the learned subspaces. This is
    what more simulations shrink.

So "simulate more" is the right move only when estimation is the binding
constraint. When truncation is, the move is more RANK -- and more simulations
matter only because the whitening step's plug-in bias grows like sqrt(d/n), so
a larger rank has to be paid for with a larger sample. That is the
growing-capacity schedule of example 07, arrived at from the other direction.

A PRACTICAL CATCH, and the thing this script actually tests. If you FIT at
rank d, then sum_{k>d} sigma_hat_k^2 is identically zero -- there are no
singular values past the last one you estimated -- so the diagnostic is
vacuous from a single fit at the rank you intend to use. It is informative
only if you fit at a generous rank D and then USE some d < D, reading the
discarded tail as the error estimate.

Which raises the question this script exists to answer: how do you know D was
large enough? The proposed self-check is whether the spectrum has visibly
decayed by k = D, i.e. whether sigma_hat_D / sigma_hat_1 is small. If the
spectrum is still substantial at the last index you estimated, there is mass
beyond it that the fit cannot see and its reported tail understates. That
check uses only the fit, so a user can run it. Whether it actually detects
under-capture is measurable here, because the true chi^2 is closed-form and
the captured fraction can be computed.

Run:  python examples/lfi/11_simulate_more.py
"""

import time

import torch

from posterior_operator import PosteriorOperator
from posterior_operator.simulators import DirichletMultinomial

SEED = 0
FIT_RANK = 128
LAYER_SIZE = 128
EPOCHS = 250
N_RATIO = 1500
N_GRID = (5_000, 20_000, 80_000)
TRUNCATIONS = (8, 32, 128)
CONFIGS = (("chi2 ~ 5", 5, 8), ("chi2 ~ 113", 5, 50))


def log_marginal(sim, y):
    alpha, total = sim.alpha, sim.alpha.sum()
    return (
        torch.lgamma(alpha + y).sum(-1)
        - torch.lgamma(total + y.sum(-1))
        - torch.lgamma(alpha).sum()
        + torch.lgamma(total)
    )


def exact_deflated_ratio(sim, theta, y):
    full = sim._full_simplex(theta)
    return torch.exp(y @ torch.log(full.clamp_min(1e-300)).T
                     - log_marginal(sim, y).unsqueeze(-1)) - 1.0


def one_fit(sim, n_sim, seed):
    g = torch.Generator().manual_seed(seed + 99)
    theta_fit, y_fit = sim.sample_joint(n_sim, generator=torch.Generator().manual_seed(seed + 1))
    torch.manual_seed(seed)
    operator = PosteriorOperator(
        theta_dim=sim.theta_dim, data_dim=sim.data_dim, rank=FIT_RANK, layer_size=LAYER_SIZE
    )
    operator.fit(theta_fit, y_fit, epochs=EPOCHS, lr=1e-3, seed=seed)

    theta_ratio = sim.sample_prior(N_RATIO, generator=g)
    _, y_ratio = sim.sample_joint(N_RATIO, generator=g)
    truth = exact_deflated_ratio(sim, theta_ratio, y_ratio)
    true_chi2 = float((truth**2).mean())

    data_scaled = operator._prepare_data(y_ratio)
    theta_scaled = (
        operator._theta_scaler.transform(theta_ratio)
        if operator._theta_scaler is not None
        else theta_ratio
    )
    spectrum = operator.singular_values
    errors = {}
    for d in TRUNCATIONS:
        estimate = operator.operator.deflated_ratio(data_scaled, theta_scaled, rank=d)
        errors[d] = (
            float(spectrum[d:].pow(2).sum()).__pow__(0.5),
            float(((estimate - truth) ** 2).mean()).__pow__(0.5),
        )
    return {
        "captured": float(spectrum.pow(2).sum()) / true_chi2,
        "sigma1": float(spectrum[0]),
        "sigma_last": float(spectrum[-1]),
        "decay": float(spectrum[-1] / spectrum[0]),
        "errors": errors,
        "true_chi2": true_chi2,
    }


def main() -> None:
    print("=" * 92)
    print("Does simulating more reduce the error the spectrum reports?")
    print("=" * 92)
    print(f"  Every fit uses rank D = {FIT_RANK}; the sample size is what varies.")
    print("  'captured' is sum sigma_hat^2 / true chi^2, computable only because the")
    print("  ratio is closed form here. 'decay' is sigma_hat_D / sigma_hat_1 -- the")
    print("  OBSERVABLE proxy for whether the fitted rank was generous enough.\n")

    for label, k, n_trials in CONFIGS:
        sim = DirichletMultinomial(n_categories=k, n_trials=n_trials, concentration=2.0)
        print(f"[{label}]  K={k}, N={n_trials}, dim={sim.theta_dim}")
        header = f"{'n_sim':>9}{'captured':>11}{'sigma_1':>10}{'sigma_D':>10}{'decay':>9}"
        for d in TRUNCATIONS:
            header += f"{'rep@' + str(d):>10}{'true@' + str(d):>10}"
        print(header)
        print("-" * len(header))
        for n_sim in N_GRID:
            started = time.perf_counter()
            r = one_fit(sim, n_sim, SEED)
            e = r["errors"]
            row = (f"{n_sim:>9}{r['captured']:>10.1%}{r['sigma1']:>10.4f}"
                   f"{r['sigma_last']:>10.4f}{r['decay']:>9.3f}")
            for d in TRUNCATIONS:
                row += f"{e[d][0]:>10.3f}{e[d][1]:>10.3f}"
            print(row + f"   [{time.perf_counter() - started:.0f}s]")
        print(f"  true chi^2 = {r['true_chi2']:.2f}\n")

    print("=" * 92)
    print("Reading")
    print("=" * 92)
    print("  Read DOWN a 'true@d' column: that is the effect of more simulations at a")
    print("  FIXED rank. If it flattens, the truncation floor has been reached and more")
    print("  data buys nothing -- the remedy is rank, not simulation. Read ACROSS a row:")
    print("  that is the effect of more rank at a fixed budget.")
    print()
    print("  The 'decay' column is the part a user can actually act on, since captured")
    print("  needs the truth. If sigma_hat_D is still an appreciable fraction of")
    print("  sigma_hat_1, the spectrum has not died within the rank that was fitted,")
    print("  so there is mass beyond D that the fit cannot see and its reported tail")
    print("  understates the error. Then D is too small -- and since the whitening's")
    print("  plug-in bias grows like sqrt(d/n), raising D is what forces you to")
    print("  simulate more. That is the honest version of 'simulate more': not a way")
    print("  to reduce truncation error, but the price of the rank that does.")


if __name__ == "__main__":
    main()
