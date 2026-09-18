"""The simulation-budget question on g-and-k, a real likelihood-free model.

Example 11 answered "does simulating more help?" on a conjugate model, where
the deflated ratio is closed form and the truncation and estimation terms can
be separated exactly. That is the right place to establish the mechanism and
the wrong place to stop, because the conclusion matters for models where no
such decomposition exists.

g-and-k is that case: defined by its quantile function, no closed-form
density, and -- since both methods condition on four octile summaries rather
than the raw sample -- no tractable likelihood for what they actually see. So
the exact L^2 error of the ratio is NOT available here. What is available is
the downstream error against a large-budget ABC reference, and the reported
spectral tail, which comes free from the fit. The question becomes the
practical one:

    as the simulation budget grows, does the answer improve, and does the
    reported tail correctly say whether rank or data is the binding constraint?

READ THE RESOLUTION LIMIT FIRST. The reference is rejection ABC, so it carries
its own error. The script measures that by tightening the tolerance fourfold
and reporting how far the reference itself moves. Differences below that floor
are not evidence of anything, and on this model the floor is around 0.04 in
units of the prior standard deviation -- the same order as some of the effects
being looked for, which is exactly why example 11 was done on a conjugate
model first.

Each fit uses a generous rank D and is then truncated, because a fit AT rank d
reports a tail of identically zero and certifies nothing.

Run:  python examples/lfi/12_gandk_budget.py
"""

import time

import torch

from posterior_operator import PosteriorOperator
from posterior_operator.baselines import NeuralPosteriorEstimator
from posterior_operator.simulators import GAndK

SEED = 0
FIT_RANK = 128
LAYER_SIZE = 128
EPOCHS = 200
N_REFERENCE = 2_000_000
ABC_KEEP = 2000
N_SCORED = 16
N_GRID = (15_000, 60_000, 240_000)
TRUNCATIONS = (8, 32, 128)


def transform(summaries):
    """Variance-stabilise the two heavy-tailed summaries, as in example 08."""
    return torch.stack(
        [
            summaries[:, 0],
            summaries[:, 1].clamp_min(1e-8).log(),
            summaries[:, 2],
            summaries[:, 3].clamp_min(1e-8).log(),
        ],
        dim=-1,
    )


def build_reference_table(sim, n, seed):
    generator = torch.Generator().manual_seed(seed)
    thetas, summaries = [], []
    remaining = n
    while remaining > 0:
        block = min(200_000, remaining)
        theta, s = sim.sample_joint(block, generator=generator)
        thetas.append(theta)
        summaries.append(transform(s))
        remaining -= block
    return torch.cat(thetas), torch.cat(summaries)


def abc_posterior(table_theta, table_z, scale, z_obs, keep):
    distance = (((table_z - z_obs) / scale) ** 2).sum(dim=-1).sqrt()
    return table_theta[torch.topk(-distance, keep).indices]


def wasserstein1(atoms, masses, sample):
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


def score(posterior, references, prior_sd, n_theta):
    """Mean W_1 to the ABC reference over coordinates and observations."""
    total = 0.0
    for j in range(n_theta):
        values, cumulative = posterior._sorted_cdf(j)
        mass = torch.diff(cumulative, dim=-1, prepend=torch.zeros(cumulative.shape[0], 1))
        for i, reference in enumerate(references):
            total += wasserstein1(values, mass[i], reference[:, j]) / prior_sd[j]
    return total / (n_theta * len(references))


def main() -> None:
    sim = GAndK(n_obs=100, summaries=True, prior_high=10.0)

    print("=" * 96)
    print("g-and-k: does a larger simulation budget help, and does the tail say why?")
    print("=" * 96)
    print(f"  No closed-form ratio here, so the error is measured against a "
          f"{N_REFERENCE:,}-draw ABC")
    print(f"  reference, {N_SCORED} observations. Every fit uses rank D = {FIT_RANK} "
          f"and is truncated.\n")

    table_theta, table_z = build_reference_table(sim, N_REFERENCE, seed=SEED + 11)
    scale = table_z.std(dim=0)
    prior_sd = table_theta.std(dim=0)

    g = torch.Generator().manual_seed(SEED + 99)
    _, y_raw = sim.sample_joint(N_SCORED, generator=g)
    y_obs = transform(y_raw)
    references = [abc_posterior(table_theta, table_z, scale, y_obs[i], ABC_KEEP)
                  for i in range(N_SCORED)]
    tight = [abc_posterior(table_theta, table_z, scale, y_obs[i], ABC_KEEP // 4)
             for i in range(N_SCORED)]

    uniform = torch.full((ABC_KEEP,), 1.0 / ABC_KEEP)
    drift = 0.0
    for j in range(4):
        for i in range(N_SCORED):
            drift += wasserstein1(references[i][:, j], uniform, tight[i][:, j]) / prior_sd[j]
    drift /= 4 * N_SCORED
    print(f"  reference self-drift when the tolerance is tightened 4x: {drift:.4f}")
    print("  -> differences below this are reference error, not method error.\n")

    header = f"{'n_sim':>9}{'chi2_hat':>10}{'sigma_1':>9}{'decay':>8}"
    for d in TRUNCATIONS:
        header += f"{'rep@' + str(d):>9}{'W1@' + str(d):>9}"
    header += f"{'NPE W1':>9}"
    print(header)
    print("-" * len(header))

    for n_sim in N_GRID:
        started = time.perf_counter()
        theta_fit, raw_fit = sim.sample_joint(n_sim, generator=torch.Generator().manual_seed(SEED + 1))
        z_fit = transform(raw_fit)

        torch.manual_seed(SEED)
        operator = PosteriorOperator(theta_dim=4, data_dim=4, rank=FIT_RANK, layer_size=LAYER_SIZE)
        operator.fit(theta_fit, z_fit, epochs=EPOCHS, lr=1e-3, seed=SEED)
        spectrum = operator.singular_values

        row = (f"{n_sim:>9}{operator.chi2_divergence:>10.2f}"
               f"{operator.maximal_correlation:>9.4f}"
               f"{float(spectrum[-1] / spectrum[0]):>8.3f}")
        for d in TRUNCATIONS:
            reported = float(spectrum[d:].pow(2).sum()).__pow__(0.5)
            posterior = operator.posterior(y_obs, rank=d).with_projection("isotonic")
            row += f"{reported:>9.3f}{score(posterior, references, prior_sd, 4):>9.4f}"

        torch.manual_seed(SEED)
        npe = NeuralPosteriorEstimator(theta_dim=4, data_dim=4, n_components=10,
                                       layer_size=LAYER_SIZE)
        npe.fit(theta_fit, z_fit, epochs=EPOCHS, lr=1e-3, seed=SEED)
        draws = npe.sample(y_obs, 4000, generator=torch.Generator().manual_seed(SEED + 3))
        flat = torch.full((4000,), 1.0 / 4000)
        npe_w1 = 0.0
        for j in range(4):
            for i in range(N_SCORED):
                npe_w1 += wasserstein1(draws[i, :, j], flat, references[i][:, j]) / prior_sd[j]
        row += f"{npe_w1 / (4 * N_SCORED):>9.4f}"
        print(row + f"   [{time.perf_counter() - started:.0f}s]", flush=True)

    print("\n" + "=" * 96)
    print("Reading")
    print("=" * 96)
    print("  Read DOWN a 'W1@d' column for the effect of budget at a fixed rank, and")
    print("  ACROSS a row for the effect of rank at a fixed budget. Example 11 predicts")
    print("  the low-rank columns are flat in the budget (truncation-limited) and the")
    print("  high-rank column improves (estimation-limited); whether that survives on a")
    print("  model with no closed-form ratio is the point of running it here.")
    print()
    print("  Compare the NPE column down the same rows. If NPE keeps improving with")
    print("  budget while the operator plateaus at every rank, the constraint is")
    print("  neither truncation nor estimation but capacity -- the third knob example 11")
    print("  identified at high chi^2 and did not test.")
    print()
    print("  Nothing below the reference-drift figure printed above should be read as")
    print("  a real difference.")


if __name__ == "__main__":
    main()
