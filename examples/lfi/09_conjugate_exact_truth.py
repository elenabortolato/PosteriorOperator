"""Where does the operator beat NPE? A conjugate model, so the truth is exact.

Every earlier comparison in this directory is scored against an *estimate* of
the posterior: quadrature on a grid (MA(2), SIR) or rejection ABC (g-and-k).
Here the posterior is available in closed form. Theta ~ Dirichlet(alpha),
Y ~ Multinomial(N, Theta), posterior exactly Dirichlet(alpha + y). No grid, no
chain, no tolerance -- posterior means, marginal densities, quantiles and exact
draws are all available to machine precision.

WHY NOT THE GAUSSIAN CONJUGATE MODEL. Because a Gaussian posterior is exactly
what a mixture density network represents, so `GaussianLinear` hands NPE a
correctly specified hypothesis class and the comparison measures nothing. On
the simplex the marginals are Beta -- skewed, bounded -- and neither method
contains the truth. That is the fair test.

THE QUESTION. Not 'is NCP better' -- the earlier scripts already answer no on
accuracy. The question is which REGIME favours it, and the two candidate axes
are separable here, which is the point of choosing this model:

  * n_categories  -> the dimension of theta. The operator answers by
    reweighting a fixed set of prior draws, so the fraction of atoms carrying
    posterior mass should fall geometrically in the dimension. If that is the
    binding constraint, NCP degrades on this axis and NPE does not.
  * n_trials      -> how informative one dataset is, hence posterior
    concentration, hence chi^2. chi^2 is what governs the rank the operator
    needs, so if THAT is the binding constraint, NCP degrades on this axis
    instead.

These knobs are NOT independent if handled naively: adding categories at a
fixed n_trials spreads the same observations over more cells, so chi^2 climbs
steeply with K (41 at K=3, 534 at K=5, and about 260000 at K=10, all at
N=100). A dimension sweep at fixed N therefore confounds exactly the two
explanations it is supposed to separate -- a first run of this script did, and
its K=3 -> K=5 degradation was uninterpretable. The dimension table below
instead tunes n_trials per K to hold chi^2 near 5, so dimension is the only
thing moving.

Run:  python examples/lfi/09_conjugate_exact_truth.py
"""

import time

import torch

from posterior_operator import PosteriorOperator
from posterior_operator.baselines import NeuralPosteriorEstimator
from posterior_operator.simulators import DirichletMultinomial

SEED = 0
N_SIM = 40000
N_EVAL = 50  # observations scored per configuration
N_EXACT = 4000  # exact posterior draws used for the W1 reference
RANK = 64
LAYER_SIZE = 128
EPOCHS = 300


def wasserstein1(atoms, masses, sample):
    """W_1 between a weighted atomic measure and an equally weighted sample."""
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


def chi_squared(sim, n_draws=200_000, seed=0):
    r"""Monte-Carlo :math:`\chi^2(\rho \| \pi \times \mu) = E_\rho[r] - 1`.

    The likelihood is tractable here, so the density ratio can be evaluated
    directly rather than read off a fitted spectrum.
    """
    g = torch.Generator().manual_seed(seed)
    theta, y = sim.sample_joint(256, generator=g)
    prior = sim.sample_prior(n_draws, generator=g)
    full = sim._full_simplex(prior)
    log_p = torch.log(full.clamp_min(1e-30))
    ratios = []
    for i in range(theta.shape[0]):
        own = sim.log_likelihood(theta[i : i + 1], y[i])[0]
        marginal = torch.logsumexp(y[i] @ log_p.T, 0) - torch.log(torch.tensor(float(n_draws)))
        ratios.append(float(torch.exp(own - marginal)))
    return float(torch.tensor(ratios).mean()) - 1.0


def evaluate(sim, seed=SEED):
    """Fit both methods on one simulation budget and score against the truth."""
    g = torch.Generator().manual_seed(seed + 99)
    theta_true, y_obs = sim.sample_joint(N_EVAL, generator=g)
    exact = sim.sample_posterior(y_obs, N_EXACT, generator=g)
    exact_mean = sim.posterior_mean(y_obs)
    prior_sd = sim.prior_sd()
    p = sim.theta_dim

    gt = torch.Generator().manual_seed(seed + 1)
    theta, y = sim.sample_joint(N_SIM, generator=gt)

    torch.manual_seed(seed)
    operator = PosteriorOperator(theta_dim=p, data_dim=sim.data_dim, rank=RANK, layer_size=LAYER_SIZE)
    operator.fit(theta, y, epochs=EPOCHS, lr=1e-3, seed=seed)
    posterior = operator.posterior(y_obs).with_projection("isotonic")

    torch.manual_seed(seed)
    npe = NeuralPosteriorEstimator(
        theta_dim=p, data_dim=sim.data_dim, n_components=10, layer_size=LAYER_SIZE
    )
    npe.fit(theta, y, epochs=EPOCHS, lr=1e-3, seed=seed)
    npe_draws = npe.sample(y_obs, N_EXACT, generator=torch.Generator().manual_seed(seed + 3))

    uniform = torch.full((N_EXACT,), 1.0 / N_EXACT)
    ncp_w1 = npe_w1 = 0.0
    ncp_cov = npe_cov = 0
    for j in range(p):
        values, cumulative = posterior._sorted_cdf(j)
        mass = torch.diff(cumulative, dim=-1, prepend=torch.zeros(cumulative.shape[0], 1))
        interval = posterior.credible_interval(alpha=0.10, coordinate=j)
        lo = torch.quantile(npe_draws[:, :, j], 0.05, dim=1)
        hi = torch.quantile(npe_draws[:, :, j], 0.95, dim=1)
        ncp_cov += int(
            ((theta_true[:, j] >= interval[:, 0]) & (theta_true[:, j] <= interval[:, 1])).sum()
        )
        npe_cov += int(((theta_true[:, j] >= lo) & (theta_true[:, j] <= hi)).sum())
        for i in range(N_EVAL):
            ncp_w1 += wasserstein1(values, mass[i], exact[i, :, j]) / prior_sd[j]
            npe_w1 += wasserstein1(npe_draws[i, :, j], uniform, exact[i, :, j]) / prior_sd[j]

    scale = p * N_EVAL
    return {
        "ncp_w1": float(ncp_w1) / scale,
        "npe_w1": float(npe_w1) / scale,
        "ncp_mean": float((posterior.mean() - exact_mean).abs().div(prior_sd).mean()),
        "npe_mean": float((npe.mean(y_obs) - exact_mean).abs().div(prior_sd).mean()),
        "ncp_cov": ncp_cov / scale,
        "npe_cov": npe_cov / scale,
        "sigma1": operator.maximal_correlation,
        "concentration": float((sim.posterior_sd(y_obs).mean(0) / prior_sd).mean()),
    }


def main() -> None:
    print("=" * 88)
    print("A conjugate model: Dirichlet-Multinomial, exact posterior in closed form")
    print("=" * 88)
    print("  Theta ~ Dirichlet(alpha), Y ~ Multinomial(N, Theta)")
    print("  posterior = Dirichlet(alpha + y), exact -- no grid, no MCMC, no ABC")
    print(f"  {N_SIM} simulations shared by both methods, rank {RANK}, "
          f"{N_EVAL} observations scored\n")
    print("  W1 is to the EXACT marginal, in units of the prior sd. Coverage is of")
    print("  nominal 90% intervals against the theta that generated the data.\n")

    # ---------------------------------------------------- dimension of theta
    print("=" * 88)
    print("Axis 1: the dimension of theta, with chi^2 HELD FIXED at about 5")
    print("=" * 88)
    print("  Holding n_trials fixed instead would confound the two axes: the same")
    print("  observations spread over more cells, so chi^2 climbs with K (41 at K=3,")
    print("  534 at K=5, ~260000 at K=10 with N=100) and any degradation could be")
    print("  blamed on either. n_trials is therefore tuned per K -- by the calibration")
    print("  in chi_squared() -- so that chi^2 is roughly constant down the column and")
    print("  the ONLY thing changing is the dimension of theta.\n")
    print(f"{'K':>4}{'dim':>5}{'N':>5}{'chi^2':>9}{'post/prior':>12}{'NCP W1':>9}{'NPE W1':>9}"
          f"{'NCP mean':>10}{'NPE mean':>10}{'NCP cov':>9}{'NPE cov':>9}{'  winner':>9}")
    print("-" * 88)
    for k, n_trials in ((3, 15), (5, 8), (10, 5), (20, 5)):
        sim = DirichletMultinomial(n_categories=k, n_trials=n_trials, concentration=2.0)
        started = time.perf_counter()
        r = evaluate(sim)
        c2 = chi_squared(sim)
        win = "NCP" if r["ncp_w1"] < r["npe_w1"] else "NPE"
        print(f"{k:>4}{sim.theta_dim:>5}{n_trials:>5}{c2:>9.2f}{r['concentration']:>12.3f}"
              f"{r['ncp_w1']:>9.4f}{r['npe_w1']:>9.4f}{r['ncp_mean']:>10.4f}{r['npe_mean']:>10.4f}"
              f"{r['ncp_cov']:>9.3f}{r['npe_cov']:>9.3f}{win:>9}"
              f"   [{time.perf_counter() - started:.0f}s]")

    # ------------------------------------------------- posterior information
    print("\n" + "=" * 88)
    print("Axis 2: information per dataset, at fixed dimension (K = 5)")
    print("  N = 2 is almost no information, so chi^2 is small and the operator's\n  truncation has little to represent -- the regime it should be best in.")
    print("=" * 88)
    print(f"{'N':>6}{'chi^2':>9}{'post/prior':>12}{'sigma_1':>9}{'NCP W1':>9}{'NPE W1':>9}"
          f"{'NCP mean':>10}{'NPE mean':>10}{'NCP cov':>9}{'NPE cov':>9}{'  winner':>9}")
    print("-" * 88)
    for n_trials in (2, 5, 15, 50, 250):
        sim = DirichletMultinomial(n_categories=5, n_trials=n_trials, concentration=2.0)
        started = time.perf_counter()
        r = evaluate(sim)
        c2 = chi_squared(sim)
        win = "NCP" if r["ncp_w1"] < r["npe_w1"] else "NPE"
        print(f"{n_trials:>6}{c2:>9.2f}{r['concentration']:>12.3f}{r['sigma1']:>9.4f}"
              f"{r['ncp_w1']:>9.4f}{r['npe_w1']:>9.4f}{r['ncp_mean']:>10.4f}{r['npe_mean']:>10.4f}"
              f"{r['ncp_cov']:>9.3f}{r['npe_cov']:>9.3f}{win:>9}"
              f"   [{time.perf_counter() - started:.0f}s]")

    print("\n" + "=" * 88)
    print("Reading")
    print("=" * 88)
    print("  Two knobs, two candidate explanations, and they move independently here.")
    print("  If NCP degrades down the FIRST table it is the atom representation: the")
    print("  posterior is a reweighting of fixed prior draws, and the share of draws")
    print("  near the posterior falls geometrically in the dimension. If it degrades")
    print("  down the SECOND it is chi^2 and the truncation rank, which is the failure")
    print("  already diagnosed on SIR. If it degrades down both, the usable regime is")
    print("  the top-left corner only: few parameters and weakly informative data.")
    print()
    print("  Watch the coverage columns independently of W1. An estimator can be wide")
    print("  and honest or narrow and overconfident, and those call for different")
    print("  remedies; W1 alone does not distinguish them.")


if __name__ == "__main__":
    main()
