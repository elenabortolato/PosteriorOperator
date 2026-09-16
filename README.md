# Posterior Operator — Neural Conditional Probability (NCP)

A PyTorch implementation of **Neural Conditional Probability for Uncertainty
Quantification** (Kostic, Pacreau, Turri, Novelli, Lounici & Pontil, *NeurIPS
2024*) — [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/705b97ecb07ae86524d438abac97a3e2-Abstract-Conference.html)
· [arXiv:2407.01171](https://arxiv.org/abs/2407.01171).

Learn a truncated singular value decomposition of the conditional expectation
operator **once, unconditionally**; then read conditional means, variances,
densities, CDFs, quantiles and confidence regions off that decomposition in
closed form, for any conditioning value, with no retraining.

Written from the method description rather than ported from the authors'
[reference code](https://github.com/CSML-IIT-UCL/NCP), which carries no licence.

---

## The method

Let $(X, Y) \sim \pi_{XY}$ with marginals $\pi_X, \pi_Y$. The conditional
expectation operator $E : L^2(\pi_Y) \to L^2(\pi_X)$, $(Eg)(x) = \mathbb{E}[g(Y)
\mid X = x]$, has kernel $p(x,y)/(\pi_X(x)\pi_Y(y))$. Subtracting its trivial
component $\mathbb{1} \otimes \mathbb{1}$ leaves the **deflated density ratio**,
whose truncated SVD is what NCP learns:

$$
\frac{p(y \mid x)}{\pi_Y(y)} \;=\; 1 \;+\; \underbrace{\sum_{k=1}^{d} \sigma_k\, u_k(x)\, v_k(y)}_{r(x,\,y)},
\qquad \sigma_k \in [0, 1].
$$

Two networks supply $u$ and $v$; $\sigma$ is a learned vector. Once you have
$r$, every conditional quantity is a weighted statistic over a reference sample
$\{y_j\}_{j=1}^m \sim \pi_Y$:

$$
\widehat{p}(\cdot \mid x) = \sum_{j=1}^m w_j(x)\,\delta_{y_j},
\qquad w_j(x) = \tfrac1m\big(1 + r(x, y_j)\big).
$$

### Training objective

With $h(x,y) = \sum_k s_k u_k(x) v_k(y)$, the loss is the squared
$L^2(\pi_X \otimes \pi_Y)$ distance to $r$, up to an $h$-independent constant:

$$
\mathcal{L} = \mathbb{E}_{\pi_X \otimes \pi_Y}\big[h^2\big]
  - 2\Big(\mathbb{E}_{\pi_{XY}}[h] - \mathbb{E}_{\pi_X \otimes \pi_Y}[h]\Big)
  = \lVert h - r\rVert^2_{L^2(\pi_X \otimes \pi_Y)} - \lVert r \rVert^2,
$$

using $\mathbb{E}_{\pi_X \otimes \pi_Y}[h\,r] = \mathbb{E}_{\pi_{XY}}[h] -
\mathbb{E}_{\pi_X \otimes \pi_Y}[h]$. **No conditioning value and no
normalising constant appear anywhere** — that is exactly why one fit serves
every $x$. Two unbiased estimators are provided:

| `mode` | estimator | cost | notes |
| --- | --- | --- | --- |
| `"ustat"` (default) | all $n(n-1)$ off-diagonal pairs | $O(nd^2)$ | lower variance; computed Gram-free, no $n \times n$ matrix |
| `"split"` | two independent halves | $O(nd)$ | the paper's split form; cheaper, noisier |

$\mathcal{L}$ is invariant to $u \mapsto A^{-\top}u,\ v \mapsto Av$, so a
penalty pulls $\mathbb{E}[uu^\top]$ and $\mathbb{E}[vv^\top]$ towards the
identity (`"orthonormality"`, or `"log_fro"` which additionally diverges as the
embedding collapses).

### The whitening step

Gradient descent identifies the two *subspaces* but neither an orthonormal
basis for them nor the singular values. A closed-form step
(`fit_statistics`) fixes both: it computes the exact $L^2$ projection of $r$
onto the learned subspaces,

$$
\hat r(x,y) = \varphi_c(x)^\top C_\varphi^{-1} C_{\varphi\psi} C_\psi^{-1} \psi_c(y),
\qquad \varphi = \operatorname{diag}(\sqrt{s})\,u, \quad \psi = \operatorname{diag}(\sqrt{s})\,v,
$$

and re-expresses it as $\sum_k \hat\sigma_k \tilde u_k(x) \tilde v_k(y)$ with
$\tilde u, \tilde v$ centered and orthonormal and $\hat\sigma_k$ the canonical
correlations between the two feature spaces. No gradients, one pass over the
data.

---

## Install

```bash
pip install -e .          # torch + numpy
pip install -e ".[dev]"   # + pytest, matplotlib for the tests and figures
```

## Quick start

```python
import torch
from posterior_operator import Heteroscedastic, build_ncp, train_ncp

data = Heteroscedastic()
g = torch.Generator().manual_seed(0)
x, y = data.sample(8000, generator=g)

operator = build_ncp(x_dim=1, y_dim=1, latent_dim=32)
train_ncp(operator, x, y, epochs=600, validation_data=data.sample(2000, generator=g))

# One fit, then condition on anything — no retraining.
posterior = operator.condition(torch.tensor([[-1.5], [0.0], [1.5]]))

posterior.mean()                      # (3, 1) conditional means
posterior.std()                       # (3, 1) conditional standard deviations
posterior.covariance()                # (3, 1, 1) conditional covariance
posterior.quantile([0.05, 0.5, 0.95]) # (3, 3) conditional quantiles
posterior.interval(alpha=0.1)         # (3, 2) shortest 90% interval
posterior.cdf()                       # step CDF on the reference atoms
posterior.sample(1000)                # (3, 1000, 1) draws
posterior.expectation(lambda y: y**3) # any observable
operator.singular_values              # the estimated spectrum
```

Conditional **densities** additionally need an estimate of $\pi_Y$, since NCP
models the ratio $p(y\mid x)/\pi_Y(y)$:

```python
from posterior_operator import GaussianKDE

grid = torch.linspace(-5, 5, 401)
marginal = GaussianKDE(operator.reference_y)
pdf = posterior.density(grid, marginal)                        # (3, 401)
mask, level = posterior.highest_density_region(grid, marginal, alpha=0.1)
```

`highest_density_region` may be **disjoint**, which is the point on multimodal
targets: the shortest interval has to bridge the low-density valley between
modes, the HDR does not.

---

## Results

`python examples/04_benchmark_table.py --seeds 3` (n=8000, 600 epochs, 2×64 MLPs,
`latent_dim=32`). Hellinger/KS are against the **exact** conditional law; the
oracle columns use the true conditional quantiles.

| dataset | hellinger | KS | cov₉₀ | width | oracle | pinball | oracle | σ̂₁ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LinearGaussian | 0.057 | 0.025 | **0.902** | 1.669 | 1.645 | 0.125 | 0.125 | 0.895 |
| Heteroscedastic | 0.066 | 0.029 | **0.901** | 2.646 | 2.639 | 0.201 | 0.201 | 0.696 |
| BimodalMixture | 0.080 | 0.020 | **0.905** | 2.465 | 2.869 | 0.205 | 0.205 | 0.782 |
| StudentT | 0.062 | — | **0.899** | 2.337 | — | 0.186 | — | 0.650 |

Coverage lands within 0.005 of nominal on all four, and the pinball loss
matches the oracle to three decimals. The oracle width is the *equal-tailed*
true interval, so on `BimodalMixture` the shortest interval is legitimately
narrower — that column is a reference point, not a lower bound.

On `LinearGaussian` the operator spectrum is also known: for a jointly Gaussian
pair, $E$ diagonalises in the Hermite basis with $\sigma_k = \rho^k$. The fit
recovers $\hat\sigma_1 = 0.8943$ against $\rho = 0.8944$
(`examples/01_gaussian_validation.py`).

![heteroscedastic](figures/heteroscedastic.png)

Intervals widen with the local noise — conditional coverage holds region by
region (0.882–0.906 across quarters of the input range), not merely on average.

![multimodal](figures/multimodal.png)

At $x=0$ the conditional mean sits at 3.6% of the peak density. The shortest
interval spans the valley (width 2.99); the HDR splits into two pieces
(width 2.55) for the same 0.93 of true mass.

## Examples

| script | what it shows |
| --- | --- |
| `01_gaussian_validation.py` | validation against a fully closed-form model, including the Hermite spectrum |
| `02_heteroscedastic_intervals.py` | intervals adapting to input-dependent noise; coverage by region |
| `03_multimodal_regions.py` | shortest interval vs highest-density region on a bimodal target |
| `04_benchmark_table.py` | the table above |

---

## Practical notes

Two things are worth knowing before tuning, both measured rather than assumed.

**`latent_dim` is a ceiling, not a cost.** It caps the rank of the
representable dependence. Unneeded directions get singular values near zero and
can be dropped at inference with `condition(x, rank=r)`; there is no need to
tune it downward. Rank truncation stays a valid probability law, but a learned
singular direction is a *mixture* of the true ones at finite sample, so a
low-rank conditional mean is shrunk towards the marginal rather than merely
noisier.

**The whitening step is a plug-in CCA, so its singular values are biased
upward.** The bias grows with `latent_dim` and shrinks with sample size, and it
is a property of the estimator, not of reusing training data: it appears
identically with *untrained* embeddings, and whitening on a fully independent
sample does not remove it. Consequently:

- **Sample splitting is not the fix it looks like.** `train_ncp(...,
  stats_fraction=0.2)` measurably *worsens* the spectrum, the conditional mean,
  the density and the coverage on the Gaussian benchmark, because it only costs
  training data. The default is `0.0` — whiten on everything.
- **`stats_reg` is the lever.** Raising it towards `1e-3` cuts the maximum
  spectrum error from 0.054 to 0.022, at the price of over-smoothing the ratio
  — which widens intervals past nominal and costs density accuracy. The default
  `1e-6` favours calibrated conditional quantities; raise it if the reported
  spectrum is itself the output you care about.

Other defaults: standardise `X` and `Y` (`Standardizer`) — the whitening step
inverts a feature covariance, so poorly scaled inputs show up as an
ill-conditioned inverse. Prefer full-batch training, since the U-statistic uses
all $n(n-1)$ cross pairs and larger batches cut its variance. Conditional
weights are clipped at zero and renormalised by default (`clip=False` exposes
the raw signed estimate, and `cdf(monotone=True)` repairs the resulting
non-monotone CDF by isotonic regression).

## Layout

```
posterior_operator/
  nn.py          MLP embeddings; singular values parametrised as exp(-w^2) in (0,1]
  losses.py      the objective, its two unbiased estimators, and the penalties
  operator.py    NCPOperator: embeddings, whitening, conditional weights
  inference.py   ConditionalDistribution (moments, CDF, quantiles, regions), GaussianKDE
  training.py    dependency-free training loop with early stopping
  data.py        four synthetic generators with closed-form ground truth
  metrics.py     Hellinger, TV, KL, JS, KS, W1, coverage, pinball
tests/           135 tests; run with `pytest`
examples/        the four scripts above
```

Tests pin the objective against a finite joint distribution where $r$ and every
expectation are exact, the whitening algebra against a hand-written
least-squares projection, and the inference functionals against brute-force
searches.

## Citation

```bibtex
@inproceedings{kostic2024neural,
  title     = {Neural Conditional Probability for Uncertainty Quantification},
  author    = {Kostic, Vladimir R. and Pacreau, Gr{\'e}goire and Turri, Giacomo
               and Novelli, Pietro and Lounici, Karim and Pontil, Massimiliano},
  booktitle = {Advances in Neural Information Processing Systems 37 (NeurIPS)},
  year      = {2024}
}
```
