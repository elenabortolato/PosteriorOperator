r"""Likelihood-free posterior-functional inference: train once, ask whatever next.

This is the simulation-based specialisation of the NCP operator. The joint law
is the prior predictive :math:`\rho(d\theta, dy) = \pi(d\theta)\,p(dy \mid
\theta)`, the conditioning variable is the *data* and the response is the
*parameter*, so the conditional expectation operator

.. math:: E_{\Theta \mid Y} : L^2_\pi(\Theta) \to L^2_\mu(Y),
          \qquad [E_{\Theta\mid Y} f](y) = \mathbb{E}[f(\Theta) \mid Y = y]

maps a functional of the parameter to its posterior expectation as a function
of the data. The target is the **posterior functional** :math:`T_f(y_0) =
\mathbb{E}[f(\Theta) \mid Y = y_0]`, and the estimator is

.. math::
    \widehat{T}_f(y_0) = \frac1n \sum_{i=1}^n f(\theta_i)
      + \sum_{k=1}^d \hat\sigma_k\, \hat u_k(y_0)
        \Big[\frac1n \sum_{i=1}^n \hat v_k(\theta_i) f(\theta_i)\Big],

which is exactly a weighted average over the prior draws with masses
:math:`w_i(y_0) = \frac1n\big(1 + \sum_k \hat\sigma_k \hat u_k(y_0) \hat
v_k(\theta_i)\big)` -- so every posterior functional, CDF, quantile, credible
region and Bayes action is a statistic of one reweighted prior sample, obtained
without ever evaluating :math:`p(y \mid \theta)`.

Two quantities of the fit are diagnostics in their own right:

* :attr:`PosteriorOperator.maximal_correlation` is :math:`\hat\sigma_1`, the
  Hirschfeld-Gebelein-Renyi maximal correlation between :math:`\Theta` and
  :math:`Y` -- a model-free measure of how strongly the best one-dimensional
  feature of the parameter can be identified from the data.
* :attr:`PosteriorOperator.chi2_divergence` is :math:`\sum_k \hat\sigma_k^2 =
  \chi^2(\rho \,\Vert\, \pi \times \mu)`, the squared Hilbert-Schmidt norm of
  the deflated operator, which is finite exactly when the operator is compact.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .data import Standardizer
from .inference import ConditionalDistribution, isotonic_regression
from .losses import NCPLoss
from .nn import MLP
from .operator import NCPOperator
from .training import train_ncp

__all__ = ["PosteriorOperator", "PosteriorSample"]


class _ScaledOperator:
    """Adapter presenting the operator in original parameter units.

    :meth:`ConditionalDistribution.density` evaluates the deflated ratio on a
    user-supplied grid; the grid arrives in the parameter's natural units while
    the network was trained on standardised inputs, so it is rescaled here.
    """

    def __init__(self, operator: NCPOperator, theta_scaler: Optional[Standardizer]):
        self._operator = operator
        self._theta_scaler = theta_scaler

    def deflated_ratio(self, data_scaled: Tensor, theta: Tensor, rank: Optional[int] = None) -> Tensor:
        if self._theta_scaler is not None:
            theta = self._theta_scaler.transform(theta)
        return self._operator.deflated_ratio(data_scaled, theta, rank=rank)


class PosteriorSample(ConditionalDistribution):
    r"""A posterior, represented as reweighted prior draws.

    The atoms are the simulated parameters :math:`\theta_i` and the masses are
    :math:`w_i(y_0) = \frac1n(1 + \hat r(y_0, \theta_i))`, so this inherits the
    whole functional toolkit of
    :class:`~posterior_operator.inference.ConditionalDistribution` and adds the
    posterior vocabulary below.

    **Signed versus clipped masses.** Nothing constrains a density-*ratio*
    estimate to be positive, so some :math:`w_i(y_0)` can be negative. The two
    families of query handle that differently, because they need different
    things:

    * **Moment-type functionals** -- :meth:`functional`,
      :meth:`~posterior_operator.inference.ConditionalDistribution.mean`,
      ``covariance``, ``variance``, :meth:`probability`,
      :meth:`posterior_risk` -- use the masses as they come. That makes them
      the estimator of Eq. (2) verbatim, which is what its theory covers.
      Clipping measurably *inflates* second moments: it discards the negative
      mass whose job is to carve probability away from the prior's tails, and
      on an MA(2) experiment where the posterior is five times tighter than the
      prior it widens the reported standard deviation by a factor of 3.6
      instead of 1.6.
    * **Order-statistic queries** -- :meth:`credible_interval`, ``quantile``,
      ``cdf``, ``sample``, :meth:`marginal_histogram`,
      :meth:`effective_sample_size` -- need a genuine probability measure (a
      monotone CDF, non-negative sampling weights), so they clip at zero and
      renormalise internally. :meth:`as_probability` exposes that measure.
    """

    @property
    def theta(self) -> Tensor:
        """The prior draws serving as atoms, shape ``(n_draws, theta_dim)``."""
        return self.atoms

    def as_probability(self) -> "PosteriorSample":
        """The same posterior with masses clipped at zero and renormalised.

        A no-op when every mass is already non-negative.
        """
        if not bool((self.weights < 0).any()):
            return self
        clipped = self.weights.clamp_min(0.0)
        total = clipped.sum(dim=-1, keepdim=True)
        degenerate = total.abs() < torch.finfo(clipped.dtype).eps
        if bool(degenerate.any()):
            uniform = torch.full_like(clipped, 1.0 / clipped.shape[1])
            clipped = torch.where(degenerate, uniform, clipped)
            total = torch.where(degenerate, torch.ones_like(total), total)
        return PosteriorSample(
            weights=clipped / total,
            atoms=self.atoms,
            operator=self._operator,
            x=self._x,
            rank=self._rank,
        )

    def with_projection(self, projection: str) -> "PosteriorSample":
        r"""The same posterior with a different signed-to-probability projection.

        ``projection`` selects how order-statistic queries turn the signed
        masses into something a quantile can be read off:

        ``"clip"``
            Clamp the negative masses at zero and renormalise. The default,
            and what :meth:`as_probability` returns.
        ``"isotonic"``
            Accumulate the masses *as they come*, signs included, into a signed
            CDF, and take its least-squares projection onto non-decreasing
            functions. Nothing is discarded.

        **Why the choice matters.** The negative masses are not numerical
        noise; they are what carves probability out of the prior's tails, and
        there are a lot of them -- 45% on the g-and-k experiment in
        ``examples/lfi/08_gandk_npe.py``. Clipping throws that information
        away, so every quantile and interval reverts towards the prior even
        though the moment functionals, which keep the signs, stay accurate.
        Measured on that experiment against a two-million-draw ABC reference,
        averaged over the four parameters in units of the prior standard
        deviation:

        =========================  =========  ==========  =====
        ..                         clipped    isotonic    NPE
        =========================  =========  ==========  =====
        :math:`W_1` to the ABC     0.318      0.092       0.095
        90% interval coverage      0.98       0.92        0.92
        =========================  =========  ==========  =====

        So the projection accounts for essentially the whole of the operator's
        apparent deficit against NPE on distribution shape. It is not yet the
        default only because it changes the behaviour of every order-statistic
        query; prefer it unless you need to reproduce the old numbers.

        Note that the projection is defined per scalar observable -- it depends
        on the order the atoms are visited in, and a multivariate posterior has
        no canonical order. :meth:`sample`, which needs joint draws, therefore
        keeps using the clipped measure whatever this is set to.
        """
        if projection not in ("clip", "isotonic"):
            raise ValueError(f"projection must be 'clip' or 'isotonic', got {projection!r}")
        out = PosteriorSample(
            weights=self.weights,
            atoms=self.atoms,
            operator=self._operator,
            x=self._x,
            rank=self._rank,
        )
        out._projection = projection
        return out

    # Order-statistic queries route through a genuine probability measure.
    # Overriding the primitive the base class builds them from covers cdf,
    # quantile, median, interval and credible_interval in one place.
    def _sorted_cdf(self, observable):  # type: ignore[override]
        if getattr(self, "_projection", "clip") != "isotonic":
            return ConditionalDistribution._sorted_cdf(self.as_probability(), observable)
        # Sort by the observable first: the CDF, and hence its monotone
        # projection, is only defined once the atoms are in that order.
        values = self._values(observable)
        order = torch.argsort(values)
        cumulative = isotonic_regression(self.weights[:, order].cumsum(dim=-1)).clamp_min(0.0)
        total = cumulative[:, -1:]
        degenerate = total.abs() < torch.finfo(cumulative.dtype).eps
        if bool(degenerate.any()):
            ramp = torch.linspace(
                0.0, 1.0, cumulative.shape[1], dtype=cumulative.dtype
            ).expand_as(cumulative)
            cumulative = torch.where(degenerate, ramp, cumulative)
            total = torch.where(degenerate, torch.ones_like(total), total)
        return values[order], (cumulative / total).clamp(0.0, 1.0)

    def sample(self, n_samples: int, generator: Optional[torch.Generator] = None) -> Tensor:
        """Draw parameter values from the posterior, via the clipped measure."""
        return ConditionalDistribution.sample(self.as_probability(), n_samples, generator=generator)

    def functional(self, f: Callable[[Tensor], Tensor]) -> Tensor:
        r"""The posterior functional :math:`\widehat{T}_f(y_0)` for any ``f``.

        ``f`` maps the parameter draws ``(n_draws, theta_dim)`` to ``(n_draws,)``
        or ``(n_draws, k)``. This is the estimator of the paper's Eq. (2),
        written as a weighted average; nothing about it depends on ``f`` having
        been known at training time.
        """
        return self.expectation(f)

    def probability(self, region: Callable[[Tensor], Tensor]) -> Tensor:
        r""":math:`\mathbb{P}(\Theta \in B \mid Y = y_0)` for an indicator ``region``.

        ``region`` maps ``(n_draws, theta_dim)`` to a boolean or 0/1 tensor of
        shape ``(n_draws,)``; this is the functional :math:`f = \mathbb{1}_B`.
        """
        indicator = region(self.atoms)
        if indicator.shape[0] != self.n_atoms:
            raise ValueError(f"region returned {indicator.shape[0]} rows for {self.n_atoms} draws")
        return self.weights @ indicator.reshape(self.n_atoms).to(self.weights.dtype)

    def credible_interval(self, alpha: float = 0.05, coordinate: int = 0) -> Tensor:
        r"""Shortest :math:`1 - \alpha` credible interval for one coordinate."""
        return self.interval(alpha=alpha, observable=coordinate)

    def posterior_risk(
        self,
        loss: Callable[[Tensor, Tensor], Tensor],
        actions: Tensor,
    ) -> Tensor:
        r"""Posterior expected loss :math:`\mathbb{E}[L(a, \Theta) \mid Y = y_0]`.

        Args:
            loss: callable mapping ``(actions, draws)`` to a ``(n_actions,
                n_draws)`` loss matrix.
            actions: candidate actions, shape ``(n_actions, ...)``.

        Returns shape ``(n_x, n_actions)``.
        """
        matrix = loss(actions, self.atoms)
        if matrix.shape != (actions.shape[0], self.n_atoms):
            raise ValueError(
                f"loss must return ({actions.shape[0]}, {self.n_atoms}), got {tuple(matrix.shape)}"
            )
        return self.weights @ matrix.T.to(self.weights.dtype)

    def bayes_action(self, loss: Callable[[Tensor, Tensor], Tensor], actions: Tensor) -> Tensor:
        r"""The risk-minimising action :math:`\arg\min_a \mathbb{E}[L(a, \Theta) \mid Y=y_0]`.

        Returns the chosen action per conditioning value, shape
        ``(n_x, ...)`` following the trailing shape of ``actions``.
        """
        return actions[self.posterior_risk(loss, actions).argmin(dim=-1)]

    def marginal_histogram(
        self, coordinate: int = 0, bins: Union[int, Tensor] = 20
    ) -> Tuple[Tensor, Tensor]:
        r"""Marginal posterior histogram of one coordinate.

        The functionals :math:`f_{j,k} = \mathbb{1}\{\theta_j \in B_{j,k}\}` over
        a partition of the coordinate's range, evaluated all at once.

        Returns ``(edges, probabilities)`` of shapes ``(n_bins + 1,)`` and
        ``(n_x, n_bins)``; the probabilities sum to one along the last axis.
        """
        probability = self.as_probability()
        values = self.atoms[:, coordinate]
        if isinstance(bins, int):
            if bins < 1:
                raise ValueError(f"bins must be positive, got {bins}")
            edges = torch.linspace(float(values.min()), float(values.max()), bins + 1, dtype=values.dtype)
        else:
            edges = torch.as_tensor(bins, dtype=values.dtype).reshape(-1)
            if edges.numel() < 2:
                raise ValueError("explicit bin edges need at least 2 entries")
        # Right-closed bins, with the leftmost edge inclusive.
        idx = (torch.bucketize(values.contiguous(), edges, right=True) - 1).clamp(0, edges.numel() - 2)
        onehot = torch.zeros(self.n_atoms, edges.numel() - 1, dtype=self.weights.dtype)
        onehot[torch.arange(self.n_atoms), idx] = 1.0
        return edges, probability.weights @ onehot

    def reweight(self, log_weights: Tensor) -> "PosteriorSample":
        r"""Retarget to a different prior by self-normalised importance weights.

        If the simulations came from a proposal :math:`q \neq \pi`, the
        functional under :math:`\pi` is recovered as

        .. math:: \mathbb{E}_\pi[f(\Theta) \mid Y = y]
                  = \frac{\mathbb{E}_q[f(\Theta) w(\Theta) \mid Y = y]}
                         {\mathbb{E}_q[w(\Theta) \mid Y = y]},
                  \qquad w(\theta) = \frac{\pi(\theta)}{q(\theta)},

        which on the discrete representation is just multiplying each atom's
        mass by :math:`w(\theta_i)` and renormalising. The operator is *not*
        refitted, so this makes one training run valid for any prior
        :math:`\pi \ll q` chosen afterwards -- at the cost of the usual
        importance-sampling variance, which
        :meth:`effective_sample_size` reports.

        A caveat that the population identity hides: the reweighting multiplies
        the *estimated* proposal posterior, so the estimator's error interacts
        with the weights. If the low-rank posterior is over-dispersed, the
        product leans on the new prior more heavily than the truth does, and
        retargeting to a concentrated prior amplifies that error rather than
        cancelling it. Read :meth:`effective_sample_size` alongside the answer.

        Args:
            log_weights: ``(n_draws,)`` values of :math:`\log w(\theta_i)`, up
                to an additive constant. ``-inf`` is allowed and removes a draw,
                which is how a hard support constraint is expressed.
        """
        log_w = torch.as_tensor(log_weights, dtype=self.weights.dtype).reshape(-1)
        if log_w.numel() != self.n_atoms:
            raise ValueError(f"expected {self.n_atoms} log weights, got {log_w.numel()}")
        finite = log_w[torch.isfinite(log_w)]
        if finite.numel() == 0:
            raise ValueError("every log weight is -inf, so the target prior has no support here")
        # Shift before exponentiating; the self-normalisation cancels the shift.
        scaled = torch.exp(log_w - finite.max())

        source = self
        combined = source.weights * scaled.unsqueeze(0)
        total = combined.sum(dim=-1, keepdim=True)
        if bool((total <= torch.finfo(total.dtype).eps).any()):
            # Signed masses can cancel to nothing once the weights restrict the
            # support (a truncating prior is the usual culprit). The clipped
            # measure always survives, so fall back to it -- for every row, so
            # the result stays homogeneous -- and say so.
            warnings.warn(
                "signed masses cancelled to zero for at least one observation after "
                "reweighting, which happens when the target prior truncates the "
                "proposal's support; falling back to the clipped probability measure. "
                "Moment estimates from this object are no longer the signed Eq. (2) "
                "estimator. Call .as_probability() before .reweight() to opt in "
                "explicitly and silence this.",
                UserWarning,
                stacklevel=2,
            )
            source = self.as_probability()
            combined = source.weights * scaled.unsqueeze(0)
            total = combined.sum(dim=-1, keepdim=True)
            if bool((total <= 0).any()):
                raise ValueError(
                    "the target prior has no posterior mass for at least one observation: it "
                    "is not supported where the proposal placed its draws"
                )
        return PosteriorSample(
            weights=combined / total,
            atoms=self.atoms,
            operator=self._operator,
            x=self._x,
            rank=self._rank,
        )

    def bootstrap_functional(
        self,
        f: Callable[[Tensor], Tensor],
        n_resamples: int = 200,
        alpha: float = 0.05,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Percentile confidence interval for :math:`\widehat{T}_f(y_0)`, by resampling draws.

        Resamples the parameter draws with replacement, keeping the fitted
        networks fixed, and recomputes the functional. This is a *confidence*
        interval for the estimate, not a credible interval for
        :math:`\Theta` -- :meth:`credible_interval` is the latter.

        What it captures and what it misses matters. It quantifies the Monte
        Carlo error of the final average over draws, which is the
        parametric-rate part of the estimator. It does **not** capture the
        error in :math:`(\hat u, \hat v, \hat\sigma)`, because the networks are
        held fixed. Since :math:`T_f(y_0)` *evaluates* the learned
        :math:`\hat u` at one fixed :math:`y_0` rather than averaging it over
        the population, that second component does not vanish at
        :math:`n^{-1/2}` in general, and no amount of resampling the draws
        recovers it. Expect this interval to under-cover; ``examples/lfi/
        07_functional_confidence_intervals.py`` measures by how much.

        Args:
            f: the functional, as in :meth:`functional`.
            n_resamples: bootstrap replicates.
            alpha: two-sided level, so the interval is the ``alpha/2`` and
                ``1 - alpha/2`` percentiles of the replicates.
            generator: RNG for the resampling.

        Returns:
            ``(estimate, interval)`` of shapes ``(n_x, k)`` and ``(n_x, k, 2)``.
        """
        if n_resamples < 2:
            raise ValueError(f"n_resamples must be at least 2, got {n_resamples}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        values = f(self.atoms)
        values = values.reshape(self.n_atoms, -1).to(self.weights.dtype)
        estimate = self.weights @ values

        replicates = torch.empty(n_resamples, *estimate.shape, dtype=estimate.dtype)
        # Resamples are processed in blocks rather than one at a time: the loop
        # body is tiny, so Python overhead otherwise dominates the cost.
        n_x = self.weights.shape[0]
        block = max(1, min(n_resamples, 4_000_000 // max(n_x * self.n_atoms, 1)))
        for start in range(0, n_resamples, block):
            size = min(block, n_resamples - start)
            idx = torch.randint(self.n_atoms, (size, self.n_atoms), generator=generator)
            # Renormalising keeps each replicate a weighting of the same total
            # mass, so the spread reflects the draws and not the resample size.
            resampled = self.weights[:, idx]  # (n_x, size, n_atoms)
            resampled = resampled / resampled.sum(dim=-1, keepdim=True)
            replicates[start : start + size] = torch.einsum("xbm,bmk->bxk", resampled, values[idx])

        levels = torch.tensor([alpha / 2, 1 - alpha / 2], dtype=estimate.dtype)
        bounds = torch.quantile(replicates, levels, dim=0)  # (2, n_x, k)
        return estimate, bounds.permute(1, 2, 0)

    def effective_sample_size(self) -> Tensor:
        r"""Kish effective sample size :math:`(\sum_i w_i)^2 / \sum_i w_i^2`.

        Ranges from 1 (all mass on one draw) to ``n_draws`` (uniform). Read it
        as how many prior draws are actually supporting the answer: a posterior
        much tighter than the prior, or a :meth:`reweight` to a distant prior,
        both show up as a small value. Computed on the clipped measure, since
        signed masses have no such interpretation. Returns shape ``(n_x,)``.
        """
        w = self.as_probability().weights
        return 1.0 / (w**2).sum(dim=-1).clamp_min(torch.finfo(w.dtype).tiny)

    def __repr__(self) -> str:
        return f"PosteriorSample(n_obs={len(self)}, n_draws={self.n_atoms}, theta_dim={self.y_dim})"


class PosteriorOperator:
    r"""Amortised posterior-functional inference for a simulator.

    Wraps an :class:`~posterior_operator.operator.NCPOperator` in the
    likelihood-free orientation -- conditioning on data, distributed over
    parameters -- and standardises both sides internally, which matters because
    the whitening step inverts a feature covariance.

    Args:
        theta_dim: dimension of the parameter.
        data_dim: dimension of the data or its summaries.
        rank: number of singular directions :math:`d` to learn. This is the
            truncation rank of the operator, and by Corollary 2 of the paper
            the only place information is lost relative to full posterior
            recovery -- so prefer generous values and truncate at query time
            via ``rank=`` on :meth:`posterior`.
        n_hidden, layer_size, dropout, activation: architecture of both branches.

    Example:
        >>> from posterior_operator.simulators import MA2
        >>> sim = MA2()
        >>> theta, y = sim.sample_joint(20000)
        >>> op = PosteriorOperator(theta_dim=2, data_dim=sim.data_dim, rank=32)
        >>> op.fit(theta, y, epochs=400)                       # once
        >>> post = op.posterior(y[:1])                         # then ask anything
        >>> post.mean(), post.credible_interval(0.05), post.probability(lambda t: t[:, 0] > 0)
    """

    def __init__(
        self,
        theta_dim: int,
        data_dim: int,
        rank: int = 32,
        n_hidden: int = 2,
        layer_size: int = 64,
        dropout: float = 0.0,
        activation: Optional[Callable[[], nn.Module]] = None,
        standardize: bool = True,
    ):
        self.theta_dim = int(theta_dim)
        self.data_dim = int(data_dim)
        self.rank = int(rank)
        act = nn.GELU if activation is None else activation
        common = dict(output_dim=rank, n_hidden=n_hidden, layer_size=layer_size, dropout=dropout, activation=act)
        # In this orientation the operator conditions on the data and is
        # distributed over the parameter, so the "x" branch embeds y.
        self.operator = NCPOperator(
            x_embedding=MLP(input_dim=self.data_dim, **common),
            y_embedding=MLP(input_dim=self.theta_dim, **common),
            latent_dim=rank,
        )
        self.standardize = bool(standardize)
        self._theta_scaler: Optional[Standardizer] = None
        self._data_scaler: Optional[Standardizer] = None
        self._theta_draws: Optional[Tensor] = None

    # ----------------------------------------------------------------- fitting

    def fit(
        self,
        theta: Tensor,
        data: Tensor,
        loss: Optional[NCPLoss] = None,
        validation_split: float = 0.2,
        **train_kwargs: Any,
    ) -> Dict[str, Any]:
        r"""Fit the operator on prior-predictive draws :math:`(\theta_i, y_i) \sim \rho`.

        Args:
            theta: ``(n, theta_dim)`` parameter draws from the prior (or proposal).
            data: ``(n, data_dim)`` corresponding simulator outputs.
            loss: training objective; defaults to ``NCPLoss()``.
            validation_split: fraction held out for early stopping. Set to
                ``0`` to train for the full number of epochs on everything.
            **train_kwargs: forwarded to
                :func:`~posterior_operator.training.train_ncp` (``epochs``,
                ``lr``, ``batch_size``, ``patience``, ``stats_reg``, ...).
        """
        theta = torch.as_tensor(theta, dtype=torch.float32).reshape(-1, self.theta_dim)
        data = torch.as_tensor(data, dtype=torch.float32).reshape(theta.shape[0], -1)
        if data.shape[1] != self.data_dim:
            raise ValueError(f"expected data_dim={self.data_dim}, got {data.shape[1]}")

        if self.standardize:
            self._theta_scaler = Standardizer().fit(theta)
            self._data_scaler = Standardizer().fit(data)
            theta_s = self._theta_scaler.transform(theta)
            data_s = self._data_scaler.transform(data)
        else:
            theta_s, data_s = theta, data

        validation = None
        if validation_split > 0:
            if not 0 < validation_split < 1:
                raise ValueError(f"validation_split must lie in [0, 1), got {validation_split}")
            n_val = max(2, int(round(theta.shape[0] * validation_split)))
            if n_val >= theta.shape[0] - 1:
                raise ValueError(f"validation_split={validation_split} leaves too little training data")
            perm = torch.randperm(theta.shape[0])
            val_idx, fit_idx = perm[:n_val], perm[n_val:]
            validation = (data_s[val_idx], theta_s[val_idx])
            data_s, theta_s, theta = data_s[fit_idx], theta_s[fit_idx], theta[fit_idx]

        train_kwargs.setdefault("patience", 40)
        train_kwargs.setdefault("val_every", 5)
        history = train_ncp(
            self.operator,
            data_s,  # conditioning variable
            theta_s,  # response
            loss=loss,
            validation_data=validation,
            **train_kwargs,
        )
        # The reference atoms live in standardised units inside the operator;
        # keep the originals so every posterior is reported in natural units.
        self._theta_draws = (
            self._theta_scaler.inverse_transform(self.operator.reference_y)
            if self._theta_scaler is not None
            else self.operator.reference_y.clone()
        )
        return history

    # --------------------------------------------------------------- inference

    def _prepare_data(self, y_obs: Tensor) -> Tensor:
        y = torch.as_tensor(y_obs, dtype=torch.float32).reshape(-1, self.data_dim)
        return self._data_scaler.transform(y) if self._data_scaler is not None else y

    def posterior(
        self,
        y_obs: Tensor,
        rank: Optional[int] = None,
        clip: bool = False,
        theta_draws: Optional[Tensor] = None,
    ) -> PosteriorSample:
        r"""The posterior :math:`\pi(\cdot \mid y_0)` for one or many observations.

        Args:
            y_obs: ``(n_obs, data_dim)`` observations, or a single flat vector.
            rank: truncate to the top ``rank`` singular directions.
            clip: clamp negative masses to zero and renormalise up front.
                Off by default, so moment-type functionals are the estimator of
                Eq. (2) verbatim; order-statistic queries clip internally
                anyway, and clipping early inflates second moments. See
                :class:`PosteriorSample`.
            theta_draws: alternative parameter draws to reweight, instead of the
                ones stored at fit time.

        Returns a :class:`PosteriorSample` -- one row per observation.
        """
        if self._theta_draws is None:
            raise RuntimeError("call fit() before posterior()")
        data_s = self._prepare_data(y_obs)
        if theta_draws is None:
            atoms = self._theta_draws
            atoms_s = self.operator.reference_y
        else:
            atoms = torch.as_tensor(theta_draws, dtype=torch.float32).reshape(-1, self.theta_dim)
            atoms_s = self._theta_scaler.transform(atoms) if self._theta_scaler is not None else atoms
        weights, _ = self.operator.conditional_weights(data_s, y_reference=atoms_s, rank=rank, clip=clip)
        return PosteriorSample(
            weights=weights,
            atoms=atoms,
            operator=_ScaledOperator(self.operator, self._theta_scaler),
            x=data_s,
            rank=rank,
        )

    # ------------------------------------------------------------- diagnostics

    @property
    def singular_values(self) -> Tensor:
        r"""Estimated spectrum :math:`\hat\sigma_1 \geq \dots \geq \hat\sigma_d`."""
        return self.operator.singular_values

    @property
    def maximal_correlation(self) -> float:
        r""":math:`\hat\sigma_1`, the Hirschfeld-Gebelein-Renyi maximal correlation.

        The supremum of :math:`\operatorname{Corr}(f(\Theta), g(Y))` over all
        square-integrable :math:`f, g`, so a model-free measure of how strongly
        the best one-dimensional feature of the parameter is identified by the
        data. Near 0 means the data are uninformative about *every* feature of
        :math:`\theta`; near 1 means some feature is pinned down almost exactly.
        """
        return float(self.singular_values[0])

    @property
    def chi2_divergence(self) -> float:
        r""":math:`\sum_k \hat\sigma_k^2 = \chi^2(\rho \Vert \pi \times \mu)`.

        The squared Hilbert-Schmidt norm of the deflated operator. Finiteness of
        the population version is exactly the compactness condition the
        low-rank expansion relies on, so a value that keeps climbing as ``rank``
        grows is a warning that the spectrum is not summable -- the
        near-deterministic-simulator regime, where the joint law concentrates
        near a lower-dimensional manifold.
        """
        return float((self.singular_values**2).sum())

    def spectrum_report(self, top: int = 8) -> str:
        """A short text report of the spectrum and the two diagnostics."""
        sv = self.singular_values
        shown = sv[: min(top, sv.numel())]
        lines = [
            f"rank d = {self.rank}",
            f"sigma_1 (HGR maximal correlation) = {self.maximal_correlation:.4f}",
            f"sum sigma_k^2 (chi-square divergence) = {self.chi2_divergence:.4f}",
            "spectrum: " + ", ".join(f"{v:.4f}" for v in shown.tolist()) + ("..." if sv.numel() > top else ""),
        ]
        return "\n".join(lines)

    def singular_function_theta(self, theta: Tensor, rank: Optional[int] = None) -> Tensor:
        r"""Evaluate the right singular functions :math:`\hat v_k(\theta)`.

        Useful for reading off *which* features of the parameter the data
        identify: on a non-identified model, :math:`\hat v_1` aligns with the
        identified direction.
        """
        theta = torch.as_tensor(theta, dtype=torch.float32).reshape(-1, self.theta_dim)
        if self._theta_scaler is not None:
            theta = self._theta_scaler.transform(theta)
        return self.operator.embed_y(theta, rank=rank)

    def singular_function_data(self, y_obs: Tensor, rank: Optional[int] = None) -> Tensor:
        r"""Evaluate the left singular functions :math:`\hat u_k(y)`."""
        return self.operator.embed_x(self._prepare_data(y_obs), rank=rank)

    def parameter_count(self) -> int:
        """Number of trainable parameters, for like-for-like cost comparisons."""
        return sum(p.numel() for p in self.operator.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        state = "fitted" if self.operator.is_fitted else "unfitted"
        return f"PosteriorOperator(theta_dim={self.theta_dim}, data_dim={self.data_dim}, rank={self.rank}, {state})"


def reference_posterior_moments(
    grid: Tensor, weights: Tensor
) -> Tuple[Tensor, Tensor]:
    """Mean and covariance of a discrete reference posterior, for validation.

    Args:
        grid: ``(m, theta_dim)`` support points.
        weights: ``(m,)`` normalised masses.
    """
    w = weights.reshape(-1, 1)
    mean = (w * grid).sum(dim=0)
    centred = grid - mean
    cov = (w * centred).T @ centred
    return mean, cov
