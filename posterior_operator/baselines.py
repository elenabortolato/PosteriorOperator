r"""Baselines the operator has to justify itself against.

Two reference points, matching the two alternatives the posterior-functional
argument is set against:

* :class:`DirectRegression` -- fit one regression per functional. By the
  elementary fact that :math:`T_f(y) = \mathbb{E}[f(\Theta) \mid Y = y]` is the
  :math:`L^2`-optimal regression of :math:`f(\theta_i)` on :math:`y_i`, this is
  a consistent estimator of that single functional and nothing else. It is the
  thing to beat when only one functional is ever wanted, and the thing that
  becomes expensive when many are.
* :class:`NeuralPosteriorEstimator` -- a conditional mixture density network
  over :math:`\theta`, i.e. neural posterior estimation in its original form.
  It targets the full posterior density and then integrates against
  :math:`f`, so it answers new functionals without retraining too -- the
  comparison is about cost and accuracy, not about amortization per se.

Both share the training loop's conventions (early stopping, seeded batching)
so that comparisons are not confounded by the optimiser.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .data import Standardizer
from .nn import MLP

__all__ = ["DirectRegression", "NeuralPosteriorEstimator"]


def _reset_parameters(module: nn.Module) -> None:
    """Re-run every submodule's own initialiser, under the current RNG state."""
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def _train(
    module: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
    epochs: int,
    lr: float,
    batch_size: Optional[int],
    patience: Optional[int],
    validation_split: float,
    seed: Optional[int],
    verbose: bool,
) -> Dict[str, Any]:
    """Shared minibatch loop with early stopping, so baselines are not handicapped."""
    generator = None
    if seed is not None:
        torch.manual_seed(int(seed))
        # The module was built before fit() was called, so its weights were drawn
        # from whatever RNG state existed then. Re-initialise under the seed so
        # that fit(seed=s) is reproducible on its own, without the caller having
        # to seed before construction.
        _reset_parameters(module)
        generator = torch.Generator().manual_seed(int(seed))

    n = inputs.shape[0]
    val_inputs = val_targets = None
    if validation_split > 0:
        n_val = max(2, int(round(n * validation_split)))
        perm = torch.randperm(n, generator=generator)
        val_idx, fit_idx = perm[:n_val], perm[n_val:]
        val_inputs, val_targets = inputs[val_idx], targets[val_idx]
        inputs, targets = inputs[fit_idx], targets[fit_idx]
        n = inputs.shape[0]

    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    size = n if batch_size is None else min(int(batch_size), n)
    history: Dict[str, Any] = {"train_loss": [], "val_loss": [], "best_epoch": None}
    best, best_state, stale = math.inf, None, 0

    for epoch in range(epochs):
        module.train()
        perm = torch.randperm(n, generator=generator)
        total = 0.0
        for start in range(0, n, size):
            idx = perm[start : start + size]
            optimizer.zero_grad(set_to_none=True)
            value = loss_fn(inputs[idx], targets[idx])
            value.backward()
            optimizer.step()
            total += float(value.detach()) * idx.numel()
        history["train_loss"].append(total / max(n, 1))

        if val_inputs is not None:
            module.eval()
            with torch.no_grad():
                val = float(loss_fn(val_inputs, val_targets))
            history["val_loss"].append(val)
            if val < best - 1e-12:
                best, stale = val, 0
                best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
                history["best_epoch"] = epoch
            else:
                stale += 1
                if patience is not None and stale >= patience:
                    break
        if verbose and epoch % 100 == 0:
            print(f"  epoch {epoch:5d} train {history['train_loss'][-1]:+.5f}")

    if best_state is not None:
        module.load_state_dict(best_state)
    module.eval()
    return history


class DirectRegression:
    r"""One regression per functional: :math:`\hat{m}_f(y) \approx \mathbb{E}[f(\Theta) \mid Y=y]`.

    Args:
        data_dim: dimension of the data or its summaries.
        output_dim: dimension of ``f(theta)``.
        n_hidden, layer_size, activation: architecture.

    Example:
        >>> model = DirectRegression(data_dim=5, output_dim=2)
        >>> model.fit(y, theta)              # f = identity: the posterior mean
        >>> model.predict(y_obs)
    """

    def __init__(
        self,
        data_dim: int,
        output_dim: int,
        n_hidden: int = 2,
        layer_size: int = 64,
        activation: Optional[Callable[[], nn.Module]] = None,
    ):
        self.data_dim = int(data_dim)
        self.output_dim = int(output_dim)
        self.net = MLP(
            input_dim=self.data_dim,
            output_dim=self.output_dim,
            n_hidden=n_hidden,
            layer_size=layer_size,
            activation=nn.GELU if activation is None else activation,
            bias=True,
        )
        self._x_scaler: Optional[Standardizer] = None
        self._y_scaler: Optional[Standardizer] = None

    def fit(
        self,
        data: Tensor,
        values: Tensor,
        epochs: int = 400,
        lr: float = 1e-3,
        batch_size: Optional[int] = 512,
        patience: Optional[int] = 40,
        validation_split: float = 0.2,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        r"""Regress ``values`` :math:`= f(\theta_i)` on ``data`` :math:`= y_i`."""
        data = torch.as_tensor(data, dtype=torch.float32).reshape(-1, self.data_dim)
        values = torch.as_tensor(values, dtype=torch.float32).reshape(data.shape[0], -1)
        if values.shape[1] != self.output_dim:
            raise ValueError(f"expected output_dim={self.output_dim}, got {values.shape[1]}")
        self._x_scaler = Standardizer().fit(data)
        self._y_scaler = Standardizer().fit(values)
        x_s = self._x_scaler.transform(data)
        y_s = self._y_scaler.transform(values)

        def loss_fn(batch_x: Tensor, batch_y: Tensor) -> Tensor:
            return ((self.net(batch_x) - batch_y) ** 2).mean()

        return _train(
            self.net, loss_fn, x_s, y_s, epochs, lr, batch_size, patience, validation_split, seed, verbose
        )

    @torch.no_grad()
    def predict(self, y_obs: Tensor) -> Tensor:
        """The fitted functional at new observations, shape ``(n_obs, output_dim)``."""
        if self._x_scaler is None or self._y_scaler is None:
            raise RuntimeError("call fit() before predict()")
        data = torch.as_tensor(y_obs, dtype=torch.float32).reshape(-1, self.data_dim)
        return self._y_scaler.inverse_transform(self.net(self._x_scaler.transform(data)))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


class NeuralPosteriorEstimator:
    r"""Neural posterior estimation: a conditional Gaussian mixture over :math:`\theta`.

    Models :math:`q(\theta \mid y) = \sum_{c=1}^{C} \pi_c(y)\,
    N(\theta; \mu_c(y), \operatorname{diag} s_c(y)^2)` and maximises
    :math:`\sum_i \log q(\theta_i \mid y_i)` over prior-predictive draws --
    the original form of NPE. Once fitted, moments come out in closed form and
    any other functional by Monte Carlo, so it amortises over functionals too;
    what differs from the operator is that it estimates the whole density
    rather than a low-rank factorisation of the dependence.

    Args:
        theta_dim, data_dim: parameter and data dimensions.
        n_components: number of mixture components.
        n_hidden, layer_size: architecture of the shared trunk.
        min_scale: floor on the component scales, for numerical stability.
    """

    def __init__(
        self,
        theta_dim: int,
        data_dim: int,
        n_components: int = 10,
        n_hidden: int = 2,
        layer_size: int = 64,
        min_scale: float = 1e-3,
    ):
        self.theta_dim = int(theta_dim)
        self.data_dim = int(data_dim)
        self.n_components = int(n_components)
        self.min_scale = float(min_scale)
        outputs = self.n_components * (1 + 2 * self.theta_dim)
        self.net = MLP(
            input_dim=self.data_dim,
            output_dim=outputs,
            n_hidden=n_hidden,
            layer_size=layer_size,
            activation=nn.GELU,
            bias=True,
        )
        self._x_scaler: Optional[Standardizer] = None
        self._theta_scaler: Optional[Standardizer] = None

    # ------------------------------------------------------------------ core

    def _mixture(self, data_scaled: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return ``(log_weights, means, scales)`` in standardised parameter units."""
        raw = self.net(data_scaled)
        c, p = self.n_components, self.theta_dim
        logits = raw[:, :c]
        means = raw[:, c : c + c * p].reshape(-1, c, p)
        log_scales = raw[:, c + c * p :].reshape(-1, c, p)
        scales = torch.nn.functional.softplus(log_scales) + self.min_scale
        return torch.log_softmax(logits, dim=-1), means, scales

    def _log_prob_scaled(self, data_scaled: Tensor, theta_scaled: Tensor) -> Tensor:
        log_weights, means, scales = self._mixture(data_scaled)
        z = (theta_scaled.unsqueeze(1) - means) / scales
        per_component = (-0.5 * z**2 - torch.log(scales) - 0.5 * math.log(2 * math.pi)).sum(dim=-1)
        return torch.logsumexp(log_weights + per_component, dim=-1)

    def fit(
        self,
        theta: Tensor,
        data: Tensor,
        epochs: int = 400,
        lr: float = 1e-3,
        batch_size: Optional[int] = 512,
        patience: Optional[int] = 40,
        validation_split: float = 0.2,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        r"""Maximise the conditional log-likelihood on prior-predictive draws."""
        theta = torch.as_tensor(theta, dtype=torch.float32).reshape(-1, self.theta_dim)
        data = torch.as_tensor(data, dtype=torch.float32).reshape(theta.shape[0], -1)
        if data.shape[1] != self.data_dim:
            raise ValueError(f"expected data_dim={self.data_dim}, got {data.shape[1]}")
        self._x_scaler = Standardizer().fit(data)
        self._theta_scaler = Standardizer().fit(theta)
        x_s = self._x_scaler.transform(data)
        t_s = self._theta_scaler.transform(theta)

        def loss_fn(batch_x: Tensor, batch_t: Tensor) -> Tensor:
            return -self._log_prob_scaled(batch_x, batch_t).mean()

        return _train(
            self.net, loss_fn, x_s, t_s, epochs, lr, batch_size, patience, validation_split, seed, verbose
        )

    # ------------------------------------------------------------- inference

    def _prepare(self, y_obs: Tensor) -> Tensor:
        if self._x_scaler is None:
            raise RuntimeError("call fit() before querying the posterior")
        data = torch.as_tensor(y_obs, dtype=torch.float32).reshape(-1, self.data_dim)
        return self._x_scaler.transform(data)

    @torch.no_grad()
    def log_prob(self, y_obs: Tensor, theta: Tensor) -> Tensor:
        r""":math:`\log q(\theta \mid y)` for paired rows, shape ``(n,)``.

        Includes the Jacobian of the internal standardisation, so the value is a
        density in the parameter's natural units.
        """
        data_scaled = self._prepare(y_obs)
        theta = torch.as_tensor(theta, dtype=torch.float32).reshape(-1, self.theta_dim)
        jacobian = torch.log(self._theta_scaler.scale).sum()
        return self._log_prob_scaled(data_scaled, self._theta_scaler.transform(theta)) - jacobian

    @torch.no_grad()
    def mean(self, y_obs: Tensor) -> Tensor:
        r"""Posterior mean :math:`\sum_c \pi_c \mu_c`, in natural units."""
        log_weights, means, _ = self._mixture(self._prepare(y_obs))
        scaled = (log_weights.exp().unsqueeze(-1) * means).sum(dim=1)
        return self._theta_scaler.inverse_transform(scaled)

    @torch.no_grad()
    def covariance(self, y_obs: Tensor) -> Tensor:
        r"""Posterior covariance of the mixture, shape ``(n_obs, p, p)``.

        Diagonal components, so the law of total covariance gives
        :math:`\sum_c \pi_c(\operatorname{diag} s_c^2 + \mu_c \mu_c^\top) -
        \bar\mu \bar\mu^\top`, rescaled to natural units.
        """
        log_weights, means, scales = self._mixture(self._prepare(y_obs))
        weights = log_weights.exp().unsqueeze(-1)
        mean = (weights * means).sum(dim=1)
        second = (weights.unsqueeze(-1) * (torch.einsum("ncp,ncq->ncpq", means, means))).sum(dim=1)
        second = second + (weights * scales**2).sum(dim=1).diag_embed()
        cov_scaled = second - torch.einsum("np,nq->npq", mean, mean)
        scale = self._theta_scaler.scale.reshape(-1)
        return cov_scaled * torch.outer(scale, scale)

    @torch.no_grad()
    def sample(self, y_obs: Tensor, n_samples: int, generator: Optional[torch.Generator] = None) -> Tensor:
        """Draw from the fitted posterior, shape ``(n_obs, n_samples, p)``."""
        log_weights, means, scales = self._mixture(self._prepare(y_obs))
        n_obs = log_weights.shape[0]
        picks = torch.multinomial(log_weights.exp(), n_samples, replacement=True, generator=generator)
        rows = torch.arange(n_obs).unsqueeze(-1)
        chosen_means = means[rows, picks]
        chosen_scales = scales[rows, picks]
        noise = torch.randn(chosen_means.shape, generator=generator)
        drawn = chosen_means + chosen_scales * noise
        return self._theta_scaler.inverse_transform(drawn.reshape(-1, self.theta_dim)).reshape(
            n_obs, n_samples, self.theta_dim
        )

    @torch.no_grad()
    def functional(
        self,
        y_obs: Tensor,
        f: Callable[[Tensor], Tensor],
        n_samples: int = 20000,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        r"""Monte Carlo estimate of :math:`\mathbb{E}_q[f(\Theta) \mid Y = y]`.

        Unlike the operator's weighted average, a new functional here costs
        ``n_samples`` fresh draws and evaluations of ``f`` per observation.
        """
        draws = self.sample(y_obs, n_samples, generator=generator)
        n_obs = draws.shape[0]
        values = f(draws.reshape(-1, self.theta_dim))
        return values.reshape(n_obs, n_samples, -1).mean(dim=1)

    @torch.no_grad()
    def quantile(
        self,
        y_obs: Tensor,
        levels,
        coordinate: int = 0,
        n_samples: int = 20000,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Marginal posterior quantiles of one coordinate, by Monte Carlo."""
        draws = self.sample(y_obs, n_samples, generator=generator)[:, :, coordinate]
        lv = torch.as_tensor(levels, dtype=draws.dtype).reshape(-1)
        return torch.quantile(draws, lv, dim=-1, interpolation="linear").T

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)
