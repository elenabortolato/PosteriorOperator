"""A small, dependency-free training loop for :class:`NCPOperator`.

Training is *unconditional*: one pass over the joint sample, no conditioning
value anywhere in the objective. The loop ends by calling
:meth:`NCPOperator.fit_statistics`, which is the closed-form whitening step
that turns the learned subspaces into a usable singular value decomposition.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch import Tensor, nn

from .losses import NCPLoss
from .operator import NCPOperator

__all__ = ["train_ncp"]


def _batches(n: int, batch_size: int, generator: Optional[torch.Generator], device: torch.device):
    perm = torch.randperm(n, generator=generator, device=device)
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        if idx.numel() >= 2:  # the objective needs at least one off-diagonal pair
            yield idx


def train_ncp(
    operator: NCPOperator,
    x,
    y,
    loss: Optional[NCPLoss] = None,
    epochs: int = 1000,
    batch_size: Optional[int] = None,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    validation_data: Optional[Tuple[Any, Any]] = None,
    patience: Optional[int] = None,
    val_every: int = 1,
    grad_clip: Optional[float] = None,
    fit_statistics: bool = True,
    stats_data: Optional[Tuple[Any, Any]] = None,
    stats_fraction: float = 0.0,
    stats_reg: float = 1e-6,
    seed: Optional[int] = None,
    verbose: bool = False,
    log_every: int = 50,
) -> Dict[str, Any]:
    """Fit an :class:`NCPOperator` on a paired sample from the joint distribution.

    Args:
        operator: the operator to train, modified in place.
        x, y: paired training sample, array-like of shapes ``(n, x_dim)`` and ``(n, y_dim)``.
        loss: objective; defaults to ``NCPLoss(mode="ustat", gamma=1e-3)``.
        epochs: maximum number of passes over the data.
        batch_size: minibatch size. ``None`` means full batch, which is the
            better default here: the U-statistic term uses all
            :math:`n(n-1)` cross pairs, so bigger batches reduce its variance
            substantially.
        validation_data: ``(x_val, y_val)`` used for early stopping and for
            selecting the returned weights.
        patience: stop after this many validation checks without improvement.
            ``None`` disables early stopping.
        val_every: evaluate the validation loss every this many epochs.
        grad_clip: optional global gradient-norm clip.
        fit_statistics: run the closed-form whitening step at the end.
        stats_data: explicit ``(x, y)`` sample for the whitening step,
            overriding ``stats_fraction``. Useful when more data is available
            for the closed-form step than for the gradient updates.
        stats_fraction: fraction of ``(x, y)`` withheld from the gradient
            updates and used only for the whitening step. Defaults to ``0``,
            i.e. whiten on all the training data. Sample splitting here is
            *not* the fix it looks like: on the Gaussian benchmark, holding out
            20% measurably worsens the spectrum, the conditional mean, the
            density and the coverage, because the whitening bias described
            under ``stats_reg`` is not caused by sample reuse and the split
            only costs training data. Raise it above ``0`` when an independent
            whitening sample is wanted for its own sake.
        stats_reg: Tikhonov shift passed to :meth:`NCPOperator.fit_statistics`.
            The whitening step is a plug-in canonical-correlation estimate
            between two ``latent_dim``-dimensional feature spaces, so its
            singular values are biased upward -- an effect that is present even
            with untrained embeddings and grows with ``latent_dim``. Raising
            this towards ``1e-3`` shrinks that bias (max spectrum error
            ``0.054 -> 0.022`` on the Gaussian benchmark) but over-smooths the
            ratio, widening intervals past their nominal level and costing
            density accuracy. The default favours calibrated conditional
            quantities; raise it if the reported spectrum itself is the output
            you care about.
        seed: seed for batch shuffling and parameter-independent randomness.
        verbose: print progress.
        log_every: printing interval when ``verbose``.

    Returns:
        History dict with ``train_loss``, ``val_loss``, ``val_epochs``,
        ``best_epoch`` and ``best_val_loss``. The operator is left holding the
        best validation weights when early stopping is in play.
    """
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if val_every < 1:
        raise ValueError(f"val_every must be positive, got {val_every}")

    loss = loss if loss is not None else NCPLoss()
    device = operator.device
    x_t, y_t = operator.prepare_x(x), operator.prepare_y(y)
    if x_t.shape[0] != y_t.shape[0]:
        raise ValueError(f"X has {x_t.shape[0]} rows but Y has {y_t.shape[0]}")
    n = x_t.shape[0]
    if n < 2:
        raise ValueError("at least 2 training samples are required")

    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    # Carve out the whitening slice before anything touches the data, so it is
    # independent of both the gradient updates and the early-stopping choice.
    stats_x = stats_y = None
    if fit_statistics and stats_data is None and stats_fraction > 0:
        if not 0.0 < stats_fraction < 1.0:
            raise ValueError(f"stats_fraction must lie in [0, 1), got {stats_fraction}")
        n_stats = int(round(n * stats_fraction))
        if n_stats < operator.latent_dim + 2 or n - n_stats < 2:
            raise ValueError(
                f"stats_fraction={stats_fraction} carves {n_stats} of {n} samples, which cannot "
                f"support latent_dim={operator.latent_dim}; pass more data, a smaller latent_dim, "
                "an explicit stats_data, or stats_fraction=0"
            )
        split = torch.randperm(n, generator=generator, device=device)
        stats_x, stats_y = x_t[split[:n_stats]], y_t[split[:n_stats]]
        x_t, y_t = x_t[split[n_stats:]], y_t[split[n_stats:]]
        n = x_t.shape[0]

    bs = n if batch_size is None else min(int(batch_size), n)
    if bs < 2:
        raise ValueError(f"batch_size must be at least 2, got {batch_size}")

    val_x = val_y = None
    if validation_data is not None:
        val_x = operator.prepare_x(validation_data[0])
        val_y = operator.prepare_y(validation_data[1])
        if val_x.shape[0] != val_y.shape[0]:
            raise ValueError("validation X and Y must have the same number of rows")
        if val_x.shape[0] < 2:
            raise ValueError("at least 2 validation samples are required")

    opt_kwargs = dict(optimizer_kwargs or {})
    opt_kwargs.setdefault("weight_decay", weight_decay)
    optimizer = optimizer_cls(operator.parameters(), lr=lr, **opt_kwargs)

    history: Dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "val_epochs": [],
        "best_epoch": None,
        "best_val_loss": None,
    }
    best_val = math.inf
    best_state: Optional[Dict[str, Tensor]] = None
    stale = 0

    for epoch in range(epochs):
        operator.train()
        epoch_loss, seen = 0.0, 0
        for idx in _batches(n, bs, generator, device):
            optimizer.zero_grad(set_to_none=True)
            u, v, s = operator.raw_embeddings(x_t[idx], y_t[idx])
            value = loss(u, v, s)
            if not torch.isfinite(value):
                raise RuntimeError(
                    f"loss became {value.item()} at epoch {epoch}; lower the learning rate "
                    "or raise the penalty weight gamma"
                )
            value.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(operator.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += float(value.detach()) * idx.numel()
            seen += idx.numel()
        train_loss = epoch_loss / max(seen, 1)
        history["train_loss"].append(train_loss)

        improved = False
        if val_x is not None and (epoch % val_every == 0 or epoch == epochs - 1):
            operator.eval()
            with torch.no_grad():
                u, v, s = operator.raw_embeddings(val_x, val_y)
                val_loss = float(loss(u, v, s))
            history["val_loss"].append(val_loss)
            history["val_epochs"].append(epoch)
            if val_loss < best_val - 1e-12:
                best_val, improved = val_loss, True
                best_state = copy.deepcopy({k: t.detach().clone() for k, t in operator.state_dict().items()})
                history["best_epoch"], history["best_val_loss"] = epoch, val_loss
                stale = 0
            else:
                stale += 1

        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            msg = f"epoch {epoch:5d}  train {train_loss:+.6f}"
            if history["val_loss"]:
                msg += f"  val {history['val_loss'][-1]:+.6f}{'  *' if improved else ''}"
            print(msg)

        if patience is not None and val_x is not None and stale >= patience:
            if verbose:
                print(f"early stop at epoch {epoch} (best epoch {history['best_epoch']})")
            break

    if best_state is not None:
        operator.load_state_dict(best_state)
    operator.eval()

    if fit_statistics:
        if stats_data is not None:
            stats_x = operator.prepare_x(stats_data[0])
            stats_y = operator.prepare_y(stats_data[1])
            if stats_x.shape[0] != stats_y.shape[0]:
                raise ValueError("stats_data X and Y must have the same number of rows")
        elif stats_x is None:  # stats_fraction == 0: knowingly whiten in-sample
            stats_x, stats_y = x_t, y_t
        operator.fit_statistics(stats_x, stats_y, reg=stats_reg, generator=generator)
        history["stats_samples"] = int(stats_x.shape[0])

    history["train_samples"] = int(n)
    return history
