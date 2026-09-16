"""Neural building blocks for the NCP embeddings."""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    """Plain feed-forward embedding ``R^input_dim -> R^output_dim``.

    The last layer is linear and (by default) bias-free: NCP only ever uses the
    embeddings after they have been centered on the empirical marginal, so a
    trailing bias is a redundant parameter.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_hidden: int = 2,
        layer_size: int | Sequence[int] = 64,
        activation: Callable[[], nn.Module] = nn.GELU,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if n_hidden < 0:
            raise ValueError(f"n_hidden must be non-negative, got {n_hidden}")
        sizes = [layer_size] * n_hidden if isinstance(layer_size, int) else list(layer_size)
        if len(sizes) != n_hidden:
            raise ValueError(f"layer_size has {len(sizes)} entries but n_hidden={n_hidden}")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for width in sizes:
            layers.append(nn.Linear(in_dim, width))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = width
        layers.append(nn.Linear(in_dim, output_dim, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SingularValues(nn.Module):
    r"""Learnable singular values constrained to :math:`(0, 1]`.

    Parametrised as :math:`s_i = \exp(-w_i^2)`, which keeps every value inside
    :math:`(0, 1]` without a projection step (the singular values of the
    deflated conditional expectation operator are bounded by 1) and lets the
    optimiser switch a direction off smoothly by growing :math:`|w_i|`.
    """

    def __init__(self, latent_dim: int, init_scale: float | None = None):
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        # Small initial weights => all singular values start near 1, so every
        # latent direction receives gradient signal from the first step.
        std = 2.0 / latent_dim if init_scale is None else init_scale
        self.weights = nn.Parameter(torch.randn(latent_dim) * std)

    @property
    def values(self) -> Tensor:
        """The singular values :math:`s \\in (0, 1]^d`."""
        return torch.exp(-self.weights**2)

    def forward(self, x: Tensor) -> Tensor:
        """Scale the trailing dimension of ``x`` by the singular values."""
        return x * self.values
