r"""Neural Conditional Probability (NCP) -- operator-theoretic conditional distributions.

An implementation of *Neural Conditional Probability for Uncertainty
Quantification* (Kostic, Pacreau, Turri, Novelli, Lounici and Pontil,
NeurIPS 2024), built from the method description rather than ported from the
authors' code.

The idea in one line: learn a truncated singular value decomposition of the
conditional expectation operator once, unconditionally, then read every
conditional quantity off that decomposition in closed form.

.. math::
    \frac{p(y \mid x)}{\pi_Y(y)} = 1 + \sum_{k=1}^{d} \sigma_k\, u_k(x)\, v_k(y)

Two neural networks supply :math:`u` and :math:`v`; training never sees a
conditioning value, so a fitted operator re-conditions on new :math:`x` -- and
switches between conditional means, quantiles, densities and confidence
regions -- without any retraining.

Quick start
-----------
.. code-block:: python

    import torch
    from posterior_operator import Heteroscedastic, build_ncp, train_ncp

    data = Heteroscedastic()
    x, y = data.sample(4000, generator=torch.Generator().manual_seed(0))

    operator = build_ncp(x_dim=1, y_dim=1, latent_dim=32)
    train_ncp(operator, x, y, epochs=800, validation_data=data.sample(1000))

    posterior = operator.condition(torch.tensor([[-1.5], [0.0], [1.5]]))
    posterior.mean()              # conditional means
    posterior.std()               # conditional standard deviations
    posterior.quantile([0.05, 0.95])
    posterior.interval(alpha=0.1)  # shortest 90% conditional interval
"""

from .data import (
    BimodalMixture,
    Heteroscedastic,
    LinearGaussian,
    Standardizer,
    StudentT,
    SyntheticConditional,
    train_val_split,
)
from .inference import ConditionalDistribution, GaussianKDE, isotonic_regression
from .lfi import PosteriorOperator, PosteriorSample, reference_posterior_moments
from .losses import (
    NCPLoss,
    centering_penalty,
    orthonormality_penalty,
    split_objective,
    ustat_objective,
)
from .nn import MLP, SingularValues
from .operator import NCPOperator
from .simulators import MA2, GaussianLinear, Simulator, SumIdentified
from .training import train_ncp

__version__ = "0.1.0"

__all__ = [
    "MA2",
    "MLP",
    "BimodalMixture",
    "ConditionalDistribution",
    "GaussianKDE",
    "GaussianLinear",
    "Heteroscedastic",
    "LinearGaussian",
    "NCPLoss",
    "NCPOperator",
    "PosteriorOperator",
    "PosteriorSample",
    "SingularValues",
    "Simulator",
    "Standardizer",
    "StudentT",
    "SumIdentified",
    "SyntheticConditional",
    "build_ncp",
    "centering_penalty",
    "isotonic_regression",
    "orthonormality_penalty",
    "reference_posterior_moments",
    "split_objective",
    "train_ncp",
    "train_val_split",
    "ustat_objective",
]


def build_ncp(
    x_dim: int,
    y_dim: int,
    latent_dim: int = 32,
    n_hidden: int = 2,
    layer_size: int = 64,
    dropout: float = 0.0,
    activation=None,
) -> NCPOperator:
    """Build an :class:`NCPOperator` with symmetric MLP embeddings.

    The defaults follow the paper's experimental setting: two hidden layers per
    branch, which is enough to match or beat heavier conditional density
    estimators on the standard benchmarks.

    Args:
        x_dim, y_dim: input dimensions of the conditioning and response variables.
        latent_dim: number of singular directions :math:`d` to learn. It caps
            the rank of the representable conditional dependence, so prefer
            generous values -- unneeded directions get singular values near
            zero and can be dropped at inference via ``rank=``.
        n_hidden, layer_size, dropout, activation: passed to both :class:`MLP` branches.
    """
    import torch.nn as nn

    act = nn.GELU if activation is None else activation
    common = dict(
        output_dim=latent_dim, n_hidden=n_hidden, layer_size=layer_size, dropout=dropout, activation=act
    )
    return NCPOperator(
        x_embedding=MLP(input_dim=x_dim, **common),
        y_embedding=MLP(input_dim=y_dim, **common),
        latent_dim=latent_dim,
    )
