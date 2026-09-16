r"""The NCP operator: learned embeddings plus the post-training whitening step.

After training, the raw embeddings :math:`u_\theta, v_\phi` span two
:math:`d`-dimensional subspaces of :math:`L^2(\pi_X)` and :math:`L^2(\pi_Y)`,
but they are identified only up to a linear map and the learned ``s`` is only a
loose estimate of the singular values. :meth:`NCPOperator.fit_statistics`
resolves both at once: it computes the exact :math:`L^2`-projection of the
deflated ratio onto those two subspaces,

.. math::
    \hat{r}(x, y)
      = \varphi_c(x)^\top\, C_\varphi^{-1} C_{\varphi\psi} C_\psi^{-1}\, \psi_c(y),
    \qquad \varphi = \operatorname{diag}(\sqrt{s})\, u,\;
           \psi = \operatorname{diag}(\sqrt{s})\, v,

and re-expresses it in singular-value form :math:`\hat{r}(x, y) = \sum_k
\hat{\sigma}_k \tilde{u}_k(x)\, \tilde{v}_k(y)` where the :math:`\tilde{u}_k`
are centered and orthonormal in :math:`L^2(\hat\pi_X)` (likewise
:math:`\tilde{v}_k`) and :math:`\hat\sigma_k \in [0, 1]` are the canonical
correlations between the two feature spaces. This is a closed-form step with no
gradient descent, and it is what makes the singular values reportable and the
representation stable enough to differentiate/integrate against.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import Tensor, nn

from .nn import SingularValues

__all__ = ["NCPOperator"]


def _as_2d(x: Any, *, name: str, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Coerce array-likes to a 2D float tensor of shape ``(n, features)``."""
    t = x if isinstance(x, Tensor) else torch.as_tensor(x)
    t = t.to(device=device, dtype=dtype)
    if t.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    if t.ndim > 2:
        t = t.reshape(t.shape[0], -1)
    return t


def _inv_sqrt_psd(cov: Tensor, reg: float) -> Tensor:
    """Symmetric inverse square root of a PSD matrix, with relative Tikhonov shift.

    ``reg`` is scaled by the mean eigenvalue so the amount of regularisation
    does not depend on how the embeddings happen to be scaled.
    """
    d = cov.shape[0]
    cov = 0.5 * (cov + cov.T)
    shift = reg * torch.diagonal(cov).mean().clamp_min(torch.finfo(cov.dtype).tiny)
    cov = cov + shift * torch.eye(d, dtype=cov.dtype, device=cov.device)
    evals, evecs = torch.linalg.eigh(cov)
    floor = evals.max().clamp_min(0) * d * torch.finfo(cov.dtype).eps
    inv_sqrt = torch.where(evals > floor, evals.clamp_min(floor).rsqrt(), torch.zeros_like(evals))
    return (evecs * inv_sqrt) @ evecs.T


class NCPOperator(nn.Module):
    r"""Neural Conditional Probability operator.

    Args:
        x_embedding: module mapping ``(n, x_dim) -> (n, latent_dim)``.
        y_embedding: module mapping ``(n, y_dim) -> (n, latent_dim)``.
        latent_dim: number of latent directions :math:`d`, i.e. the rank of the
            truncated singular value decomposition being learned.

    The module keeps everything inference needs in buffers, so a
    ``state_dict`` round-trip restores a ready-to-use operator: the whitening
    maps, the singular values, and the reference :math:`Y` sample that stands
    in for :math:`\pi_Y`.
    """

    def __init__(self, x_embedding: nn.Module, y_embedding: nn.Module, latent_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.x_embedding = x_embedding
        self.y_embedding = y_embedding
        self.singular_layer = SingularValues(latent_dim)

        d = self.latent_dim
        self.register_buffer("_mean_phi", torch.zeros(d))
        self.register_buffer("_mean_psi", torch.zeros(d))
        self.register_buffer("_left_map", torch.eye(d))
        self.register_buffer("_right_map", torch.eye(d))
        self.register_buffer("_singular_values", torch.zeros(d))
        self.register_buffer("_fitted", torch.zeros((), dtype=torch.bool))
        self.register_buffer("_reference_y", torch.zeros(0, 0))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Resize the reference-sample buffer before the copy.

        ``_reference_y`` holds however many atoms were kept at fit time, so a
        freshly constructed operator (which starts with an empty buffer) would
        otherwise fail the strict shape check on ``load_state_dict``.
        """
        key = prefix + "_reference_y"
        incoming = state_dict.get(key)
        if incoming is not None and incoming.shape != self._reference_y.shape:
            self._reference_y = torch.empty(
                incoming.shape, dtype=self._reference_y.dtype, device=self._reference_y.device
            )
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # ------------------------------------------------------------------ utils

    @property
    def device(self) -> torch.device:
        return self.singular_layer.weights.device

    @property
    def dtype(self) -> torch.dtype:
        return self.singular_layer.weights.dtype

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit_statistics` has been run."""
        return bool(self._fitted.item())

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "the whitening statistics are not available yet; call "
                "fit_statistics(X, Y) on a sample from the joint distribution "
                "(training.train_ncp does this for you at the end of training)"
            )

    @property
    def singular_values(self) -> Tensor:
        r"""Estimated singular values :math:`\hat\sigma_1 \geq \dots \geq \hat\sigma_d`."""
        self._require_fitted()
        return self._singular_values.clone()

    @property
    def reference_y(self) -> Tensor:
        r"""The stored sample standing in for :math:`\pi_Y` during inference."""
        self._require_fitted()
        if self._reference_y.numel() == 0:
            raise RuntimeError("no reference Y sample was stored; pass y_reference explicitly")
        return self._reference_y

    def prepare_x(self, x) -> Tensor:
        return _as_2d(x, name="X", dtype=self.dtype, device=self.device)

    def prepare_y(self, y) -> Tensor:
        return _as_2d(y, name="Y", dtype=self.dtype, device=self.device)

    # --------------------------------------------------------------- training

    def raw_embeddings(self, x, y) -> Tuple[Tensor, Tensor, Tensor]:
        """Return ``(u(X), v(Y), s)`` -- the quantities the training loss consumes."""
        u = self.x_embedding(self.prepare_x(x))
        v = self.y_embedding(self.prepare_y(y))
        return u, v, self.singular_layer.values

    def forward(self, x, y) -> Tuple[Tensor, Tensor, Tensor]:
        return self.raw_embeddings(x, y)

    def _scaled_features(self, x, y) -> Tuple[Tensor, Tensor]:
        r"""The symmetric factorisation :math:`\varphi = \sqrt{s}\,u`, :math:`\psi = \sqrt{s}\,v`.

        Splitting ``s`` evenly across the two sides makes :math:`h(x, y) =
        \varphi(x) \cdot \psi(y)` and keeps the two whitening problems on the
        same scale.
        """
        sqrt_s = self.singular_layer.values.sqrt()
        phi = self.x_embedding(self.prepare_x(x)) * sqrt_s
        psi = self.y_embedding(self.prepare_y(y)) * sqrt_s
        return phi, psi

    # ------------------------------------------------- post-training statistics

    @torch.no_grad()
    def fit_statistics(
        self,
        x,
        y,
        reg: float = 1e-6,
        chunk_size: int = 4096,
        store_reference: bool = True,
        max_reference: Optional[int] = 20000,
        generator: Optional[torch.Generator] = None,
    ) -> "NCPOperator":
        r"""Compute the whitening maps and singular values on a joint sample.

        Use as much data as possible, normally the full training set. This is a
        plug-in canonical-correlation estimate between two ``latent_dim``-
        dimensional feature spaces, so its singular values carry an upward
        finite-sample bias that grows with ``latent_dim`` and shrinks with the
        sample size. That bias is a property of the estimator, not of sample
        reuse: it appears just the same with untrained embeddings and on a
        sample independent of training, so holding data back from the fit does
        not remove it. ``reg`` is the lever that does.

        Args:
            x, y: paired sample from :math:`\pi_{XY}`. It also supplies the
                reference atoms for inference, so it should be large enough to
                resolve the quantiles of interest.
            reg: relative Tikhonov shift for the two feature covariances.
                Raising it towards ``1e-3`` counters the upward bias in the
                reported singular values at the cost of over-smoothing the
                density ratio, which widens intervals beyond their nominal
                level. See ``stats_reg`` in
                :func:`~posterior_operator.training.train_ncp`.
            chunk_size: minibatch size for the streaming moment accumulation.
            store_reference: keep (a subsample of) ``y`` as the empirical stand-in
                for :math:`\pi_Y` used by :meth:`condition`.
            max_reference: cap on the stored reference sample; ``None`` stores all.
            generator: RNG used for subsampling the reference set.
        """
        was_training = self.training
        self.eval()
        try:
            x_t, y_t = self.prepare_x(x), self.prepare_y(y)
            n = x_t.shape[0]
            if n < self.latent_dim + 2:
                raise ValueError(
                    f"need more samples than latent dimensions to estimate the feature "
                    f"covariances, got n={n} and latent_dim={self.latent_dim}"
                )

            d = self.latent_dim
            acc_dtype = torch.float64
            sum_phi = torch.zeros(d, dtype=acc_dtype, device=self.device)
            sum_psi = torch.zeros(d, dtype=acc_dtype, device=self.device)
            sum_pp = torch.zeros(d, d, dtype=acc_dtype, device=self.device)
            sum_qq = torch.zeros(d, d, dtype=acc_dtype, device=self.device)
            sum_pq = torch.zeros(d, d, dtype=acc_dtype, device=self.device)

            for start in range(0, n, chunk_size):
                stop = min(start + chunk_size, n)
                phi, psi = self._scaled_features(x_t[start:stop], y_t[start:stop])
                phi, psi = phi.to(acc_dtype), psi.to(acc_dtype)
                sum_phi += phi.sum(dim=0)
                sum_psi += psi.sum(dim=0)
                sum_pp += phi.T @ phi
                sum_qq += psi.T @ psi
                sum_pq += phi.T @ psi

            mean_phi = sum_phi / n
            mean_psi = sum_psi / n
            # Centered (co)variances from uncentered sums; accumulated in float64
            # so the subtraction below stays well-conditioned.
            cov_phi = (sum_pp - n * torch.outer(mean_phi, mean_phi)) / (n - 1)
            cov_psi = (sum_qq - n * torch.outer(mean_psi, mean_psi)) / (n - 1)
            cov_cross = (sum_pq - n * torch.outer(mean_phi, mean_psi)) / (n - 1)

            inv_sqrt_phi = _inv_sqrt_psd(cov_phi, reg)
            inv_sqrt_psi = _inv_sqrt_psd(cov_psi, reg)
            # Canonical-correlation matrix; its SVD is the singular value
            # decomposition of the projected deflated ratio.
            m = inv_sqrt_phi @ cov_cross @ inv_sqrt_psi
            left, svals, right_h = torch.linalg.svd(m)

            self._mean_phi = mean_phi.to(self.dtype)
            self._mean_psi = mean_psi.to(self.dtype)
            self._left_map = (inv_sqrt_phi @ left).to(self.dtype)
            self._right_map = (inv_sqrt_psi @ right_h.T).to(self.dtype)
            # Canonical correlations live in [0, 1]; the plug-in estimate can
            # overshoot slightly on small samples.
            self._singular_values = svals.clamp(0.0, 1.0).to(self.dtype)
            self._fitted = torch.ones((), dtype=torch.bool, device=self.device)

            if store_reference:
                ref = y_t
                if max_reference is not None and n > max_reference:
                    idx = torch.randperm(n, generator=generator, device=self.device)[:max_reference]
                    ref = y_t[idx]
                self._reference_y = ref.clone()
            else:
                self._reference_y = torch.zeros(0, 0, dtype=self.dtype, device=self.device)
        finally:
            self.train(was_training)
        return self

    # -------------------------------------------------------------- inference

    def _rank_slice(self, rank: Optional[int]) -> int:
        if rank is None:
            return self.latent_dim
        if not 1 <= rank <= self.latent_dim:
            raise ValueError(f"rank must be in [1, {self.latent_dim}], got {rank}")
        return rank

    @torch.no_grad()
    def embed_x(self, x, rank: Optional[int] = None) -> Tensor:
        r"""Whitened, centered left singular functions :math:`\tilde{u}(x)`, shape ``(n, rank)``."""
        self._require_fitted()
        r = self._rank_slice(rank)
        phi = self.x_embedding(self.prepare_x(x)) * self.singular_layer.values.sqrt()
        return (phi - self._mean_phi) @ self._left_map[:, :r]

    @torch.no_grad()
    def embed_y(self, y, rank: Optional[int] = None) -> Tensor:
        r"""Whitened, centered right singular functions :math:`\tilde{v}(y)`, shape ``(n, rank)``."""
        self._require_fitted()
        r = self._rank_slice(rank)
        psi = self.y_embedding(self.prepare_y(y)) * self.singular_layer.values.sqrt()
        return (psi - self._mean_psi) @ self._right_map[:, :r]

    @torch.no_grad()
    def deflated_ratio(self, x, y, rank: Optional[int] = None) -> Tensor:
        r"""The matrix :math:`\hat{r}(x_i, y_j) = p(y_j \mid x_i)/\pi_Y(y_j) - 1`.

        Returns shape ``(n_x, n_y)``: every conditioning value against every
        evaluation point. This single bilinear form is the primitive behind
        every downstream quantity -- densities, CDFs, moments, regions.
        """
        self._require_fitted()
        r = self._rank_slice(rank)
        u = self.embed_x(x, rank=r)
        v = self.embed_y(y, rank=r)
        return (u * self._singular_values[:r]) @ v.T

    @torch.no_grad()
    def conditional_weights(
        self,
        x,
        y_reference=None,
        rank: Optional[int] = None,
        clip: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        r"""Discrete conditional law of :math:`Y \mid X = x` over reference atoms.

        With :math:`\hat\pi_Y = \frac{1}{m}\sum_j \delta_{y_j}` the plug-in
        marginal, :math:`p(y \mid x) = \pi_Y(y)\,(1 + \hat{r}(x, y))` puts mass
        :math:`w_j(x) = \frac{1}{m}\,(1 + \hat{r}(x, y_j))` on atom
        :math:`y_j`. The masses sum to one by construction (the right singular
        functions are centered on the fitting sample) but can go negative,
        since nothing constrains a density-*ratio* estimate to be positive.

        Args:
            clip: clamp negative masses to zero and renormalise, yielding a
                genuine probability vector. Turn it off to inspect the raw
                signed estimate.

        Returns:
            ``(weights, atoms)`` of shapes ``(n_x, m)`` and ``(m, y_dim)``.
        """
        self._require_fitted()
        atoms = self.reference_y if y_reference is None else self.prepare_y(y_reference)
        m = atoms.shape[0]
        if m < 2:
            raise ValueError(f"need at least 2 reference atoms, got {m}")
        weights = (1.0 + self.deflated_ratio(x, atoms, rank=rank)) / m
        if clip:
            weights = weights.clamp_min(0.0)
        total = weights.sum(dim=-1, keepdim=True)
        degenerate = total.abs() < torch.finfo(weights.dtype).eps
        if bool(degenerate.any()):
            # No mass survived clipping: fall back to the marginal rather than
            # dividing by ~0 and emitting NaNs.
            weights = torch.where(degenerate, torch.full_like(weights, 1.0 / m), weights)
            total = torch.where(degenerate, torch.ones_like(total), total)
        return weights / total, atoms

    def condition(
        self,
        x,
        y_reference=None,
        rank: Optional[int] = None,
        clip: bool = True,
    ):
        """Return a :class:`~posterior_operator.inference.ConditionalDistribution`.

        One call re-conditions the trained operator on new ``x`` values; no
        retraining is involved, which is the practical payoff of the
        unconditional training objective.
        """
        from .inference import ConditionalDistribution

        weights, atoms = self.conditional_weights(x, y_reference=y_reference, rank=rank, clip=clip)
        return ConditionalDistribution(
            weights=weights,
            atoms=atoms,
            operator=self,
            x=self.prepare_x(x),
            rank=rank,
        )

    def parameters_summary(self) -> str:
        """Short human-readable description, handy in logs and notebooks."""
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        head = f"NCPOperator(latent_dim={self.latent_dim}, trainable_params={n_params}"
        if not self.is_fitted:
            return head + ", fitted=False)"
        sv = self._singular_values
        top = ", ".join(f"{v:.3f}" for v in sv[: min(5, sv.numel())].tolist())
        return head + f", fitted=True, top_singular_values=[{top}])"
