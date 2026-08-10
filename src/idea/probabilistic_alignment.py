"""Probabilistic Calibration and Matching Alignment (PCMA).

PCMA quantifies the *typicality* of a candidate exemplar: whether the
candidate is compatible with the distribution of its own class, rather
than merely close to the query.

Each class ``c`` is modelled as a Gaussian manifold ``N(mu_c, Sigma_c)``
estimated from the annotated candidate pool. The squared 2-Wasserstein
distance between a deterministic embedding ``z_e`` (a Dirac measure) and
the class manifold is

    W2^2(z_e, N_c) = ||z_e - mu_c||^2 + Tr(Sigma_c)

and is converted into a bounded typicality score by an exponential
kernel

    S_typ(e) = exp(-W2^2(z_e, N_c) / sigma^2)

The two terms have different roles. Within a class, ``Tr(Sigma_c)`` is
constant, so candidate ordering is determined by ``||z_e - mu_c||^2``. Across
classes, the trace acts as a class-level reliability prior: compact class
distributions yield more stable empirical prototypes, whereas dispersed
classes carry greater prototype uncertainty. Although ``Sigma_c`` defines the
Gaussian mathematically, the score depends only on
``Tr(Sigma_c) = sum_d Var(z_d)``. The implementation therefore stores this
scalar sufficient statistic and never constructs or inverts a covariance
matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ClassGaussian", "PCMAConfig", "ProbabilisticAlignmentModule"]


@dataclass
class ClassGaussian:
    """Prototype and scalar dispersion of a single class."""

    label: str
    mu: np.ndarray
    """Mean embedding, shape ``(D,)``."""
    trace: float
    """Mean squared distance to the prototype, ``Tr(Sigma_c)``."""
    n_samples: int


@dataclass
class PCMAConfig:
    """Configuration for :class:`ProbabilisticAlignmentModule`."""

    sigma_sq: float = 1.0
    """Bandwidth ``sigma^2`` of the exponential kernel."""

    def __post_init__(self) -> None:
        if self.sigma_sq <= 0:
            raise ValueError(f"sigma_sq must be > 0, got {self.sigma_sq}")


class ProbabilisticAlignmentModule:
    """Estimate class Gaussians and score exemplar typicality.

    The module is fitted once on the annotated candidate pool (the training
    split) and then queried during exemplar selection. Fitting involves no
    gradient updates: it only computes prototypes and scalar dispersions of the
    frozen calibrated embeddings.

    Example
    -------
    >>> pcma = ProbabilisticAlignmentModule(PCMAConfig(sigma_sq=1.0))
    >>> pcma.fit(features, labels)                      # doctest: +SKIP
    >>> pcma.typicality(z_e, "forest")                  # doctest: +SKIP
    0.83...
    """

    def __init__(self, config: Optional[PCMAConfig] = None) -> None:
        self.config = config or PCMAConfig()
        self._manifolds: Dict[str, ClassGaussian] = {}
        self._feature_dim: Optional[int] = None

    # Fitting
    def fit(
        self,
        features: np.ndarray,
        labels: Sequence[str],
    ) -> Dict[str, ClassGaussian]:
        """Estimate one prototype and scalar dispersion per class.

        Parameters
        ----------
        features:
            Calibrated embeddings of the candidate pool, shape ``(N, D)``.
        labels:
            Ground-truth class label of each row of ``features``.

        Returns
        -------
        Mapping from class label to its :class:`ClassGaussian`.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D (N, D), got shape {features.shape}")
        if len(labels) != features.shape[0]:
            raise ValueError(
                f"labels length {len(labels)} does not match features rows {features.shape[0]}"
            )

        self._feature_dim = features.shape[1]

        by_class: Dict[str, List[int]] = {}
        for idx, label in enumerate(labels):
            by_class.setdefault(label, []).append(idx)

        manifolds: Dict[str, ClassGaussian] = {}
        for label, indices in by_class.items():
            block = features[indices]
            mu = block.mean(axis=0)
            centered = block - mu
            # Equivalent to sum_d Var(z_d) with population variance (ddof=0).
            trace = float(np.mean(np.sum(centered**2, axis=1)))
            manifolds[label] = ClassGaussian(
                label=label, mu=mu, trace=trace, n_samples=block.shape[0]
            )

        self._manifolds = manifolds

        logger.info(
            "PCMA fitted on %d samples across %d classes (D=%d, sigma^2=%.3g)",
            features.shape[0],
            len(manifolds),
            self._feature_dim,
            self.config.sigma_sq,
        )
        for label in sorted(manifolds):
            g = manifolds[label]
            logger.debug(
                "  class %-24s n=%-4d ||mu||=%.4f Tr(Sigma)=%.6f",
                label,
                g.n_samples,
                float(np.linalg.norm(g.mu)),
                g.trace,
            )
        return manifolds

    @property
    def is_fitted(self) -> bool:
        return bool(self._manifolds)

    @property
    def manifolds(self) -> Dict[str, ClassGaussian]:
        return dict(self._manifolds)

    def _require_fitted(self) -> None:
        if not self._manifolds:
            raise RuntimeError(
                "ProbabilisticAlignmentModule.fit() must be called before scoring"
            )

    # Scoring
    def wasserstein_sq(self, z: np.ndarray, label: str) -> float:
        """Return ``W2^2(z, N_label) = ||z - mu||^2 + Tr(Sigma)``."""
        self._require_fitted()
        manifold = self._manifolds.get(label)
        if manifold is None:
            raise KeyError(f"no Gaussian manifold estimated for class {label!r}")

        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if z.shape[0] != manifold.mu.shape[0]:
            raise ValueError(
                f"embedding dim {z.shape[0]} does not match manifold dim {manifold.mu.shape[0]}"
            )

        mean_term = float(np.sum((z - manifold.mu) ** 2))
        return mean_term + manifold.trace

    def typicality(self, z: np.ndarray, label: str) -> float:
        """Return ``S_typ(e) = exp(-W2^2 / sigma^2)`` in ``(0, 1]``."""
        w2 = self.wasserstein_sq(z, label)
        return float(np.exp(-w2 / self.config.sigma_sq))

    def typicality_batch(
        self,
        features: np.ndarray,
        labels: Sequence[str],
    ) -> np.ndarray:
        """Vectorised :meth:`typicality` over a batch of candidates.

        Candidates whose class has no estimated manifold receive a score of
        ``0.0``: an exemplar we cannot vouch for is not treated as typical.
        """
        self._require_fitted()
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if len(labels) != features.shape[0]:
            raise ValueError(
                f"labels length {len(labels)} does not match features rows {features.shape[0]}"
            )

        scores = np.zeros(features.shape[0], dtype=np.float64)
        # Group by label so each class mean/trace is touched once.
        index_by_label: Dict[str, List[int]] = {}
        for idx, label in enumerate(labels):
            index_by_label.setdefault(label, []).append(idx)

        for label, indices in index_by_label.items():
            manifold = self._manifolds.get(label)
            if manifold is None:
                logger.warning(
                    "no manifold for class %r; assigning S_typ=0 to %d candidate(s)",
                    label,
                    len(indices),
                )
                continue
            block = features[indices]
            mean_term = np.sum((block - manifold.mu[None, :]) ** 2, axis=1)
            w2 = mean_term + manifold.trace
            scores[indices] = np.exp(-w2 / self.config.sigma_sq)

        return scores

    # Diagnostics
    def summary(self) -> str:
        """Human-readable description of the fitted manifolds."""
        if not self._manifolds:
            return "PCMA: not fitted"
        lines = [
            f"PCMA: {len(self._manifolds)} class manifolds, "
            f"D={self._feature_dim}, sigma^2={self.config.sigma_sq:g}",
        ]
        for label in sorted(self._manifolds):
            g = self._manifolds[label]
            lines.append(
                f"  {label:<24s} n={g.n_samples:<4d} Tr(Sigma)={g.trace:.6f}"
            )
        return "\n".join(lines)
