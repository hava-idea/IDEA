"""Dual-stream feature calibration (Section 3.1, Eq. 3-6).

The calibrated representation ``z`` is the single feature space in which every
downstream operation happens: FAISS indexing, PCMA typicality, SAC
discriminativeness, and the relational/global selection scores. Producing one
vector -- rather than keeping two streams and blending their similarities
separately -- is what makes the rest of the pipeline consistent with the paper.

Streams
-------
Semantic stream
    ``Phi_sem(x)`` in R^4096: the frozen MLLM's visual representation. In
    InternVL2.5-8B this is the output of ``extract_feature`` (the post
    pixel-shuffle, post-projection visual tokens), mean-pooled over the token
    axis. See :mod:`idea.mllm_backend`.

Structural stream
    ``Phi_str(x)`` in R^64: the LSConv backbone output (Eq. 3-5). Large-kernel
    perception captures geometric skeletons; small-kernel aggregation modulates
    local texture. See :mod:`idea.lsconv`.

Fusion (Eq. 6)
--------------
::

    z = L2Norm(Concat[w_sem * Phi_sem, w_str * Phi_str])

Two implementation details are needed to make Eq. 6 behave as intended, and both
are applied here explicitly rather than being left implicit:

1. **Per-stream standardisation.** ``Phi_sem`` and ``Phi_str`` come from
   different networks and have unrelated scales; ``Phi_str`` in particular is an
   unbounded post-BatchNorm activation. Concatenating raw vectors would let
   whichever stream happens to have the larger norm dominate, and ``w_sem`` /
   ``w_str`` would no longer control the trade-off. We therefore z-score each
   stream with statistics estimated on the candidate pool, then L2-normalise
   each stream to unit norm *before* applying the weights. After this, the
   weights are the only thing controlling the balance.

2. **Amplitude vs similarity weighting.** With unit-norm streams, the cosine
   similarity between two calibrated vectors decomposes exactly as::

       cos(z_a, z_b) = (w_sem^2 * cos_sem + w_str^2 * cos_str)
                       / (w_sem^2 + w_str^2)

   so Eq. 6's weights act on *amplitudes* and contribute in proportion to their
   squares. With ``w_sem=0.7, w_str=0.3`` the effective similarity weights are
   0.845 / 0.155, not 0.7 / 0.3. ``weight_mode`` exposes this choice:

   * ``"amplitude"`` (default) -- literal Eq. 6.
   * ``"similarity"`` -- pre-square-roots the weights so the effective
     similarity contributions equal the nominal ``w_sem`` / ``w_str``.

   The default is the literal reading of the paper. ``"similarity"`` exists
   because it is the interpretation people usually mean by "70/30 late fusion",
   and having both makes the distinction auditable instead of buried.

Statistics must be fitted on the candidate pool (training split) only, and the
same fitted transform is then applied to validation and test queries. Fitting on
anything other than the pool would leak information across splits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-8


@dataclass
class CalibrationConfig:
    """Configuration for :class:`FeatureCalibrator`."""

    w_sem: float = 0.7
    w_str: float = 0.3
    weight_mode: str = "amplitude"
    use_structural: bool = True
    standardise: bool = True

    def __post_init__(self) -> None:
        if self.weight_mode not in {"amplitude", "similarity"}:
            raise ValueError(
                f"weight_mode must be 'amplitude' or 'similarity', got {self.weight_mode!r}"
            )
        if self.w_sem < 0 or self.w_str < 0:
            raise ValueError("fusion weights must be non-negative")
        if self.w_sem == 0 and self.w_str == 0:
            raise ValueError("at least one fusion weight must be positive")
        if not self.use_structural and self.w_str != 0:
            logger.debug("use_structural=False; w_str=%.3f will be ignored", self.w_str)

    def effective_weights(self) -> Tuple[float, float]:
        """Weights actually applied to the unit-norm streams."""
        if self.weight_mode == "similarity":
            return float(np.sqrt(self.w_sem)), float(np.sqrt(self.w_str))
        return float(self.w_sem), float(self.w_str)

    def similarity_contributions(self) -> Tuple[float, float]:
        """Fraction of ``cos(z_a, z_b)`` contributed by each stream."""
        a, b = self.effective_weights()
        total = a * a + b * b
        if total <= 0:
            return 0.0, 0.0
        return a * a / total, b * b / total


class FeatureCalibrator:
    """Fuses the semantic and structural streams into the calibrated ``z``.

    Usage
    -----
    >>> calibrator = FeatureCalibrator(CalibrationConfig())
    >>> calibrator.fit(pool_semantic, pool_structural)   # candidate pool only
    >>> z_pool = calibrator.transform(pool_semantic, pool_structural)
    >>> z_query = calibrator.transform(query_semantic, query_structural)
    """

    def __init__(self, config: Optional[CalibrationConfig] = None) -> None:
        self.config = config or CalibrationConfig()
        self._sem_mean: Optional[np.ndarray] = None
        self._sem_std: Optional[np.ndarray] = None
        self._str_mean: Optional[np.ndarray] = None
        self._str_std: Optional[np.ndarray] = None
        self._fitted = False
        self._sem_dim: Optional[int] = None
        self._str_dim: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def output_dim(self) -> int:
        if not self._fitted:
            raise RuntimeError("calibrator not fitted")
        dim = int(self._sem_dim or 0)
        if self._uses_structural:
            dim += int(self._str_dim or 0)
        return dim

    def fit(
        self,
        semantic: np.ndarray,
        structural: Optional[np.ndarray] = None,
    ) -> "FeatureCalibrator":
        """Estimate per-stream standardisation statistics on the candidate pool.

        Parameters
        ----------
        semantic:
            ``(N, D_sem)`` frozen-MLLM features of the candidate pool.
        structural:
            ``(N, D_str)`` LSConv features of the same pool. May be ``None``
            when the structural stream is disabled (LSConv ablation).
        """
        semantic = self._as_2d(semantic, "semantic")
        self._sem_dim = semantic.shape[1]
        self._sem_mean = semantic.mean(axis=0)
        self._sem_std = self._safe_std(semantic)

        if self._uses_structural:
            if structural is None:
                raise ValueError(
                    "use_structural=True but no structural features were provided to fit()"
                )
            structural = self._as_2d(structural, "structural")
            if structural.shape[0] != semantic.shape[0]:
                raise ValueError(
                    f"stream length mismatch: semantic {semantic.shape[0]} vs "
                    f"structural {structural.shape[0]}"
                )
            self._str_dim = structural.shape[1]
            self._str_mean = structural.mean(axis=0)
            self._str_std = self._safe_std(structural)

        self._fitted = True

        sem_share, str_share = self.config.similarity_contributions()
        logger.info(
            "FeatureCalibrator fitted on %d pool samples: "
            "sem_dim=%s, str_dim=%s, z_dim=%d",
            semantic.shape[0],
            self._sem_dim,
            self._str_dim if self._uses_structural else "disabled",
            self.output_dim,
        )
        logger.info(
            "Fusion (Eq. 6): w_sem=%.3f, w_str=%.3f, mode=%s "
            "-> similarity contribution sem=%.1f%%, str=%.1f%%",
            self.config.w_sem,
            self.config.w_str,
            self.config.weight_mode,
            100.0 * sem_share,
            100.0 * str_share,
        )
        return self

    def transform(
        self,
        semantic: np.ndarray,
        structural: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Produce the calibrated representation ``z`` (Eq. 6).

        Returns
        -------
        ``(N, D_sem + D_str)`` L2-normalised array, or ``(N, D_sem)`` when the
        structural stream is disabled. Accepts a single vector and still returns
        a 2-D array, so callers can index uniformly.
        """
        if not self._fitted:
            raise RuntimeError("call fit() on the candidate pool before transform()")

        semantic = self._as_2d(semantic, "semantic")
        if semantic.shape[1] != self._sem_dim:
            raise ValueError(
                f"semantic dim mismatch: expected {self._sem_dim}, got {semantic.shape[1]}"
            )

        sem_weight, str_weight = self.config.effective_weights()
        sem_block = self._prepare_stream(semantic, self._sem_mean, self._sem_std)
        blocks = [sem_weight * sem_block]

        if self._uses_structural:
            if structural is None:
                raise ValueError(
                    "use_structural=True but no structural features were provided to transform()"
                )
            structural = self._as_2d(structural, "structural")
            if structural.shape[1] != self._str_dim:
                raise ValueError(
                    f"structural dim mismatch: expected {self._str_dim}, got {structural.shape[1]}"
                )
            if structural.shape[0] != semantic.shape[0]:
                raise ValueError(
                    f"stream length mismatch: semantic {semantic.shape[0]} vs "
                    f"structural {structural.shape[0]}"
                )
            str_block = self._prepare_stream(structural, self._str_mean, self._str_std)
            blocks.append(str_weight * str_block)

        fused = np.concatenate(blocks, axis=1)
        norms = np.linalg.norm(fused, axis=1, keepdims=True)
        return fused / np.maximum(norms, _EPS)

    def decompose_similarity(
        self,
        semantic_a: np.ndarray,
        structural_a: Optional[np.ndarray],
        semantic_b: np.ndarray,
        structural_b: Optional[np.ndarray],
    ) -> dict:
        """Split ``cos(z_a, z_b)`` into its per-stream contributions.

        Diagnostic helper. Useful for verifying that the structural stream is
        actually influencing retrieval rather than being numerically swamped.
        """
        z_a = self.transform(semantic_a, structural_a)[0]
        z_b = self.transform(semantic_b, structural_b)[0]

        sem_dim = int(self._sem_dim or 0)
        sem_part = float(np.dot(z_a[:sem_dim], z_b[:sem_dim]))
        str_part = float(np.dot(z_a[sem_dim:], z_b[sem_dim:])) if self._uses_structural else 0.0

        return {
            "total": sem_part + str_part,
            "semantic": sem_part,
            "structural": str_part,
        }

    @property
    def _uses_structural(self) -> bool:
        return self.config.use_structural and self.config.w_str > 0

    def _prepare_stream(
        self,
        features: np.ndarray,
        mean: Optional[np.ndarray],
        std: Optional[np.ndarray],
    ) -> np.ndarray:
        """Standardise (optional) then L2-normalise one stream to unit norm."""
        block = features
        if self.config.standardise and mean is not None and std is not None:
            block = (block - mean) / std
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        return block / np.maximum(norms, _EPS)

    @staticmethod
    def _safe_std(features: np.ndarray) -> np.ndarray:
        """Per-dimension std with near-constant dimensions neutralised.

        A randomly initialised convolutional stream produces some channels that
        are effectively constant. Dividing by their tiny std would amplify pure
        numerical noise into a dominant feature direction, so those dimensions
        are left unscaled instead.
        """
        std = features.std(axis=0)
        return np.where(std < 1e-6, 1.0, std)

    @staticmethod
    def _as_2d(features: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(features, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(f"{name} features must be 1-D or 2-D, got shape {array.shape}")
        return array
