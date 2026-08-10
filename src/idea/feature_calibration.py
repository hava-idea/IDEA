"""Dual-stream normalization and fusion for IDEA.

The frozen MLLM semantic stream and Structure Pooling stream are independently
L2-normalized, concatenated without explicit weights, and normalized again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def l2_normalize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a row-wise L2-normalized 2-D array."""
    array = np.asarray(features, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"features must be 1-D or 2-D, got shape {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


@dataclass
class CalibrationConfig:
    use_structural: bool = True


class FeatureCalibrator:
    """Create ``z = L2Norm([L2Norm(f_sem); L2Norm(f_str)])``."""

    def __init__(self, config: Optional[CalibrationConfig] = None) -> None:
        self.config = config or CalibrationConfig()
        self._sem_dim: Optional[int] = None
        self._str_dim: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        return self._sem_dim is not None

    @property
    def output_dim(self) -> int:
        if self._sem_dim is None:
            raise RuntimeError("calibrator not fitted")
        return self._sem_dim + (self._str_dim or 0)

    def fit(
        self,
        semantic: np.ndarray,
        structural: Optional[np.ndarray] = None,
    ) -> "FeatureCalibrator":
        """Record dimensions for compatibility; no statistics are estimated."""
        semantic = l2_normalize(semantic)
        self._sem_dim = semantic.shape[1]
        if self.config.use_structural:
            if structural is None:
                raise ValueError("structural features are required")
            structural = l2_normalize(structural)
            if structural.shape[0] != semantic.shape[0]:
                raise ValueError("semantic and structural stream lengths differ")
            self._str_dim = structural.shape[1]
        return self

    def transform(
        self,
        semantic: np.ndarray,
        structural: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        semantic = l2_normalize(semantic)
        if self._sem_dim is None:
            self._sem_dim = semantic.shape[1]
        if semantic.shape[1] != self._sem_dim:
            raise ValueError(
                f"semantic dim mismatch: expected {self._sem_dim}, got {semantic.shape[1]}"
            )
        if not self.config.use_structural:
            return semantic
        if structural is None:
            raise ValueError("structural features are required")
        structural = l2_normalize(structural)
        if structural.shape[0] != semantic.shape[0]:
            raise ValueError("semantic and structural stream lengths differ")
        if self._str_dim is None:
            self._str_dim = structural.shape[1]
        if structural.shape[1] != self._str_dim:
            raise ValueError(
                f"structural dim mismatch: expected {self._str_dim}, got {structural.shape[1]}"
            )
        return l2_normalize(np.concatenate([semantic, structural], axis=1))
