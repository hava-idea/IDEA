"""Self-Supervised Alignment Contrast (SAC).

Implements the SAC contrastive score defined in Section 3.2.

SAC answers a different question from PCMA. PCMA asks whether a candidate is
*typical* of its own class. SAC asks whether a candidate is *separable* from the
classes that the frozen MLLM actually confuses it with. A candidate can be a
perfectly typical member of its class and still be a poor demonstration if it
sits on the boundary with a confusable class.

Definitions
-----------
For a candidate ``e`` with ground-truth class ``c``:

Initial contrast::

    S_disc_init(e) = sim(z_e, z_c^+) - max_{j in H(c)} sim(z_e, z_j^-)

Normalised contrastive alignment probability::

                          exp(sim(z_e, z_c^+) / tau)
    S_disc(e) = ---------------------------------------------------------
                exp(sim(z_e, z_c^+) / tau) + sum_j exp(sim(z_e, z_j^-) / tau)

where

* ``z_c^+`` is the prototype of the ground-truth class, i.e. the mean embedding
  of all candidate-pool examples of that class;
* ``H(c)`` is the hard negative class set for ``c``, estimated from zero-shot
  confusion on the *validation* split (never the test split);
* ``tau`` is the temperature (0.07 by default, following the SimCLR/MoCo
  convention). Lower ``tau`` makes the score more sensitive to the boundary.

``S_disc`` lies in (0, 1). It approaches 1 when the candidate is much closer to
its own prototype than to any hard negative prototype, and drops toward 0 when a
hard negative prototype is closer -- exactly the case we want to suppress.

Note on scope
-------------
This module only *scores* candidates. It performs no gradient updates: the
prototypes are means over frozen features and ``H`` is a lookup table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _l2_normalise(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalisation."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


@dataclass
class ClassPrototypes:
    """Per-class prototype embeddings ``z_c^+``.

    The prototype is the mean of the calibrated embeddings of every
    candidate-pool example belonging to that class.
    Prototypes are L2-normalised so that ``sim`` is a plain dot product.
    """

    prototypes: Dict[str, np.ndarray] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    dim: int = 0

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        labels: Sequence[str],
    ) -> "ClassPrototypes":
        """Estimate prototypes from the candidate pool.

        Parameters
        ----------
        features:
            ``(N, D)`` calibrated embeddings ``z`` of the candidate pool
            (the training split only).
        labels:
            Length-``N`` ground-truth labels.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"expected 2-D features, got shape {features.shape}")
        if len(labels) != features.shape[0]:
            raise ValueError(
                f"features/labels length mismatch: {features.shape[0]} vs {len(labels)}"
            )

        prototypes: Dict[str, np.ndarray] = {}
        counts: Dict[str, int] = {}
        for label in sorted(set(labels)):
            mask = np.fromiter((lab == label for lab in labels), dtype=bool, count=len(labels))
            member = features[mask]
            centroid = member.mean(axis=0)
            prototypes[label] = _l2_normalise(centroid)[0]
            counts[label] = int(mask.sum())

        logger.info(
            "SAC prototypes fitted: %d classes, dim=%d, min/max members=%d/%d",
            len(prototypes),
            features.shape[1],
            min(counts.values()) if counts else 0,
            max(counts.values()) if counts else 0,
        )
        return cls(prototypes=prototypes, counts=counts, dim=features.shape[1])

    @property
    def classes(self) -> List[str]:
        return sorted(self.prototypes)

    def get(self, label: str) -> np.ndarray:
        if label not in self.prototypes:
            raise KeyError(f"no prototype for class {label!r}")
        return self.prototypes[label]

    def similarity_to_all(self, embedding: np.ndarray) -> Dict[str, float]:
        """Cosine similarity from one embedding to every class prototype."""
        vector = _l2_normalise(embedding)[0]
        return {
            label: float(np.dot(vector, proto))
            for label, proto in self.prototypes.items()
        }


@dataclass
class HardNegativeSet:
    """The validation-confusion hard negative class sets ``H(c)``.

    ``H`` is estimated from zero-shot confusion on the validation split: for
    class ``c`` we keep the classes that the frozen MLLM most often predicts when
    the true label is ``c``. This is a prior about the *model's* confusions, which
    is why it must come from validation data and not from the test queries.
    """

    hard_negatives: Dict[str, List[str]] = field(default_factory=dict)
    top_k: int = 5
    source: str = "validation_confusion"

    @classmethod
    def from_confusion(
        cls,
        confusion: Mapping[str, Mapping[str, int]],
        classes: Sequence[str],
        top_k: int = 5,
    ) -> "HardNegativeSet":
        """Build ``H`` from a validation confusion table.

        Parameters
        ----------
        confusion:
            ``confusion[true_label][predicted_label] = count`` for
            ``predicted_label != true_label``, measured on the validation split.
        classes:
            The full class list, used to keep the output well-defined for
            classes with no observed confusion.
        top_k:
            Number of hard negative classes to retain per class.
        """
        hard: Dict[str, List[str]] = {}
        for label in classes:
            row = confusion.get(label, {})
            ranked = sorted(
                ((other, count) for other, count in row.items() if other != label and other in classes),
                key=lambda item: (-item[1], item[0]),
            )
            hard[label] = [other for other, _ in ranked[:top_k]]

        empty = [label for label, negatives in hard.items() if not negatives]
        if empty:
            logger.info(
                "SAC: %d/%d classes had no validation confusion; "
                "S_disc is 1.0 for those classes (%s)",
                len(empty),
                len(classes),
                ", ".join(empty[:5]) + ("..." if len(empty) > 5 else ""),
            )
        return cls(hard_negatives=hard, top_k=top_k)

    def get(self, label: str) -> List[str]:
        return self.hard_negatives.get(label, [])

    def summary(self) -> str:
        lines = [f"HardNegativeSet (source={self.source}, top_k={self.top_k})"]
        for label in sorted(self.hard_negatives):
            negatives = self.hard_negatives[label]
            lines.append(f"  {label:<24s} <- {', '.join(negatives) if negatives else '(none)'}")
        return "\n".join(lines)


@dataclass
class SACConfig:
    """Configuration for the SAC scorer."""

    temperature: float = 0.07
    hard_negative_k: int = 5
    use_hard_negatives: bool = True

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.hard_negative_k < 1:
            raise ValueError(f"hard_negative_k must be >= 1, got {self.hard_negative_k}")


class SACScorer:
    """Computes the SAC discriminativeness score ``S_disc``.

    Example
    -------
    >>> prototypes = ClassPrototypes.fit(pool_z, pool_labels)
    >>> hard = HardNegativeSet.from_confusion(val_confusion, classes, top_k=5)
    >>> scorer = SACScorer(prototypes, hard, SACConfig())
    >>> scores = scorer.score_batch(candidate_z, candidate_labels)
    """

    def __init__(
        self,
        prototypes: ClassPrototypes,
        hard_negative_set: HardNegativeSet,
        config: Optional[SACConfig] = None,
    ) -> None:
        self.prototypes = prototypes
        self.hard_negatives = hard_negative_set
        self.config = config or SACConfig()

    def initial_contrast(self, embedding: np.ndarray, label: str) -> float:
        """Return ``sim(z_e, z_c^+) - max_j sim(z_e, z_j^-)``.

        Kept as a separate method because it is the interpretable form (a signed
        margin), whereas the normalized form is used in scoring.
        """
        vector = _l2_normalise(embedding)[0]
        positive = float(np.dot(vector, self.prototypes.get(label)))

        negatives = self._negative_labels(label)
        if not negatives:
            return positive

        hardest = max(
            float(np.dot(vector, self.prototypes.get(other))) for other in negatives
        )
        return positive - hardest

    def score(self, embedding: np.ndarray, label: str) -> float:
        """Return the normalized contrastive alignment probability in (0, 1]."""
        vector = _l2_normalise(embedding)[0]
        tau = self.config.temperature

        positive = float(np.dot(vector, self.prototypes.get(label)))
        negatives = self._negative_labels(label)

        # Subtract the max logit before exponentiating: mathematically identical
        # to the definition but avoids overflow, which matters because tau is small
        # (sim/tau reaches ~14 at tau=0.07).
        logits = [positive / tau]
        logits.extend(
            float(np.dot(vector, self.prototypes.get(other))) / tau for other in negatives
        )
        shift = max(logits)
        exponentials = np.exp(np.asarray(logits) - shift)
        return float(exponentials[0] / exponentials.sum())

    def score_batch(
        self,
        embeddings: np.ndarray,
        labels: Sequence[str],
    ) -> np.ndarray:
        """Vectorised ``S_disc`` over many candidates.

        Returns
        -------
        ``(N,)`` array of scores in (0, 1).
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if len(labels) != embeddings.shape[0]:
            raise ValueError(
                f"embeddings/labels length mismatch: {embeddings.shape[0]} vs {len(labels)}"
            )
        return np.array(
            [self.score(embeddings[i], labels[i]) for i in range(embeddings.shape[0])],
            dtype=np.float64,
        )

    def diagnostics(self, scores: np.ndarray) -> Dict[str, float]:
        """Summary statistics of a score array.

        ``S_disc`` collapsing to a narrow band near 1.0 means the hard negative
        prototypes are too far away to matter -- either ``H`` is badly estimated
        or ``tau`` is too large for this feature space. Logging the spread makes
        that failure visible instead of silent.
        """
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size == 0:
            return {}
        return {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "dynamic_range": float(scores.max() - scores.min()),
        }

    def _negative_labels(self, label: str) -> List[str]:
        if not self.config.use_hard_negatives:
            return []
        return [
            other
            for other in self.hard_negatives.get(label)
            if other in self.prototypes.prototypes and other != label
        ]
