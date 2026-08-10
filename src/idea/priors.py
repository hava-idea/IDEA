"""Retrieval priors estimated from the validation split.

This module owns the two priors used by the selection engine:

* ``C_hard`` -- the set of failure-prone classes used by ``S_hard``;
* ``H``      -- the per-class hard negative sets used by SAC.

Both are estimated from **zero-shot inference on the validation split**, exactly
as stated in Section 4.1 of the paper:

    "The validation split is used to identify the hard category set C_hard and
     hard negative set H through zero-shot validation."

Why this module exists as its own file
--------------------------------------
Keeping prior estimation in one place makes the data-flow constraint auditable:
the only split that may be read here is ``val``. The test split is the query set
and must never contribute to any prior, otherwise the hardness term and SAC would
carry test-set information into exemplar selection -- a transductive leak that
would invalidate the reported accuracy.

``estimate_priors`` therefore takes validation predictions and labels only, and
``Priors.assert_no_test_contamination`` provides a runtime check that callers can
use to make the guarantee explicit in logs and tests.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class ClassDifficulty:
    """Per-class zero-shot difficulty statistics measured on validation.

    Attributes
    ----------
    accuracy:
        Zero-shot accuracy for this class on the validation split.
    error_rate:
        ``1 - accuracy``.
    confusion_complexity:
        Number of distinct wrong classes predicted for this class, divided by the
        total number of classes. A class that is confused with many others is
        harder than one that is confused with a single neighbour, even at equal
        error rate.
    difficulty_score:
        ``error_rate + 0.5 * confusion_complexity``. A continuous difficulty
        measure retained for diagnostics only; it is not used by the paper's
        binary ``S_hard`` score.
    confused_with:
        Counter of predicted labels for the misclassified samples of this class.
    """

    label: str
    total: int
    correct: int
    accuracy: float
    error_rate: float
    confusion_complexity: float
    difficulty_score: float
    confused_with: Dict[str, int] = field(default_factory=dict)


@dataclass
class Priors:
    """Container for all validation-derived retrieval priors.

    ``source_split`` is recorded so that any downstream consumer -- and any
    reader of the logs -- can confirm the priors did not come from the test set.
    """

    classes: List[str]
    hard_classes: List[str]
    hard_negatives: Dict[str, List[str]]
    confusion: Dict[str, Dict[str, int]]
    difficulty: Dict[str, ClassDifficulty]
    hard_class_threshold: float
    hard_negative_k: int
    source_split: str = "val"
    n_samples: int = 0

    def is_hard(self, label: str) -> bool:
        """Binary indicator used by the paper's ``S_hard`` score."""
        return label in set(self.hard_classes)

    def difficulty_score(self, label: str) -> float:
        """Continuous difficulty in ``[0, 1.5]``; 0.0 for unseen classes."""
        entry = self.difficulty.get(label)
        return entry.difficulty_score if entry is not None else 0.0

    def assert_no_test_contamination(self) -> None:
        """Fail loudly if these priors were not built from validation data.

        Cheap runtime guard against the leakage class of bug. Called by the
        runner before selection begins and asserted in the test suite.
        """
        if self.source_split != "val":
            raise RuntimeError(
                "retrieval priors must be estimated on the validation split, "
                f"but source_split={self.source_split!r}. Estimating C_hard or H "
                "from the test split leaks test information into exemplar "
                "selection and invalidates the reported accuracy."
            )

    def summary(self) -> str:
        lines = [
            "Validation-derived retrieval priors",
            f"  source split          : {self.source_split} ({self.n_samples} samples)",
            f"  classes               : {len(self.classes)}",
            f"  hard-class threshold  : accuracy < {self.hard_class_threshold:.2f}",
            f"  |C_hard|              : {len(self.hard_classes)}",
            f"  hard negatives per cls: top-{self.hard_negative_k}",
        ]
        if self.hard_classes:
            lines.append("  C_hard:")
            for label in self.hard_classes:
                entry = self.difficulty.get(label)
                if entry is None:
                    lines.append(f"    {label}")
                else:
                    lines.append(
                        f"    {label:<24s} acc={entry.accuracy:.2%} "
                        f"difficulty={entry.difficulty_score:.3f}"
                    )
        return "\n".join(lines)

    def to_json(self, path: str | Path) -> None:
        """Persist priors so a run can be audited or replayed."""
        payload = {
            "source_split": self.source_split,
            "n_samples": self.n_samples,
            "classes": self.classes,
            "hard_class_threshold": self.hard_class_threshold,
            "hard_negative_k": self.hard_negative_k,
            "hard_classes": self.hard_classes,
            "hard_negatives": self.hard_negatives,
            "confusion": self.confusion,
            "difficulty": {
                label: {
                    "total": entry.total,
                    "correct": entry.correct,
                    "accuracy": entry.accuracy,
                    "error_rate": entry.error_rate,
                    "confusion_complexity": entry.confusion_complexity,
                    "difficulty_score": entry.difficulty_score,
                    "confused_with": entry.confused_with,
                }
                for label, entry in self.difficulty.items()
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("priors written to %s", path)


def build_confusion(
    true_labels: Sequence[str],
    predictions: Sequence[Optional[str]],
) -> Dict[str, Dict[str, int]]:
    """Build ``confusion[true][predicted] = count`` over misclassified samples.

    Samples whose prediction could not be parsed (``None``) are skipped rather
    than counted as errors, so an unparseable response never fabricates a
    confusion edge.
    """
    if len(true_labels) != len(predictions):
        raise ValueError(
            f"labels/predictions length mismatch: {len(true_labels)} vs {len(predictions)}"
        )

    confusion: Dict[str, Counter] = defaultdict(Counter)
    for truth, predicted in zip(true_labels, predictions):
        if predicted is None or predicted == truth:
            continue
        confusion[truth][predicted] += 1
    return {truth: dict(row) for truth, row in confusion.items()}


def estimate_priors(
    true_labels: Sequence[str],
    predictions: Sequence[Optional[str]],
    classes: Sequence[str],
    *,
    hard_class_threshold: float = 0.5,
    hard_negative_k: int = 5,
    source_split: str = "val",
) -> Priors:
    """Estimate ``C_hard`` and ``H`` from zero-shot validation results.

    Parameters
    ----------
    true_labels, predictions:
        Ground-truth labels and zero-shot predictions on the **validation**
        split. ``predictions`` may contain ``None`` for unparseable responses.
    classes:
        Full class list of the dataset.
    hard_class_threshold:
        A class enters ``C_hard`` when its validation accuracy falls below this
        value. Default 0.5, i.e. the frozen model is wrong more often than right.
    hard_negative_k:
        Number of hard negative classes retained per class for ``H``.
    source_split:
        Recorded for auditability. Must remain ``"val"``; the runner asserts this
        before selection so that a future refactor cannot silently pass test
        data.

    Returns
    -------
    Priors
    """
    if source_split != "val":
        raise ValueError(
            f"priors must be estimated from the validation split, got {source_split!r}"
        )
    if not 0.0 < hard_class_threshold <= 1.0:
        raise ValueError(
            f"hard_class_threshold must lie in (0, 1], got {hard_class_threshold}"
        )

    classes = list(classes)
    confusion = build_confusion(true_labels, predictions)

    totals: Counter = Counter()
    corrects: Counter = Counter()
    for truth, predicted in zip(true_labels, predictions):
        totals[truth] += 1
        if predicted == truth:
            corrects[truth] += 1

    n_classes = max(len(classes), 1)
    difficulty: Dict[str, ClassDifficulty] = {}
    for label in classes:
        total = totals.get(label, 0)
        if total == 0:
            continue
        correct = corrects.get(label, 0)
        accuracy = correct / total
        error_rate = 1.0 - accuracy
        confused_with = confusion.get(label, {})
        confusion_complexity = len(confused_with) / n_classes
        difficulty[label] = ClassDifficulty(
            label=label,
            total=total,
            correct=correct,
            accuracy=accuracy,
            error_rate=error_rate,
            confusion_complexity=confusion_complexity,
            difficulty_score=error_rate + 0.5 * confusion_complexity,
            confused_with=dict(confused_with),
        )

    hard_classes = sorted(
        label for label, entry in difficulty.items() if entry.accuracy < hard_class_threshold
    )

    hard_negatives: Dict[str, List[str]] = {}
    class_set = set(classes)
    for label in classes:
        row = confusion.get(label, {})
        ranked = sorted(
            (
                (other, count)
                for other, count in row.items()
                if other != label and other in class_set
            ),
            key=lambda item: (-item[1], item[0]),
        )
        hard_negatives[label] = [other for other, _ in ranked[:hard_negative_k]]

    evaluated = sum(totals.values())
    logger.info(
        "priors estimated on %s: %d samples, |C_hard|=%d/%d (acc < %.2f)",
        source_split,
        evaluated,
        len(hard_classes),
        len(classes),
        hard_class_threshold,
    )

    priors = Priors(
        classes=classes,
        hard_classes=hard_classes,
        hard_negatives=hard_negatives,
        confusion=confusion,
        difficulty=difficulty,
        hard_class_threshold=hard_class_threshold,
        hard_negative_k=hard_negative_k,
        source_split=source_split,
        n_samples=len(true_labels),
    )
    priors.assert_no_test_contamination()
    return priors


def empty_priors(classes: Sequence[str], *, hard_negative_k: int = 5) -> Priors:
    """Priors with no hard classes and no confusions.

    Used by ablations that disable the hardness term and by baselines, which by
    protocol must not receive any IDEA-specific prior.
    """
    classes = list(classes)
    return Priors(
        classes=classes,
        hard_classes=[],
        hard_negatives={label: [] for label in classes},
        confusion={},
        difficulty={},
        hard_class_threshold=0.5,
        hard_negative_k=hard_negative_k,
        source_split="val",
        n_samples=0,
    )
