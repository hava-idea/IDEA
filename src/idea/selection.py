"""Budget-constrained adaptive decision engine from Section 3.3.

This module assembles the final many-shot context. It sits on top of the
Intrinsic Layer (PCMA typicality, SAC discriminativeness) and implements the
Relational and Global layers:

**Relational Layer** -- query-aware similarity folds intrinsic quality
into query proximity, so a candidate scores well only when it is both close to
the query *and* reliable as a demonstration::

    S_sim(e, q) = w1 * S_typ(e) + w2 * S_disc(e) + sim(z_e, z_q)

**Global Layer** -- sequence-level terms control redundancy, label
skew and hard-class coverage, and an MMR-style loop picks exemplars one at a
time::

    S_div(e, S_t) = 1 - max_{e' in S_t} sim(z_e, z_e')
    S_bal(e, S_t) = 1 / (Freq(label(e), S_t) + 1)
    S_hard(e)     = 1[y_e in C_hard]

    e* = argmax_{e in C \\ S_t} [ a*S_sim + b*S_bal + g*S_div + d*S_hard ]

Selection is genuinely iterative: ``S_div`` and ``S_bal`` depend on what has
already been chosen, so the ranking is recomputed at every step. A single
sorted pass would drop both terms and reduce the engine to plain retrieval.

Both budgets are enforced. A candidate is admissible only while the shot count
is under ``max_shots`` *and* its token cost fits the remaining token budget,
where the cost includes the exemplar's image tokens -- 256 per 448x448 tile
under InternVL2.5's pixel-shuffle -- not just its text.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

logger = logging.getLogger(__name__)

#: Visual tokens contributed by one 448x448 tile under InternVL2.5's
#: pixel-shuffle (32x32 patches -> 0.5 downsample -> 16x16 = 256 tokens).
IMAGE_TOKEN_COST = 256


@dataclass
class Candidate:
    """An annotated pool item eligible for selection.

    Attributes
    ----------
    id:
        Unique identifier.
    image_path:
        Path to the image, used when building the prompt.
    label:
        Ground-truth class.
    embedding:
        Calibrated representation ``z``, L2-normalised.
    text_token_cost:
        Tokens contributed by this exemplar's text, excluding image tokens.
    """

    id: str
    image_path: str
    label: str
    embedding: np.ndarray
    text_token_cost: int = 0

    @property
    def token_cost(self) -> int:
        """Total prompt cost: text tokens plus this exemplar's image tokens."""
        return self.text_token_cost + IMAGE_TOKEN_COST


@dataclass
class SelectionWeights:
    """Decision weights for relational and global selection scores.

    Attributes
    ----------
    alpha, beta, gamma, delta:
        Priorities of similarity, balance, diversity and hardness.
    omega1, omega2:
        Intrinsic quality adjustment coefficients on typicality and
        discriminativeness in the relational score.
    """

    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.2
    delta: float = 0.3
    omega1: float = 0.2
    omega2: float = 0.15

    def as_dict(self) -> Dict[str, float]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "omega1": self.omega1,
            "omega2": self.omega2,
        }


@dataclass(frozen=True)
class AdaptiveDiagnostics:
    """Validation diagnostics used by the one-pass weight update."""

    r_bal: float
    r_div: float
    r_hard: float
    r_var: float
    r_conf: float

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


def normalize_to_sum(values: np.ndarray, target_sum: float) -> np.ndarray:
    """Scale non-negative values while preserving a requested total."""
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("cannot normalize weights with a non-positive sum")
    return values * target_sum / total


def adapt_selection_weights(
    base: SelectionWeights,
    diagnostics: AdaptiveDiagnostics,
) -> SelectionWeights:
    """Apply the paper's one-pass multiplicative reliability update."""
    global_base = np.array(
        [base.alpha, base.beta, base.gamma, base.delta], dtype=np.float64
    )
    global_reliability = np.array(
        [0.0, diagnostics.r_bal, diagnostics.r_div, diagnostics.r_hard]
    )
    global_new = normalize_to_sum(
        global_base * (1.0 + global_reliability), global_base.sum()
    )

    intrinsic_base = np.array([base.omega1, base.omega2], dtype=np.float64)
    intrinsic_new = normalize_to_sum(
        intrinsic_base
        * np.array([1.0 + diagnostics.r_var, 1.0 + diagnostics.r_conf]),
        intrinsic_base.sum(),
    )
    return SelectionWeights(*map(float, [*global_new, *intrinsic_new]))


def imbalance_diagnostic(labels: Sequence[str], n_classes: int) -> float:
    """Return ``1 - H(p) / log(|C|)`` for context labels."""
    if not labels or n_classes <= 1:
        return 0.0
    counts = np.asarray(list(Counter(labels).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return float(np.clip(1.0 - entropy / np.log(n_classes), 0.0, 1.0))


def redundancy_diagnostic(embeddings: np.ndarray) -> float:
    """Mean nearest-neighbor cosine redundancy mapped to ``[0, 1]``."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or len(embeddings) < 2:
        return 0.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-8)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    return float(np.clip(np.mean((1.0 + similarities.max(axis=1)) / 2.0), 0, 1))


def compute_adaptive_diagnostics(
    *,
    context_labels: Sequence[Sequence[str]],
    context_embeddings: Sequence[np.ndarray],
    train_embeddings: np.ndarray,
    train_labels: Sequence[str],
    val_true: Sequence[str],
    val_predictions: Sequence[Optional[str]],
    hard_classes: Set[str],
    n_classes: int,
) -> AdaptiveDiagnostics:
    """Compute the five validation diagnostics used for reweighting."""
    r_bal_values = [imbalance_diagnostic(labels, n_classes) for labels in context_labels]
    r_div_values = [redundancy_diagnostic(block) for block in context_embeddings]

    hard_indices = [i for i, label in enumerate(val_true) if label in hard_classes]
    r_hard = (
        1.0
        - np.mean([val_predictions[i] == val_true[i] for i in hard_indices])
        if hard_indices
        else 0.0
    )

    train_embeddings = np.asarray(train_embeddings, dtype=np.float64)
    mean_variances = []
    for label in sorted(set(train_labels)):
        mask = np.asarray([item == label for item in train_labels])
        block = train_embeddings[mask]
        # Mean intra-class feature variance: (1 / D) * sum_d Var(z_d).
        mean_variances.append(float(np.mean(np.var(block, axis=0))))

    class_errors = []
    for label in sorted(set(val_true)):
        indices = [i for i, truth in enumerate(val_true) if truth == label]
        class_errors.append(
            float(
                np.mean(
                    [
                        val_predictions[i] is not None
                        and val_predictions[i] != label
                        for i in indices
                    ]
                )
            )
        )

    return AdaptiveDiagnostics(
        r_bal=float(np.mean(r_bal_values)) if r_bal_values else 0.0,
        r_div=float(np.mean(r_div_values)) if r_div_values else 0.0,
        r_hard=float(r_hard),
        r_var=(
            float(np.clip(np.mean(mean_variances), 0, 1))
            if mean_variances
            else 0.0
        ),
        r_conf=float(np.mean(class_errors)) if class_errors else 0.0,
    )


@dataclass
class SelectionTrace:
    """Diagnostics for one selection call, useful for auditing context quality."""

    selected_ids: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    token_cost: int = 0
    stopped_on: str = "shot_budget"
    per_step_scores: List[float] = field(default_factory=list)

    @property
    def label_distribution(self) -> Dict[str, int]:
        return dict(Counter(self.labels))

    @property
    def num_classes_covered(self) -> int:
        return len(set(self.labels))

    @property
    def label_entropy(self) -> float:
        """Shannon entropy (nats) of the selected label distribution.

        A low value means the prompt is dominated by a few classes, which is the
        failure mode ``S_bal`` is designed to mitigate.
        """
        if not self.labels:
            return 0.0
        counts = np.array(list(Counter(self.labels).values()), dtype=np.float64)
        probs = counts / counts.sum()
        return float(-(probs * np.log(probs)).sum())


class AdaptiveSelectionEngine:
    """Iterative MMR-style exemplar selection under shot and token budgets.

    Parameters
    ----------
    candidates:
        The annotated candidate pool. Embeddings must be the calibrated ``z``
        vectors and must all share a dimension.
    typicality:
        ``S_typ`` per candidate id, from :mod:`idea.probabilistic_alignment`.
        Missing ids default to 0.
    discriminativeness:
        ``S_disc`` per candidate id, from :mod:`idea.self_supervised_alignment`.
        Missing ids default to 0.
    hard_classes:
        ``C_hard``, estimated from the validation split. Drives ``S_hard``.
    weights:
        Decision weights.
    """

    def __init__(
        self,
        candidates: Sequence[Candidate],
        typicality: Optional[Dict[str, float]] = None,
        discriminativeness: Optional[Dict[str, float]] = None,
        hard_classes: Optional[Set[str]] = None,
        weights: Optional[SelectionWeights] = None,
    ) -> None:
        if not candidates:
            raise ValueError("candidate pool is empty")

        self.candidates = list(candidates)
        self.weights = weights or SelectionWeights()
        self.hard_classes = set(hard_classes or ())

        dims = {c.embedding.shape[-1] for c in self.candidates}
        if len(dims) != 1:
            raise ValueError(f"candidate embeddings have inconsistent dimensions: {sorted(dims)}")

        # Pool matrix for vectorised similarity. Embeddings are already
        # L2-normalised by the calibrator, so a dot product is cosine.
        self._pool = np.asarray(
            [c.embedding for c in self.candidates], dtype=np.float32
        )
        self._labels = np.asarray([c.label for c in self.candidates], dtype=object)
        self._token_costs = np.asarray([c.token_cost for c in self.candidates], dtype=np.int64)

        typ = typicality or {}
        disc = discriminativeness or {}
        self._typicality = np.asarray(
            [float(typ.get(c.id, 0.0)) for c in self.candidates], dtype=np.float32
        )
        self._discriminativeness = np.asarray(
            [float(disc.get(c.id, 0.0)) for c in self.candidates], dtype=np.float32
        )
        self._is_hard = np.asarray(
            [1.0 if c.label in self.hard_classes else 0.0 for c in self.candidates],
            dtype=np.float32,
        )

        missing_typ = sum(1 for c in self.candidates if c.id not in typ)
        missing_disc = sum(1 for c in self.candidates if c.id not in disc)
        if missing_typ or missing_disc:
            logger.warning(
                "intrinsic scores missing for some candidates "
                "(typicality: %d, discriminativeness: %d of %d) -- defaulting to 0",
                missing_typ,
                missing_disc,
                len(self.candidates),
            )

    def relational_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute relational similarity for every candidate.

        Parameters
        ----------
        query_embedding:
            Calibrated query representation ``z_q``, L2-normalised.

        Returns
        -------
        ``(N,)`` array of query-aware similarity scores.
        """
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._pool.shape[1]:
            raise ValueError(
                f"query dimension {query.shape[0]} does not match pool dimension "
                f"{self._pool.shape[1]}"
            )

        cosine = self._pool @ query
        return (
            self.weights.omega1 * self._typicality
            + self.weights.omega2 * self._discriminativeness
            + cosine
        )

    def select(
        self,
        query_embedding: np.ndarray,
        max_shots: int,
        token_budget: Optional[int] = None,
        exclude_ids: Optional[Set[str]] = None,
    ) -> tuple[List[Candidate], SelectionTrace]:
        """Build the exemplar sequence for one query.

        Parameters
        ----------
        query_embedding:
            Calibrated query representation ``z_q``.
        max_shots:
            Shot budget: the maximum number of exemplars to place in the prompt.
        token_budget:
            Optional cumulative token budget. Each candidate consumes
            ``text_token_cost + IMAGE_TOKEN_COST``. Selection stops when the next
            best admissible candidate would exceed the budget. ``None`` means the
            shot budget alone applies.
        exclude_ids:
            Candidate ids to skip, e.g. to prevent a query from retrieving
            itself when the pool and query set overlap.

        Returns
        -------
        ``(selected, trace)`` -- the chosen candidates in prompt order, and
        diagnostics including the realised label distribution and token cost.
        """
        if max_shots <= 0:
            return [], SelectionTrace(stopped_on="shot_budget")

        base_scores = self.relational_similarity(query_embedding)

        num_candidates = len(self.candidates)
        available = np.ones(num_candidates, dtype=bool)
        if exclude_ids:
            for index, candidate in enumerate(self.candidates):
                if candidate.id in exclude_ids:
                    available[index] = False

        # Running max similarity to the selected set.
        # An empty selected set means no redundancy penalty, so S_div = 1.
        max_sim_to_selected = np.full(num_candidates, -1.0, dtype=np.float32)
        label_counts: Counter = Counter()

        selected: List[Candidate] = []
        trace = SelectionTrace()
        spent_tokens = 0

        while len(selected) < max_shots:
            if not available.any():
                trace.stopped_on = "pool_exhausted"
                break

            # S_div: 1 - max similarity to anything already selected.
            diversity = 1.0 - max_sim_to_selected
            if not selected:
                diversity = np.ones(num_candidates, dtype=np.float32)

            # S_bal: inverse frequency of the candidate's label in S_t.
            balance = np.asarray(
                [1.0 / (label_counts[label] + 1) for label in self._labels],
                dtype=np.float32,
            )

            utility = (
                self.weights.alpha * base_scores
                + self.weights.beta * balance
                + self.weights.gamma * diversity
                + self.weights.delta * self._is_hard
            )

            admissible = available.copy()
            if token_budget is not None:
                remaining = token_budget - spent_tokens
                admissible &= self._token_costs <= remaining
            if not admissible.any():
                trace.stopped_on = "token_budget"
                break

            masked = np.where(admissible, utility, -np.inf)
            best = int(np.argmax(masked))

            candidate = self.candidates[best]
            selected.append(candidate)
            trace.selected_ids.append(candidate.id)
            trace.labels.append(candidate.label)
            trace.per_step_scores.append(float(utility[best]))

            available[best] = False
            label_counts[candidate.label] += 1
            spent_tokens += int(self._token_costs[best])
            max_sim_to_selected = np.maximum(max_sim_to_selected, self._pool @ candidate.embedding)

        trace.token_cost = spent_tokens
        if len(selected) >= max_shots:
            trace.stopped_on = "shot_budget"

        logger.debug(
            "selected %d exemplars (%s), %d classes, %d tokens, label entropy %.3f",
            len(selected),
            trace.stopped_on,
            trace.num_classes_covered,
            trace.token_cost,
            trace.label_entropy,
        )
        return selected, trace
