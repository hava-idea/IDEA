"""Baseline exemplar selection methods for comparison against IDEA.

Methods
-------
* **Random** -- uniform random sample (no embeddings needed).
* **Cosine** -- top-k nearest neighbors by cosine similarity via FAISS.
* **RICES** -- Retrieval-based ICL with class-balanced round-robin selection.
  See Yang et al., 2023 (https://arxiv.org/abs/2209.01511).
* **DiverseICL** -- greedy farthest-first diversity selection.

All methods share the same call signature::

    selected = method_select(
        query_embedding,
        candidates,
        k=20,
        token_budget=None,
    )

where *candidates* is a list of :class:`~idea.selection.Candidate` and the
return value is an ordered list of the selected candidates (prompt order:
first = earliest in context).

Token budget
------------
Every image contributes ``IMAGE_TOKEN_COST = 256`` visual tokens. Text cost is
stored on each ``Candidate``. Selection stops early if adding the next candidate
would exceed ``token_budget``; the returned list may therefore contain fewer than
``k`` items. Pass ``token_budget=None`` to disable the budget.

FAISS index note
----------------
``IndexFlatIP`` computes inner products. With L2-normalised vectors this equals
cosine similarity. The ``search`` call returns **(distances, indices)** where
``distances`` are the inner-product scores in **descending** order (highest =
most similar). Do NOT apply ``1 - score`` before ranking; that would invert the
order and select the *least* similar examples.
"""

from __future__ import annotations

import logging
import random as _random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Visual tokens consumed per image (InternVL2.5 pixel-shuffle 448×448 → 256).
IMAGE_TOKEN_COST = 256


def _fits_budget(selected: list, candidate, token_budget: Optional[int]) -> bool:
    """Return True if adding *candidate* keeps total tokens within budget."""
    if token_budget is None:
        return True
    used = sum(IMAGE_TOKEN_COST + c.text_token_cost for c in selected)
    return used + IMAGE_TOKEN_COST + candidate.text_token_cost <= token_budget


# Random baseline

def random_select(
    query_embedding: np.ndarray,  # unused, kept for uniform signature
    candidates: list,
    k: int = 20,
    token_budget: Optional[int] = None,
    seed: Optional[int] = None,
) -> list:
    """Select *k* examples by uniform random sampling without replacement.

    Parameters
    ----------
    query_embedding:
        Ignored. Present only to match the common baseline signature.
    candidates:
        Pool of :class:`~idea.selection.Candidate` objects.
    k:
        Maximum number of examples to return.
    token_budget:
        Hard cap on total token cost (images + text). ``None`` = no cap.
    seed:
        Optional RNG seed for reproducibility.
    """
    rng = _random.Random(seed)
    pool = candidates[:]
    rng.shuffle(pool)
    selected = []
    for cand in pool:
        if len(selected) >= k:
            break
        if _fits_budget(selected, cand, token_budget):
            selected.append(cand)
    return selected


# Cosine (top-k nearest neighbours)

def cosine_select(
    query_embedding: np.ndarray,
    candidates: list,
    k: int = 20,
    token_budget: Optional[int] = None,
) -> list:
    """Select *k* candidates most similar to the query by cosine similarity.

    Uses a FAISS ``IndexFlatIP`` over L2-normalised embeddings. The index is
    built from *candidates* at call time, so there is no persistent state.

    Bug note
    --------
    The intermediate version of this code applied ``1 - scores`` before sorting,
    which reversed the ranking and caused the *least* similar examples to be
    selected. This function uses the raw inner-product scores from FAISS, which
    are already in descending similarity order.
    """
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("cosine_select requires faiss. Run: pip install faiss-cpu") from exc

    if not candidates:
        return []

    embeddings = np.stack([c.embedding for c in candidates], axis=0).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    embeddings = embeddings / norms

    q = query_embedding.astype(np.float32).reshape(1, -1)
    q_norm = np.linalg.norm(q)
    if q_norm > 1e-12:
        q = q / q_norm

    n, d = embeddings.shape
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    n_retrieve = min(n, k * 4)  # over-retrieve to leave room for budget filter
    scores, idxs = index.search(q, n_retrieve)
    # scores[0] is sorted descending — highest inner product (= cosine) first.
    # No inversion needed.

    selected = []
    for idx in idxs[0]:
        if idx < 0:
            break
        if len(selected) >= k:
            break
        cand = candidates[int(idx)]
        if _fits_budget(selected, cand, token_budget):
            selected.append(cand)

    return selected


# RICES (class-balanced nearest-neighbour retrieval)

def rices_select(
    query_embedding: np.ndarray,
    candidates: list,
    k: int = 20,
    token_budget: Optional[int] = None,
) -> list:
    """Retrieval-based ICL with class-balanced round-robin selection (RICES).

    Algorithm
    ---------
    1. For each class, rank its candidates by cosine similarity to the query.
    2. Iterate round-robin over classes (sorted alphabetically for
       determinism), taking the top-ranked unused candidate from each class.
    3. Stop when *k* examples have been collected or the budget is exhausted.

    This produces a diverse, class-balanced selection without requiring the
    diversity metric of DiverseICL, and mirrors the RICES baseline used in the
    paper's Table 2.
    """
    if not candidates:
        return []

    # Group by class
    by_class: Dict[str, list] = defaultdict(list)
    for cand in candidates:
        by_class[cand.label].append(cand)

    q = query_embedding.astype(np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm > 1e-12:
        q = q / q_norm

    # Within each class, rank by cosine similarity descending
    ranked: Dict[str, list] = {}
    for label, pool in by_class.items():
        embs = np.stack([c.embedding for c in pool], axis=0).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        embs = embs / norms
        sims = embs @ q
        order = np.argsort(-sims)
        ranked[label] = [pool[i] for i in order]

    class_order = sorted(ranked.keys())
    pointers = {c: 0 for c in class_order}
    selected = []

    while len(selected) < k:
        advanced = False
        for label in class_order:
            if len(selected) >= k:
                break
            ptr = pointers[label]
            pool = ranked[label]
            # Advance past already-exhausted candidates
            while ptr < len(pool):
                cand = pool[ptr]
                ptr += 1
                if _fits_budget(selected, cand, token_budget):
                    selected.append(cand)
                    pointers[label] = ptr
                    advanced = True
                    break
            else:
                pointers[label] = ptr
        if not advanced:
            break

    return selected


# DiverseICL (greedy farthest-first selection)

def diverse_select(
    query_embedding: np.ndarray,
    candidates: list,
    k: int = 20,
    token_budget: Optional[int] = None,
) -> list:
    """Greedy farthest-first diversity selection (DiverseICL).

    Algorithm
    ---------
    Given a candidate pool C, let S be the selected set (empty initially).

    At each step, select the candidate ``e*`` that maximises the minimum cosine
    distance to any already-selected example::

        e* = argmax_{e in C \\ S}  min_{s in S}  (1 - cos(e, s))

    When ``S`` is empty, the first candidate is chosen by cosine similarity to
    the query. Each later demonstration maximizes its minimum cosine distance
    to the selected set.

    Runtime: O(k * |C|) inner products -- acceptable for |C| <= a few thousand.
    """
    if not candidates:
        return []

    embs = np.stack([c.embedding for c in candidates], axis=0).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    embs = embs / norms  # L2-normalised; inner product = cosine similarity

    q = query_embedding.astype(np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm > 1e-12:
        q = q / q_norm

    n = len(candidates)
    selected_indices: List[int] = []
    available = list(range(n))

    # Cosine similarity from each candidate to the query, used for tie-breaking
    # and for seeding the first selection.
    query_sims = embs @ q  # (n,)

    # min_dist[i] = min cosine distance from candidate i to the selected set.
    # Initialised to inf (nothing selected yet). Distance = 1 - cos_sim.
    min_dist = np.full(n, np.inf)

    for step in range(k):
        if not available:
            break

        if step == 0:
            # Seed: pick the candidate most similar to the query.
            avail_arr = np.array(available)
            best_local = int(np.argmax(query_sims[avail_arr]))
            chosen = available[best_local]
        else:
            avail_arr = np.array(available)
            scores = min_dist[avail_arr]
            best_local = int(np.argmax(scores))
            chosen = available[best_local]

        cand = candidates[chosen]
        if not _fits_budget(
            [candidates[i] for i in selected_indices], cand, token_budget
        ):
            # Skip this candidate and try the next best.
            available.remove(chosen)
            continue

        selected_indices.append(chosen)
        available.remove(chosen)

        # Update min distances using the newly-selected candidate's similarities.
        new_sims = embs @ embs[chosen]  # cosine similarities to chosen
        new_dists = 1.0 - new_sims     # cosine distances
        min_dist = np.minimum(min_dist, new_dists)

    return [candidates[i] for i in selected_indices]
